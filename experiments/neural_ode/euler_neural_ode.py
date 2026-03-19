"""
Neural ODE Architecture Sweep — Compressible Euler

Systematic comparison of 3 neural ODE architectures for learning reduced-order
dynamics of the compressible Euler equations:
  - Small:  2 hidden layers × 64 units, tanh
  - Medium: 3 hidden layers × 128 units, tanh  (baseline from 05)
  - Large:  4 hidden layers × 256 units, tanh

Training: 5000 steps with cosine LR annealing (1e-3 → 0), 20-member ensembles.
Uses EXACT same data pipeline as experiments/euler/04_conditional_integral.py.

Data regimes:
  1. Dense data, low noise    (250 samples, 1% noise)
  2. Sparse data, low noise   (55 samples, 3% noise)
  3. Dense data, high noise   (250 samples, 10% noise)

Usage:
    python euler_neural_ode.py                     # run all architectures × regimes
    python euler_neural_ode.py --arch small medium  # selected architectures
    python euler_neural_ode.py --regime dense_low_noise
    python euler_neural_ode.py --save-data          # generate .npz data without training
"""

import sys
import os
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import diffrax
import optax
from scipy.interpolate import interp1d

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EULER_DIR = os.path.join(SCRIPT_DIR, '..', 'euler')
ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, EULER_DIR)
sys.path.insert(0, ROOT_DIR)

import config
from config import Basis
from core import generate_trajectory

# ── Configuration ────────────────────────────────────────────────────────────
SCHEMAS = [
    {"name": "dense_low_noise",  "label": "Dense data, low noise",  "NUM_SAMPLES": 250, "NOISE_LEVEL": 0.01},
    {"name": "sparse_low_noise", "label": "Sparse data, low noise", "NUM_SAMPLES": 55,  "NOISE_LEVEL": 0.03},
    {"name": "dense_high_noise", "label": "Dense data, high noise", "NUM_SAMPLES": 250, "NOISE_LEVEL": 0.10},
]

ARCHITECTURES = {
    "small":  {"hidden_dim": 64,  "num_layers": 2, "color": "tab:green"},
    "medium": {"hidden_dim": 128, "num_layers": 3, "color": "tab:orange"},
    "large":  {"hidden_dim": 256, "num_layers": 4, "color": "tab:red"},
}

TRAINING_SPAN = (0, 0.08)
PREDICTION_SPAN = (0, 0.15)
NUM_MODES = 6
ENSEMBLE_SIZE = 20
NUM_TRAIN_STEPS = 5000
INIT_LR = 1e-3
SEED = 42
NUM_EVAL_POINTS = 400

FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# Neural ODE model
# =============================================================================
class NeuralODE(eqx.Module):
    """MLP dynamics: dq/dt = f_theta(q), with tanh activations."""
    layers: list

    def __init__(self, in_dim, hidden_dim, num_layers, *, key):
        keys = random.split(key, num_layers + 1)
        self.layers = []
        d_in = in_dim
        for i in range(num_layers):
            self.layers.append(eqx.nn.Linear(d_in, hidden_dim, key=keys[i]))
            d_in = hidden_dim
        self.layers.append(eqx.nn.Linear(hidden_dim, in_dim, key=keys[num_layers]))

    def __call__(self, t, y, args):
        x = y
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


def count_params(model):
    """Count trainable parameters in an Equinox model."""
    params = eqx.filter(model, eqx.is_array)
    return sum(p.size for p in jax.tree.leaves(params))


# =============================================================================
# Training (with cosine LR schedule)
# =============================================================================
def train_single(model, t_train, y_train, q0, num_steps, key):
    """Train one neural ODE via trajectory matching with cosine LR annealing.
    Returns (model, losses).
    """
    t_train_jnp = jnp.array(t_train)
    y_train_jnp = jnp.array(y_train)  # (num_modes, num_samples)
    q0_jnp = jnp.array(q0)

    # Precompute constants outside JIT to avoid ConcretizationTypeError
    t0 = float(t_train[0])
    t1 = float(t_train[-1])
    dt0 = float(t_train[1] - t_train[0])

    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_train_jnp)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    schedule = optax.cosine_decay_schedule(init_value=INIT_LR, decay_steps=num_steps)
    opt = optax.adam(schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(model):
        term = diffrax.ODETerm(model)
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=t0, t1=t1,
            dt0=dt0,
            y0=q0_jnp,
            saveat=saveat,
            adjoint=adjoint,
            max_steps=16384,
            throw=False,
        )
        y_pred = sol.ys  # (len(t_train), num_modes)
        return jnp.mean((y_pred - y_train_jnp.T) ** 2)

    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state_new = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss

    recorded_losses = []
    for i in range(num_steps):
        model, opt_state, loss = step(model, opt_state)
        if i % 500 == 0 or i == num_steps - 1:
            loss_val = float(loss)
            recorded_losses.append(loss_val)
            print(f"      step {i:5d}/{num_steps}  loss={loss_val:.6f}", flush=True)

    return model, recorded_losses


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_ensemble(trained_models, q0, t_pred, t_full, true_comp, t_samp):
    """Evaluate an ensemble of trained models. Returns results dict."""
    t_pred_jnp = jnp.array(t_pred)
    q0_jnp = jnp.array(q0)

    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_pred_jnp)

    # Precompute constants outside any JIT context
    t0_pred = float(t_pred[0])
    t1_pred = float(t_pred[-1])
    dt0_pred = float(t_pred[1] - t_pred[0])

    rom_solves = []
    for model in trained_models:
        try:
            term = diffrax.ODETerm(model)
            sol = diffrax.diffeqsolve(
                term, solver,
                t0=t0_pred, t1=t1_pred,
                dt0=dt0_pred,
                y0=q0_jnp,
                saveat=saveat,
                max_steps=16384,
                throw=False,
            )
            traj = np.array(sol.ys).T  # (num_modes, len(t_pred))
            if np.all(np.isfinite(traj)):
                rom_solves.append(traj)
        except Exception:
            pass

    n_stable = len(rom_solves)
    n_total = len(trained_models)
    stability_pct = n_stable / max(n_total, 1) * 100

    train_error = pred_error = float('inf')
    ci_coverage = ci_width = float('nan')

    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        train_mask = t_pred <= TRAINING_SPAN[1]
        pred_mask = t_pred > TRAINING_SPAN[1]

        ti = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)

        train_error = float(np.linalg.norm(rom_med[:, train_mask] - ta[:, train_mask]) /
                           np.linalg.norm(ta[:, train_mask]))
        pred_error = float(np.linalg.norm(rom_med[:, pred_mask] - ta[:, pred_mask]) /
                          np.linalg.norm(ta[:, pred_mask]))

        q05 = np.percentile(rom_arr, 5, axis=0)
        q95 = np.percentile(rom_arr, 95, axis=0)
        ci_width = float(np.mean(q95 - q05))
        ci_coverage = float(np.mean((ta >= q05) & (ta <= q95)))

    return {
        'rom_solves': rom_solves,
        'n_stable': n_stable, 'n_total': n_total,
        'stability_pct': stability_pct,
        'train_error': train_error, 'pred_error': pred_error,
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
    }


# =============================================================================
# Plotting
# =============================================================================
def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_trajectories(result, arch_name, arch_cfg, save_dir):
    """Plot 6-mode trajectory comparison (train/pred/full columns)."""
    plt = _setup_matplotlib()

    rom_solves = result['rom_solves']
    if len(rom_solves) == 0:
        return

    snaps_comp = result['snaps_comp']
    true_comp = result['true_comp']
    t_full = result['t_full']
    t_pred = result['t_pred']
    t_samp = result['t_samp']
    schema = result['schema']
    color = arch_cfg['color']

    rom_arr = np.array(rom_solves)
    rom_med = np.median(rom_arr, axis=0)
    rom_q05 = np.percentile(rom_arr, 5, axis=0)
    rom_q95 = np.percentile(rom_arr, 95, axis=0)

    true_interp = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
    true_at_pred = true_interp(t_pred)

    train_end = TRAINING_SPAN[1]
    train_mask = t_pred <= train_end
    pred_mask = t_pred > train_end
    t_train_win = t_pred[train_mask]
    t_pred_win = t_pred[pred_mask]

    fig, ax = plt.subplots(NUM_MODES, 3, figsize=(15, 2.5 * NUM_MODES),
                           sharey='row', sharex='col')
    if NUM_MODES == 1:
        ax = ax.reshape(1, -1)

    for i in range(NUM_MODES):
        # Training window
        ax[i, 0].plot(t_samp, snaps_comp[i], 'k*', ms=3, label='Obs')
        ax[i, 0].plot(t_train_win, true_at_pred[i, train_mask],
                     color='tab:gray', lw=1.5, label='Truth')
        ax[i, 0].plot(t_train_win, rom_med[i, train_mask],
                     color=color, ls='--', lw=2, alpha=0.9, label='Median')
        ax[i, 0].fill_between(t_train_win,
                              rom_q05[i, train_mask], rom_q95[i, train_mask],
                              color=color, alpha=0.15, label='90% CI')
        ax[i, 0].set_ylabel(f'Mode {i}')

        # Prediction window
        ax[i, 1].plot(t_pred_win, true_at_pred[i, pred_mask], color='tab:gray', lw=1.5)
        ax[i, 1].plot(t_pred_win, rom_med[i, pred_mask],
                     color=color, ls='--', lw=2, alpha=0.9)
        ax[i, 1].fill_between(t_pred_win,
                              rom_q05[i, pred_mask], rom_q95[i, pred_mask],
                              color=color, alpha=0.15)

        # Full span
        ax[i, 2].plot(t_samp, snaps_comp[i], 'k*', ms=3)
        ax[i, 2].plot(t_pred, true_at_pred[i], color='tab:gray', lw=1.5)
        ax[i, 2].plot(t_pred, rom_med[i], color=color, ls='--', lw=2, alpha=0.9)
        ax[i, 2].fill_between(t_pred, rom_q05[i], rom_q95[i],
                              color=color, alpha=0.15)
        ax[i, 2].axvline(train_end, color='k', ls=':', lw=0.8, alpha=0.5)

        yvals = true_at_pred[i]
        ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
        pad = max(abs(ymax - ymin) * 0.3, 1e-6)
        for j in range(3):
            ax[i, j].set_ylim(ymin - pad, ymax + pad)

    ax[0, 0].set_title('Training Window')
    ax[0, 1].set_title('Prediction Window')
    ax[0, 2].set_title('Full Span')
    ax[0, 0].legend(fontsize=7, loc='upper right')
    for j in range(3):
        ax[-1, j].set_xlabel('Time')

    dim_str = f"{arch_cfg['num_layers']}×{arch_cfg['hidden_dim']}"
    fig.suptitle(f"Neural ODE ({arch_name} {dim_str}) — {schema['label']}", fontsize=14)
    fig.tight_layout()
    path = os.path.join(save_dir, f"euler_{arch_name}_{schema['name']}_trajectories.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


def plot_loss(losses, arch_name, arch_cfg, schema, save_dir):
    """Plot training loss convergence."""
    plt = _setup_matplotlib()

    mean_loss = np.mean(losses, axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(mean_loss, lw=0.8, color=arch_cfg['color'])
    ax[0].set_xlabel('Training Step')
    ax[0].set_ylabel('MSE Loss')
    ax[0].set_title('Loss Convergence (mean over ensemble)')
    ax[0].grid(True, alpha=0.3)

    half = len(mean_loss) // 2
    ax[1].plot(range(half, len(mean_loss)), mean_loss[half:], lw=0.8, color=arch_cfg['color'])
    ax[1].set_xlabel('Training Step')
    ax[1].set_ylabel('MSE Loss')
    ax[1].set_title('Loss (last 50%)')
    ax[1].grid(True, alpha=0.3)

    dim_str = f"{arch_cfg['num_layers']}×{arch_cfg['hidden_dim']}"
    fig.suptitle(f"Training Loss — {arch_name} ({dim_str}) — {schema['label']}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(save_dir, f"euler_{arch_name}_{schema['name']}_loss.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


def plot_architecture_comparison(all_results, save_dir):
    """Bar chart comparing all architectures across data regimes."""
    plt = _setup_matplotlib()

    metrics = ['train_error', 'pred_error', 'ci_coverage', 'stability_pct']
    metric_labels = ['Train Error', 'Pred Error', 'CI Coverage', 'Stability %']

    regime_names = [s['label'] for s in SCHEMAS]
    arch_names = list(ARCHITECTURES.keys())
    n_regimes = len(regime_names)
    n_archs = len(arch_names)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    x = np.arange(n_regimes)
    width = 0.25

    for ax, metric, label in zip(axes, metrics, metric_labels):
        for j, arch in enumerate(arch_names):
            vals = []
            for schema in SCHEMAS:
                key = (arch, schema['name'])
                if key in all_results:
                    v = all_results[key].get(metric, float('nan'))
                    if metric in ('train_error', 'pred_error'):
                        v = v * 100  # convert to percentage
                    vals.append(v)
                else:
                    vals.append(float('nan'))
            color = ARCHITECTURES[arch]['color']
            ax.bar(x + j * width, vals, width, label=arch, color=color, alpha=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(regime_names, fontsize=8, rotation=15, ha='right')
        if metric in ('train_error', 'pred_error'):
            ax.set_ylabel(f'{label} (%)')
        elif metric == 'ci_coverage':
            ax.set_ylabel(label)
            ax.axhline(0.9, color='k', ls=':', lw=0.8, alpha=0.5, label='90% target')
        else:
            ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Neural ODE Architecture Comparison — Euler', fontsize=14)
    fig.tight_layout()
    path = os.path.join(save_dir, "euler_architecture_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\n  📊 Saved: {path}")
    plt.close(fig)


# =============================================================================
# Data save/load helpers (Kaggle compatibility)
# =============================================================================
def save_data(regime_data, path):
    """Save generated data as .npz for Kaggle use."""
    np.savez(path, **regime_data)
    print(f"  💾 Saved data: {path}")


def load_data(path):
    """Load pre-saved data from .npz file."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def generate_regime_data(schema):
    """Generate data for one regime. Returns dict of arrays."""
    np.random.seed(SEED)
    (fom, t_full, true_states, t_samp, snaps_samp) = \
        generate_trajectory(config, config.time_domain, TRAINING_SPAN,
                           schema['NUM_SAMPLES'], schema['NOISE_LEVEL'])
    basis = Basis(num_vectors=NUM_MODES)
    basis.fit(snaps_samp)
    snaps_comp = basis.compress(snaps_samp)
    true_comp = basis.compress(true_states)
    q0 = snaps_comp[:, 0]

    return {
        't_full': t_full,
        't_samp': t_samp,
        'snaps_comp': snaps_comp,
        'true_comp': true_comp,
        'q0': q0,
        'cumulative_energy': np.array(basis.cumulative_energy),
    }


# =============================================================================
# Run single architecture × regime experiment
# =============================================================================
def run_experiment(schema, arch_name, arch_cfg):
    """Run one architecture on one data regime. Returns results dict."""
    np.random.seed(SEED)

    noise_level = schema['NOISE_LEVEL']
    num_samples = schema['NUM_SAMPLES']
    dim_str = f"{arch_cfg['num_layers']}×{arch_cfg['hidden_dim']}"

    print(f"\n{'─'*70}")
    print(f"  {arch_name} ({dim_str}) — {schema['label']}  "
          f"({num_samples} samples, {noise_level:.0%} noise)")
    print(f"{'─'*70}")

    # ── Data generation ──────────────────────────────────────────────────
    data_path = os.path.join(SCRIPT_DIR, f"data_euler_{schema['name']}.npz")
    if os.path.exists(data_path):
        print(f"  Loading pre-saved data: {data_path}")
        regime_data = load_data(data_path)
        t_full = regime_data['t_full']
        t_samp = regime_data['t_samp']
        snaps_comp = regime_data['snaps_comp']
        true_comp = regime_data['true_comp']
        q0 = regime_data['q0']
        cum_energy = float(regime_data['cumulative_energy'])
    else:
        regime_data = generate_regime_data(schema)
        t_full = regime_data['t_full']
        t_samp = regime_data['t_samp']
        snaps_comp = regime_data['snaps_comp']
        true_comp = regime_data['true_comp']
        q0 = regime_data['q0']
        cum_energy = float(regime_data['cumulative_energy'])

    print(f"  POD energy: {cum_energy:.4%}")

    # ── Count parameters ─────────────────────────────────────────────────
    dummy_key = random.PRNGKey(0)
    dummy_model = NeuralODE(
        in_dim=NUM_MODES,
        hidden_dim=arch_cfg['hidden_dim'],
        num_layers=arch_cfg['num_layers'],
        key=dummy_key,
    )
    n_params = count_params(dummy_model)
    print(f"  Parameters: {n_params:,}")

    # ── Train ensemble ───────────────────────────────────────────────────
    print(f"  Training {ENSEMBLE_SIZE} ensemble members ({NUM_TRAIN_STEPS} steps, "
          f"cosine LR {INIT_LR}→0)...", flush=True)
    t0 = time.time()
    trained_models = []
    all_losses = []

    for m in range(ENSEMBLE_SIZE):
        key = random.PRNGKey(SEED + m)
        model = NeuralODE(
            in_dim=NUM_MODES,
            hidden_dim=arch_cfg['hidden_dim'],
            num_layers=arch_cfg['num_layers'],
            key=key,
        )
        model, losses = train_single(model, t_samp, snaps_comp, q0, NUM_TRAIN_STEPS, key)
        trained_models.append(model)
        all_losses.append(losses)
        print(f"    member {m+1:2d}/{ENSEMBLE_SIZE}  final_loss={losses[-1]:.6f}")

    runtime = time.time() - t0
    print(f"  Ensemble training took {runtime:.0f}s")

    # ── Evaluate ─────────────────────────────────────────────────────────
    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], NUM_EVAL_POINTS)
    eval_result = evaluate_ensemble(trained_models, q0, t_pred, t_full, true_comp, t_samp)

    print(f"\n  Results ({runtime:.0f}s):")
    print(f"    Stability: {eval_result['n_stable']}/{eval_result['n_total']} "
          f"({eval_result['stability_pct']:.0f}%)")
    print(f"    Train error: {eval_result['train_error']:.4%}  |  "
          f"Pred error: {eval_result['pred_error']:.4%}")
    print(f"    CI coverage: {eval_result['ci_coverage']:.2%} (target: 90%)")
    mean_final_loss = np.mean([l[-1] for l in all_losses])
    print(f"    Mean final loss: {mean_final_loss:.6f}")

    result = {
        **eval_result,
        'schema': schema,
        'arch_name': arch_name,
        'arch_cfg': arch_cfg,
        'n_params': n_params,
        'runtime': runtime,
        'losses': all_losses,
        'snaps_comp': snaps_comp,
        'true_comp': true_comp,
        't_full': t_full,
        't_pred': t_pred,
        't_samp': t_samp,
    }
    return result


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Neural ODE Architecture Sweep — Euler')
    parser.add_argument('--arch', nargs='+', choices=list(ARCHITECTURES.keys()),
                       default=list(ARCHITECTURES.keys()),
                       help='Architectures to test (default: all)')
    parser.add_argument('--regime', nargs='+',
                       choices=[s['name'] for s in SCHEMAS],
                       default=[s['name'] for s in SCHEMAS],
                       help='Data regimes to test (default: all)')
    parser.add_argument('--save-data', action='store_true',
                       help='Generate and save .npz data files without training')
    args = parser.parse_args()

    # ── Save-data mode ───────────────────────────────────────────────────
    if args.save_data:
        print("Generating and saving data files...")
        for schema in SCHEMAS:
            if schema['name'] not in args.regime:
                continue
            data_path = os.path.join(SCRIPT_DIR, f"data_euler_{schema['name']}.npz")
            regime_data = generate_regime_data(schema)
            save_data(regime_data, data_path)
        print("Done. Use these .npz files on Kaggle.")
        return

    # ── Filter selections ────────────────────────────────────────────────
    selected_schemas = [s for s in SCHEMAS if s['name'] in args.regime]
    selected_archs = {k: v for k, v in ARCHITECTURES.items() if k in args.arch}

    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("=" * 70)
    print("Neural ODE Architecture Sweep — Compressible Euler")
    print("=" * 70)
    print(f"Architectures: {list(selected_archs.keys())}")
    print(f"Regimes:       {len(selected_schemas)}")
    for s in selected_schemas:
        print(f"  • {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  noise={s['NOISE_LEVEL']:.0%}")
    print(f"Training:  steps={NUM_TRAIN_STEPS}, lr=cosine({INIT_LR}→0), "
          f"ensemble={ENSEMBLE_SIZE}")

    # ── Run all experiments ──────────────────────────────────────────────
    all_results = {}
    for schema in selected_schemas:
        for arch_name, arch_cfg in selected_archs.items():
            result = run_experiment(schema, arch_name, arch_cfg)
            plot_trajectories(result, arch_name, arch_cfg, FIGURE_DIR)
            plot_loss(result['losses'], arch_name, arch_cfg, schema, FIGURE_DIR)
            all_results[(arch_name, schema['name'])] = result

    # ── Comparison plot ──────────────────────────────────────────────────
    if len(all_results) > 1:
        plot_architecture_comparison(all_results, FIGURE_DIR)

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print(f"ARCHITECTURE COMPARISON — Euler")
    print(f"{'='*90}")
    print(f"{'Arch':<8s} {'Regime':<28s} {'Params':>7s} {'Train':>8s} "
          f"{'Pred':>8s} {'CI_cov':>7s} {'Stab':>5s} {'Time':>6s}")
    print(f"{'-'*8} {'-'*28} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*5} {'-'*6}")
    for schema in selected_schemas:
        for arch_name in selected_archs:
            key = (arch_name, schema['name'])
            if key not in all_results:
                continue
            r = all_results[key]
            print(f"{arch_name:<8s} {schema['label']:<28s} {r['n_params']:>7,d} "
                  f"{r['train_error']:>7.2%} {r['pred_error']:>7.2%} "
                  f"{r['ci_coverage']:>6.1%} {r['stability_pct']:>4.0f}% "
                  f"{r['runtime']:>5.0f}s")


if __name__ == "__main__":
    main()

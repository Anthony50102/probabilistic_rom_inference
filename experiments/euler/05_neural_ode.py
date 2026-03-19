"""
05 — Neural ODE Ensemble Baseline (Compressible Euler)

Black-box baseline for comparison with Bayesian OpInf (04):
  - MLP: 6 → 128 → 128 → 128 → 6  (tanh activations)
  - Training: trajectory matching via MSE against noisy reduced observations
  - UQ: ensemble of 20 independently trained networks

Uses the EXACT same data pipeline as 04_conditional_integral.py:
  1. FOM solve → subsample in training span → add noise → POD compression
  2. Evaluate on t_pred = linspace(0, 0.15, 400), compare with cubic-interpolated truth

Data regimes (same as 04):
  1. Dense data, low noise    (250 samples, 1% noise)
  2. Sparse data, low noise   (55 samples, 3% noise)
  3. Dense data, high noise   (250 samples, 10% noise)

Usage:
    python 05_neural_ode.py                  # run all 3 regimes
    python 05_neural_ode.py dense_low_noise  # run one regime
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import diffrax
import optax
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis
from core import generate_trajectory

# ── Data regime definitions ──────────────────────────────────────────────────
SCHEMAS = [
    {
        "name": "dense_low_noise",
        "label": "Dense data, low noise",
        "NUM_SAMPLES": 250,
        "NOISE_LEVEL": 0.01,
        "NUM_EVAL_POINTS": 400,
    },
    {
        "name": "sparse_low_noise",
        "label": "Sparse data, low noise",
        "NUM_SAMPLES": 55,
        "NOISE_LEVEL": 0.03,
        "NUM_EVAL_POINTS": 200,
    },
    {
        "name": "dense_high_noise",
        "label": "Dense data, high noise",
        "NUM_SAMPLES": 250,
        "NOISE_LEVEL": 0.10,
        "NUM_EVAL_POINTS": 400,
    },
]

# ── Model hyperparameters ────────────────────────────────────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=6,
    HIDDEN_DIM=128,
    NUM_LAYERS=3,
    ACTIVATION='tanh',
    ENSEMBLE_SIZE=20,
    NUM_TRAIN_STEPS=2000,
    LEARNING_RATE=1e-3,
    SEED=42,
)

TRAINING_SPAN = (0, 0.08)
PREDICTION_SPAN = (0, 0.15)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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


# =============================================================================
# Training
# =============================================================================
def train_single(model, t_train, y_train, q0, num_steps, lr, key):
    """Train one neural ODE via trajectory matching. Returns (model, losses)."""
    t_train_jnp = jnp.array(t_train)
    y_train_jnp = jnp.array(y_train)  # (num_modes, num_samples)
    q0_jnp = jnp.array(q0)

    # Precompute constants outside JIT to avoid tracing errors
    t0 = float(t_train[0])
    t1 = float(t_train[-1])
    dt0 = float(t_train[1] - t_train[0])

    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_train_jnp)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    opt = optax.adam(lr)
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

    losses = []
    for i in range(num_steps):
        model, opt_state, loss = step(model, opt_state)
        losses.append(float(loss))

    return model, losses


# =============================================================================
# Run experiment
# =============================================================================
def run_experiment(schema):
    """Run one data regime. Returns results dict."""
    p = MODEL_PARAMS
    np.random.seed(p['SEED'])

    noise_level = schema['NOISE_LEVEL']
    num_samples = schema['NUM_SAMPLES']
    num_modes = p['NUM_MODES']

    print(f"\n{'='*70}")
    print(f"  {schema['label']}  ({num_samples} samples, {noise_level:.0%} noise)")
    print(f"{'='*70}")

    # ── Data generation (EXACTLY as in 04) ───────────────────────────────
    (fom, t_full, true_states, t_samp, snaps_samp) = \
        generate_trajectory(config, config.time_domain, TRAINING_SPAN, num_samples, noise_level)
    basis = Basis(num_vectors=num_modes)
    basis.fit(snaps_samp)
    snaps_comp = basis.compress(snaps_samp)   # (6, num_samples)
    true_comp = basis.compress(true_states)   # (6, len(t_full))
    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    q0 = snaps_comp[:, 0]

    # ── Train ensemble ───────────────────────────────────────────────────
    ensemble_size = p['ENSEMBLE_SIZE']
    num_steps = p['NUM_TRAIN_STEPS']
    lr = p['LEARNING_RATE']
    base_key = random.PRNGKey(p['SEED'])

    print(f"  Training {ensemble_size} ensemble members ({num_steps} steps each)...")
    t0 = time.time()
    trained_models = []
    all_losses = []

    for m in range(ensemble_size):
        key = random.PRNGKey(p['SEED'] + m)
        model = NeuralODE(
            in_dim=num_modes,
            hidden_dim=p['HIDDEN_DIM'],
            num_layers=p['NUM_LAYERS'],
            key=key,
        )
        model, losses = train_single(model, t_samp, snaps_comp, q0, num_steps, lr, key)
        trained_models.append(model)
        all_losses.append(losses)
        final_loss = losses[-1]
        print(f"    member {m+1:2d}/{ensemble_size}  final_loss={final_loss:.6f}")

    runtime = time.time() - t0
    print(f"  Ensemble training took {runtime:.0f}s")

    # ── Evaluate ─────────────────────────────────────────────────────────
    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    t_pred_jnp = jnp.array(t_pred)
    q0_jnp = jnp.array(q0)

    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_pred_jnp)

    rom_solves = []
    t0_pred = float(t_pred[0])
    t1_pred = float(t_pred[-1])
    dt0_pred = float(t_pred[1] - t_pred[0])
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
    n_total = ensemble_size
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

    print(f"\n  Results ({runtime:.0f}s):")
    print(f"    Stability: {n_stable}/{n_total} ({stability_pct:.0f}%)")
    print(f"    Train error: {train_error:.4%}  |  Pred error: {pred_error:.4%}")
    print(f"    CI coverage: {ci_coverage:.2%} (target: 90%)")
    mean_final_loss = np.mean([l[-1] for l in all_losses])
    print(f"    Mean final loss: {mean_final_loss:.6f}")

    return {
        'schema': schema,
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'n_stable': n_stable, 'n_total': n_total,
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'runtime': runtime, 'losses': all_losses,
        'rom_solves': rom_solves,
        'snaps_comp': snaps_comp, 'true_comp': true_comp,
        't_full': t_full, 't_pred': t_pred, 't_samp': t_samp,
        'training_span': TRAINING_SPAN, 'num_modes': num_modes,
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_results(result, save_dir=None):
    """Generate ROM trajectory, loss convergence, and ensemble spread plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    schema = result['schema']
    prefix = f"05_{schema['name']}"
    losses = result['losses']  # list of lists (per ensemble member)
    rom_solves = result['rom_solves']
    snaps_comp = result['snaps_comp']
    true_comp = result['true_comp']
    t_full = result['t_full']
    t_pred = result['t_pred']
    t_samp = result['t_samp']
    training_span = result['training_span']
    num_modes = result['num_modes']

    # ── 1. ROM Trajectory Plot (3-column) ────────────────────────────
    if len(rom_solves) > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        rom_q05 = np.percentile(rom_arr, 5, axis=0)
        rom_q95 = np.percentile(rom_arr, 95, axis=0)

        true_interp = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        true_at_pred = true_interp(t_pred)

        train_end = training_span[1]
        train_mask = t_pred <= train_end
        pred_mask = t_pred > train_end
        t_train_win = t_pred[train_mask]
        t_pred_win = t_pred[pred_mask]

        fig, ax = plt.subplots(num_modes, 3, figsize=(15, 2.5 * num_modes),
                               sharey='row', sharex='col')
        if num_modes == 1:
            ax = ax.reshape(1, -1)

        for i in range(num_modes):
            # Training window
            ax[i, 0].plot(t_samp, snaps_comp[i], 'k*', ms=3, label='Obs')
            ax[i, 0].plot(t_train_win, true_at_pred[i, train_mask],
                         color='tab:gray', lw=1.5, label='Truth')
            ax[i, 0].plot(t_train_win, rom_med[i, train_mask],
                         color='tab:orange', ls='--', lw=2, alpha=0.9, label='Median')
            ax[i, 0].fill_between(t_train_win,
                                  rom_q05[i, train_mask], rom_q95[i, train_mask],
                                  color='tab:orange', alpha=0.15, label='90% CI')
            ax[i, 0].set_ylabel(f'Mode {i}')

            # Prediction window
            ax[i, 1].plot(t_pred_win, true_at_pred[i, pred_mask], color='tab:gray', lw=1.5)
            ax[i, 1].plot(t_pred_win, rom_med[i, pred_mask],
                         color='tab:orange', ls='--', lw=2, alpha=0.9)
            ax[i, 1].fill_between(t_pred_win,
                                  rom_q05[i, pred_mask], rom_q95[i, pred_mask],
                                  color='tab:orange', alpha=0.15)

            # Full span
            ax[i, 2].plot(t_samp, snaps_comp[i], 'k*', ms=3)
            ax[i, 2].plot(t_pred, true_at_pred[i], color='tab:gray', lw=1.5)
            ax[i, 2].plot(t_pred, rom_med[i], color='tab:orange', ls='--', lw=2, alpha=0.9)
            ax[i, 2].fill_between(t_pred, rom_q05[i], rom_q95[i],
                                  color='tab:orange', alpha=0.15)
            ax[i, 2].axvline(train_end, color='k', ls=':', lw=0.8, alpha=0.5)

            # y-limits from truth
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

        fig.suptitle(f"Neural ODE — {schema['label']}", fontsize=14)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_rom_trajectories.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)

    # ── 2. Training Loss Plot ────────────────────────────────────────
    mean_loss = np.mean(losses, axis=0)
    fig_loss, ax_loss = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss[0].plot(mean_loss, lw=0.8, color='tab:blue')
    ax_loss[0].set_xlabel('Training Step')
    ax_loss[0].set_ylabel('MSE Loss')
    ax_loss[0].set_title('Loss Convergence (mean over ensemble)')
    ax_loss[0].grid(True, alpha=0.3)
    half = len(mean_loss) // 2
    ax_loss[1].plot(range(half, len(mean_loss)), mean_loss[half:], lw=0.8, color='tab:blue')
    ax_loss[1].set_xlabel('Training Step')
    ax_loss[1].set_ylabel('MSE Loss')
    ax_loss[1].set_title('Loss (last 50%)')
    ax_loss[1].grid(True, alpha=0.3)
    fig_loss.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_loss.png")
    fig_loss.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig_loss)

    # ── 3. Ensemble Spread Plot ──────────────────────────────────────
    if len(rom_solves) > 0:
        rom_arr = np.array(rom_solves)
        true_interp = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        true_at_pred = true_interp(t_pred)
        train_end = training_span[1]

        n_show = min(2, num_modes)
        fig, axes = plt.subplots(n_show, 2, figsize=(14, 4 * n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_show):
            # Left: individual ensemble trajectories
            for m in range(len(rom_solves)):
                axes[i, 0].plot(t_pred, rom_arr[m, i], color='tab:orange',
                               alpha=0.1, lw=0.6)
            axes[i, 0].plot(t_pred, true_at_pred[i], color='tab:gray', lw=1.5,
                           label='Truth')
            axes[i, 0].axvline(train_end, color='k', ls=':', lw=0.8, alpha=0.5)
            axes[i, 0].set_ylabel(f'Mode {i}')
            axes[i, 0].set_title(f'Mode {i} — Ensemble Trajectories')
            axes[i, 0].legend(fontsize=7)
            axes[i, 0].grid(True, alpha=0.2)

            # Right: per-member error histogram
            train_mask = t_pred <= train_end
            pred_mask = t_pred > train_end
            train_errs = []
            pred_errs = []
            for m in range(len(rom_solves)):
                te = np.linalg.norm(rom_arr[m, i, train_mask] - true_at_pred[i, train_mask]) / \
                     max(np.linalg.norm(true_at_pred[i, train_mask]), 1e-12)
                pe = np.linalg.norm(rom_arr[m, i, pred_mask] - true_at_pred[i, pred_mask]) / \
                     max(np.linalg.norm(true_at_pred[i, pred_mask]), 1e-12)
                train_errs.append(te)
                pred_errs.append(pe)
            axes[i, 1].hist(train_errs, bins=10, alpha=0.6, color='tab:blue',
                           label='Train err')
            axes[i, 1].hist(pred_errs, bins=10, alpha=0.6, color='tab:red',
                           label='Pred err')
            axes[i, 1].set_xlabel('Relative Error')
            axes[i, 1].set_ylabel('Count')
            axes[i, 1].set_title(f'Mode {i} — Per-member Error Distribution')
            axes[i, 1].legend(fontsize=7)
            axes[i, 1].grid(True, alpha=0.2)

        fig.suptitle(f"Ensemble Trajectories — {schema['label']}", fontsize=14)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_ensemble_spread.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main(schema_names=None):
    """Run selected (or all) data regimes."""
    schemas = SCHEMAS
    if schema_names:
        schemas = [s for s in SCHEMAS if s['name'] in schema_names]
        if not schemas:
            print(f"Unknown schema(s): {schema_names}")
            print(f"Available: {[s['name'] for s in SCHEMAS]}")
            return

    print("=" * 70)
    print("05 — Neural ODE Ensemble — Compressible Euler")
    print("=" * 70)
    print(f"Regimes: {len(schemas)}")
    for s in schemas:
        print(f"  • {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  noise={s['NOISE_LEVEL']:.0%}")
    print(f"Model:  hidden={MODEL_PARAMS['HIDDEN_DIM']}, layers={MODEL_PARAMS['NUM_LAYERS']}, "
          f"lr={MODEL_PARAMS['LEARNING_RATE']}, steps={MODEL_PARAMS['NUM_TRAIN_STEPS']}, "
          f"ensemble={MODEL_PARAMS['ENSEMBLE_SIZE']}")

    results = []
    for schema in schemas:
        r = run_experiment(schema)
        plot_results(r)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"SUMMARY — Neural ODE Ensemble (Euler)")
    print(f"{'='*80}")
    print(f"{'Regime':<28s} {'Samp':>4s} {'Noise':>5s} {'Stab':>5s} "
          f"{'Train':>8s} {'Pred':>8s} {'CI_cov':>7s} {'Time':>6s}")
    print(f"{'-'*28} {'-'*4} {'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
    for r in results:
        s = r['schema']
        print(f"{s['label']:<28s} {s['NUM_SAMPLES']:>4d} "
              f"{s['NOISE_LEVEL']:>4.0%} {r['stability_pct']:>4.0f}% "
              f"{r['train_error']:>7.2%} {r['pred_error']:>7.2%} "
              f"{r['ci_coverage']:>6.1%} {r['runtime']:>5.0f}s")


if __name__ == "__main__":
    schema_names = sys.argv[1:] if len(sys.argv) > 1 else None
    main(schema_names)

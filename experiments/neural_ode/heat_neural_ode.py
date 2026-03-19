"""
Neural ODE Architecture Sweep — Cubic Heat Equation

Systematic comparison of 3 neural ODE architectures for learning reduced-order
dynamics of the cubic heat equation with external input:
  - Small:  2 hidden layers × 64 units, tanh
  - Medium: 3 hidden layers × 128 units, tanh  (baseline from 05)
  - Large:  4 hidden layers × 256 units, tanh

Training: 5000 steps with cosine LR annealing (1e-3 → 0), 20-member ensembles.
Uses EXACT same data pipeline as experiments/heat/05_neural_ode.py.

Multi-trajectory problem: 5 training ICs + 1 test IC.
NeuralODE input: [q̂(5 dims), u(2 dims)] = 7 dims → output 5 dims.

Data regimes:
  1. Dense data, low noise     (65 samples, 1% noise)
  2. Sparse data, medium noise (20 samples, 5% noise)
  3. Dense data, high noise    (65 samples, 10% noise)

Usage:
    python heat_neural_ode.py                     # run all architectures × regimes
    python heat_neural_ode.py --arch small medium  # selected architectures
    python heat_neural_ode.py --regime dense_low_noise
    python heat_neural_ode.py --save-data          # generate .npz data without training
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEAT_DIR = os.path.join(SCRIPT_DIR, '..', 'heat')
ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, HEAT_DIR)
sys.path.insert(0, ROOT_DIR)

import config
from config import Basis, input_func_factory, input_parameters, test_parameters
from step1_generate_data import TrajectorySampler

# ── Configuration ────────────────────────────────────────────────────────────
SCHEMAS = [
    {"name": "dense_low_noise", "label": "Dense data, low noise",
     "NUM_SAMPLES": 65, "NOISE_LEVEL": 0.01, "NUM_EVAL_POINTS": 150},
    {"name": "sparse_medium_noise", "label": "Sparse data, medium noise",
     "NUM_SAMPLES": 20, "NOISE_LEVEL": 0.05, "NUM_EVAL_POINTS": 100},
    {"name": "dense_high_noise", "label": "Dense data, high noise",
     "NUM_SAMPLES": 65, "NOISE_LEVEL": 0.10, "NUM_EVAL_POINTS": 150},
]

ARCHITECTURES = {
    "small":  {"hidden_dim": 64,  "num_layers": 2, "color": "tab:green"},
    "medium": {"hidden_dim": 128, "num_layers": 3, "color": "tab:orange"},
    "large":  {"hidden_dim": 256, "num_layers": 4, "color": "tab:red"},
}

TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 2.0)
NUM_MODES = 5
NUM_ICS = 5
ENSEMBLE_SIZE = 5
NUM_TRAIN_STEPS = 2000
INIT_LR = 1e-3
SEED = 42

FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# Neural ODE model
# =============================================================================
class NeuralODE(eqx.Module):
    """MLP mapping [q̂, u] → dq̂/dt."""
    layers: list

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, *, key):
        keys = jax.random.split(key, num_layers + 1)
        dims = [in_dim] + [hidden_dim] * num_layers + [out_dim]
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


def count_params(model):
    """Count trainable parameters in an Equinox model."""
    params = eqx.filter(model, eqx.is_array)
    return sum(p.size for p in jax.tree.leaves(params))


# =============================================================================
# Input function helper (JAX-compatible)
# =============================================================================
def _make_jax_input_func(params):
    """JAX-compatible scalar input function u(t) → R^2."""
    a, b = params
    def _u(t):
        return jnp.array([
            a * jnp.sin(2.0 * jnp.pi * t),
            b * jnp.sin(4.0 * jnp.pi * t),
        ])
    return _u


# =============================================================================
# ODE integration
# =============================================================================
def _solve_trajectory(model, q0, t_obs, input_func):
    """Integrate neural ODE from q0 at times t_obs with input_func.

    Returns predicted states (len(t_obs), num_modes).
    """
    def ode_fn(t, y, args):
        u = input_func(t)
        inp = jnp.concatenate([y, u])
        return args(inp)

    term = diffrax.ODETerm(ode_fn)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_obs)
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)
    sol = diffrax.diffeqsolve(
        term, solver, t0=t_obs[0], t1=t_obs[-1], dt0=0.01,
        y0=q0, args=model, saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=16384, throw=False,
    )
    return sol.ys  # (len(t_obs), num_modes)


# =============================================================================
# Training (multi-trajectory with cosine LR)
# =============================================================================
@eqx.filter_jit
def _train_step(model, opt_state, all_q0, all_t_obs, all_y_obs,
                all_input_params, opt):
    """One gradient step over all training ICs."""

    def loss_fn(model):
        total_loss = 0.0
        n_ics = len(all_q0)
        for ic in range(n_ics):
            q0 = all_q0[ic]
            t_obs = all_t_obs[ic]
            y_obs = all_y_obs[ic]
            a, b = all_input_params[ic]
            input_func = _make_jax_input_func((a, b))
            y_pred = _solve_trajectory(model, q0, t_obs, input_func)
            # y_pred: (len(t_obs), num_modes), y_obs: (num_modes, len(t_obs))
            total_loss = total_loss + jnp.mean((y_pred - y_obs.T) ** 2)
        return total_loss / n_ics

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_single_member(key, all_q0, all_t_obs, all_y_obs,
                        all_input_params, num_steps,
                        num_modes, hidden_dim, num_layers):
    """Train one ensemble member with cosine LR. Returns (model, losses)."""
    in_dim = num_modes + 2  # state + 2D input
    model = NeuralODE(in_dim, num_modes, hidden_dim, num_layers, key=key)
    schedule = optax.cosine_decay_schedule(init_value=INIT_LR, decay_steps=num_steps)
    opt = optax.adam(schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    recorded_losses = []
    for s in range(num_steps):
        model, opt_state, loss = _train_step(
            model, opt_state,
            all_q0, all_t_obs, all_y_obs, all_input_params, opt,
        )
        if s % 500 == 0 or s == num_steps - 1:
            loss_val = float(loss)
            recorded_losses.append(loss_val)
            print(f"      step {s:5d}/{num_steps}  loss={loss_val:.6f}", flush=True)

    return model, np.array(recorded_losses)


def train_ensemble(all_q0, all_t_obs, all_y_obs, all_input_params,
                   hidden_dim, num_layers, num_modes):
    """Train full ensemble. Returns list of (model, losses)."""
    base_key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(base_key, ENSEMBLE_SIZE)

    ensemble = []
    for m in range(ENSEMBLE_SIZE):
        print(f"    ── Ensemble member {m + 1}/{ENSEMBLE_SIZE} ──")
        model, losses = train_single_member(
            keys[m], all_q0, all_t_obs, all_y_obs, all_input_params,
            NUM_TRAIN_STEPS, num_modes, hidden_dim, num_layers,
        )
        ensemble.append((model, losses))
    return ensemble


# =============================================================================
# Evaluation (multi-IC)
# =============================================================================
def evaluate_ensemble(ensemble, q0, t_pred, input_params):
    """Integrate all ensemble members from q0, return stable trajectories.

    Returns array of shape (n_stable, num_modes, len(t_pred)).
    """
    input_func = _make_jax_input_func(input_params)
    t_pred_jnp = jnp.array(t_pred)
    solves = []
    for model, _ in ensemble:
        try:
            y_pred = _solve_trajectory(model, jnp.array(q0), t_pred_jnp, input_func)
            y_np = np.array(y_pred)  # (len(t_pred), num_modes)
            if np.all(np.isfinite(y_np)):
                solves.append(y_np.T)  # → (num_modes, len(t_pred))
        except Exception:
            pass
    if solves:
        return np.stack(solves, axis=0)  # (n_stable, num_modes, len(t_pred))
    return np.empty((0, q0.shape[0], len(t_pred)))


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
    """Generate data for one regime (all ICs). Returns dict of arrays."""
    np.random.seed(SEED)

    noise_level = schema['NOISE_LEVEL']
    num_samples = schema['NUM_SAMPLES']
    num_eval_points = schema['NUM_EVAL_POINTS']
    train_params = input_parameters[:NUM_ICS]

    sampler = TrajectorySampler(
        training_span=TRAINING_SPAN,
        num_samples=num_samples,
        noiselevel=noise_level,
        num_regression_points=num_eval_points,
        synced=False,
    )
    (all_true_states, all_time_sampled, all_snapshots,
     all_training_inputs) = sampler.multisample(train_params)

    # POD basis fitted on all training snapshots
    snapshots_train = np.hstack(all_snapshots)
    basis = Basis(num_vectors=NUM_MODES)
    basis.fit(snapshots_train)

    # Compress training data
    all_snapshots_comp = [basis.compress(s) for s in all_snapshots]
    all_true_comp = [basis.compress(s) for s in all_true_states]

    # Generate test IC data
    (test_true_list, test_t_list, test_snap_list,
     test_inp_list) = sampler.multisample([test_parameters])
    test_true_comp = basis.compress(test_true_list[0])
    test_snaps_comp = basis.compress(test_snap_list[0])
    test_t_samp = test_t_list[0]

    # Pack into dict for saving
    data = {
        'cumulative_energy': np.array(basis.cumulative_energy),
        'test_true_comp': test_true_comp,
        'test_snaps_comp': test_snaps_comp,
        'test_t_samp': test_t_samp,
    }
    for i in range(NUM_ICS):
        data[f'snaps_comp_{i}'] = all_snapshots_comp[i]
        data[f'true_comp_{i}'] = all_true_comp[i]
        data[f't_samp_{i}'] = all_time_sampled[i]
    return data


# =============================================================================
# Run single architecture × regime experiment
# =============================================================================
def run_experiment(schema, arch_name, arch_cfg):
    """Run one architecture on one data regime. Returns results dict."""
    np.random.seed(SEED)

    noise_level = schema['NOISE_LEVEL']
    num_samples = schema['NUM_SAMPLES']
    num_eval_points = schema['NUM_EVAL_POINTS']
    dim_str = f"{arch_cfg['num_layers']}×{arch_cfg['hidden_dim']}"
    train_params = input_parameters[:NUM_ICS]

    print(f"\n{'─'*70}")
    print(f"  {arch_name} ({dim_str}) — {schema['label']}  "
          f"({num_samples} samples/IC, {noise_level:.0%} noise, {NUM_ICS} ICs)")
    print(f"{'─'*70}")

    # ── Data generation ──────────────────────────────────────────────────
    data_path = os.path.join(SCRIPT_DIR, f"data_heat_{schema['name']}.npz")
    if os.path.exists(data_path):
        print(f"  Loading pre-saved data: {data_path}")
        regime_data = load_data(data_path)
        cum_energy = float(regime_data['cumulative_energy'])
        all_snapshots_comp = [regime_data[f'snaps_comp_{i}'] for i in range(NUM_ICS)]
        all_true_comp = [regime_data[f'true_comp_{i}'] for i in range(NUM_ICS)]
        all_time_sampled = [regime_data[f't_samp_{i}'] for i in range(NUM_ICS)]
        test_true_comp = regime_data['test_true_comp']
        test_snaps_comp = regime_data['test_snaps_comp']
        test_t_samp = regime_data['test_t_samp']
    else:
        sampler = TrajectorySampler(
            training_span=TRAINING_SPAN,
            num_samples=num_samples,
            noiselevel=noise_level,
            num_regression_points=num_eval_points,
            synced=False,
        )
        (all_true_states, all_time_sampled, all_snapshots,
         all_training_inputs) = sampler.multisample(train_params)

        # POD basis
        snapshots_train = np.hstack(all_snapshots)
        basis = Basis(num_vectors=NUM_MODES)
        basis.fit(snapshots_train)
        cum_energy = basis.cumulative_energy
        print(f"  POD energy: {cum_energy:.4%}")

        all_snapshots_comp = [basis.compress(s) for s in all_snapshots]
        all_true_comp = [basis.compress(s) for s in all_true_states]

        # Test IC
        (test_true_list, test_t_list, test_snap_list,
         test_inp_list) = sampler.multisample([test_parameters])
        test_true_comp = basis.compress(test_true_list[0])
        test_snaps_comp = basis.compress(test_snap_list[0])
        test_t_samp = test_t_list[0]

    print(f"  POD energy: {float(cum_energy) if not isinstance(cum_energy, float) else cum_energy:.4%}")

    # ── Count parameters ─────────────────────────────────────────────────
    in_dim = NUM_MODES + 2  # 5 state dims + 2 input dims = 7
    dummy_model = NeuralODE(
        in_dim=in_dim, out_dim=NUM_MODES,
        hidden_dim=arch_cfg['hidden_dim'],
        num_layers=arch_cfg['num_layers'],
        key=random.PRNGKey(0),
    )
    n_params = count_params(dummy_model)
    print(f"  Parameters: {n_params:,}")

    # ── Prepare JAX training data ────────────────────────────────────────
    all_q0 = [jnp.array(sc[:, 0]) for sc in all_snapshots_comp]
    all_t_obs = [jnp.array(ts) for ts in all_time_sampled]
    all_y_obs = [jnp.array(sc) for sc in all_snapshots_comp]
    all_input_params_jnp = [
        jnp.array([float(p[0]), float(p[1])]) for p in train_params
    ]

    # ── Train ensemble ───────────────────────────────────────────────────
    print(f"  Training {ENSEMBLE_SIZE} ensemble members ({NUM_TRAIN_STEPS} steps, "
          f"cosine LR {INIT_LR}→0)...")
    t0 = time.time()
    ensemble = train_ensemble(
        all_q0, all_t_obs, all_y_obs, all_input_params_jnp,
        arch_cfg['hidden_dim'], arch_cfg['num_layers'], NUM_MODES,
    )
    runtime = time.time() - t0
    print(f"  Ensemble training took {runtime:.0f}s")

    # ── Evaluation (all ICs) ─────────────────────────────────────────────
    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    train_mask = t_pred <= TRAINING_SPAN[1]
    pred_mask = t_pred > TRAINING_SPAN[1]

    eval_params = list(train_params) + [test_parameters]
    eval_snaps_comp = all_snapshots_comp + [test_snaps_comp]
    eval_true_comp = all_true_comp + [test_true_comp]
    eval_t_samp = all_time_sampled + [test_t_samp]
    eval_labels = [f"Train IC {i} {train_params[i]}" for i in range(NUM_ICS)] + \
                  [f"Test IC {test_parameters}"]

    all_rom_solves = []
    all_n_stable = []
    all_train_errors = []
    all_pred_errors = []

    for ic_idx, (params, true_c) in enumerate(zip(eval_params, eval_true_comp)):
        q0 = eval_snaps_comp[ic_idx][:, 0]
        ic_solves = evaluate_ensemble(ensemble, q0, t_pred, params)
        all_rom_solves.append(ic_solves)
        all_n_stable.append(len(ic_solves))

        ti = interp1d(config.time_domain, true_c,
                      kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)

        if len(ic_solves) > 0:
            rom_med = np.median(ic_solves, axis=0)
            te = float(np.linalg.norm(rom_med[:, train_mask] - ta[:, train_mask]) /
                       np.linalg.norm(ta[:, train_mask]))
            pe = float(np.linalg.norm(rom_med[:, pred_mask] - ta[:, pred_mask]) /
                       np.linalg.norm(ta[:, pred_mask]))
        else:
            te, pe = float('inf'), float('inf')

        all_train_errors.append(te)
        all_pred_errors.append(pe)
        label = eval_labels[ic_idx]
        print(f"    {label}: {all_n_stable[-1]}/{ENSEMBLE_SIZE} stable, "
              f"train={te:.4%}, pred={pe:.4%}")

    # Aggregate over training ICs
    train_ic_stable = sum(all_n_stable[:NUM_ICS])
    train_ic_total = ENSEMBLE_SIZE * NUM_ICS
    train_errors_fin = [e for e in all_train_errors[:NUM_ICS] if np.isfinite(e)]
    pred_errors_fin = [e for e in all_pred_errors[:NUM_ICS] if np.isfinite(e)]
    train_error = np.mean(train_errors_fin) if train_errors_fin else float('inf')
    pred_error = np.mean(pred_errors_fin) if pred_errors_fin else float('inf')
    stability_pct = train_ic_stable / max(train_ic_total, 1) * 100

    print(f"\n    Overall: {train_ic_stable}/{train_ic_total} ({stability_pct:.0f}%)")
    print(f"    Avg train: {train_error:.4%}  |  Avg pred: {pred_error:.4%}")
    print(f"    Test IC: {all_n_stable[-1]}/{ENSEMBLE_SIZE} stable, "
          f"train={all_train_errors[-1]:.4%}, pred={all_pred_errors[-1]:.4%}")

    # CI coverage
    ci_width = ci_coverage = float('nan')
    if train_ic_stable > 0:
        all_in_ci, all_widths = [], []
        for ic_idx in range(NUM_ICS):
            if all_n_stable[ic_idx] > 0:
                ti = interp1d(config.time_domain, eval_true_comp[ic_idx],
                              kind='cubic', fill_value='extrapolate')
                ta = ti(t_pred)
                q05 = np.percentile(all_rom_solves[ic_idx], 5, axis=0)
                q95 = np.percentile(all_rom_solves[ic_idx], 95, axis=0)
                all_widths.append(np.mean(q95 - q05))
                all_in_ci.append(np.mean((ta >= q05) & (ta <= q95)))
        if all_widths:
            ci_width = np.mean(all_widths)
            ci_coverage = np.mean(all_in_ci)
            print(f"    CI coverage: {ci_coverage:.2%} (target: 90%)")

    all_member_losses = np.stack([losses for _, losses in ensemble], axis=0)

    return {
        'schema': schema,
        'arch_name': arch_name,
        'arch_cfg': arch_cfg,
        'n_params': n_params,
        'train_error': train_error,
        'pred_error': pred_error,
        'stability_pct': stability_pct,
        'n_stable': train_ic_stable,
        'n_total': train_ic_total,
        'test_train_error': all_train_errors[-1],
        'test_pred_error': all_pred_errors[-1],
        'test_n_stable': all_n_stable[-1],
        'ci_coverage': ci_coverage,
        'ci_width': ci_width,
        'runtime': runtime,
        'all_member_losses': all_member_losses,
        # Per-IC data for multi-trajectory plotting
        'all_rom_solves': all_rom_solves,
        'all_snaps_comp': eval_snaps_comp,
        'all_true_comp': eval_true_comp,
        'all_t_samp': eval_t_samp,
        'all_n_stable': all_n_stable,
        'eval_labels': eval_labels,
        't_full': config.time_domain,
        't_pred': t_pred,
        'training_span': TRAINING_SPAN,
        'num_modes': NUM_MODES,
        'max_samp': ENSEMBLE_SIZE,
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_trajectories(result, save_dir):
    """Plot multi-IC trajectory grid (rows=ICs, cols=modes)."""
    os.makedirs(save_dir, exist_ok=True)

    schema = result['schema']
    arch_name = result['arch_name']
    arch_cfg = result['arch_cfg']
    t_pred = result['t_pred']
    t_full = result['t_full']
    training_span = result['training_span']
    num_modes = result['num_modes']
    all_rom_solves = result['all_rom_solves']
    all_snaps_comp = result['all_snaps_comp']
    all_true_comp = result['all_true_comp']
    all_t_samp = result['all_t_samp']
    all_n_stable = result['all_n_stable']
    eval_labels = result['eval_labels']
    max_samp = result['max_samp']
    color = arch_cfg['color']

    n_ics_total = len(all_rom_solves)
    has_any_stable = any(ns > 0 for ns in all_n_stable)

    if not has_any_stable:
        print("  ⚠ No stable Neural ODE solves — skipping trajectory plot")
        return

    fig, axes = plt.subplots(
        n_ics_total, num_modes,
        figsize=(4 * num_modes, 2.5 * n_ics_total),
        sharex=True, squeeze=False,
    )

    for row in range(n_ics_total):
        rom_solves = all_rom_solves[row]
        n_stable = all_n_stable[row]
        label = eval_labels[row]

        true_interp = interp1d(t_full, all_true_comp[row],
                               kind='cubic', fill_value='extrapolate')
        true_at_pred = true_interp(t_pred)

        for col in range(num_modes):
            ax = axes[row, col]

            ax.axvspan(training_span[0], training_span[1],
                       color='gray', alpha=0.10, zorder=0)

            ax.plot(t_pred, true_at_pred[col],
                    color='tab:gray', lw=2,
                    label='True' if (row == 0 and col == 0) else None)

            ax.plot(all_t_samp[row], all_snaps_comp[row][col],
                    'k*', ms=4, zorder=5,
                    label='Data' if (row == 0 and col == 0) else None)

            if n_stable > 0:
                ax.plot(
                    t_pred,
                    np.median(rom_solves[:, col, :], axis=0),
                    color=color, ls='--', lw=2, alpha=0.9,
                    label='Median' if (row == 0 and col == 0) else None,
                )
                ax.fill_between(
                    t_pred,
                    np.percentile(rom_solves[:, col, :], 5, axis=0),
                    np.percentile(rom_solves[:, col, :], 95, axis=0),
                    color=color, alpha=0.15,
                    label='90% CI' if (row == 0 and col == 0) else None,
                )

            ax.axvline(training_span[1], color='k', ls=':', lw=0.8, alpha=0.5)

            yvals = true_at_pred[col]
            ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
            pad = max(abs(ymax - ymin) * 0.3, 1e-6)
            ax.set_ylim(ymin - pad, ymax + pad)

            if row == 0:
                ax.set_title(f'Mode {col + 1}')
            if col == 0:
                ax.set_ylabel(f'{label}\n({n_stable}/{max_samp} stable)',
                              fontsize=8)
            if row == n_ics_total - 1:
                ax.set_xlabel('Time')

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_leg, loc='upper center',
                   ncol=len(handles), fontsize=9,
                   bbox_to_anchor=(0.5, 1.02))

    dim_str = f"{arch_cfg['num_layers']}×{arch_cfg['hidden_dim']}"
    fig.suptitle(f"Neural ODE ({arch_name} {dim_str}) — {schema['label']}",
                 fontsize=14, y=1.05)
    fig.tight_layout()
    path = os.path.join(save_dir, f"heat_{arch_name}_{schema['name']}_trajectories.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


def plot_loss(result, save_dir):
    """Plot training loss convergence."""
    os.makedirs(save_dir, exist_ok=True)

    schema = result['schema']
    arch_name = result['arch_name']
    arch_cfg = result['arch_cfg']
    all_member_losses = result['all_member_losses']
    color = arch_cfg['color']

    mean_loss = np.mean(all_member_losses, axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(mean_loss, lw=0.8, color=color)
    ax[0].set_xlabel('Training Step')
    ax[0].set_ylabel('MSE Loss')
    ax[0].set_title('Loss Convergence (mean over ensemble)')
    ax[0].grid(True, alpha=0.3)

    half = len(mean_loss) // 2
    ax[1].plot(range(half, len(mean_loss)), mean_loss[half:], lw=0.8, color=color)
    ax[1].set_xlabel('Training Step')
    ax[1].set_ylabel('MSE Loss')
    ax[1].set_title('Loss (last 50%)')
    ax[1].grid(True, alpha=0.3)

    dim_str = f"{arch_cfg['num_layers']}×{arch_cfg['hidden_dim']}"
    fig.suptitle(f"Training Loss — {arch_name} ({dim_str}) — {schema['label']}",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(save_dir, f"heat_{arch_name}_{schema['name']}_loss.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


def plot_architecture_comparison(all_results, save_dir):
    """Bar chart comparing all architectures across data regimes."""
    os.makedirs(save_dir, exist_ok=True)

    metrics = ['train_error', 'pred_error', 'ci_coverage', 'stability_pct']
    metric_labels = ['Train Error', 'Pred Error', 'CI Coverage', 'Stability %']

    regime_names = [s['label'] for s in SCHEMAS]
    arch_names = list(ARCHITECTURES.keys())
    n_regimes = len(regime_names)

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

    fig.suptitle('Neural ODE Architecture Comparison — Heat', fontsize=14)
    fig.tight_layout()
    path = os.path.join(save_dir, "heat_architecture_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\n  📊 Saved: {path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Neural ODE Architecture Sweep — Heat')
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
            data_path = os.path.join(SCRIPT_DIR, f"data_heat_{schema['name']}.npz")
            regime_data = generate_regime_data(schema)
            save_data(regime_data, data_path)
        print("Done. Use these .npz files on Kaggle.")
        return

    # ── Filter selections ────────────────────────────────────────────────
    selected_schemas = [s for s in SCHEMAS if s['name'] in args.regime]
    selected_archs = {k: v for k, v in ARCHITECTURES.items() if k in args.arch}

    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("=" * 70)
    print("Neural ODE Architecture Sweep — Cubic Heat Equation")
    print("=" * 70)
    print(f"Architectures: {list(selected_archs.keys())}")
    print(f"Regimes:       {len(selected_schemas)}")
    for s in selected_schemas:
        print(f"  • {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  "
              f"noise={s['NOISE_LEVEL']:.0%}")
    print(f"Training:  steps={NUM_TRAIN_STEPS}, lr=cosine({INIT_LR}→0), "
          f"ensemble={ENSEMBLE_SIZE}")
    print(f"ICs:       {NUM_ICS} training + 1 test ({test_parameters})")

    # ── Run all experiments ──────────────────────────────────────────────
    all_results = {}
    for schema in selected_schemas:
        for arch_name, arch_cfg in selected_archs.items():
            result = run_experiment(schema, arch_name, arch_cfg)
            plot_trajectories(result, FIGURE_DIR)
            plot_loss(result, FIGURE_DIR)
            all_results[(arch_name, schema['name'])] = result

    # ── Comparison plot ──────────────────────────────────────────────────
    if len(all_results) > 1:
        plot_architecture_comparison(all_results, FIGURE_DIR)

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"ARCHITECTURE COMPARISON — Heat")
    print(f"{'='*100}")
    print(f"{'Arch':<8s} {'Regime':<28s} {'Params':>7s} {'Train':>8s} "
          f"{'Pred':>8s} {'TestPred':>8s} {'CI_cov':>7s} {'Stab':>5s} {'Time':>6s}")
    print(f"{'-'*8} {'-'*28} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*5} {'-'*6}")
    for schema in selected_schemas:
        for arch_name in selected_archs:
            key = (arch_name, schema['name'])
            if key not in all_results:
                continue
            r = all_results[key]
            print(f"{arch_name:<8s} {schema['label']:<28s} {r['n_params']:>7,d} "
                  f"{r['train_error']:>7.2%} {r['pred_error']:>7.2%} "
                  f"{r['test_pred_error']:>7.2%} "
                  f"{r['ci_coverage']:>6.1%} {r['stability_pct']:>4.0f}% "
                  f"{r['runtime']:>5.0f}s")


if __name__ == "__main__":
    main()

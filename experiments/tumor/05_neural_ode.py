"""
05 — Neural ODE Ensemble Baseline (Tumor Growth) — Multi-Trajectory

Black-box baseline for comparison with Bayesian OpInf:
  - MLP: r → 128 → 128 → 128 → r  (tanh activations)
  - Training: multi-trajectory MSE over 5 (k, d) parameter regimes
  - Evaluation: 5 training ICs + 1 unseen test IC
  - UQ: ensemble of 20 independently trained networks

Uses the EXACT same data pipeline as the Bayesian OpInf experiment:
  1. FOM solve → subsample in training span → add noise
  2. Clean POD basis (fit on concatenated noise-free snapshots)
  3. t_pred starts at TRAINING_SPAN[0] (skip MRI IC transient)

Data regime:
  Dense data, low noise (80 samples, 1% noise)

Usage:
    python 05_neural_ode.py
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
from config import (
    Basis, TumorTwinFOM,
    load_multitraj_fom_data, TRAINING_PARAMS, TEST_PARAMS, get_fom_data_path,
)
from core.plotting import plot_full_order_error

# ── Data regime definitions ──────────────────────────────────────────────────
SCHEMAS = [
    {
        "name": "dense_low_noise",
        "label": "Dense data, low noise",
        "NUM_SAMPLES": 80,
        "NOISE_LEVEL": 0.01,
        "NUM_EVAL_POINTS": 200,
    },
]

# ── Model hyperparameters ────────────────────────────────────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=4,
    NUM_ICS=5,
    HIDDEN_DIM=128,
    NUM_LAYERS=3,
    ACTIVATION='tanh',
    ENSEMBLE_SIZE=20,
    NUM_TRAIN_STEPS=3000,
    LEARNING_RATE=1e-3,
    SEED=42,
)

TRAINING_SPAN = config.TRAINING_SPAN
PREDICTION_SPAN = (TRAINING_SPAN[0], config.PREDICTION_DAYS)
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
# Training — Multi-IC loss
# =============================================================================
def _solve_trajectory(model, q0, t_obs):
    """Integrate neural ODE from q0 at observation times.

    Returns predicted states (len(t_obs), num_modes).
    """
    term = diffrax.ODETerm(model)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_obs)
    adjoint = diffrax.RecursiveCheckpointAdjoint()
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=t_obs[0], t1=t_obs[-1],
        dt0=t_obs[1] - t_obs[0],
        y0=q0, saveat=saveat,
        adjoint=adjoint,
        max_steps=16384, throw=False,
    )
    return sol.ys  # (len(t_obs), num_modes)


@eqx.filter_jit
def _train_step(model, opt_state, all_q0, all_t_obs, all_y_obs, opt):
    """One gradient step over all training ICs."""

    def loss_fn(model):
        total_loss = 0.0
        n_ics = len(all_q0)
        for ic in range(n_ics):
            y_pred = _solve_trajectory(model, all_q0[ic], all_t_obs[ic])
            # y_pred: (len(t_obs), num_modes), y_obs: (num_modes, len(t_obs))
            total_loss = total_loss + jnp.mean((y_pred - all_y_obs[ic].T) ** 2)
        return total_loss / n_ics

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_single_member(key, all_q0, all_t_obs, all_y_obs,
                        num_steps, lr, num_modes, hidden_dim, num_layers):
    """Train one ensemble member on all trajectories. Returns (model, losses)."""
    model = NeuralODE(
        in_dim=num_modes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        key=key,
    )
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    losses = []
    for step in range(num_steps):
        model, opt_state, loss = _train_step(
            model, opt_state, all_q0, all_t_obs, all_y_obs, opt,
        )
        losses.append(float(loss))
        if step % 500 == 0 or step == num_steps - 1:
            print(f"      step {step:5d}/{num_steps}  loss={losses[-1]:.6f}")

    return model, np.array(losses)


def train_ensemble(all_q0, all_t_obs, all_y_obs, p):
    """Train full ensemble. Returns list of (model, losses)."""
    ensemble_size = p['ENSEMBLE_SIZE']
    base_key = jax.random.PRNGKey(p['SEED'])
    keys = jax.random.split(base_key, ensemble_size)

    ensemble = []
    for m in range(ensemble_size):
        print(f"    ── Ensemble member {m + 1}/{ensemble_size} ──")
        model, losses = train_single_member(
            keys[m], all_q0, all_t_obs, all_y_obs,
            p['NUM_TRAIN_STEPS'], p['LEARNING_RATE'],
            p['NUM_MODES'], p['HIDDEN_DIM'], p['NUM_LAYERS'],
        )
        ensemble.append((model, losses))
    return ensemble


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_ensemble(ensemble, q0, t_pred):
    """Integrate all ensemble members from q0, return stable trajectories.

    Returns array of shape (n_stable, num_modes, len(t_pred)).
    """
    t_pred_jnp = jnp.array(t_pred)
    q0_jnp = jnp.array(q0)
    t0_pred = float(t_pred[0])
    t1_pred = float(t_pred[-1])
    dt0_pred = float(t_pred[1] - t_pred[0])

    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_pred_jnp)

    solves = []
    for model, _ in ensemble:
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
                solves.append(traj)
        except Exception:
            pass
    if solves:
        return np.stack(solves, axis=0)  # (n_stable, num_modes, len(t_pred))
    return np.empty((0, q0.shape[0], len(t_pred)))


# =============================================================================
# Run experiment
# =============================================================================
def run_experiment(schema):
    """Run one data regime. Returns results dict."""
    p = MODEL_PARAMS
    np.random.seed(p['SEED'])

    noise_level = schema['NOISE_LEVEL']
    num_samples = schema['NUM_SAMPLES']
    num_eval_points = schema['NUM_EVAL_POINTS']
    num_modes = p['NUM_MODES']
    num_ics = p['NUM_ICS']

    print(f"\n{'='*70}")
    print(f"  {schema['label']}  ({num_samples} samples, {noise_level:.0%} noise)")
    print(f"{'='*70}")

    # ── 1. GENERATE TRAINING DATA (5 trajectories) ───────────────────
    t_pred_full = np.linspace(TRAINING_SPAN[0], config.PREDICTION_DAYS, num_eval_points)
    (all_foms, t_full, all_true_states, all_t_samp, all_snaps_noisy) = \
        load_multitraj_fom_data(t_pred_full, TRAINING_SPAN, num_samples, noise_level,
                                param_list=TRAINING_PARAMS)

    # ── 2. BUILD POD BASIS (from concatenated clean snapshots) ────────
    all_snaps_clean = [fom.get_states(t_samp)
                       for fom, t_samp in zip(all_foms, all_t_samp)]
    basis = Basis(num_vectors=num_modes)
    basis.fit(np.hstack(all_snaps_clean))
    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    # ── 3. COMPRESS ALL SNAPSHOTS AND TRUTH ──────────────────────────
    all_snaps_comp = [basis.compress(s) for s in all_snaps_noisy]
    all_true_comp = [basis.compress(s) for s in all_true_states]

    # ── 4. PREPARE JAX TRAINING DATA ─────────────────────────────────
    all_q0 = [jnp.array(sc[:, 0]) for sc in all_snaps_comp]
    all_t_obs = [jnp.array(ts) for ts in all_t_samp]
    all_y_obs = [jnp.array(sc) for sc in all_snaps_comp]

    # ── 5. TRAIN ENSEMBLE ─────────────────────────────────────────────
    print(f"\n  Training {p['ENSEMBLE_SIZE']} ensemble members "
          f"({p['NUM_TRAIN_STEPS']} steps each, {num_ics} ICs)...")
    t0_wall = time.time()
    ensemble = train_ensemble(all_q0, all_t_obs, all_y_obs, p)
    runtime = time.time() - t0_wall
    print(f"  Training time: {runtime:.0f}s")

    # ── 6. GENERATE TEST IC DATA ─────────────────────────────────────
    test_fom = TumorTwinFOM(get_fom_data_path(*TEST_PARAMS))
    test_true = test_fom.get_states(t_pred_full)
    test_true_comp = basis.compress(test_true)
    test_t_samp = np.sort(
        np.random.uniform(TRAINING_SPAN[0], TRAINING_SPAN[1], size=num_samples))
    test_t_samp[0] = TRAINING_SPAN[0]
    test_t_samp[-1] = TRAINING_SPAN[1]
    test_snaps_noisy = test_fom.noise(test_fom.get_states(test_t_samp), noise_level)
    test_snaps_comp = basis.compress(test_snaps_noisy)

    # ── 7. EVALUATE ALL ICs (5 train + 1 test) ───────────────────────
    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], num_eval_points)
    max_samp = p['ENSEMBLE_SIZE']

    eval_params = list(TRAINING_PARAMS) + [TEST_PARAMS]
    eval_snaps_comp = all_snaps_comp + [test_snaps_comp]
    eval_true_comp = all_true_comp + [test_true_comp]
    eval_t_samp = all_t_samp + [test_t_samp]
    eval_labels = [f"Train (k={k:.3f}, d={d:.3f})" for k, d in TRAINING_PARAMS] + \
                  [f"Test (k={TEST_PARAMS[0]:.3f}, d={TEST_PARAMS[1]:.3f})"]

    all_rom_solves = []
    all_n_stable = []
    all_train_errors = []
    all_pred_errors = []
    train_mask = t_pred <= TRAINING_SPAN[1]
    pred_mask = t_pred > TRAINING_SPAN[1]

    for ic_idx, (params, true_c) in enumerate(zip(eval_params, eval_true_comp)):
        q0 = eval_snaps_comp[ic_idx][:, 0]
        ic_solves = evaluate_ensemble(ensemble, q0, t_pred)
        all_rom_solves.append(ic_solves)
        all_n_stable.append(len(ic_solves))

        ti = interp1d(t_full, true_c, kind='cubic', fill_value='extrapolate')
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
        print(f"    {label}: {all_n_stable[-1]}/{max_samp} stable, "
              f"train={te:.4%}, pred={pe:.4%}")

    # Aggregate over training ICs
    train_ic_stable = sum(all_n_stable[:num_ics])
    train_ic_total = max_samp * num_ics
    train_errors_fin = [e for e in all_train_errors[:num_ics] if np.isfinite(e)]
    pred_errors_fin = [e for e in all_pred_errors[:num_ics] if np.isfinite(e)]
    train_error = np.mean(train_errors_fin) if train_errors_fin else float('inf')
    pred_error = np.mean(pred_errors_fin) if pred_errors_fin else float('inf')
    stability_pct = train_ic_stable / max(train_ic_total, 1) * 100

    print(f"\n    Overall: {train_ic_stable}/{train_ic_total} ({stability_pct:.0f}%)")
    print(f"    Avg train: {train_error:.4%}  |  Avg pred: {pred_error:.4%}")
    print(f"    Test IC: {all_n_stable[-1]}/{max_samp} stable, "
          f"train={all_train_errors[-1]:.4%}, pred={all_pred_errors[-1]:.4%}")

    # CI coverage (over training ICs)
    ci_width = ci_coverage = float('nan')
    if train_ic_stable > 0:
        all_in_ci, all_widths = [], []
        for ic_idx in range(num_ics):
            if all_n_stable[ic_idx] > 0:
                ti = interp1d(t_full, eval_true_comp[ic_idx],
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

    # Collect per-member losses for plotting
    all_member_losses = np.stack([losses for _, losses in ensemble], axis=0)

    return {
        'schema': schema,
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'n_stable': train_ic_stable, 'n_total': train_ic_total,
        'test_train_error': all_train_errors[-1],
        'test_pred_error': all_pred_errors[-1],
        'test_n_stable': all_n_stable[-1],
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'runtime': runtime,
        'all_member_losses': all_member_losses,
        # Per-IC data for multi-trajectory plotting
        'all_rom_solves': all_rom_solves,
        'all_snaps_comp': eval_snaps_comp,
        'all_true_comp': eval_true_comp,
        'all_t_samp': eval_t_samp,
        'all_n_stable': all_n_stable,
        'eval_params': eval_params,
        'eval_labels': eval_labels,
        't_full': t_full,
        't_pred': t_pred,
        'training_span': TRAINING_SPAN,
        'num_modes': num_modes,
        'max_samp': max_samp,
        'basis': basis,
        'all_true_states': all_true_states + [test_true],
        'all_foms': all_foms + [test_fom],
        'losses': all_member_losses,
        'all_train_errors': all_train_errors,
        'all_pred_errors': all_pred_errors,
    }


# =============================================================================
# Plotting — Spatial heatmaps (test IC)
# =============================================================================
def plot_spatial_comparison(result, save_dir=None):
    """Plot 3D tumor density slices for the test IC: FOM truth vs Neural ODE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(save_dir, exist_ok=True)

    schema = result['schema']
    prefix = f"05_{schema['name']}"
    basis = result['basis']
    t_full = result['t_full']
    t_pred = result['t_pred']
    all_rom_solves = result['all_rom_solves']
    all_true_states = result['all_true_states']
    all_foms = result['all_foms']

    # Use test IC (last index)
    test_idx = len(all_rom_solves) - 1
    test_solves = all_rom_solves[test_idx]
    test_true = all_true_states[test_idx]
    test_fom = all_foms[test_idx]

    if len(test_solves) == 0:
        print("  ⚠ No stable test IC predictions — skipping spatial plot")
        return

    ens_med = np.median(test_solves, axis=0)

    timepoints_to_show = [5, 15, 30, 45, 60, 90]
    n_times = len(timepoints_to_show)
    fig, axes = plt.subplots(3, n_times, figsize=(3.5 * n_times, 10))

    for col, t_target in enumerate(timepoints_to_show):
        idx_full = np.argmin(np.abs(t_full - t_target))
        fom_state = test_true[:, idx_full]

        idx_pred = np.argmin(np.abs(t_pred - t_target))
        rom_full = basis.decompress(ens_med[:, idx_pred])

        fom_slices = test_fom.get_center_slices(fom_state)
        rom_slices = test_fom.get_center_slices(rom_full)
        err_slices = test_fom.get_center_slices(np.abs(fom_state - rom_full))

        im0 = axes[0, col].imshow(fom_slices['axial'].T, origin='lower',
                                   cmap='hot_r', vmin=0, vmax=1, aspect='equal')
        axes[0, col].set_title(f'Day {t_full[idx_full]:.0f}', fontsize=11)

        im1 = axes[1, col].imshow(rom_slices['axial'].T, origin='lower',
                                   cmap='hot_r', vmin=0, vmax=1, aspect='equal')

        im2 = axes[2, col].imshow(err_slices['axial'].T, origin='lower',
                                   cmap='Oranges', vmin=0, aspect='equal')

        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        if col == 0:
            axes[0, col].set_ylabel('FOM Truth', fontsize=12, fontweight='bold')
            axes[1, col].set_ylabel('Neural ODE', fontsize=12, fontweight='bold')
            axes[2, col].set_ylabel('|Error|', fontsize=12, fontweight='bold')

    fig.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.8, label='Cellularity')
    fig.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.8, label='Cellularity')
    fig.colorbar(im2, ax=axes[2, :].tolist(), shrink=0.8, label='|Error|')
    k_t, d_t = TEST_PARAMS
    fig.suptitle(f'Test IC (k={k_t:.3f}, d={d_t:.3f}): FOM vs Neural ODE (axial)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_spatial_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


# =============================================================================
# Plotting — Tumor volume (all ICs)
# =============================================================================
def plot_tumor_volume(result, save_dir=None):
    """Plot total tumor burden for all ICs: FOM truth vs Neural ODE ensemble.

    Uses reduced-space dot product to avoid decompressing full 646K DOF fields.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(save_dir, exist_ok=True)

    schema = result['schema']
    prefix = f"05_{schema['name']}"
    basis = result['basis']
    t_full = result['t_full']
    t_pred = result['t_pred']
    all_rom_solves = result['all_rom_solves']
    all_true_states = result['all_true_states']
    all_foms = result['all_foms']
    eval_labels = result['eval_labels']
    num_ics = len(all_rom_solves)

    # Precompute efficient volume projection: vol = vol_proj @ q + shift_vol
    V = basis.entries   # (n_dof, r)
    ones = np.ones(V.shape[0])
    vol_proj = V.T @ ones          # (r,)
    shift_vol = ones @ basis.shift_  # scalar

    train_colors = plt.cm.tab10(np.linspace(0, 0.5, num_ics - 1))
    test_color = 'tab:red'

    fig, ax = plt.subplots(figsize=(10, 6))

    for ic_idx in range(num_ics):
        fom = all_foms[ic_idx]
        true_states = all_true_states[ic_idx]
        voxel_vol = float(np.prod(fom.spacing))
        is_test = (ic_idx == num_ics - 1)
        color = test_color if is_test else train_colors[ic_idx]
        lw_truth = 2.5 if is_test else 1.5
        alpha_truth = 1.0 if is_test else 0.6

        # FOM truth volume
        fom_vol = np.array([true_states[:, i].sum() * voxel_vol
                            for i in range(true_states.shape[1])])
        label_truth = eval_labels[ic_idx]
        ax.plot(t_full, fom_vol, color=color, lw=lw_truth, alpha=alpha_truth,
                label=label_truth)

        # ROM ensemble volume
        ic_solves = all_rom_solves[ic_idx]
        if len(ic_solves) > 0:
            ens_vols = np.array([vol_proj @ ic_solves[s] + shift_vol
                                 for s in range(len(ic_solves))]) * voxel_vol
            ens_med = np.median(ens_vols, axis=0)
            ens_lo = np.percentile(ens_vols, 5, axis=0)
            ens_hi = np.percentile(ens_vols, 95, axis=0)
            ax.plot(t_pred, ens_med, color=color, ls='--', lw=1.5, alpha=0.8)
            ax.fill_between(t_pred, ens_lo, ens_hi, color=color, alpha=0.10)

    ax.axvline(TRAINING_SPAN[1], color='gray', ls='--', alpha=0.5, label='Train/Predict')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Total Tumor Burden (mm³)')
    ax.set_title('Tumor Volume: All ICs (solid=FOM, dashed=NeuralODE)')
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_tumor_volume.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


# =============================================================================
# Plotting — Multi-IC ROM trajectories, loss, ensemble spread
# =============================================================================
def plot_results(result, save_dir=None):
    """Generate multi-IC ROM trajectory, loss, ensemble spread, spatial, and volume plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    schema = result['schema']
    prefix = f"05_{schema['name']}"
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
    all_member_losses = result['all_member_losses']

    # ── 1. Multi-trajectory ROM plot ─────────────────────────────────
    n_ics_total = len(all_rom_solves)
    has_any_stable = any(ns > 0 for ns in all_n_stable)

    if not has_any_stable:
        print("  ⚠ No stable Neural ODE solves — skipping trajectory plot")
    else:
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
                        color='tab:orange', ls='--', lw=2, alpha=0.9,
                        label='Median' if (row == 0 and col == 0) else None,
                    )
                    ax.fill_between(
                        t_pred,
                        np.percentile(rom_solves[:, col, :], 5, axis=0),
                        np.percentile(rom_solves[:, col, :], 95, axis=0),
                        color='tab:orange', alpha=0.15,
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
                    ax.set_xlabel('Time (days)')

        handles, labels_leg = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels_leg, loc='upper center',
                       ncol=len(handles), fontsize=9,
                       bbox_to_anchor=(0.5, 1.02))

        fig.suptitle(f"Neural ODE — {schema['label']}", fontsize=14, y=1.05)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_rom_trajectories.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)

    # ── 2. Loss Convergence Plot ─────────────────────────────────────
    mean_loss = np.mean(all_member_losses, axis=0)
    fig_loss, ax_loss = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss[0].plot(mean_loss, lw=0.8, color='tab:orange')
    ax_loss[0].set_xlabel('Training Step')
    ax_loss[0].set_ylabel('MSE Loss')
    ax_loss[0].set_title('Loss Convergence (ensemble mean)')
    ax_loss[0].grid(True, alpha=0.3)
    half = len(mean_loss) // 2
    ax_loss[1].plot(range(half, len(mean_loss)), mean_loss[half:],
                    lw=0.8, color='tab:orange')
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
    n_modes_show = min(2, num_modes)
    ics_to_show = list(range(min(2, n_ics_total)))
    if n_ics_total > 2:
        ics_to_show.append(n_ics_total - 1)  # test IC

    fig_sp, axes_sp = plt.subplots(
        len(ics_to_show), n_modes_show,
        figsize=(6 * n_modes_show, 3 * len(ics_to_show)),
        sharex=True, squeeze=False,
    )
    for ri, ic_idx in enumerate(ics_to_show):
        rom_solves = all_rom_solves[ic_idx]
        true_interp = interp1d(t_full, all_true_comp[ic_idx],
                               kind='cubic', fill_value='extrapolate')
        true_at_pred = true_interp(t_pred)
        for ci in range(n_modes_show):
            ax = axes_sp[ri, ci]
            ax.plot(t_pred, true_at_pred[ci], color='tab:gray', lw=2, label='True')
            if len(rom_solves) > 0:
                for m in range(len(rom_solves)):
                    ax.plot(t_pred, rom_solves[m, ci, :],
                            color='tab:orange', alpha=0.1, lw=0.5)
            ax.axvline(training_span[1], color='k', ls=':', lw=0.8, alpha=0.5)
            if ri == 0:
                ax.set_title(f'Mode {ci + 1}')
            if ci == 0:
                ax.set_ylabel(eval_labels[ic_idx], fontsize=8)
            if ri == len(ics_to_show) - 1:
                ax.set_xlabel('Time (days)')
    fig_sp.suptitle(f"Ensemble Spread — {schema['label']}", fontsize=12, y=1.02)
    fig_sp.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_ensemble_spread.png")
    fig_sp.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig_sp)

    # ── 4. Full-Order Error Plot (first training IC, capped at 20) ───
    basis = result.get('basis')
    all_true_states_full = result.get('all_true_states')
    if basis is not None and all_true_states_full is not None and len(all_rom_solves) > 0:
        first_ic_solves = all_rom_solves[0]
        if len(first_ic_solves) > 0:
            first_true_full = all_true_states_full[0]
            rom_arr_capped = first_ic_solves[:min(20, len(first_ic_solves))]
            fig_foe, axes_foe = plot_full_order_error(
                rom_solves=rom_arr_capped,
                basis=basis,
                true_states=first_true_full,
                time_domain_full=t_full,
                time_domain_eval=t_pred,
                training_span=training_span,
                error_type='relative',
            )
            fig_foe.suptitle(f'Full-Order Error (IC 0) — {schema["label"]}', fontsize=14)
            path = os.path.join(save_dir, f"{prefix}_full_order_error.png")
            fig_foe.savefig(path, dpi=200, bbox_inches='tight')
            print(f"  📊 Saved: {path}")
            plt.close(fig_foe)

    # ── 5. Per-IC Error Bar Chart ────────────────────────────────────
    all_train_errors = result.get('all_train_errors', [])
    all_pred_errors = result.get('all_pred_errors', [])
    n_train_ics = len(TRAINING_PARAMS)

    if all_train_errors and all_pred_errors:
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        x = np.arange(n_ics_total)
        width = 0.35
        train_errs_pct = [e * 100 if np.isfinite(e) else 0.0 for e in all_train_errors]
        pred_errs_pct = [e * 100 if np.isfinite(e) else 0.0 for e in all_pred_errors]

        colors_train = ['tab:blue'] * n_train_ics + ['tab:cyan']
        colors_pred = ['tab:red'] * n_train_ics + ['tab:orange']
        for i in range(n_ics_total):
            ax_bar.bar(x[i] - width/2, train_errs_pct[i], width,
                       color=colors_train[i], alpha=0.7,
                       label='Train' if i == 0 else None)
            ax_bar.bar(x[i] + width/2, pred_errs_pct[i], width,
                       color=colors_pred[i], alpha=0.7,
                       label='Pred' if i == 0 else None)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([l.replace('Train ', 'Tr ').replace('Test ', 'Te ')
                                for l in eval_labels], rotation=30, ha='right', fontsize=8)
        ax_bar.set_ylabel('Relative Error (%)')
        ax_bar.set_title(f'Per-IC Error — {schema["label"]}')
        ax_bar.legend()
        ax_bar.grid(True, alpha=0.2, axis='y')
        fig_bar.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_per_ic_error.png")
        fig_bar.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig_bar)

    # ── 6. Spatial Heatmap Comparison (test IC) ──────────────────────
    plot_spatial_comparison(result, save_dir=save_dir)

    # ── 7. Tumor Volume Plot (all ICs) ──────────────────────────────
    plot_tumor_volume(result, save_dir=save_dir)


# =============================================================================
# Save predictions for cross-method comparison
# =============================================================================
def save_predictions(result, save_dir=None):
    """Save predictions for cross-method comparison."""
    schema = result['schema']
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, "results", "comparison", schema['name'])
    os.makedirs(save_dir, exist_ok=True)

    all_rom_solves = result['all_rom_solves']
    method_name = "05_neural_ode"

    save_dict = {
        't_pred': result['t_pred'],
        'train_error': result['train_error'],
        'pred_error': result['pred_error'],
        'stability_pct': result['stability_pct'],
        'ci_coverage': result.get('ci_coverage', float('nan')),
        'ci_width': result.get('ci_width', float('nan')),
        'runtime': result['runtime'],
        'n_ics': len(all_rom_solves),
        'test_train_error': result.get('test_train_error', float('nan')),
        'test_pred_error': result.get('test_pred_error', float('nan')),
    }
    for ic_idx, solves in enumerate(all_rom_solves):
        if len(solves) > 0:
            save_dict[f'rom_solves_{ic_idx}'] = np.array(solves)
        else:
            save_dict[f'rom_solves_{ic_idx}'] = np.empty(
                (0, result['num_modes'], len(result['t_pred'])))

    path = os.path.join(save_dir, f"{method_name}.npz")
    np.savez(path, **save_dict)
    print(f"  💾 Saved predictions: {path}")


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
    print("05 — Neural ODE Ensemble — Tumor Growth (Multi-Trajectory)")
    print("=" * 70)
    print(f"Regimes: {len(schemas)}")
    for s in schemas:
        print(f"  • {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  noise={s['NOISE_LEVEL']:.0%}")
    print(f"Model:  {MODEL_PARAMS['NUM_LAYERS']}×{MODEL_PARAMS['HIDDEN_DIM']} "
          f"{MODEL_PARAMS['ACTIVATION']}, ensemble={MODEL_PARAMS['ENSEMBLE_SIZE']}, "
          f"lr={MODEL_PARAMS['LEARNING_RATE']}, steps={MODEL_PARAMS['NUM_TRAIN_STEPS']}")
    print(f"ICs:    {MODEL_PARAMS['NUM_ICS']} training + 1 test ({TEST_PARAMS})")

    results = []
    for schema in schemas:
        r = run_experiment(schema)
        plot_results(r)
        save_predictions(r)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"SUMMARY — Neural ODE Ensemble (Tumor Growth, Multi-Traj)")
    print(f"{'='*90}")
    print(f"{'Regime':<28s} {'Samp':>4s} {'Noise':>5s} {'Stab':>5s} "
          f"{'Train':>8s} {'Pred':>8s} {'TestPr':>8s} {'CI_cov':>7s} {'Time':>6s}")
    print(f"{'-'*28} {'-'*4} {'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
    for r in results:
        s = r['schema']
        print(f"{s['label']:<28s} {s['NUM_SAMPLES']:>4d} "
              f"{s['NOISE_LEVEL']:>4.0%} {r['stability_pct']:>4.0f}% "
              f"{r['train_error']:>7.2%} {r['pred_error']:>7.2%} "
              f"{r['test_pred_error']:>7.2%} "
              f"{r['ci_coverage']:>6.1%} {r['runtime']:>5.0f}s")


if __name__ == "__main__":
    schema_names = sys.argv[1:] if len(sys.argv) > 1 else None
    main(schema_names)

"""
05 — Neural ODE Ensemble Baseline (Tumor Growth + Chemotherapy) — Single Trajectory

Black-box baseline for comparison with Bayesian OpInf chemo (04_unified_chemo.py).
The MLP takes the reduced state q AND the chemo input α(t) as inputs:

    dq/dt = f_θ(q, α(t))     (concat scalar α as MLP input → in-dim = r+1)

Training: single noisy chemo trajectory, MSE on rollout.
UQ: ensemble of independently trained networks, 5–95% bands.

Mirrors the chemo Bayesian OpInf script:
  - Same FOM data (load_chemo_fom_data)
  - Same training span / prediction span
  - Same α(t) tabulation (make_jax_input_func) so dynamics see exact same input

Data regimes:
  1. Dense data, low noise    (80 samples, 1% noise)
  2. Dense data, medium noise (80 samples, 3% noise)
  3. Dense data, high noise   (80 samples, 5% noise)

Usage:
    python 05_neural_ode_chemo.py                  # all 3 regimes
    python 05_neural_ode_chemo.py dense_low_noise  # one regime
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
    Basis, TumorTwinFOM, load_chemo_fom_data, make_jax_input_func,
)
from core.plotting import plot_full_order_error

# Match chemo OpInf script.
# Match 04 chemo: tighter training span on sparser/larger-dose schedule.
TRAINING_SPAN = (5.0, 70.0)
PREDICTION_DAYS = 110.0
FOM_DATA_PATH = os.path.join(
    os.path.dirname(__file__), 'data',
    'TNBC_demo_001_fom_chemo_sparse5_sens0p5.npz'
)

SCHEMAS = [
    {"name": "dense_low_noise",    "label": "Dense data, low noise",
     "NUM_SAMPLES": 200, "NOISE_LEVEL": 0.01, "NUM_EVAL_POINTS": 400},
    {"name": "dense_medium_noise", "label": "Dense data, medium noise",
     "NUM_SAMPLES": 200, "NOISE_LEVEL": 0.03, "NUM_EVAL_POINTS": 400},
    {"name": "dense_high_noise",   "label": "Dense data, high noise",
     "NUM_SAMPLES": 200, "NOISE_LEVEL": 0.05, "NUM_EVAL_POINTS": 400},
]

MODEL_PARAMS = dict(
    NUM_MODES=4,
    HIDDEN_DIM=128,
    NUM_LAYERS=3,
    ENSEMBLE_SIZE=20,
    NUM_TRAIN_STEPS=6000,
    LEARNING_RATE=5e-4,
    GRAD_CLIP=1.0,
    SEED=42,
    # Outlier filtering: deep ensembles trained from random inits commonly
    # have a fraction of members converge to bad local minima. Their
    # trajectories blow out the 5-95% percentile bands even when the median
    # is fine. We drop any member whose final training loss exceeds
    # `LOSS_OUTLIER_FACTOR` times the median final loss across the ensemble.
    LOSS_OUTLIER_FACTOR=3.0,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# Neural ODE: f_θ(q, α(t))   — α concatenated as extra input
# =============================================================================
class ChemoNeuralODE(eqx.Module):
    """MLP dynamics: dq/dt = f_θ([q; α(t)]) → r outputs.

    The chemo input function is supplied at integration time via the
    diffrax `args` parameter (avoids closing over a non-JAX object).
    """
    layers: list

    def __init__(self, num_modes, hidden_dim, num_layers, *, key):
        keys = random.split(key, num_layers + 1)
        self.layers = []
        d_in = num_modes + 1  # state (r) + α (1)
        for i in range(num_layers):
            self.layers.append(eqx.nn.Linear(d_in, hidden_dim, key=keys[i]))
            d_in = hidden_dim
        # output: r derivatives
        self.layers.append(eqx.nn.Linear(hidden_dim, num_modes,
                                         key=keys[num_layers]))

    def __call__(self, t, y, args):
        # `args` carries the JAX-tabulated input function (interpolator).
        ifn = args
        alpha = jnp.atleast_1d(ifn(t))  # shape (1,)
        x = jnp.concatenate([y, alpha], axis=-1)
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


def _solve_trajectory(model, q0, t_obs, ifn_jax):
    """Integrate the MLP from q0 over t_obs with chemo input ifn_jax."""
    term = diffrax.ODETerm(model)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_obs)
    adjoint = diffrax.RecursiveCheckpointAdjoint()
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=t_obs[0], t1=t_obs[-1],
        dt0=t_obs[1] - t_obs[0],
        y0=q0,
        args=ifn_jax,
        saveat=saveat,
        adjoint=adjoint,
        max_steps=16384,
        throw=False,
    )
    return sol.ys  # (len(t_obs), num_modes)


@eqx.filter_jit
def _train_step(model, opt_state, q0, t_obs, y_obs, ifn_jax, opt):
    def loss_fn(model):
        y_pred = _solve_trajectory(model, q0, t_obs, ifn_jax)
        return jnp.mean((y_pred - y_obs.T) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_single_member(key, q0, t_obs, y_obs, ifn_jax, p):
    """Train one ensemble member."""
    model = ChemoNeuralODE(
        num_modes=p['NUM_MODES'],
        hidden_dim=p['HIDDEN_DIM'],
        num_layers=p['NUM_LAYERS'],
        key=key,
    )
    opt = optax.chain(
        optax.clip_by_global_norm(p['GRAD_CLIP']),
        optax.adam(p['LEARNING_RATE']),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    losses = []
    num_steps = p['NUM_TRAIN_STEPS']
    for step in range(num_steps):
        model, opt_state, loss = _train_step(
            model, opt_state, q0, t_obs, y_obs, ifn_jax, opt,
        )
        losses.append(float(loss))
        if step % 500 == 0 or step == num_steps - 1:
            print(f"      step {step:5d}/{num_steps}  loss={losses[-1]:.6f}")
    return model, np.array(losses)


def train_ensemble(q0, t_obs, y_obs, ifn_jax, p):
    base_key = jax.random.PRNGKey(p['SEED'])
    keys = jax.random.split(base_key, p['ENSEMBLE_SIZE'])
    ensemble = []
    for m in range(p['ENSEMBLE_SIZE']):
        print(f"    ── Ensemble member {m + 1}/{p['ENSEMBLE_SIZE']} ──")
        ensemble.append(train_single_member(
            keys[m], q0, t_obs, y_obs, ifn_jax, p))
    return ensemble


def evaluate_ensemble(ensemble, q0, t_pred, ifn_jax):
    t_pred_jnp = jnp.array(t_pred)
    q0_jnp = jnp.array(q0)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_pred_jnp)

    solves = []
    for model, _ in ensemble:
        try:
            term = diffrax.ODETerm(model)
            sol = diffrax.diffeqsolve(
                term, solver,
                t0=float(t_pred[0]),
                t1=float(t_pred[-1]),
                dt0=float(t_pred[1] - t_pred[0]),
                y0=q0_jnp,
                args=ifn_jax,
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
        return np.stack(solves, axis=0)
    return np.empty((0, q0.shape[0], len(t_pred)))


# =============================================================================
# Run experiment
# =============================================================================
def run_experiment(schema):
    p = MODEL_PARAMS
    np.random.seed(p['SEED'])

    noise_level = schema['NOISE_LEVEL']
    num_samples = schema['NUM_SAMPLES']
    num_eval_points = schema['NUM_EVAL_POINTS']
    num_modes = p['NUM_MODES']

    print(f"\n{'='*70}")
    print(f"  {schema['label']}  ({num_samples} samples, {noise_level:.0%} noise)")
    print(f"{'='*70}")

    # ── Data (chemo, single trajectory) ──────────────────────────────────
    t_pred = np.linspace(TRAINING_SPAN[0], PREDICTION_DAYS, num_eval_points)

    fom, t_full, true_states, t_samp, snaps_noisy, ifn, chemo_meta = \
        load_chemo_fom_data(FOM_DATA_PATH, t_pred, TRAINING_SPAN,
                            num_samples, noise_level, seed=p['SEED'])

    ifn_jax = make_jax_input_func(
        ifn, float(t_pred[0]), float(t_pred[-1]), n_points=4001)

    print(f"  Chemo: {len(chemo_meta['dose_days'])} doses, "
          f"sens={chemo_meta['sensitivity']:.2f}")

    # ── POD basis (clean snapshots) ──────────────────────────────────────
    snaps_clean = fom.get_states(t_samp)
    basis = Basis(num_vectors=num_modes)
    basis.fit(snaps_clean)
    snaps_comp = basis.compress(snaps_noisy)
    true_comp = basis.compress(true_states)
    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    # ── Train ensemble ───────────────────────────────────────────────────
    q0 = jnp.array(snaps_comp[:, 0])
    t_obs = jnp.array(t_samp)
    y_obs = jnp.array(snaps_comp)

    print(f"\n  Training {p['ENSEMBLE_SIZE']} ensemble members "
          f"({p['NUM_TRAIN_STEPS']} steps each)...")
    t0 = time.time()
    ensemble = train_ensemble(q0, t_obs, y_obs, ifn_jax, p)
    runtime = time.time() - t0
    print(f"  Training time: {runtime:.0f}s")

    # ── Filter outlier ensemble members by final train loss ──────────────
    # Some random inits get stuck in bad local minima and produce
    # trajectories that blow out percentile bands. Drop members whose
    # final loss is much larger than the ensemble median.
    final_losses = np.array([losses[-1] for _, losses in ensemble])
    med_loss = float(np.median(final_losses))
    cutoff = med_loss * p['LOSS_OUTLIER_FACTOR']
    keep_mask = final_losses <= cutoff
    n_kept = int(keep_mask.sum())
    n_dropped = len(ensemble) - n_kept
    print(f"\n  Ensemble loss filter: median={med_loss:.4g}, "
          f"cutoff={cutoff:.4g}  →  kept {n_kept}/{len(ensemble)} "
          f"(dropped {n_dropped} outlier{'s' if n_dropped != 1 else ''})")
    if n_dropped > 0:
        for i, fl in enumerate(final_losses):
            if not keep_mask[i]:
                print(f"    ✗ member {i:2d}: final loss {fl:.4g} (>{cutoff:.4g})")
    kept_ensemble = [m for m, k in zip(ensemble, keep_mask) if k]

    # ── Evaluate ─────────────────────────────────────────────────────────
    rom_solves = evaluate_ensemble(kept_ensemble, q0, t_pred, ifn_jax)
    n_stable = len(rom_solves)
    n_total = p['ENSEMBLE_SIZE']
    stability_pct = n_stable / n_total * 100

    train_error = pred_error = float('inf')
    ci_coverage = ci_width = float('nan')
    train_mask = t_pred <= TRAINING_SPAN[1]
    pred_mask = t_pred > TRAINING_SPAN[1]

    if n_stable > 0:
        rom_med = np.median(rom_solves, axis=0)
        ti = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)
        train_error = float(np.linalg.norm(rom_med[:, train_mask] - ta[:, train_mask]) /
                            np.linalg.norm(ta[:, train_mask]))
        pred_error = float(np.linalg.norm(rom_med[:, pred_mask] - ta[:, pred_mask]) /
                           np.linalg.norm(ta[:, pred_mask]))
        q05 = np.percentile(rom_solves, 5, axis=0)
        q95 = np.percentile(rom_solves, 95, axis=0)
        ci_width = float(np.mean(q95 - q05))
        ci_coverage = float(np.mean((ta >= q05) & (ta <= q95)))

    all_member_losses = np.stack([losses for _, losses in ensemble], axis=0)

    print(f"\n  Results ({runtime:.0f}s):")
    print(f"    Stability: {n_stable}/{n_total} ({stability_pct:.0f}%)")
    print(f"    Train error: {train_error:.4%}  |  Pred error: {pred_error:.4%}")
    print(f"    CI coverage: {ci_coverage:.2%} (target: 90%)")

    return {
        'schema': schema,
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'n_stable': n_stable, 'n_total': n_total,
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'runtime': runtime,
        'losses': all_member_losses,
        'rom_solves': rom_solves,
        'kept_ensemble': kept_ensemble,
        'q0': q0,
        'snaps_comp': snaps_comp, 'true_comp': true_comp,
        't_full': t_full, 't_pred': t_pred, 't_samp': t_samp,
        'training_span': TRAINING_SPAN, 'num_modes': num_modes,
        'true_states': true_states, 'basis': basis,
        'fom': fom,
        'chemo_meta': chemo_meta,
    }


# =============================================================================
# Plotting (single trajectory; mirrors 04 chemo layout)
# =============================================================================
def plot_results(result, save_dir=None):
    """Standard diagnostic figures via the centralized plotting package."""
    from core.plotting import RunResult, figures
    if save_dir is None:
        save_dir = FIGURE_DIR
    run = RunResult.from_flat(result, "05_neural_ode_chemo")
    figures.standard(run, save_dir, f"05_{result['schema']['name']}",
                     layout="windows")


def plot_uncertainty_panel(result, save_dir, timepoints_to_show=None):
    """3-row × N-col panel: FOM truth | Neural ODE median | 5–95% width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schema = result['schema']
    prefix = f"05_chemo_{schema['name']}"
    basis = result['basis']
    t_full = result['t_full']
    t_pred = result['t_pred']
    rom_solves = result['rom_solves']
    true_states = result['true_states']
    fom = result['fom']

    if len(rom_solves) == 0:
        print("  ⚠ No stable ensemble members — skipping uncertainty panel")
        return

    rom_arr = np.array(rom_solves)
    rom_med = np.median(rom_arr, axis=0)

    if timepoints_to_show is None:
        timepoints_to_show = [5, 30, 50, 70, 90, 105]
    n_times = len(timepoints_to_show)

    fig, axes = plt.subplots(3, n_times, figsize=(3.5 * n_times, 10),
                              constrained_layout=True)

    # First pass: compute everything, find global width max for color scale.
    panels = []
    width_max = 0.0
    for col, t_target in enumerate(timepoints_to_show):
        idx_pred = np.argmin(np.abs(t_pred - t_target))
        idx_full = np.argmin(np.abs(t_full - t_target))

        fom_state = true_states[:, idx_full]
        rom_full_med = basis.decompress(rom_med[:, idx_pred])

        n_e = rom_arr.shape[0]
        full_states = np.stack(
            [basis.decompress(rom_arr[s, :, idx_pred])
             for s in range(n_e)], axis=0
        )
        q05 = np.percentile(full_states, 5, axis=0)
        q95 = np.percentile(full_states, 95, axis=0)
        width = q95 - q05

        fom_slices = fom.get_center_slices(fom_state)
        rom_slices = fom.get_center_slices(rom_full_med)
        width_slices = fom.get_center_slices(width)
        panels.append((fom_slices, rom_slices, width_slices, t_full[idx_full]))
        width_max = max(width_max, float(width_slices['axial'].max()))

    width_max = max(width_max, 1e-9)

    for col, (fom_slices, rom_slices, width_slices, t_actual) in enumerate(
            panels):
        im0 = axes[0, col].imshow(fom_slices['axial'].T, origin='lower',
                                   cmap='hot_r', vmin=0, vmax=1, aspect='equal')
        axes[0, col].set_title(f'Day {t_actual:.0f}', fontsize=11)
        im1 = axes[1, col].imshow(rom_slices['axial'].T, origin='lower',
                                   cmap='hot_r', vmin=0, vmax=1, aspect='equal')
        im2 = axes[2, col].imshow(width_slices['axial'].T, origin='lower',
                                   cmap='viridis', vmin=0, vmax=width_max,
                                   aspect='equal')
        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        if col == 0:
            axes[0, col].set_ylabel('FOM Truth', fontsize=12, fontweight='bold')
            axes[1, col].set_ylabel('Neural ODE Median', fontsize=12,
                                    fontweight='bold')
            axes[2, col].set_ylabel('5–95% Width', fontsize=12,
                                    fontweight='bold')

    fig.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.8, label='Cellularity',
                 pad=0.02)
    fig.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.8, label='Cellularity',
                 pad=0.02)
    fig.colorbar(im2, ax=axes[2, :].tolist(), shrink=0.8, label='Width (5–95%)',
                 pad=0.02)
    fig.suptitle(f'Uncertainty Panel (axial slice) — {schema["label"]}',
                 fontsize=14)
    path = os.path.join(save_dir, f"{prefix}_uncertainty_panel.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


def plot_spatial_comparison(result, save_dir, timepoints_to_show=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schema = result['schema']
    prefix = f"05_chemo_{schema['name']}"
    basis = result['basis']
    t_full = result['t_full']
    t_pred = result['t_pred']
    rom_solves = result['rom_solves']
    true_states = result['true_states']
    fom = result['fom']

    if len(rom_solves) == 0:
        print("  ⚠ No stable Neural ODE solves — skipping spatial plot")
        return

    rom_med = np.median(rom_solves, axis=0)

    if timepoints_to_show is None:
        timepoints_to_show = [5, 15, 30, 45, 60, 90]

    n_times = len(timepoints_to_show)
    fig, axes = plt.subplots(3, n_times, figsize=(3.5 * n_times, 10),
                              constrained_layout=True)

    for col, t_target in enumerate(timepoints_to_show):
        idx_full = np.argmin(np.abs(t_full - t_target))
        fom_state = true_states[:, idx_full]

        idx_pred = np.argmin(np.abs(t_pred - t_target))
        rom_full = basis.decompress(rom_med[:, idx_pred])

        fom_slices = fom.get_center_slices(fom_state)
        rom_slices = fom.get_center_slices(rom_full)
        err_slices = fom.get_center_slices(np.abs(fom_state - rom_full))

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

    fig.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.8, label='Cellularity',
                 pad=0.02)
    fig.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.8, label='Cellularity',
                 pad=0.02)
    fig.colorbar(im2, ax=axes[2, :].tolist(), shrink=0.8, label='|Error|',
                 pad=0.02)
    fig.suptitle(f'Tumor + Chemo: FOM vs Neural ODE — {schema["label"]}',
                 fontsize=14)
    path = os.path.join(save_dir, f"{prefix}_spatial_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


def plot_tumor_volume(result, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schema = result['schema']
    prefix = f"05_chemo_{schema['name']}"
    basis = result['basis']
    t_full = result['t_full']
    t_pred = result['t_pred']
    rom_solves = result['rom_solves']
    true_states = result['true_states']
    fom = result['fom']
    training_span = result['training_span']

    V = basis.entries
    ones = np.ones(V.shape[0])
    vol_proj = V.T @ ones
    shift_vol = ones @ basis.shift_
    voxel_vol = float(np.prod(fom.spacing))

    fom_vol = np.array([true_states[:, i].sum() * voxel_vol
                        for i in range(true_states.shape[1])])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_full, fom_vol, color='tab:gray', lw=2.5, label='FOM truth')

    if len(rom_solves) > 0:
        ens_vols = np.array([vol_proj @ rom_solves[s] + shift_vol
                             for s in range(len(rom_solves))]) * voxel_vol
        ens_med = np.median(ens_vols, axis=0)
        ens_lo = np.percentile(ens_vols, 5, axis=0)
        ens_hi = np.percentile(ens_vols, 95, axis=0)
        ax.plot(t_pred, ens_med, color='tab:orange', ls='--', lw=2,
                label='Neural ODE median')
        ax.fill_between(t_pred, ens_lo, ens_hi, color='tab:orange', alpha=0.20,
                        label='Neural ODE 5-95%')

    ax.axvspan(training_span[0], training_span[1], color='gray', alpha=0.08,
               label='Training span')
    ax.axvline(training_span[1], color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Total Tumor Burden (mm³)')
    ax.set_title(f"Tumor Volume Over Time — {schema['label']}")
    ax.legend(loc='best')
    fig.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_tumor_volume.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
DOSE_VARIATION = False


def evaluate_dose_variation(result, dose_scales=(0.8, 1.0, 1.2),
                            save_dir=None):
    """Re-integrate trained Neural ODE ensemble at modified dose scales.

    Mirrors 04's evaluate_dose_variation: keeps the trained network weights
    fixed and only changes the α(t) input function fed to the integrator.
    Tests whether the MLP's input coupling generalizes to dose levels not
    seen at training. FOM at each dose scale must already be saved as
    `..._sparse5_sens0p5_dose<scale>.npz` (1.0 is the headline file).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schema = result['schema']
    prefix = f"05_chemo_{schema['name']}"
    if save_dir is None:
        save_dir = FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    kept_ensemble = result['kept_ensemble']
    basis = result['basis']
    t_pred = result['t_pred']
    chemo_meta = result['chemo_meta']
    training_span = result['training_span']
    fom_default = result['fom']
    q0 = result['q0']

    # Volume projection (same as plot_tumor_volume).
    V = basis.entries
    ones = np.ones(V.shape[0])
    vol_proj = V.T @ ones
    shift_vol = ones @ basis.shift_
    voxel_vol = float(np.prod(fom_default.spacing))

    spec = chemo_meta['chemo_spec']
    t0_chemo = chemo_meta['t0']
    fom_data_dir = os.path.join(SCRIPT_DIR, 'data')

    print(f"\n  ── Dose variation evaluation: scales {dose_scales} ──")
    fig, axes = plt.subplots(1, len(dose_scales),
                             figsize=(5 * len(dose_scales), 4.5),
                             sharey=True)
    if len(dose_scales) == 1:
        axes = [axes]

    summary = []
    for i, scale in enumerate(dose_scales):
        print(f"\n    Dose × {scale:g}")
        ifn_scaled = config.chemo_input_func_factory(spec, t0_chemo,
                                                    dose_scale=scale)
        ifn_jax_scaled = make_jax_input_func(
            ifn_scaled, float(t_pred[0]), float(t_pred[-1]), n_points=4001)

        rom_solves_scaled = evaluate_ensemble(
            kept_ensemble, q0, t_pred, ifn_jax_scaled)
        n_stable = len(rom_solves_scaled)
        n_total = len(kept_ensemble)
        print(f"      Stable: {n_stable}/{n_total}")

        scale_tag = f"dose{scale:g}".replace('.', 'p')
        fom_path = os.path.join(
            fom_data_dir,
            f'TNBC_demo_001_fom_chemo_sparse5_sens0p5_{scale_tag}.npz'
        )
        if not os.path.exists(fom_path) and abs(scale - 1.0) < 1e-9:
            fom_path = FOM_DATA_PATH
        if not os.path.exists(fom_path):
            print(f"      ⚠ FOM file missing for scale {scale}: {fom_path} "
                  f"— skipping panel")
            axes[i].set_title(f'Dose × {scale:g}\n(no FOM)')
            continue

        fom_s = TumorTwinFOM(fom_path)
        true_states_s = fom_s.get_states(t_pred)
        fom_vol = np.array([true_states_s[:, k].sum() * voxel_vol
                            for k in range(true_states_s.shape[1])])

        ax = axes[i]
        ax.plot(t_pred, fom_vol, color='tab:gray', lw=2.5, label='FOM Truth')
        if n_stable > 0:
            rom_arr_s = np.array(rom_solves_scaled)
            rom_vols = np.array([vol_proj @ rom_arr_s[s] + shift_vol
                                 for s in range(rom_arr_s.shape[0])]) * voxel_vol
            rom_med_v = np.median(rom_vols, axis=0)
            rom_lo = np.percentile(rom_vols, 5, axis=0)
            rom_hi = np.percentile(rom_vols, 95, axis=0)
            ax.plot(t_pred, rom_med_v, color='tab:orange', lw=2, ls='--',
                    label='Neural ODE median')
            ax.fill_between(t_pred, rom_lo, rom_hi, color='tab:orange',
                            alpha=0.15, label='Neural ODE 5–95%')
            err = float(np.linalg.norm(rom_med_v - fom_vol)
                        / np.linalg.norm(fom_vol))
            summary.append((scale, n_stable, err))
            ax.text(0.04, 0.94,
                    f'rel err (volume): {err:.2%}\nstab {n_stable}/{n_total}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.axvspan(training_span[0], training_span[1],
                   color='gray', alpha=0.10)
        ax.axvline(training_span[1], color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Time (days)')
        ax.set_title(f'Dose × {scale:g}', fontsize=12)
        if i == 0:
            ax.set_ylabel('Total Tumor Burden (mm³)')
        ax.legend(loc='upper right', fontsize=9, frameon=True)

    fig.suptitle(f'Dose-variation evaluation (Neural ODE) — {schema["label"]}',
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_dose_variation.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\n  📊 Saved: {path}")
    plt.close(fig)

    print("\n  Dose-variation summary:")
    print(f"  {'Scale':>6s} {'Stab':>5s} {'Volume rel err':>16s}")
    for scale, n_stable, err in summary:
        print(f"  {scale:>6.2g} {n_stable:>5d} {err:>15.2%}")
    return summary


def main(schema_names=None):
    print("=" * 70)
    print(" 05 — Neural ODE Ensemble (Tumor + Chemo, single trajectory)")
    print("=" * 70)

    if schema_names:
        sel = [s for s in SCHEMAS if s['name'] in schema_names]
    else:
        sel = SCHEMAS
    print(f"Regimes: {len(sel)}")
    for s in sel:
        print(f"  • {s['label']:30s} samples={s['NUM_SAMPLES']:>3} "
              f"noise={s['NOISE_LEVEL']:.0%}")
    p = MODEL_PARAMS
    print(f"Model: ensemble={p['ENSEMBLE_SIZE']}, hidden={p['HIDDEN_DIM']}, "
          f"steps={p['NUM_TRAIN_STEPS']}, lr={p['LEARNING_RATE']}")

    summary = []
    for schema in sel:
        r = run_experiment(schema)
        plot_results(r)

        # save predictions for cross-method comparison
        out_dir = os.path.join(SCRIPT_DIR, 'results', 'comparison',
                               schema['name'])
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, '05_neural_ode.npz'),
            t_pred=r['t_pred'], t_full=r['t_full'],
            rom_solves=r['rom_solves'], true_comp=r['true_comp'],
            train_error=r['train_error'], pred_error=r['pred_error'],
            ci_coverage=r['ci_coverage'], ci_width=r['ci_width'],
            stability_pct=r['stability_pct'], runtime=r['runtime'],
        )
        print(f"  💾 Saved predictions: {out_dir}/05_neural_ode.npz")

        if DOSE_VARIATION:
            try:
                evaluate_dose_variation(r)
            except Exception as e:
                print(f"  ⚠ Dose-variation eval failed: {e}")

        summary.append((schema['label'], schema['NUM_SAMPLES'],
                        schema['NOISE_LEVEL'],
                        r['stability_pct'], r['train_error'], r['pred_error'],
                        r['ci_coverage'], r['runtime']))

    print("\n" + "=" * 80)
    print("SUMMARY — Neural ODE (Tumor + Chemo)")
    print("=" * 80)
    print(f"{'Regime':28s} {'Samp':>4s} {'Noise':>5s} {'Stab':>5s} "
          f"{'Train':>8s} {'Pred':>8s} {'CI_cov':>7s} {'Time':>6s}")
    print("-" * 80)
    for lbl, ns, nl, st, te, pe, ci, rt in summary:
        print(f"{lbl:28s} {ns:>4d} {nl:>4.0%} {st:>4.0f}% "
              f"{te:>7.2%} {pe:>7.2%} {ci:>6.1%} {rt:>5.0f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('schemas', nargs='*',
                        help='Schema names to run (default: all)')
    parser.add_argument('--dose-variation', action='store_true',
                        help='Evaluate at dose scales {0.8, 1.0, 1.2}')
    args = parser.parse_args()
    DOSE_VARIATION = args.dose_variation
    main(args.schemas if args.schemas else None)

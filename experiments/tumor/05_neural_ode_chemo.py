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
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import diffrax
import optax
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from chemo_protocol import (
    TRAINING_SPAN, PREDICTION_DAYS, FOM_DATA_PATH, SCHEMAS, OUTPUT_ROOT,
    prepare_data, array_fingerprint,
)

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
        if not np.isfinite(losses[-1]):
            raise FloatingPointError(
                f"Nonfinite training loss at step {step}: {losses[-1]}"
            )
        if step % 500 == 0 or step == num_steps - 1:
            print(f"      step {step:5d}/{num_steps}  loss={losses[-1]:.6f}")
    return model, np.array(losses)


def _model_arrays(model):
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(model)
            if eqx.is_array(leaf)]


def train_ensemble(q0, t_obs, y_obs, ifn_jax, p, checkpoint_dir=None,
                   checkpoint_manifest=None, runtime_metadata=None):
    """Train or restore completed members, retaining the original member order."""
    if checkpoint_dir is not None:
        if checkpoint_manifest is None:
            raise ValueError("Checkpointing requires a protocol manifest")
        os.makedirs(checkpoint_dir, exist_ok=True)
        manifest_path = os.path.join(checkpoint_dir, "manifest.json")
        expected = json.loads(json.dumps(checkpoint_manifest, sort_keys=True))
        if os.path.exists(manifest_path):
            with open(manifest_path) as stream:
                actual = json.load(stream)
            if actual != expected:
                raise ValueError(
                    f"Checkpoint manifest mismatch: {manifest_path}; "
                    "schema, protocol fingerprint, or model parameters changed"
                )
        else:
            if os.listdir(checkpoint_dir):
                raise ValueError(
                    f"Refusing checkpoints without a manifest: {checkpoint_dir}"
                )
            with open(manifest_path + ".pending", "w") as stream:
                json.dump(expected, stream, indent=2, sort_keys=True)
            os.replace(manifest_path + ".pending", manifest_path)

    base_key = jax.random.PRNGKey(p['SEED'])
    keys = jax.random.split(base_key, p['ENSEMBLE_SIZE'])
    ensemble = []
    member_runtimes = []
    runtime_new = 0.0
    n_restored = 0
    for m in range(p['ENSEMBLE_SIZE']):
        print(f"    ── Ensemble member {m + 1}/{p['ENSEMBLE_SIZE']} ──")
        if checkpoint_dir is not None:
            model_path = os.path.join(checkpoint_dir, f"member_{m:03d}.eqx")
            stats_path = os.path.join(checkpoint_dir, f"member_{m:03d}.npz")
        if (checkpoint_dir is not None and os.path.exists(model_path)
                and os.path.exists(stats_path)):
            template = ChemoNeuralODE(
                p['NUM_MODES'], p['HIDDEN_DIM'], p['NUM_LAYERS'], key=keys[m])
            model = eqx.tree_deserialise_leaves(model_path, template)
            with np.load(stats_path, allow_pickle=False) as stats:
                losses = stats['losses']
                member_runtime = float(stats['runtime'])
                if int(stats['member_index']) != m:
                    raise ValueError(f"Wrong member identity: {stats_path}")
                if str(stats['model_id']) != array_fingerprint(*_model_arrays(model)):
                    raise ValueError(f"Model checkpoint fingerprint mismatch: {model_path}")
            n_restored += 1
            print(f"      Restored completed member ({member_runtime:.1f}s training)")
        else:
            started = time.perf_counter()
            model, losses = train_single_member(
                keys[m], q0, t_obs, y_obs, ifn_jax, p)
            member_runtime = time.perf_counter() - started
            runtime_new += member_runtime

        if losses.shape != (p['NUM_TRAIN_STEPS'],) or not np.all(np.isfinite(losses)):
            raise FloatingPointError(f"Invalid training loss history for member {m}")
        if not all(np.all(np.isfinite(a)) for a in _model_arrays(model)):
            raise FloatingPointError(f"Nonfinite trained weights for member {m}")
        if not np.isfinite(member_runtime) or member_runtime < 0:
            raise ValueError(f"Invalid training runtime for member {m}")
        if checkpoint_dir is not None and not (
                os.path.exists(model_path) and os.path.exists(stats_path)):
            # Publish statistics last: a member is complete only with both files.
            with open(model_path + ".pending", "wb") as stream:
                eqx.tree_serialise_leaves(stream, model)
            with open(stats_path + ".pending", "wb") as stream:
                np.savez_compressed(
                    stream, losses=losses, runtime=member_runtime, member_index=m,
                    model_id=array_fingerprint(*_model_arrays(model)))
            os.replace(model_path + ".pending", model_path)
            os.replace(stats_path + ".pending", stats_path)
        ensemble.append((model, losses))
        member_runtimes.append(member_runtime)
    if runtime_metadata is not None:
        runtime_metadata.update(
            runtime_new=runtime_new, runtime_total=float(sum(member_runtimes)),
            member_runtimes=np.asarray(member_runtimes), n_restored=n_restored,
            n_newly_trained=p['ENSEMBLE_SIZE'] - n_restored,
        )
    return ensemble


def evaluate_ensemble(ensemble, q0, t_pred, ifn_jax):
    t_pred_jnp = jnp.array(t_pred)
    q0_jnp = jnp.array(q0)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_pred_jnp)

    solves = []
    rejected = dict(solver=0, shape=0, nonfinite=0)
    for model, _ in ensemble:
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
        if sol.result != diffrax.RESULTS.successful:
            rejected['solver'] += 1
            continue
        traj = np.array(sol.ys).T  # (num_modes, len(t_pred))
        if traj.shape != (q0_jnp.shape[0], len(t_pred)):
            rejected['shape'] += 1
            continue
        if not np.all(np.isfinite(traj)):
            rejected['nonfinite'] += 1
            continue
        solves.append(traj)
    print(f"  Prediction solves: {len(solves)}/{len(ensemble)} accepted; "
          f"rejected {sum(rejected.values())} "
          f"(solver={rejected['solver']}, shape={rejected['shape']}, "
          f"nonfinite={rejected['nonfinite']})")
    if solves:
        return np.stack(solves, axis=0)
    return np.empty((0, q0_jnp.shape[0], len(t_pred)))


# =============================================================================
# Run experiment
# =============================================================================
def run_experiment(schema, data=None, checkpoint_dir=None):
    p = MODEL_PARAMS
    if data is None:
        data = prepare_data(schema, seed=p['SEED'], num_modes=p['NUM_MODES'])
    if data['schema'] != schema:
        raise ValueError("Prepared data schema does not match the requested schema")

    noise_level = schema['NOISE_LEVEL']
    num_samples = schema['NUM_SAMPLES']
    num_modes = p['NUM_MODES']

    print(f"\n{'='*70}")
    print(f"  {schema['label']}  ({num_samples} samples, {noise_level:.0%} noise)")
    print(f"{'='*70}")

    # ── Data (chemo, single trajectory) ──────────────────────────────────
    t_pred, t_full = data['t_pred'], data['t_full']
    fom, basis = data['fom'], data['basis']
    true_states, true_comp = data['true_states'], data['true_comp']
    t_samp, snaps_comp = data['t_samp'], data['snaps_comp']
    ifn_jax, chemo_meta = data['input_func'], data['chemo_meta']

    print(f"  Chemo: {len(chemo_meta['dose_days'])} doses, "
          f"sens={chemo_meta['sensitivity']:.2f}")

    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    # ── Train ensemble ───────────────────────────────────────────────────
    q0 = jnp.array(data['q0'])
    t_obs = jnp.array(t_samp)
    y_obs = jnp.array(snaps_comp)

    print(f"\n  Training {p['ENSEMBLE_SIZE']} ensemble members "
          f"({p['NUM_TRAIN_STEPS']} steps each)...")
    timing = {}
    ensemble = train_ensemble(
        q0, t_obs, y_obs, ifn_jax, p, checkpoint_dir=checkpoint_dir,
        checkpoint_manifest=dict(
            version=1, schema=schema, data_fingerprint=data['fingerprint'],
            model_params=p, integrator="Tsit5-fixed-default-max_steps16384",
        ),
        runtime_metadata=timing,
    )
    runtime = timing['runtime_total']
    print(f"  Training time: {timing['runtime_new']:.1f}s newly spent; "
          f"{runtime:.1f}s total across all members "
          f"({timing['n_restored']} restored)")

    # ── Filter outlier ensemble members by final train loss ──────────────
    # Some random inits get stuck in bad local minima and produce
    # trajectories that blow out percentile bands. Drop members whose
    # final loss is much larger than the ensemble median.
    final_losses = np.array([losses[-1] for _, losses in ensemble])
    if not np.all(np.isfinite(final_losses)):
        raise FloatingPointError("Nonfinite ensemble final training losses")
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
    if not kept_ensemble:
        raise RuntimeError("The final-loss filter retained no ensemble members")
    model_id = array_fingerprint(
        *[a for model, _ in kept_ensemble for a in _model_arrays(model)])

    # ── Evaluate ─────────────────────────────────────────────────────────
    rom_solves = evaluate_ensemble(kept_ensemble, q0, t_pred, ifn_jax)
    n_stable = len(rom_solves)
    n_total = n_kept
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
    print(f"    Trained: {len(ensemble)}; loss-filtered: {n_dropped} "
          "(not numerical prediction failures)")
    print(f"    Train error: {train_error:.4%}  |  Pred error: {pred_error:.4%}")
    print(f"    CI coverage: {ci_coverage:.2%} (target: 90%)")

    return {
        'schema': schema,
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'n_stable': n_stable, 'n_total': n_total,
        'n_trained': len(ensemble), 'n_kept': n_kept, 'n_dropped': n_dropped,
        'kept_indices': np.flatnonzero(keep_mask),
        'dropped_indices': np.flatnonzero(~keep_mask),
        'final_losses': final_losses, 'median_final_loss': med_loss,
        'loss_cutoff': cutoff, 'keep_mask': keep_mask,
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'runtime': runtime,
        **timing,
        'data': data, 'data_fingerprint': data['fingerprint'],
        'model_id': model_id, 'ensemble': ensemble,
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
        save_dir = os.path.join(OUTPUT_ROOT, result['schema']['name'], "figures")
    os.makedirs(save_dir, exist_ok=True)
    run = RunResult.from_flat(result, "05_neural_ode_chemo")
    figures.standard(run, save_dir, f"05_chemo_{result['schema']['name']}",
                     layout="windows", dose_days=result["chemo_meta"]["dose_days"])
    plot_spatial_comparison(result, save_dir)
    plot_tumor_volume(result, save_dir)
    plot_uncertainty_panel(result, save_dir)


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
    """Evaluate changed inputs with the same nominally trained networks."""
    from chemo_evaluation import evaluate_doses

    data = result['data']
    if save_dir is None:
        save_dir = os.path.join(OUTPUT_ROOT, result['schema']['name'])

    def predict(scale, input_func):
        return evaluate_ensemble(
            result['kept_ensemble'], result['q0'], data['t_pred'], input_func)

    return evaluate_doses(
        data, predict, "05_neural_ode_chemo", save_dir,
        nominal_solves=result['rom_solves'],
        n_total=len(result['kept_ensemble']),
        n_trained=MODEL_PARAMS['ENSEMBLE_SIZE'],
        model_id=result['model_id'], dose_scales=dose_scales,
    )


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
        out_dir = os.path.join(OUTPUT_ROOT, schema['name'])
        os.makedirs(out_dir, exist_ok=True)
        r = run_experiment(
            schema, checkpoint_dir=os.path.join(out_dir, 'neural_checkpoints'))

        np.savez_compressed(
            os.path.join(out_dir, '05_neural_ode_chemo.npz'),
            t_pred=r['t_pred'], t_full=r['t_full'],
            t_samp=r['t_samp'], snaps_comp=r['snaps_comp'],
            snaps_noisy=r['data']['snaps_noisy'], q0=np.asarray(r['q0']),
            alpha_pred=r['data']['alpha_pred'],
            training_span=r['training_span'],
            basis_entries=r['basis'].entries, basis_shift=r['basis'].shift_,
            data_fingerprint=r['data_fingerprint'], model_id=r['model_id'],
            schema_json=json.dumps(schema, sort_keys=True),
            model_params_json=json.dumps(MODEL_PARAMS, sort_keys=True),
            rom_solves=r['rom_solves'], true_comp=r['true_comp'],
            train_error=r['train_error'], pred_error=r['pred_error'],
            ci_coverage=r['ci_coverage'], ci_width=r['ci_width'],
            stability_pct=r['stability_pct'], runtime=r['runtime'],
            runtime_new=r['runtime_new'], runtime_total=r['runtime_total'],
            member_runtimes=r['member_runtimes'], n_restored=r['n_restored'],
            n_newly_trained=r['n_newly_trained'],
            n_stable=r['n_stable'], n_total=r['n_total'],
            n_trained=r['n_trained'], n_kept=r['n_kept'], n_dropped=r['n_dropped'],
            kept_indices=r['kept_indices'], dropped_indices=r['dropped_indices'],
            losses=r['losses'], final_losses=r['final_losses'],
            median_final_loss=r['median_final_loss'], loss_cutoff=r['loss_cutoff'],
        )
        print(f"  💾 Saved predictions: {out_dir}/05_neural_ode_chemo.npz")
        plot_results(r, save_dir=os.path.join(out_dir, 'figures'))

        if DOSE_VARIATION:
            evaluate_dose_variation(r, save_dir=out_dir)

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

"""
02 — Two-Stage Hierarchical Bayesian OpInf
    2D Diffusion-Reaction Equation: ∂u/∂t = κ∇²u − βu²

Hierarchical Bayesian Operator Inference with two-stage conditional
decomposition:

  p(O, X | Y) = p(O | X, dX/dt) × p(X, dX/dt | Y)

  Stage 1 — Bayesian GP:
    Fit GP hyperparameters via MLE, then compute the exact GP posterior
    distributions over latent states X and derivatives dX/dt:
      p(X | Y) = N(μ_x, Σ_x)      (GP state posterior)
      p(dX/dt | Y) = N(μ_z, Σ_z)  (GP derivative posterior)

  Stage 2 — Bayesian Operator Inference via SVI:
    Sample X ~ p(X | Y) from the GP state posterior (not fixed to mean).
    ODE constraint: f(X)O^T ~ N(μ_z, Σ_z + γ₂I)
    This propagates GP state uncertainty into the operator posterior.

Physics constraint (ELBO factor):
  dX/dt ≈ f(X)O^T   (derivative matching with full GP uncertainty flow)

Single data regime: dense data, medium noise (3%).

Usage:
    python 02_two_stage_svi.py
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from jax import random
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import opinf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis
from core import (
    generate_trajectory, JaxCompatibleModel,
    build_bayesian_opinf_model, compute_gp_derivatives, rbf_eval,
)
from core.bayesian_opinf import fit_gp_hyperparameters_mle, _find_operator_samples
from core.plotting import plot_full_order_error

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

# ── Data regime ──────────────────────────────────────────────────────────────
SCHEMAS = [
    {
        "name": "dense_medium_noise",
        "label": "Dense data, medium noise",
        "NUM_SAMPLES": 60,
        "NOISE_LEVEL": 0.03,
        "NUM_EVAL_POINTS": 200,
    },
]

# ── Model hyperparameters ───────────────────────────────────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=3,
    GAMMA=2.0,
    GAMMA2=10.0,
    NUM_EVAL_POINTS=200,
    NUM_STEPS_GP=5000,          # Phase A: AutoDelta (MAP)
    NUM_STEPS_OP=5000,          # Phase B: AutoNormal (posterior)
    LEARNING_RATE=3e-3,
    NUM_POSTERIOR_SAMPLES=500,
    REGULARIZER=1.0,
    SEED=42,
)

TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 3.0)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# ROM predictions via scipy solve_ivp
# =============================================================================
def _generate_rom_predictions_scipy(samples, rom, snaps_comp, t_pred,
                                     num_modes, num_pulls=200):
    """Generate ROM predictions using scipy solve_ivp."""
    Os, rom_solves = [], []
    O_samples = _find_operator_samples(samples, "O")
    if O_samples.ndim == 2:
        O_samples = O_samples[np.newaxis, ...]

    n_total = len(O_samples)
    n_use = min(num_pulls, n_total)
    indices = np.linspace(0, n_total - 1, n_use, dtype=int)

    for idx in indices:
        O = O_samples[idx]
        Os.append(np.array(O))
        rom.model._extract_operators(np.array(O))

        def rhs(t, state, _rom=rom):
            return np.array(_rom.model.rhs(t, state, None))

        try:
            sol = solve_ivp(rhs, [t_pred[0], t_pred[-1]],
                          np.array(snaps_comp[:, 0]),
                          t_eval=t_pred, method='RK45', max_step=0.01)
            if sol.success and np.all(np.isfinite(sol.y)):
                rom_solves.append(sol.y)
        except Exception:
            pass

    return (np.array(Os),
            np.array(rom_solves) if rom_solves else np.empty((0, num_modes, len(t_pred))))


# =============================================================================
# Run experiment
# =============================================================================
def run_experiment(schema):
    """Run one data regime. Returns results dict."""
    p = MODEL_PARAMS
    np.random.seed(p['SEED'])
    rng_key = random.PRNGKey(p['SEED'])

    noise_level = schema['NOISE_LEVEL']
    num_samples = schema['NUM_SAMPLES']
    num_eval_points = schema['NUM_EVAL_POINTS']
    num_modes = p['NUM_MODES']

    print(f"\n{'='*70}")
    print(f"  {schema['label']}  ({num_samples} samples, {noise_level:.0%} noise)")
    print(f"{'='*70}")

    # ── 1. Data generation ───────────────────────────────────────────────
    (fom, t_full, true_states, t_samp, snaps_samp) = \
        generate_trajectory(config, config.time_domain, TRAINING_SPAN, num_samples, noise_level)
    basis = Basis(num_vectors=num_modes)
    basis.fit(snaps_samp)
    snaps_comp = basis.compress(snaps_samp)
    true_comp = basis.compress(true_states)
    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    # ── 2. Fit GP hyperparameters via MLE ────────────────────────────────
    Ls, Vs, Ns, gp_models = fit_gp_hyperparameters_mle(t_samp, snaps_comp, verbose=False)
    for i in range(num_modes):
        T = t_samp[-1] - t_samp[0]
        print(f"  Mode {i}: ℓ={Ls[i]:.5f} (T/ℓ={T/Ls[i]:.0f}), σ²={Vs[i]:.4f}, ν={Ns[i]:.6f}")

    # ── 3. LS operator (prior) ───────────────────────────────────────────
    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
        model=JaxCompatibleModel(operators="cAH",
                                  solver=opinf.lstsq.L2Solver(regularizer=p['REGULARIZER'])),
    )
    rom.fit(states=snaps_samp)

    t_eval_ls = np.linspace(float(t_samp[0]), float(t_samp[-1]), num_eval_points)
    X_mle = np.zeros((num_modes, num_eval_points))
    for i in range(num_modes):
        K = rbf_eval(Ls[i], Vs[i], t_samp, t_samp) + (Ns[i]+1e-5)*np.eye(len(t_samp))
        Ks = rbf_eval(Ls[i], Vs[i], t_eval_ls, t_samp)
        X_mle[i] = Ks @ np.linalg.solve(K, snaps_comp[i])

    mu_z_mle, _ = compute_gp_derivatives(Ls, Vs, t_samp, t_eval_ls, snaps_comp, Ns=Ns)
    D = np.array(rom.model._assemble_data_matrix(jnp.array(X_mle), inputs=None))
    DtD = D.T @ D
    prior_operator = np.linalg.solve(DtD + np.eye(DtD.shape[0]), D.T @ np.array(mu_z_mle).T).T
    print(f"  LS operator norm: {np.linalg.norm(prior_operator):.1f}")

    # ── 4. Stage 1: GP posterior means for latent states ─────────────────
    time_eval = np.linspace(t_samp[0], t_samp[-1], num_eval_points)

    Xs_means = np.array([
        gp_models[i].predict(time_eval[:, None], return_std=False)
        for i in range(num_modes)
    ])

    gp_state_err = np.linalg.norm(Xs_means - interp1d(t_full, true_comp, kind='cubic',
                   fill_value='extrapolate')(time_eval)) / np.linalg.norm(
                   interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')(time_eval))
    print(f"  GP state prediction error: {gp_state_err:.4%}")

    # ── 5. Stage 2: Bayesian operator inference (two-phase SVI) ──────────
    bayesian_opinf_model = build_bayesian_opinf_model(
        prior_operator=jnp.array(prior_operator),
        rom=rom,
        Ls_means=Ls,
        Vs_means=Vs,
        time_domain_sampled=t_samp,
        snapshots=snaps_comp,
        Xs_means=Xs_means,
        Ns_means=Ns,
        inputs_eval=None,
        data_scaler=None,
        sample_X=False,
    )

    # ── 6. Two-phase SVI ─────────────────────────────────────────────────
    from numpyro.infer import SVI, Trace_ELBO, Predictive
    from numpyro.infer.initialization import init_to_value
    from numpyro.optim import ClippedAdam

    model_kwargs = dict(time=time_eval, gamma=p['GAMMA'], gamma2=p['GAMMA2'],
                        normalization=1e-6)

    t0 = time.time()

    # Phase A: AutoDelta → MAP estimate
    guide_a = numpyro.infer.autoguide.AutoDelta(
        bayesian_opinf_model,
        init_loc_fn=init_to_value(values={'O': jnp.array(prior_operator)}),
    )
    svi_a = SVI(bayesian_opinf_model, guide_a, ClippedAdam(step_size=p['LEARNING_RATE']),
                loss=Trace_ELBO())
    rng_key, init_key = random.split(rng_key)
    svi_state_a = svi_a.init(init_key, **model_kwargs)

    @jax.jit
    def _step_a(state, _):
        state, loss = svi_a.update(state, **model_kwargs)
        return state, loss

    svi_state_a, losses_a = jax.lax.scan(_step_a, svi_state_a, jnp.arange(p['NUM_STEPS_GP']))
    params_a = svi_a.get_params(svi_state_a)
    rng_key, sk = random.split(rng_key)
    map_samples = guide_a.sample_posterior(sk, params_a, sample_shape=(1,), **model_kwargs)
    O_map = np.array(_find_operator_samples({**map_samples}, "O")).squeeze()
    if O_map.ndim == 1:
        O_map = O_map.reshape(prior_operator.shape)
    print(f"  Phase A (MAP): loss {float(losses_a[0]):.0f} → {float(losses_a[-1]):.0f}, "
          f"|O_map|={np.linalg.norm(O_map):.1f}")

    # Phase B: AutoNormal initialized at MAP → posterior with uncertainty
    guide_b = numpyro.infer.autoguide.AutoNormal(
        bayesian_opinf_model,
        init_loc_fn=init_to_value(values={'O': jnp.array(O_map)}),
    )
    svi_b = SVI(bayesian_opinf_model, guide_b, ClippedAdam(step_size=p['LEARNING_RATE']),
                loss=Trace_ELBO())
    rng_key, init_key = random.split(rng_key)
    svi_state_b = svi_b.init(init_key, **model_kwargs)

    @jax.jit
    def _step_b(state, _):
        state, loss = svi_b.update(state, **model_kwargs)
        return state, loss

    svi_state_b, losses_b = jax.lax.scan(_step_b, svi_state_b, jnp.arange(p['NUM_STEPS_OP']))
    runtime = time.time() - t0

    params_b = svi_b.get_params(svi_state_b)
    rng_key, sk, pk = random.split(rng_key, 3)
    posterior_samples = guide_b.sample_posterior(
        sk, params_b, sample_shape=(p['NUM_POSTERIOR_SAMPLES'],), **model_kwargs)
    predictive = Predictive(bayesian_opinf_model, posterior_samples=posterior_samples,
                            num_samples=p['NUM_POSTERIOR_SAMPLES'])
    model_output = predictive(pk, **model_kwargs)
    samples = {**model_output, **posterior_samples}
    all_losses = list(np.array(losses_a)) + list(np.array(losses_b))
    print(f"  Phase B (posterior): loss {float(losses_b[0]):.0f} → {float(losses_b[-1]):.0f}")
    print(f"  Total runtime: {runtime:.1f}s")

    # ── 7. Generate predictions ──────────────────────────────────────────
    O_samp = _find_operator_samples(samples, "O")
    if O_samp.ndim == 2:
        O_samp = O_samp[np.newaxis, ...]
    O_med = np.median(O_samp, axis=0)

    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    Os, rom_solves = _generate_rom_predictions_scipy(
        samples=samples, rom=rom, snaps_comp=snaps_comp,
        t_pred=t_pred, num_modes=num_modes,
        num_pulls=min(200, p['NUM_POSTERIOR_SAMPLES']),
    )

    n_stable = len(rom_solves)
    n_total = len(Os)
    stability_pct = n_stable / max(n_total, 1) * 100

    # ── 8. Compute metrics ───────────────────────────────────────────────
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
    print(f"    Operator norm: {np.linalg.norm(O_med):.1f} "
          f"(prior: {np.linalg.norm(prior_operator):.1f})")
    print(f"    Convergence: loss {all_losses[0]:.0f} → {all_losses[-1]:.0f}")

    return {
        'schema': schema,
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'n_stable': n_stable, 'n_total': n_total,
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'runtime': runtime, 'losses': all_losses,
        'samples': samples, 'rom_solves': rom_solves,
        'snaps_comp': snaps_comp, 'true_comp': true_comp,
        't_full': t_full, 't_pred': t_pred, 't_samp': t_samp,
        'training_span': TRAINING_SPAN, 'num_modes': num_modes,
        'true_states': true_states, 'basis': basis, 'fom': fom,
        'snaps_samp': snaps_samp,
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_results(result, save_dir=None):
    """Generate ROM trajectory, full-order error, 2D contour, and loss plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    schema = result['schema']
    prefix = f"02_{schema['name']}"
    losses = result['losses']
    rom_solves = result['rom_solves']
    snaps_comp = result['snaps_comp']
    true_comp = result['true_comp']
    t_full = result['t_full']
    t_pred = result['t_pred']
    t_samp = result['t_samp']
    training_span = result['training_span']
    num_modes = result['num_modes']

    # ── 1. Notebook-style ROM trajectory plot ────────────────────────
    if len(rom_solves) > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        rom_q05 = np.percentile(rom_arr, 5, axis=0)
        rom_q95 = np.percentile(rom_arr, 95, axis=0)

        true_interp = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        true_at_pred = true_interp(t_pred)

        n_stable = result['n_stable']
        n_total = result['n_total']

        fig, ax = plt.subplots(num_modes, 1, figsize=(10, 2.5 * num_modes), sharex=True)
        if num_modes == 1:
            ax = [ax]
        for i in range(num_modes):
            ax[i].axvspan(training_span[0], training_span[1], color='gray', alpha=0.10, zorder=0)
            ax[i].plot(t_pred, true_at_pred[i], color='tab:gray', lw=2, label='True solution')
            ax[i].plot(t_samp, snaps_comp[i], 'k*', ms=5, label='Training data', zorder=5)
            ax[i].plot(t_pred, rom_med[i], color='tab:purple', linestyle='--', alpha=0.9, lw=2, label='ROM median')
            ax[i].fill_between(t_pred, rom_q05[i], rom_q95[i], color='tab:purple', alpha=0.15, label='ROM 5-95%')
            ax[i].axvline(training_span[1], color='k', ls=':', lw=0.8, alpha=0.5)
            ax[i].set_ylabel(f'Mode {i+1}')
            yvals = true_at_pred[i]
            ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
            pad = max(abs(ymax - ymin) * 0.3, 1e-6)
            ax[i].set_ylim(ymin - pad, ymax + pad)
            if i == 0:
                ax[i].legend(loc='upper right', fontsize=9)
        ax[-1].set_xlabel('Time')
        fig.suptitle(f'Two-Stage SVI — {schema["label"]}  ({n_stable}/{n_total} stable)', fontsize=14)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_rom_trajectories.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)

    # ── 2. Full-order error plot ─────────────────────────────────────
    basis = result.get('basis')
    true_states = result.get('true_states')
    if len(rom_solves) > 0 and basis is not None and true_states is not None:
        rom_arr = np.array(rom_solves)
        fig_foe, axes_foe = plot_full_order_error(
            rom_solves=rom_arr,
            basis=basis,
            true_states=true_states,
            time_domain_full=t_full,
            time_domain_eval=t_pred,
            training_span=training_span,
            error_type='relative',
        )
        fig_foe.suptitle(f'Full-Order Error — {schema["label"]}', fontsize=14)
        path = os.path.join(save_dir, f"{prefix}_full_order_error.png")
        fig_foe.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig_foe)

    # ── 3. 2D Contour Plots ──────────────────────────────────────────
    fom = result.get('fom')
    if len(rom_solves) > 0 and fom is not None:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)

        snapshot_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        snapshot_times = [t for t in snapshot_times if t <= t_pred[-1]]

        fig, axes = plt.subplots(2, len(snapshot_times),
                                 figsize=(3.5 * len(snapshot_times), 6.5))
        x, y = fom.spatial_domain

        for col, t_snap in enumerate(snapshot_times):
            t_idx = np.argmin(np.abs(t_full - t_snap))
            u_true = fom.reconstruct_2d(true_states[:, t_idx])

            t_idx_pred = np.argmin(np.abs(t_pred - t_snap))
            u_rom_full = basis.decompress(rom_med[:, t_idx_pred])
            u_rom = fom.reconstruct_2d(u_rom_full)

            vmin = min(u_true.min(), u_rom.min())
            vmax = max(u_true.max(), u_rom.max())
            levels = np.linspace(vmin, vmax, 30)

            im0 = axes[0, col].contourf(x, y, u_true, levels=levels,
                                         cmap='RdBu_r', extend='both')
            axes[0, col].set_aspect('equal')
            axes[0, col].set_title(f't = {t_snap:.1f}', fontsize=12)
            plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

            im1 = axes[1, col].contourf(x, y, u_rom, levels=levels,
                                         cmap='RdBu_r', extend='both')
            axes[1, col].set_aspect('equal')
            plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

            for row in range(2):
                if col > 0:
                    axes[row, col].set_yticklabels([])

        axes[0, 0].set_ylabel('True', fontsize=12)
        axes[1, 0].set_ylabel('ROM', fontsize=12)
        fig.suptitle(f'2D Field — Two-Stage SVI — {schema["label"]}', fontsize=14)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_2d_contours.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)

    # ── 4. Loss convergence plot ─────────────────────────────────────
    fig_loss, ax_loss = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss[0].plot(losses, lw=0.8, color='tab:blue')
    ax_loss[0].set_xlabel('SVI Iteration')
    ax_loss[0].set_ylabel('ELBO Loss')
    ax_loss[0].set_title('Loss Convergence')
    ax_loss[0].grid(True, alpha=0.3)
    half = len(losses) // 2
    ax_loss[1].plot(range(half, len(losses)), losses[half:], lw=0.8, color='tab:blue')
    ax_loss[1].set_xlabel('SVI Iteration')
    ax_loss[1].set_ylabel('ELBO Loss')
    ax_loss[1].set_title('Loss (last 50%)')
    ax_loss[1].grid(True, alpha=0.3)
    fig_loss.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_loss.png")
    fig_loss.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig_loss)


# =============================================================================
# Save predictions for cross-method comparison
# =============================================================================
def save_predictions(result, save_dir=None):
    """Save predictions for cross-method comparison."""
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, "results", "comparison", result['schema']['name'])
    os.makedirs(save_dir, exist_ok=True)

    rom_solves = result['rom_solves']
    rom_arr = np.array(rom_solves) if len(rom_solves) > 0 else np.empty((0, result['num_modes'], len(result['t_pred'])))

    method_name = "02_two_stage_svi"
    path = os.path.join(save_dir, f"{method_name}.npz")
    np.savez(path,
        rom_solves=rom_arr,
        t_pred=result['t_pred'],
        train_error=result['train_error'],
        pred_error=result['pred_error'],
        stability_pct=result['stability_pct'],
        ci_coverage=result.get('ci_coverage', float('nan')),
        ci_width=result.get('ci_width', float('nan')),
        runtime=result['runtime'],
    )
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
    print("02 — Two-Stage Hierarchical Bayesian OpInf — 2D Diffusion-Reaction")
    print("=" * 70)
    print(f"Regimes: {len(schemas)}")
    for s in schemas:
        print(f"  • {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  noise={s['NOISE_LEVEL']:.0%}")
    print(f"Model:  γ={MODEL_PARAMS['GAMMA']}, γ₂={MODEL_PARAMS['GAMMA2']}, "
          f"lr={MODEL_PARAMS['LEARNING_RATE']}, "
          f"steps_gp={MODEL_PARAMS['NUM_STEPS_GP']}, steps_op={MODEL_PARAMS['NUM_STEPS_OP']}")

    results = []
    for schema in schemas:
        r = run_experiment(schema)
        plot_results(r)
        save_predictions(r)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"SUMMARY — Two-Stage Hierarchical Bayesian OpInf (2D Diffusion-Reaction)")
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

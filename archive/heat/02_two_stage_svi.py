"""
02 — Two-Stage Hierarchical Bayesian OpInf — Heat Equation

Hierarchical Bayesian Operator Inference with two-stage conditional
decomposition for multi-trajectory, input-dependent PDEs:

  p(O, X | Y) = p(O | X, dX/dt) × p(X, dX/dt | Y)

  Stage 1 — Bayesian GP (per-IC):
    Fit GP hyperparameters via MLE, then compute the exact GP posterior
    distributions over latent states X and derivatives dX/dt:
      p(X^(ic) | Y^(ic)) = N(μ_x, Σ_x)   (GP state posterior)
      p(dX/dt^(ic) | Y^(ic)) = N(μ_z, Σ_z) (GP derivative posterior)

  Stage 2 — Bayesian Operator Inference via SVI:
    Sample X^(ic) ~ p(X^(ic) | Y^(ic)) from GP state posteriors.
    O ~ N(O_ls, γ|O_ls|)  — SHARED operator with informative prior.
    ODE constraint per IC: f(X^(ic), u^(ic))O^T ~ N(μ_z, Σ_z + γ₂I)
    This propagates GP state uncertainty into the operator posterior.

Operators: cAHBN (constant + linear + quadratic + input + state×input)

Data regimes (same hyperparameters for all):
  1. Dense data, low noise    (65 samples, 1% noise)
  2. Sparse data, medium noise (20 samples, 5% noise)
  3. Dense data, high noise   (65 samples, 10% noise)

Usage:
    python 02_two_stage_svi.py                  # run all 3 regimes
    python 02_two_stage_svi.py dense_low_noise  # run one regime
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from jax import random
from scipy.interpolate import interp1d
import opinf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import (
    Basis, ReducedOrderModel, input_func_factory,
    input_parameters, test_parameters,
)
from step1_generate_data import TrajectorySampler
from core import (
    compute_gp_derivatives, rbf_eval,
    fit_gp_hyperparameters_mle,
    build_bayesian_opinf_model, run_svi,
)
from core.bayesian_opinf import _find_operator_samples
from core.plotting import plot_full_order_error
from heat_plotter import _generate_rom_solves

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

# ── Data regime definitions ──────────────────────────────────────────────────
SCHEMAS = [
    {
        "name": "sparse_low_noise",
        "label": "Sparse data, low noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.01,
        "NUM_EVAL_POINTS": 100,
    },
    {
        "name": "sparse_medium_noise",
        "label": "Sparse data, medium noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.03,
        "NUM_EVAL_POINTS": 100,
    },
    {
        "name": "sparse_high_noise",
        "label": "Sparse data, high noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.05,
        "NUM_EVAL_POINTS": 100,
    },
]

# ── Shared model hyperparameters (same for ALL regimes) ──────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=5,
    NUM_ICS=5,
    GAMMA=2.0,
    GAMMA2=2.0,
    NUM_STEPS=10000,
    LEARNING_RATE=3e-3,
    NUM_POSTERIOR_SAMPLES=500,
    SEED=42,
)

TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 2.0)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


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
    num_ics = p['NUM_ICS']
    train_params = input_parameters[:num_ics]

    print(f"\n{'='*70}")
    print(f"  {schema['label']}  ({num_samples} samples, {noise_level:.0%} noise)")
    print(f"{'='*70}")

    # ── Data generation (EXACTLY matching heat/04) ────────────────────────
    sampler = TrajectorySampler(
        training_span=TRAINING_SPAN,
        num_samples=num_samples,
        noiselevel=noise_level,
        num_regression_points=num_eval_points,
        synced=False,
    )
    (all_true_states, all_time_sampled, all_snapshots,
     all_training_inputs) = sampler.multisample(train_params)

    snapshots_train = np.hstack(all_snapshots)
    basis = Basis(num_vectors=num_modes)
    basis.fit(snapshots_train)
    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    all_snapshots_comp = [basis.compress(s) for s in all_snapshots]
    all_true_comp = [basis.compress(s) for s in all_true_states]

    # Build ROM with inputs
    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(all_time_sampled[0]),
        model=ReducedOrderModel(),
    )
    first_input_func = input_func_factory(train_params[0])
    first_inputs = first_input_func(all_time_sampled[0])
    rom.fit(states=all_snapshots[0], inputs=first_inputs)
    print(f"  Operator shape: {rom.model.operator_matrix.shape}")

    # Per-IC eval inputs
    all_inputs_eval = []
    for ic in range(num_ics):
        t_eval_ic = np.linspace(
            float(all_time_sampled[ic][0]),
            float(all_time_sampled[ic][-1]),
            num_eval_points,
        )
        in_func = input_func_factory(train_params[ic])
        all_inputs_eval.append(in_func(t_eval_ic))

    # ── Stage 1: MLE GP fitting (per-IC) ─────────────────────────────────
    all_mle_Ls, all_mle_Vs, all_mle_Ns = [], [], []
    all_gp_models = []
    for ic in range(num_ics):
        print(f"  GP MLE — IC {ic} ({train_params[ic]})")
        Ls, Vs, Ns, gp_models = fit_gp_hyperparameters_mle(
            all_time_sampled[ic], all_snapshots_comp[ic], verbose=False)
        all_mle_Ls.append(Ls)
        all_mle_Vs.append(Vs)
        all_mle_Ns.append(Ns)
        all_gp_models.append(gp_models)
        for j in range(num_modes):
            print(f"    Mode {j}: ℓ={Ls[j]:.5f}, σ²={Vs[j]:.4f}, ν={Ns[j]:.6f}")

    # ── LS operator (direct solve from GP derivatives, same as 04) ──────
    D_blocks, dXdt_blocks = [], []
    for ic in range(num_ics):
        t_eval_ic = np.linspace(
            float(all_time_sampled[ic][0]),
            float(all_time_sampled[ic][-1]),
            num_eval_points,
        )
        X_mle = np.zeros((num_modes, num_eval_points))
        for j in range(num_modes):
            ell, sig2, nu = all_mle_Ls[ic][j], all_mle_Vs[ic][j], all_mle_Ns[ic][j]
            K = rbf_eval(ell, sig2, all_time_sampled[ic], all_time_sampled[ic]) + \
                (nu + 1e-5) * np.eye(len(all_time_sampled[ic]))
            Ks = rbf_eval(ell, sig2, t_eval_ic, all_time_sampled[ic])
            X_mle[j] = Ks @ np.linalg.solve(K, all_snapshots_comp[ic][j])

        mu_z_ic, _ = compute_gp_derivatives(
            all_mle_Ls[ic], all_mle_Vs[ic],
            all_time_sampled[ic], t_eval_ic,
            all_snapshots_comp[ic], Ns=all_mle_Ns[ic])

        in_func = input_func_factory(train_params[ic])
        inputs_ic = in_func(t_eval_ic)
        D_ic = np.array(rom.model._assemble_data_matrix(
            jnp.array(X_mle), inputs=jnp.array(inputs_ic)))
        D_blocks.append(D_ic)
        dXdt_blocks.append(np.array(mu_z_ic).T)

    D_all = np.vstack(D_blocks)
    dXdt_all = np.vstack(dXdt_blocks)
    DtD = D_all.T @ D_all
    O_ls = np.linalg.solve(DtD + np.eye(DtD.shape[0]), D_all.T @ dXdt_all).T
    print(f"  LS operator norm: {np.linalg.norm(O_ls):.1f}, shape: {O_ls.shape}")

    # ── Stage 1 output: GP posterior means (per-IC) ────────────────────
    Xs_means_list = []
    for ic in range(num_ics):
        time_eval_ic = np.linspace(
            float(all_time_sampled[ic][0]),
            float(all_time_sampled[ic][-1]),
            num_eval_points,
        )
        Xs_means_ic = np.array([
            all_gp_models[ic][j].predict(time_eval_ic[:, None], return_std=False)
            for j in range(num_modes)
        ])
        Xs_means_list.append(Xs_means_ic)

    # ── Stage 2: Build Bayesian model (deterministic X, derivative constraint) ──
    bayesian_model = build_bayesian_opinf_model(
        prior_operator=jnp.array(O_ls),
        rom=rom,
        Ls_means=[all_mle_Ls[ic] for ic in range(num_ics)],
        Vs_means=[all_mle_Vs[ic] for ic in range(num_ics)],
        time_domain_sampled=[all_time_sampled[ic] for ic in range(num_ics)],
        snapshots=[all_snapshots_comp[ic] for ic in range(num_ics)],
        Xs_means=Xs_means_list,
        Ns_means=[all_mle_Ns[ic] for ic in range(num_ics)],
        inputs_eval=[all_inputs_eval[ic] for ic in range(num_ics)],
        data_scaler=None,
        sample_X=False,
    )

    # Evaluation time grid (use first IC's eval grid)
    time_eval = np.linspace(
        float(all_time_sampled[0][0]),
        float(all_time_sampled[0][-1]),
        num_eval_points,
    )

    # ── Two-phase SVI ────────────────────────────────────────────────────
    #   Phase A: AutoDelta (MAP) to find optimal operator location
    #   Phase B: AutoNormal initialized at MAP for uncertainty quantification
    from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
    from numpyro.infer.initialization import init_to_value
    from numpyro.optim import ClippedAdam

    model_kwargs = dict(time=jnp.array(time_eval), gamma=p['GAMMA'],
                        gamma2=p['GAMMA2'], normalization=1e-6)

    rng_key, svi_key = random.split(rng_key)
    t0 = time.time()

    # Phase A: AutoDelta → MAP estimate
    guide_a = autoguide.AutoDelta(
        bayesian_model,
        init_loc_fn=init_to_value(values={'O': jnp.array(O_ls)}),
    )
    svi_a = SVI(bayesian_model, guide_a, ClippedAdam(step_size=p['LEARNING_RATE']),
                loss=Trace_ELBO())
    rng_key, init_key = random.split(rng_key)
    svi_state_a = svi_a.init(init_key, **model_kwargs)

    @jax.jit
    def _step_a(state, _):
        state, loss = svi_a.update(state, **model_kwargs)
        return state, loss

    svi_state_a, losses_a = jax.lax.scan(_step_a, svi_state_a, jnp.arange(p['NUM_STEPS']))
    params_a = svi_a.get_params(svi_state_a)
    rng_key, sk = random.split(rng_key)
    map_samples = guide_a.sample_posterior(sk, params_a, sample_shape=(1,), **model_kwargs)
    O_map = np.array(_find_operator_samples({**map_samples}, "O")).squeeze()
    if O_map.ndim == 1:
        O_map = O_map.reshape(O_ls.shape)
    print(f"  Phase A (MAP): loss {float(losses_a[0]):.0f} → {float(losses_a[-1]):.0f}, "
          f"|O_map|={np.linalg.norm(O_map):.1f}")

    # Phase B: AutoNormal initialized at MAP → posterior with uncertainty
    guide_b = autoguide.AutoNormal(
        bayesian_model,
        init_loc_fn=init_to_value(values={'O': jnp.array(O_map)}),
    )
    svi_b = SVI(bayesian_model, guide_b, ClippedAdam(step_size=p['LEARNING_RATE']),
                loss=Trace_ELBO())
    rng_key, init_key = random.split(rng_key)
    svi_state_b = svi_b.init(init_key, **model_kwargs)

    @jax.jit
    def _step_b(state, _):
        state, loss = svi_b.update(state, **model_kwargs)
        return state, loss

    svi_state_b, losses_b = jax.lax.scan(_step_b, svi_state_b, jnp.arange(p['NUM_STEPS']))
    runtime = time.time() - t0

    params_b = svi_b.get_params(svi_state_b)
    rng_key, sk, pk = random.split(rng_key, 3)
    posterior_samples = guide_b.sample_posterior(
        sk, params_b, sample_shape=(p['NUM_POSTERIOR_SAMPLES'],), **model_kwargs)
    predictive = Predictive(bayesian_model, posterior_samples=posterior_samples,
                            num_samples=p['NUM_POSTERIOR_SAMPLES'])
    model_output = predictive(pk, **model_kwargs)
    samples = {**model_output, **posterior_samples}
    losses = list(np.array(losses_a)) + list(np.array(losses_b))
    print(f"  Phase B (posterior): loss {float(losses_b[0]):.0f} → {float(losses_b[-1]):.0f}")
    print(f"  Total runtime: {runtime:.1f}s")

    # ── Evaluate all training ICs + test IC ───────────────────────────────
    print(f"\n  Results ({runtime:.0f}s):")

    O_samp = _find_operator_samples(samples, "O")
    if O_samp.ndim == 2:
        O_samp = O_samp[np.newaxis, ...]
    O_med = np.median(O_samp, axis=0)
    print(f"    Operator norm: {np.linalg.norm(O_med):.1f} (LS: {np.linalg.norm(O_ls):.1f})")

    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    max_samp = min(200, len(O_samp))

    # Generate test IC data
    test_sampler = TrajectorySampler(
        training_span=TRAINING_SPAN,
        num_samples=num_samples,
        noiselevel=noise_level,
        num_regression_points=num_eval_points,
        synced=False,
    )
    (test_true_list, test_t_list, test_snap_list,
     test_inp_list) = test_sampler.multisample([test_parameters])
    test_true_comp = basis.compress(test_true_list[0])
    test_snaps_comp = basis.compress(test_snap_list[0])
    test_t_samp = test_t_list[0]

    # Evaluate all ICs
    eval_params = list(train_params) + [test_parameters]
    eval_snaps_comp = all_snapshots_comp + [test_snaps_comp]
    eval_true_comp = all_true_comp + [test_true_comp]
    eval_t_samp = all_time_sampled + [test_t_samp]
    eval_labels = [f"Train IC {i} {train_params[i]}" for i in range(num_ics)] + \
                  [f"Test IC {test_parameters}"]

    all_rom_solves = []
    all_n_stable = []
    all_train_errors = []
    all_pred_errors = []
    train_mask = t_pred <= TRAINING_SPAN[1]
    pred_mask = t_pred > TRAINING_SPAN[1]

    for ic_idx, (params, true_c) in enumerate(zip(eval_params, eval_true_comp)):
        q0 = eval_snaps_comp[ic_idx][:, 0]
        ic_input_func = input_func_factory(params)

        ic_solves = _generate_rom_solves(
            operator_samples=O_samp, rom=rom, q0=q0,
            time_eval=t_pred, input_func=ic_input_func,
            max_samples=max_samp,
        )
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

    # CI coverage
    ci_width = ci_coverage = float('nan')
    if train_ic_stable > 0:
        all_in_ci, all_widths = [], []
        for ic_idx in range(num_ics):
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

    return {
        'schema': schema,
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'n_stable': train_ic_stable, 'n_total': train_ic_total,
        'test_train_error': all_train_errors[-1],
        'test_pred_error': all_pred_errors[-1],
        'test_n_stable': all_n_stable[-1],
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'runtime': runtime, 'losses': losses,
        'samples': samples,
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
        'num_modes': p['NUM_MODES'],
        'basis': basis,
        'all_true_states_full': all_true_states,
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_results(result, save_dir=None):
    """Generate multi-IC ROM trajectory, loss, and full-order error plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    schema = result['schema']
    prefix = f"02_{schema['name']}"
    samples = result['samples']
    losses = result['losses']
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
    max_samp = min(200, len(_find_operator_samples(samples, "O")))

    # ── 1. Multi-trajectory ROM plot ─────────────────────────────────
    n_ics_total = len(all_rom_solves)
    has_any_stable = any(ns > 0 for ns in all_n_stable)

    if not has_any_stable:
        print("  ⚠ No stable ROM solves — skipping ROM trajectory plot")
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
                        color='tab:purple', ls='--', lw=2, alpha=0.9,
                        label='Median' if (row == 0 and col == 0) else None,
                    )
                    ax.fill_between(
                        t_pred,
                        np.percentile(rom_solves[:, col, :], 5, axis=0),
                        np.percentile(rom_solves[:, col, :], 95, axis=0),
                        color='tab:purple', alpha=0.15,
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

        fig.suptitle(f"ROM Predictions — {schema['label']}", fontsize=14, y=1.05)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_rom_trajectories.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)

    # ── 2. Loss Convergence Plot ─────────────────────────────────────
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

    # ── 3. Full-Order Error Plot (first training IC) ─────────────────
    basis = result.get('basis')
    all_true_states_full = result.get('all_true_states_full')
    if basis is not None and all_true_states_full is not None and len(all_rom_solves) > 0:
        first_ic_solves = all_rom_solves[0]
        if len(first_ic_solves) > 0:
            first_true_full = all_true_states_full[0]
            fig_foe, axes_foe = plot_full_order_error(
                rom_solves=first_ic_solves,
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


# =============================================================================
# save_predictions
# =============================================================================
def save_predictions(result, save_dir=None):
    """Save predictions for cross-method comparison."""
    schema = result['schema']
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, "results", "comparison", schema['name'])
    os.makedirs(save_dir, exist_ok=True)

    all_rom_solves = result['all_rom_solves']
    method_name = "02_two_stage_svi"

    save_dict = {
        't_pred': result['t_pred'],
        'train_error': result['train_error'],
        'pred_error': result['pred_error'],
        'stability_pct': result['stability_pct'],
        'ci_coverage': result.get('ci_coverage', float('nan')),
        'ci_width': result.get('ci_width', float('nan')),
        'runtime': result['runtime'],
        'n_ics': len(all_rom_solves),
    }
    for ic_idx, solves in enumerate(all_rom_solves):
        if len(solves) > 0:
            save_dict[f'rom_solves_{ic_idx}'] = np.array(solves)
        else:
            save_dict[f'rom_solves_{ic_idx}'] = np.empty((0, result['num_modes'], len(result['t_pred'])))

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
    print("02 — Two-Stage Hierarchical Bayesian OpInf — Cubic Heat")
    print("=" * 70)
    print(f"Regimes: {len(schemas)}")
    for s in schemas:
        print(f"  • {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  noise={s['NOISE_LEVEL']:.0%}")
    print(f"Model:  γ={MODEL_PARAMS['GAMMA']}, γ₂={MODEL_PARAMS['GAMMA2']}, "
          f"lr={MODEL_PARAMS['LEARNING_RATE']}, steps={MODEL_PARAMS['NUM_STEPS']}")
    print(f"ICs:    {MODEL_PARAMS['NUM_ICS']} training + 1 test ({test_parameters})")

    results = []
    for schema in schemas:
        r = run_experiment(schema)
        plot_results(r)
        save_predictions(r)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"SUMMARY — Two-Stage Hierarchical Bayesian OpInf (Heat)")
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

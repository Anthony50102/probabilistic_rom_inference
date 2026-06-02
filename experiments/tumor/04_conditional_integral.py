"""
04 — Conditional GP + Dual Constraint (Integral + Derivative Form) — Tumor Growth

Bayesian Operator Inference with full uncertainty propagation for single-trajectory
tumor growth (autonomous dynamics):
  θ_GP ~ LogNormal(MLE, σ)    — GP hyperparameters are sampled
  X(t) = K_* K⁻¹ y            — states computed analytically from θ_GP
  O ~ N(O_ls, γ|O_ls|)        — operator with informative prior
  γ₂ = fixed hyperparameter   — constraint noise scale (fixed)

Physics constraints (likelihood factors in ELBO):
  1. Derivative:  dX/dt ≈ D(X)O^T   (weighted by GP derivative variance)
  2. Integral:    ∫D(X)O^T ds ≈ ΔX  (prevents null basin, robust to noise)
  3. GP MLL:      log p(y|θ_GP)      (data fidelity for hyperparameters)

Operators: cA (constant + linear) — autonomous, no parametric inputs.
The (k, d) parameters are baked into the single FOM trajectory.

POD basis is computed from NOISY data (realistic: only noisy measurements
are available in practice). The GP noise parameter ν absorbs measurement
noise after projection.

Data regimes:
  1. Dense data, low noise    (80 samples, 1% noise)
  2. Dense data, medium noise (80 samples, 3% noise)
  3. Dense data, high noise   (80 samples, 5% noise)

Usage:
    python 04_conditional_integral.py                  # run all 3 regimes
    python 04_conditional_integral.py dense_low_noise  # run one regime
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from numpyro.infer.initialization import init_to_value
from numpyro.optim import ClippedAdam
from jax import random
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis, TumorTwinFOM, load_fom_data
from core import JaxCompatibleModel, compute_gp_derivatives, rbf_eval
from core.bayesian_opinf import fit_gp_hyperparameters_mle, _find_operator_samples
from core.diagnostics import plot_trace
from core.plotting import plot_full_order_error
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

# ── Data regime definitions ──────────────────────────────────────────────────
SCHEMAS = [
    {
        "name": "dense_low_noise",
        "label": "Dense data, low noise",
        "NUM_SAMPLES": 80,
        "NOISE_LEVEL": 0.01,
        "NUM_EVAL_POINTS": 200,
    },
    {
        "name": "dense_medium_noise",
        "label": "Dense data, medium noise",
        "NUM_SAMPLES": 80,
        "NOISE_LEVEL": 0.03,
        "NUM_EVAL_POINTS": 200,
    },
    {
        "name": "dense_high_noise",
        "label": "Dense data, high noise",
        "NUM_SAMPLES": 80,
        "NOISE_LEVEL": 0.05,
        "NUM_EVAL_POINTS": 200,
    },
]

# ── Shared model hyperparameters (same for ALL regimes) ──────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=4,
    GAMMA=0.05,
    GAMMA2=0.035,
    DERIV_WEIGHT=1.0,
    INTEGRAL_WEIGHT=8.0,
    MLL_WEIGHT=0.1,
    GP_PRIOR_SCALE=0.03,
    WINDOW_SIZE=20,
    NUM_STEPS=12000,
    LEARNING_RATE=3e-3,
    NUM_POSTERIOR_SAMPLES=500,
    SEED=42,
)

TRAINING_SPAN = config.TRAINING_SPAN
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# ROM solve helper (autonomous — no input function)
# =============================================================================
def _generate_rom_solves(operator_samples, rom, q0, time_eval, max_samples=200):
    """Generate ROM solves from operator samples for a single autonomous trajectory."""
    solves = []
    n = min(len(operator_samples), max_samples)
    for i in range(n):
        rom.model._extract_operators(np.array(operator_samples[i]))
        try:
            rom.model.predict(state0=q0, t=time_eval, input_func=None)
            result = rom.model.predict_result_
            if hasattr(result, 'y'):
                sol = result.y
            elif hasattr(result, 'ys'):
                sol = np.array(result.ys).T
            else:
                continue
            if sol.shape[1] == len(time_eval) and np.all(np.isfinite(sol)):
                solves.append(sol)
        except Exception:
            pass
    if solves:
        return np.array(solves)
    return np.empty((0, len(q0), len(time_eval)))


# =============================================================================
# Model builder (single trajectory, autonomous)
# =============================================================================
def build_model(
    rom, num_modes, time_sampled, snapshots_comp,
    O_prior, mle_Ls, mle_Vs, mle_Ns,
    num_eval_points=200, window_size=20,
    deriv_weight=1.0, integral_weight=1.0,
    mll_weight=0.1, gp_prior_scale=0.03,
):
    """Build the conditional integral NumPyro model for a single trajectory."""
    t_train = jnp.array(time_sampled)
    n_train = len(t_train)
    y_obs = jnp.array(snapshots_comp)

    time_eval = np.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
    t_eval = jnp.array(time_eval)
    dt_eval = float(time_eval[1] - time_eval[0])

    # Precompute kernel distance matrices
    sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
    sq_diffs_et = (t_eval[:, None] - t_train[None, :]) ** 2
    diffs_et = t_eval[:, None] - t_train[None, :]
    sq_diffs_ee = (t_eval[:, None] - t_eval[None, :]) ** 2
    I_train = jnp.eye(n_train)

    O_prior_jnp = jnp.array(O_prior)
    mle_log_ells = jnp.array([jnp.log(l) for l in mle_Ls])
    mle_log_sig2s = jnp.array([jnp.log(v) for v in mle_Vs])
    mle_log_nus = jnp.array([jnp.log(max(n, 1e-6)) for n in mle_Ns])

    # Precompute integration windows
    n_windows = num_eval_points // window_size
    ws_list = [i * window_size for i in range(n_windows)]
    we_list = [(i + 1) * window_size - 1 for i in range(n_windows)]
    if we_list[-1] < num_eval_points - 1:
        we_list[-1] = num_eval_points - 1

    trap_weights = []
    window_durations = []
    for ws, we in zip(ws_list, we_list):
        n_pts = we - ws + 1
        w = jnp.ones(n_pts) * dt_eval
        w = w.at[0].set(0.5 * dt_eval)
        w = w.at[-1].set(0.5 * dt_eval)
        trap_weights.append(w)
        window_durations.append(float(time_eval[we] - time_eval[ws]))

    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def _single_gp_conditional(ell, sig2, nu, y_i):
        """GP posterior: mean, derivative mean/var, MLL — deterministic given hypers."""
        ell2 = ell ** 2
        jitter = jnp.maximum(1e-5, sig2 * 1e-4)

        K_tt = _rbf_sq(ell, sig2, sq_diff_tt) + (nu + jitter) * I_train
        L = jnp.linalg.cholesky(K_tt)
        alpha = jax.scipy.linalg.cho_solve((L, True), y_i)

        K_et = _rbf_sq(ell, sig2, sq_diffs_et)
        X_eval = K_et @ alpha

        K_zy = -(diffs_et / ell2) * K_et
        mu_z = K_zy @ alpha

        K_ee = _rbf_sq(ell, sig2, sq_diffs_ee)
        K_zz = ((1.0 - sq_diffs_ee / ell2) / ell2) * K_ee

        V = jax.scipy.linalg.cho_solve((L, True), K_zy.T)
        deriv_var = jnp.maximum(jnp.diag(K_zz) - jnp.sum(K_zy * V.T, axis=1), 0.0)

        mll = -0.5 * (jnp.dot(y_i, alpha) +
                       2.0 * jnp.sum(jnp.log(jnp.diag(L))) +
                       n_train * jnp.log(2.0 * jnp.pi))

        return X_eval, mu_z, deriv_var, mll

    _batch_gp_conditional = jax.vmap(_single_gp_conditional)

    def model(gamma=0.05, gamma2=0.035, jitter=1e-4):

        # GP hyperparameters — sampled with informative priors
        ells = jnp.stack([
            numpyro.sample(f"lengthscale_{i}",
                          dist.LogNormal(mle_log_ells[i], gp_prior_scale))
            for i in range(num_modes)
        ])
        sig2s = jnp.stack([
            numpyro.sample(f"variance_{i}",
                          dist.LogNormal(mle_log_sig2s[i], gp_prior_scale))
            for i in range(num_modes)
        ])
        nus = jnp.stack([
            numpyro.sample(f"noise_{i}",
                          dist.LogNormal(mle_log_nus[i], gp_prior_scale))
            for i in range(num_modes)
        ])

        # GP conditional (deterministic given hypers + data)
        Xs_eval, mu_zs, deriv_vars, mlls = _batch_gp_conditional(ells, sig2s, nus, y_obs)

        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", Xs_eval[i])

        # GP marginal log-likelihood
        if mll_weight > 0:
            numpyro.factor("gp_mll", mll_weight * jnp.sum(mlls))

        # Operator with informative prior
        prior_scale = gamma * jnp.maximum(jnp.abs(O_prior_jnp), 0.5)
        O = numpyro.sample("O", dist.Normal(O_prior_jnp, prior_scale))

        # Autonomous dynamics: D(X_eval) @ O^T
        f_Xi = rom.model._assemble_data_matrix(Xs_eval, inputs=None) @ O.T

        # CONSTRAINT 1: Derivative matching
        if deriv_weight > 0:
            for i in range(num_modes):
                total_var = deriv_vars[i] + gamma2 + jitter
                numpyro.factor(f"ode_constraint_{i}",
                    deriv_weight * jnp.sum(
                        dist.Normal(f_Xi[:, i], jnp.sqrt(total_var)).log_prob(mu_zs[i])))

        # CONSTRAINT 2: Integral form
        if integral_weight > 0:
            for i in range(num_modes):
                for w_idx, (ws, we) in enumerate(zip(ws_list, we_list)):
                    delta_X_obs = Xs_eval[i, we] - Xs_eval[i, ws]
                    delta_X_pred = jnp.sum(trap_weights[w_idx] * f_Xi[ws:we+1, i])
                    constraint_std = jnp.sqrt(gamma2) * window_durations[w_idx]
                    numpyro.factor(f"integral_{i}_{w_idx}",
                        integral_weight * dist.Normal(
                            delta_X_pred, constraint_std).log_prob(delta_X_obs))

    return model, time_eval


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

    # ── Data generation ──────────────────────────────────────────────────
    t_pred = np.linspace(TRAINING_SPAN[0], config.PREDICTION_DAYS, num_eval_points)

    fom, t_full, true_states, t_samp, snaps_noisy = load_fom_data(
        t_pred, TRAINING_SPAN, num_samples, noise_level)

    # ── Adaptive POD: fit generous basis, then truncate by GP SNR ────────
    max_modes = p['NUM_MODES'] + 4   # generous initial fit
    basis_probe = Basis(num_vectors=max_modes)
    basis_probe.fit(snaps_noisy)
    snaps_comp_probe = basis_probe.compress(snaps_noisy)

    # GP MLE on all probe modes to determine signal vs noise
    Ls_probe, Vs_probe, Ns_probe, _ = fit_gp_hyperparameters_mle(
        t_samp, snaps_comp_probe, verbose=False)

    SNR_THRESHOLD = 10.0
    effective_modes = 0
    for j in range(max_modes):
        snr = Vs_probe[j] / max(Ns_probe[j], 1e-10)
        tag = "✓" if snr > SNR_THRESHOLD else "✗"
        print(f"  Probe mode {j}: σ²={Vs_probe[j]:.4f}, ν={Ns_probe[j]:.6f}, "
              f"SNR={snr:.1f} {tag}")
        if snr > SNR_THRESHOLD:
            effective_modes = j + 1
    effective_modes = max(effective_modes, 2)  # at least 2 modes
    num_modes = min(effective_modes, p['NUM_MODES'])
    print(f"  → Using {num_modes} modes (SNR threshold={SNR_THRESHOLD})")

    # Final basis with effective modes
    basis = Basis(num_vectors=num_modes)
    basis.fit(snaps_noisy)
    snaps_comp = basis.compress(snaps_noisy)
    true_comp = basis.compress(true_states)
    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    # Autonomous ROM: cA (constant + linear)
    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
        model=JaxCompatibleModel(operators="cA",
                                 solver=opinf.lstsq.L2Solver(regularizer=1e0)),
    )
    rom.fit(states=snaps_noisy)
    print(f"  Operator shape: {rom.model.operator_matrix.shape}")

    # ── MLE warm start (reuse probe results for kept modes) ────────────
    Ls = Ls_probe[:num_modes]
    Vs = Vs_probe[:num_modes]
    Ns = Ns_probe[:num_modes]
    for i in range(num_modes):
        T = t_samp[-1] - t_samp[0]
        print(f"  Mode {i}: ℓ={Ls[i]:.5f} (T/ℓ={T/Ls[i]:.0f}), σ²={Vs[i]:.4f}, ν={Ns[i]:.6f}")

    # LS operator
    t_eval_ls = np.linspace(float(t_samp[0]), float(t_samp[-1]), num_eval_points)
    X_mle = np.zeros((num_modes, num_eval_points))
    # Also compute GP posterior mean at sampled times (for denoised obs in plots)
    X_mle_at_samp = np.zeros((num_modes, len(t_samp)))
    for i in range(num_modes):
        K = rbf_eval(Ls[i], Vs[i], t_samp, t_samp) + (Ns[i] + 1e-5) * np.eye(len(t_samp))
        K_inv_y = np.linalg.solve(K, snaps_comp[i])
        Ks = rbf_eval(Ls[i], Vs[i], t_eval_ls, t_samp)
        X_mle[i] = Ks @ K_inv_y
        Ks_samp = rbf_eval(Ls[i], Vs[i], t_samp, t_samp)
        X_mle_at_samp[i] = Ks_samp @ K_inv_y

    mu_z_mle, _ = compute_gp_derivatives(Ls, Vs, t_samp, t_eval_ls, snaps_comp, Ns=Ns)
    D = np.array(rom.model._assemble_data_matrix(jnp.array(X_mle), inputs=None))
    DtD = D.T @ D
    O_ls = np.linalg.solve(DtD + np.eye(DtD.shape[0]), D.T @ np.array(mu_z_mle).T).T
    print(f"  LS operator norm: {np.linalg.norm(O_ls):.1f}, shape: {O_ls.shape}")

    # ── Build & run SVI ──────────────────────────────────────────────────
    model, time_eval = build_model(
        rom=rom, num_modes=num_modes,
        time_sampled=t_samp, snapshots_comp=snaps_comp,
        O_prior=O_ls, mle_Ls=Ls, mle_Vs=Vs, mle_Ns=Ns,
        num_eval_points=num_eval_points, window_size=p['WINDOW_SIZE'],
        deriv_weight=p['DERIV_WEIGHT'], integral_weight=p['INTEGRAL_WEIGHT'],
        mll_weight=p['MLL_WEIGHT'], gp_prior_scale=p['GP_PRIOR_SCALE'],
    )

    init_values = {'O': jnp.array(O_ls)}
    for i in range(num_modes):
        init_values[f'lengthscale_{i}'] = Ls[i]
        init_values[f'variance_{i}'] = Vs[i]
        init_values[f'noise_{i}'] = max(float(Ns[i]), 1e-6)

    model_kwargs = dict(gamma=p['GAMMA'], gamma2=p['GAMMA2'], jitter=1e-4)
    guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    optimizer = ClippedAdam(step_size=p['LEARNING_RATE'])
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key, ik = random.split(rng_key)
    t0 = time.time()
    svi_state = svi.init(ik, **model_kwargs)

    @jax.jit
    def _step(s, _):
        s, l = svi.update(s, **model_kwargs)
        return s, l

    num_steps = p['NUM_STEPS']
    seg_size = max(1, num_steps // 10)
    all_losses = []
    for seg in range(10):
        start = seg * seg_size
        end = min(start + seg_size, num_steps)
        if seg == 9:
            end = num_steps
        if start >= num_steps:
            break
        svi_state, seg_losses = jax.lax.scan(_step, svi_state, jnp.arange(end - start))
        seg_np = np.array(seg_losses)
        all_losses.extend(seg_np.tolist())
        print(f"    step {end:6d}/{num_steps}  loss={seg_np[-1]:10.2f}")

    params = svi.get_params(svi_state)
    rng_key, sk, pk = random.split(rng_key, 3)
    n_post = p['NUM_POSTERIOR_SAMPLES']
    post = guide.sample_posterior(sk, params, sample_shape=(n_post,), **model_kwargs)
    pred = Predictive(model, posterior_samples=post, num_samples=n_post)
    out = pred(pk, **model_kwargs)
    samples = {**out, **post}
    runtime = time.time() - t0

    # ── Evaluate ─────────────────────────────────────────────────────────
    O_samp = _find_operator_samples(samples, "O")
    if O_samp.ndim == 2:
        O_samp = O_samp[np.newaxis, ...]
    O_med = np.median(O_samp, axis=0)

    q0 = snaps_comp[:, 0]
    rom_solves = _generate_rom_solves(
        operator_samples=O_samp, rom=rom, q0=q0,
        time_eval=t_pred, max_samples=min(200, n_post))

    n_stable = len(rom_solves)
    n_total = min(200, n_post)
    stability_pct = n_stable / max(n_total, 1) * 100

    train_error = pred_error = float('inf')
    ci_coverage = ci_width = float('nan')

    train_mask = t_pred <= TRAINING_SPAN[1]
    pred_mask = t_pred > TRAINING_SPAN[1]

    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)

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
    print(f"    Operator norm: {np.linalg.norm(O_med):.1f} (LS: {np.linalg.norm(O_ls):.1f})")
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
        'gp_mean_at_samp': X_mle_at_samp,
        't_full': t_full, 't_pred': t_pred, 't_samp': t_samp,
        'training_span': TRAINING_SPAN, 'num_modes': num_modes,
        'true_states': true_states, 'basis': basis,
        'fom': fom,
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_results(result, save_dir=None):
    """Generate ROM trajectory, operator trace, loss, spatial, and volume plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    schema = result['schema']
    prefix = f"04_{schema['name']}"

    # ── Snapshot result for replotting ───────────────────────────────
    try:
        from core.plotting import save_plot_data
        save_plot_data(result, os.path.join(save_dir, "plot_data", f"{prefix}.pkl"))
    except Exception as _e:
        print(f"  ⚠ snapshot failed: {_e}")
    samples = result['samples']
    losses = result['losses']
    rom_solves = result['rom_solves']
    snaps_comp = result['snaps_comp']
    true_comp = result['true_comp']
    gp_mean_at_samp = result.get('gp_mean_at_samp', snaps_comp)  # fallback
    t_full = result['t_full']
    t_pred = result['t_pred']
    t_samp = result['t_samp']
    training_span = result['training_span']
    num_modes = result['num_modes']

    # ── 1. ROM Trajectory Plot ───────────────────────────────────────
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
            ax[i].axvspan(training_span[0], training_span[1],
                          color='gray', alpha=0.10, zorder=0)
            ax[i].plot(t_pred, true_at_pred[i], color='tab:gray', lw=2,
                       label='True solution')
            ax[i].plot(t_samp, snaps_comp[i], 'k.', ms=3,
                       alpha=0.3, label='Raw obs', zorder=4)
            ax[i].plot(t_samp, gp_mean_at_samp[i], 'k*', ms=5,
                       label='GP denoised', zorder=5)
            ax[i].plot(t_pred, rom_med[i], color='tab:purple', linestyle='--',
                       alpha=0.9, lw=2, label='ROM median')
            ax[i].fill_between(t_pred, rom_q05[i], rom_q95[i],
                               color='tab:purple', alpha=0.15, label='ROM 5-95%')
            ax[i].axvline(training_span[1], color='k', ls=':', lw=0.8, alpha=0.5)
            ax[i].set_ylabel(f'Mode {i + 1}')
            yvals = true_at_pred[i]
            ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
            pad = max(abs(ymax - ymin) * 0.3, 1e-6)
            ax[i].set_ylim(ymin - pad, ymax + pad)
        ax[-1].set_xlabel('Time (days)')
        handles, labels = ax[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center',
                       ncol=len(handles), fontsize=10,
                       bbox_to_anchor=(0.5, 0.95))
        fig.suptitle(f'Bayesian OpInf — {schema["label"]}', fontsize=14, y=0.995)
        fig.tight_layout()
        fig.subplots_adjust(top=0.90)
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

    # ── 3. Operator Trace Plot ───────────────────────────────────────
    try:
        fig_trace, _ = plot_trace(samples, param_name="O", n_random=6)
        path = os.path.join(save_dir, f"{prefix}_operator_traces.png")
        fig_trace.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig_trace)
    except Exception as e:
        print(f"  ⚠ Operator trace plot failed: {e}")

    # ── 4. Full-Order Error Plot ─────────────────────────────────────
    basis = result.get('basis')
    true_states = result.get('true_states')
    if len(rom_solves) > 0 and basis is not None and true_states is not None:
        rom_arr = np.array(rom_solves)
        max_foe_samples = min(20, rom_arr.shape[0])
        idx = np.linspace(0, rom_arr.shape[0] - 1, max_foe_samples, dtype=int)
        fig_foe, axes_foe = plot_full_order_error(
            rom_solves=rom_arr[idx],
            basis=basis,
            true_states=true_states,
            time_domain_full=t_full,
            time_domain_eval=t_pred,
            training_span=training_span,
            error_type='relative',
            suptitle=f'Full-Order Error — {schema["label"]}',
        )
        path = os.path.join(save_dir, f"{prefix}_full_order_error.png")
        fig_foe.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig_foe)

    # ── 5. Spatial Comparison Plot ───────────────────────────────────
    plot_spatial_comparison(result, save_dir)

    # ── 6. Tumor Volume Over Time ────────────────────────────────────
    plot_tumor_volume(result, save_dir)


def plot_spatial_comparison(result, save_dir, timepoints_to_show=None):
    """Plot 3D tumor density slices: FOM truth vs ROM prediction vs error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schema = result['schema']
    prefix = f"04_{schema['name']}"
    basis = result['basis']
    t_full = result['t_full']
    t_pred = result['t_pred']
    rom_solves = result['rom_solves']
    true_states = result['true_states']
    fom = result['fom']

    if len(rom_solves) == 0:
        print("  ⚠ No stable ROM solves — skipping spatial plot")
        return

    rom_arr = np.array(rom_solves)
    rom_med = np.median(rom_arr, axis=0)

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
                                   cmap='Reds', vmin=0, aspect='equal')

        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        if col == 0:
            axes[0, col].set_ylabel('FOM Truth', fontsize=12, fontweight='bold')
            axes[1, col].set_ylabel('Bayesian OpInf', fontsize=12, fontweight='bold')
            axes[2, col].set_ylabel('Cellularity |Δ|', fontsize=12, fontweight='bold')

    fig.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.8, label='Cellularity')
    fig.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.8, label='Cellularity')
    fig.colorbar(im2, ax=axes[2, :].tolist(), shrink=0.8, label='Cellularity |Δ|')
    fig.suptitle(f'Tumor Growth: FOM vs Bayesian OpInf (axial slice) — {schema["label"]}',
                 fontsize=14)
    path = os.path.join(save_dir, f"{prefix}_spatial_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


def plot_tumor_volume(result, save_dir):
    """Plot total tumor burden over time.

    Uses reduced-space dot product to avoid decompressing full DOF fields.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schema = result['schema']
    prefix = f"04_{schema['name']}"
    basis = result['basis']
    t_full = result['t_full']
    t_pred = result['t_pred']
    rom_solves = result['rom_solves']
    true_states = result['true_states']
    fom = result['fom']
    training_span = result['training_span']

    # Efficient volume projection vectors
    V = basis.entries   # (n_dof, r)
    ones = np.ones(V.shape[0])
    vol_proj = V.T @ ones          # (r,)
    shift_vol = ones @ basis.shift_  # scalar
    voxel_vol = float(np.prod(fom.spacing))

    fig, ax = plt.subplots(figsize=(10, 6))

    # FOM truth volume
    fom_vol = np.array([true_states[:, i].sum() * voxel_vol
                        for i in range(true_states.shape[1])])
    ax.plot(t_full, fom_vol, color='tab:gray', lw=2.5, label='FOM Truth')

    # ROM volume
    if len(rom_solves) > 0:
        rom_arr = np.array(rom_solves)
        rom_vols = np.array([vol_proj @ rom_arr[s] + shift_vol
                             for s in range(rom_arr.shape[0])]) * voxel_vol
        rom_med = np.median(rom_vols, axis=0)
        rom_lo = np.percentile(rom_vols, 5, axis=0)
        rom_hi = np.percentile(rom_vols, 95, axis=0)
        ax.plot(t_pred, rom_med, color='tab:purple', lw=2, ls='--',
                label='Bayesian OpInf median')
        ax.fill_between(t_pred, rom_lo, rom_hi, color='tab:purple', alpha=0.15,
                        label='Bayesian OpInf 90% CI')

    ax.axvline(training_span[1], color='gray', ls='--', alpha=0.5,
               label='Train/Predict')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Total Tumor Burden (mm³)')
    ax.set_title(f'Tumor Volume Over Time — {schema["label"]}')
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_tumor_volume.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


# =============================================================================
# Save predictions for cross-method comparison
# =============================================================================
def save_predictions(result, save_dir=None):
    """Save predictions for cross-method comparison."""
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, "results", "comparison",
                                result['schema']['name'])
    os.makedirs(save_dir, exist_ok=True)

    rom_solves = result['rom_solves']
    rom_arr = (np.array(rom_solves) if len(rom_solves) > 0
               else np.empty((0, result['num_modes'], len(result['t_pred']))))

    method_name = "04_conditional_integral"
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
    print("04 — Conditional GP + Dual Constraint — Tumor Growth (Single Traj)")
    print("=" * 70)
    print(f"Regimes: {len(schemas)}")
    for s in schemas:
        print(f"  • {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  "
              f"noise={s['NOISE_LEVEL']:.0%}")
    print(f"Model:  γ={MODEL_PARAMS['GAMMA']}, γ₂={MODEL_PARAMS['GAMMA2']}, "
          f"lr={MODEL_PARAMS['LEARNING_RATE']}, steps={MODEL_PARAMS['NUM_STEPS']}")

    results = []
    for schema in schemas:
        r = run_experiment(schema)
        plot_results(r)
        save_predictions(r)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"SUMMARY — Conditional GP + Dual Constraint (Tumor Growth)")
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
    args = sys.argv[1:]
    if args and args[0] == "--replot":
        # Replot from a saved snapshot: --replot <path/to/plot_data/PREFIX.pkl> [save_dir]
        from core.plotting import load_plot_data
        pkl_path = args[1]
        result = load_plot_data(pkl_path)
        if len(args) > 2:
            save_dir = args[2]
        else:
            # plot_data/ lives inside save_dir
            save_dir = os.path.dirname(os.path.dirname(os.path.abspath(pkl_path)))
        plot_results(result, save_dir=save_dir)
    else:
        schema_names = args if args else None
        main(schema_names)

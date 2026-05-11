"""
04g_marginal_O.py — Fully Bayesian Operator Inference with O marginalised analytically.

Key insight: both constraints (derivative matching and integral form) are LINEAR
in the operator O for any fixed GP-conditional state X.  With a Gaussian prior
on O, the conditional posterior p(O | θ, data) is Gaussian by conjugacy and the
marginal likelihood p(data | θ) is closed-form.

This eliminates:
  * The null-basin local mode (we never search over O)
  * The high-dimensional "narrow shell" problem (O is closed form, not optimised)
  * The need for an LS warm-start (we never initialise O)

What remains: SVI / NUTS on θ alone (~18 dims: ℓ, σ², ν per mode).
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
from config import Basis
from core import (
    generate_trajectory, JaxCompatibleModel, compute_gp_derivatives,
    rbf_eval,
)
from core.bayesian_opinf import fit_gp_hyperparameters_mle
from core.diagnostics import plot_trace
from core.plotting import plot_full_order_error
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

SCHEMAS = [
    {"name": "dense_low_noise",  "NUM_SAMPLES": 250, "NOISE_LEVEL": 0.01,
     "NUM_EVAL_POINTS": 400, "label": "Dense data, low noise"},
    {"name": "sparse_low_noise", "NUM_SAMPLES": 55, "NOISE_LEVEL": 0.03,
     "NUM_EVAL_POINTS": 200, "label": "Sparse data, low noise"},
    {"name": "dense_high_noise", "NUM_SAMPLES": 250, "NOISE_LEVEL": 0.10,
     "NUM_EVAL_POINTS": 400, "label": "Dense data, high noise"},
]

MODEL_PARAMS = dict(
    NUM_MODES=6,
    GAMMA2=10.0,
    DERIV_WEIGHT=1.0,
    INTEGRAL_WEIGHT=1.0,
    MLL_WEIGHT=1.0,
    SIGMA_O=30.0,        # O ~ N(0, SIGMA_O^2 I)  (data + γ² noise determines effective scale)
    WINDOW_SIZE=20,
    NUM_STEPS=8000,
    LEARNING_RATE=5e-3,
    NUM_POSTERIOR_SAMPLES=500,
    SEED=42,
)

TRAINING_SPAN = (0, 0.08)
PREDICTION_SPAN = (0, 0.15)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


def build_model(rom, num_modes, time_sampled, snapshots_comp,
                num_eval_points, window_size,
                deriv_weight, integral_weight, mll_weight,
                sigma_O):
    """Build marginalised-O Bayesian model.

    Returns (model, posterior_O_fn, time_eval).
      model:           numpyro model over θ (GP hypers) only
      posterior_O_fn:  callable(theta_dict) -> (O_mean[r,m], O_chol[r,m,m])
                        for closed-form posterior on O given θ
    """
    t_train = jnp.array(time_sampled)
    n_train = len(t_train)
    y_obs = jnp.array(snapshots_comp)

    time_eval = np.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
    t_eval = jnp.array(time_eval)
    dt_eval = float(time_eval[1] - time_eval[0])

    sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
    sq_diffs_et = (t_eval[:, None] - t_train[None, :]) ** 2
    diffs_et = t_eval[:, None] - t_train[None, :]
    sq_diffs_ee = (t_eval[:, None] - t_eval[None, :]) ** 2
    I_train = jnp.eye(n_train)

    # Integration windows for the integral constraint
    n_windows = num_eval_points // window_size
    ws_arr = jnp.array([i * window_size for i in range(n_windows)])
    we_list = [(i + 1) * window_size - 1 for i in range(n_windows)]
    if we_list[-1] < num_eval_points - 1:
        we_list[-1] = num_eval_points - 1
    we_arr = jnp.array(we_list)

    # Trapezoidal weight vectors per window, padded to a fixed max length so
    # we can stack them. Padding rows are zeroed out in the design matrix.
    W = n_windows
    max_len = window_size + 1
    trap_pad = np.zeros((W, num_eval_points), dtype=np.float32)
    win_durations = np.zeros(W, dtype=np.float32)
    for w_idx, (ws, we) in enumerate(zip(ws_arr.tolist(), we_arr.tolist())):
        n_pts = we - ws + 1
        w = np.ones(n_pts) * dt_eval
        w[0] = 0.5 * dt_eval
        w[-1] = 0.5 * dt_eval
        trap_pad[w_idx, ws:we + 1] = w
        win_durations[w_idx] = time_eval[we] - time_eval[ws]
    trap_pad = jnp.asarray(trap_pad)               # (W, T_eval)
    win_durations = jnp.asarray(win_durations)     # (W,)

    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def _single_gp_conditional(ell, sig2, nu, y_i):
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
        mll = -0.5 * (jnp.dot(y_i, alpha)
                      + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
                      + n_train * jnp.log(2.0 * jnp.pi))
        return X_eval, mu_z, deriv_var, mll

    _batch_gp_cond = jax.vmap(_single_gp_conditional)

    # Broad GP-hyper priors (no MLE anchoring needed once O is marginalised)
    T_span = float(t_train[-1] - t_train[0])
    broad_log_ell_loc = jnp.log(T_span / 20.0)
    broad_log_sig2_locs = jnp.array(
        [float(jnp.log(jnp.var(jnp.array(snapshots_comp[i])) + 1e-12))
         for i in range(num_modes)])

    inv_sigma_O2_default = 1.0 / (sigma_O ** 2)
    log_sigma_O2_default = 2.0 * jnp.log(sigma_O)

    def _per_mode_evidence(A, y_i, prec_i, m, inv_sigma_O2, log_sigma_O2):
        """Closed-form: log p(y_i | θ) when y_i = A · O_i + ε, O_i ~ N(0, σ_O² I).

        Returns (log_evidence, μ_i, L_i) where Λ_i = LᵢLᵢᵀ and Σ_O_i = Λ_i⁻¹.
        """
        # M_i = Aᵀ diag(prec_i) A
        Aw = A * prec_i[:, None]
        M_i = A.T @ Aw                                         # (m, m)
        Lambda_i = M_i + inv_sigma_O2 * jnp.eye(m)
        L_i = jnp.linalg.cholesky(Lambda_i)
        b_i = A.T @ (prec_i * y_i)                             # (m,)
        mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_i)
        log_det_Lambda = 2.0 * jnp.sum(jnp.log(jnp.diag(L_i)))
        quad_y = jnp.sum(prec_i * y_i ** 2)
        quad_mu = jnp.dot(mu_i, b_i)            # μᵢᵀ Λᵢ μᵢ = μᵢᵀ bᵢ
        log_det_Sigma = -jnp.sum(jnp.log(prec_i + 1e-30))  # |Σ| = ∏ 1/prec_i
        N_i = prec_i.shape[0]
        log_p = -0.5 * ((quad_y - quad_mu)
                        + log_det_Sigma
                        + m * log_sigma_O2
                        + log_det_Lambda
                        + N_i * jnp.log(2.0 * jnp.pi))
        return log_p, mu_i, L_i

    def _build_AyP(ells, sig2s, nus, gamma2):
        """Per-θ pure-jax: build (A, y_per_mode, prec_per_mode)."""
        Xs, mu_zs, deriv_vars, mlls = _batch_gp_cond(ells, sig2s, nus, y_obs)
        f_X = rom.model._assemble_data_matrix(Xs, inputs=None)   # (T_eval, m)
        int_A = trap_pad @ f_X                                   # (W, m)
        A = jnp.concatenate([f_X, int_A], axis=0)                # (T_eval+W, m)
        delta_X = Xs[:, we_arr] - Xs[:, ws_arr]                  # (r, W)
        y_per_mode = jnp.concatenate([mu_zs, delta_X], axis=1)   # (r, T_eval+W)
        prec_deriv = deriv_weight / (deriv_vars + gamma2 + 1e-4)  # (r, T_eval)
        prec_int_per_window = integral_weight / (gamma2 * win_durations ** 2)
        prec_int = jnp.broadcast_to(prec_int_per_window, (num_modes, W))
        prec_per_mode = jnp.concatenate([prec_deriv, prec_int], axis=1)
        return Xs, mu_zs, A, y_per_mode, prec_per_mode, mlls

    HIERARCHICAL = bool(int(os.environ.get("HIER_SIGMA_O", "0")))

    def model(gamma2=10.0):
        ells = jnp.stack([
            numpyro.sample(f"lengthscale_{i}",
                           dist.LogNormal(broad_log_ell_loc, 1.0))
            for i in range(num_modes)])
        sig2s = jnp.stack([
            numpyro.sample(f"variance_{i}",
                           dist.LogNormal(broad_log_sig2_locs[i], 0.5))
            for i in range(num_modes)])
        nus = jnp.stack([
            numpyro.sample(f"noise_{i}",
                           dist.LogNormal(-8.0, 1.0))
            for i in range(num_modes)])

        if HIERARCHICAL:
            sO = numpyro.sample("sigma_O", dist.HalfCauchy(sigma_O))
            inv_sO2 = 1.0 / (sO ** 2 + 1e-12)
            log_sO2 = 2.0 * jnp.log(sO + 1e-12)
        else:
            inv_sO2 = inv_sigma_O2_default
            log_sO2 = log_sigma_O2_default

        Xs, mu_zs, A, y_per_mode, prec_per_mode, mlls = _build_AyP(
            ells, sig2s, nus, gamma2)

        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", Xs[i])

        if mll_weight > 0:
            numpyro.factor("gp_mll", mll_weight * jnp.sum(mlls))

        m = A.shape[1]
        total_evidence = 0.0
        for i in range(num_modes):
            log_p_i, _, _ = _per_mode_evidence(A, y_per_mode[i], prec_per_mode[i], m,
                                                inv_sO2, log_sO2)
            total_evidence = total_evidence + log_p_i
        numpyro.factor("marg_O_evidence", total_evidence)

    @jax.jit
    def posterior_O_fn(ells, sig2s, nus, gamma2, sigma_O_val):
        """Closed-form O posterior given θ: returns (μ_O, Cholesky of Σ_O)."""
        inv_sO2 = 1.0 / (sigma_O_val ** 2 + 1e-12)
        Xs, mu_zs, A, y_per_mode, prec_per_mode, _ = _build_AyP(
            ells, sig2s, nus, gamma2)
        m = A.shape[1]

        def _one(y_i, prec_i):
            Aw = A * prec_i[:, None]
            M_i = A.T @ Aw
            Lambda_i = M_i + inv_sO2 * jnp.eye(m)
            L_i = jnp.linalg.cholesky(Lambda_i)
            b_i = A.T @ (prec_i * y_i)
            mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_i)
            # Σ_O_i = (Lᵢ Lᵢᵀ)⁻¹  →  Cholesky of Σ_O_i is L_i⁻ᵀ (lower-tri-of-inverse).
            # For sampling we need any factor C with C Cᵀ = Σ_O_i.  L_i⁻ᵀ works:
            #   (L_i⁻ᵀ)(L_i⁻ᵀ)ᵀ = L_i⁻ᵀ L_i⁻¹ = (Lᵢ Lᵢᵀ)⁻¹ = Λᵢ⁻¹ ✓
            C_i = jax.scipy.linalg.solve_triangular(L_i, jnp.eye(m), lower=True).T
            return mu_i, C_i, Xs, mu_zs

        mu_all, C_all = [], []
        for i in range(num_modes):
            mi, Ci, _, _ = _one(y_per_mode[i], prec_per_mode[i])
            mu_all.append(mi); C_all.append(Ci)
        return jnp.stack(mu_all), jnp.stack(C_all), Xs, mu_zs

    return model, posterior_O_fn, time_eval


def run_one(schema, p):
    print(f"\n{'=' * 78}\n  {schema['label']}  —  marginalised-O Bayesian\n{'=' * 78}")
    noise = schema['NOISE_LEVEL']
    nsamp = schema['NUM_SAMPLES']
    neval = schema['NUM_EVAL_POINTS']
    nmodes = p['NUM_MODES']

    np.random.seed(p['SEED'])
    rng_key = random.PRNGKey(p['SEED'])

    fom, t_full, true_states, t_samp, snaps_samp = generate_trajectory(
        config, config.time_domain, TRAINING_SPAN, nsamp, noise)
    basis = Basis(num_vectors=nmodes)
    basis.fit(snaps_samp)
    snaps_comp = basis.compress(snaps_samp)
    true_comp = basis.compress(true_states)
    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
        model=JaxCompatibleModel(operators="cAH",
                                 solver=opinf.lstsq.L2Solver(regularizer=1e0)))
    rom.fit(states=snaps_samp)

    # For diagnostics only: still compute LS estimate so we can compare
    Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(t_samp, snaps_comp, verbose=False)
    print(f"  GP-MLE hypers: ℓ ≈ {np.mean(Ls):.4f}   σ² ≈ {np.mean(Vs):.3f}   ν ≈ {np.mean(Ns):.6f}")

    model, posterior_O_fn, time_eval = build_model(
        rom=rom, num_modes=nmodes,
        time_sampled=t_samp, snapshots_comp=snaps_comp,
        num_eval_points=neval, window_size=p['WINDOW_SIZE'],
        deriv_weight=p['DERIV_WEIGHT'], integral_weight=p['INTEGRAL_WEIGHT'],
        mll_weight=p['MLL_WEIGHT'], sigma_O=p['SIGMA_O'])

    # Warm-start θ at GP-MLE values (low-dim, no null basin to worry about)
    init_values = {}
    for i in range(nmodes):
        init_values[f'lengthscale_{i}'] = jnp.asarray(Ls[i])
        init_values[f'variance_{i}'] = jnp.asarray(Vs[i])
        init_values[f'noise_{i}'] = jnp.asarray(Ns[i])
    guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))

    optimizer = ClippedAdam(step_size=p['LEARNING_RATE'])
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    model_kwargs = dict(gamma2=p['GAMMA2'])

    rng_key, ik = random.split(rng_key)
    t0 = time.time()
    state = svi.init(ik, **model_kwargs)

    @jax.jit
    def _step(s, _):
        return svi.update(s, **model_kwargs)

    nsteps = p['NUM_STEPS']
    state, losses = jax.lax.scan(_step, state, jnp.arange(nsteps))
    losses = np.array(losses)
    print(f"  SVI: {nsteps} steps   loss {losses[0]:.2f} → {losses[-1]:.2f}   ({time.time()-t0:.1f}s)")

    params = svi.get_params(state)
    rng_key, sk = random.split(rng_key)
    npost = p['NUM_POSTERIOR_SAMPLES']
    post = guide.sample_posterior(sk, params, sample_shape=(npost,), **model_kwargs)

    # Compute O posterior per θ-sample and draw one O per sample
    rng_key, ok = random.split(rng_key)

    def _stack_theta(d):
        return (jnp.stack([d[f'lengthscale_{i}'] for i in range(nmodes)], axis=-1),
                jnp.stack([d[f'variance_{i}'] for i in range(nmodes)], axis=-1),
                jnp.stack([d[f'noise_{i}'] for i in range(nmodes)], axis=-1))

    ells_s, sig2s_s, nus_s = _stack_theta(post)   # each (npost, r)
    sO_s = post.get('sigma_O', None)               # (npost,) or None
    sO_default_jnp = jnp.asarray(p['SIGMA_O'])

    @jax.jit
    def _draw_O(ells, sig2s, nus, key, sO_val):
        mu_O, C_O, _, _ = posterior_O_fn(ells, sig2s, nus, p['GAMMA2'], sO_val)
        eps = jax.random.normal(key, shape=mu_O.shape)
        O = mu_O + jnp.einsum('ijk,ik->ij', C_O, eps)
        return O, mu_O

    keys = jax.random.split(ok, npost)
    t_o = time.time()
    O_samples_list, O_mean_list = [], []
    for s in range(npost):
        sO_val = sO_default_jnp if sO_s is None else sO_s[s]
        O, mu_O = _draw_O(ells_s[s], sig2s_s[s], nus_s[s], keys[s], sO_val)
        O_samples_list.append(np.array(O))
        O_mean_list.append(np.array(mu_O))
    O_samples = np.stack(O_samples_list)           # (npost, r, m)
    O_means = np.stack(O_mean_list)
    print(f"  O posterior sampling: {time.time()-t_o:.1f}s")
    op_norms = np.linalg.norm(O_samples.reshape(npost, -1), axis=1)
    print(f"  ‖O‖: median={np.median(op_norms):.1f}   min={op_norms.min():.1f}   max={op_norms.max():.1f}")
    print(f"  ‖μ_O‖ (avg over θ-samples): {np.linalg.norm(O_means, axis=(1,2)).mean():.1f}")

    # Predict via ROM solver
    from core import generate_rom_predictions
    samples_for_rom = {'O': jnp.array(O_samples)}
    for i in range(nmodes):
        samples_for_rom[f'X_{i}'] = jnp.stack(post[f'lengthscale_{i}'])  # dummy, not used
    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    Os, Xs, rom_solves = generate_rom_predictions(
        samples=samples_for_rom, rom=rom, snapshots_compressed=snaps_comp,
        time_eval=t_pred, num_modes=nmodes, num_pulls=min(200, npost))
    n_stable, n_total = len(rom_solves), len(Os)
    stab = n_stable / max(n_total, 1) * 100

    train_err = pred_err = float('inf')
    ci_cov = ci_w = float('nan')
    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        tm = t_pred <= TRAINING_SPAN[1]; pm = t_pred > TRAINING_SPAN[1]
        ti = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)
        train_err = float(np.linalg.norm(rom_med[:, tm] - ta[:, tm]) /
                          np.linalg.norm(ta[:, tm]))
        pred_err = float(np.linalg.norm(rom_med[:, pm] - ta[:, pm]) /
                         np.linalg.norm(ta[:, pm]))
        q05 = np.percentile(rom_arr, 5, axis=0); q95 = np.percentile(rom_arr, 95, axis=0)
        ci_w = float(np.mean(q95 - q05))
        ci_cov = float(np.mean((ta >= q05) & (ta <= q95)))

    runtime = time.time() - t0
    print(f"\n  RESULTS — Marginal-O Bayesian")
    print(f"    Stability:   {n_stable}/{n_total} ({stab:.1f}%)")
    print(f"    Train error: {train_err:.2%}")
    print(f"    Pred error:  {pred_err:.2%}")
    print(f"    CI coverage: {ci_cov:.1%} (target 90%)")
    print(f"    CI width:    {ci_w:.4f}")
    print(f"    Runtime:     {runtime:.0f}s")

    out_dir = os.path.join(SCRIPT_DIR, 'results', 'comparison', schema['name'])
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, '04g_marginal_O.npz'),
             train_error=train_err, pred_error=pred_err,
             stability_pct=stab, ci_coverage=ci_cov, ci_width=ci_w, runtime=runtime,
             op_norm_median=float(np.median(op_norms)),
             losses=losses)
    return train_err, pred_err, stab, ci_cov, runtime


if __name__ == "__main__":
    args = sys.argv[1:]
    p = dict(MODEL_PARAMS)
    schema_name = args[0] if args else 'dense_low_noise'
    schema = next(s for s in SCHEMAS if s['name'] == schema_name)
    run_one(schema, p)

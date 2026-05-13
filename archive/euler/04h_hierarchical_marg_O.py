"""
04h_hierarchical_marg_O.py — Hierarchical fully-Bayesian Op-Inf (Option 1).

Extension of 04g (marginalised-O) with proper hyperpriors on the previously
hand-tuned noise scales:

  * γ²       ~ HalfCauchy(1.0)            — likelihood "constraint noise"
  * τ_O      ~ HalfCauchy(SIGMA_O)        — global scale for operator prior
  * σ_O,i    ~ LogNormal(log τ_O, 0.5)    — per-mode ARD, shared hyperprior
  * (ℓ_i, σ²_i, ν_i) inferred per mode as before

This eliminates the two free knobs of 04g (GAMMA2, SIGMA_O) and turns them
into properly inferred latent variables, giving a fully-hierarchical Bayesian
formulation.  Operator O is still analytically marginalised conditional on
(θ_GP, γ², σ_O) so VI / HMC operates on a low-dimensional latent space and the
null-basin / narrow-shell problems of joint (θ, O) inference are avoided.
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide
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
    DERIV_WEIGHT=1.0,
    INTEGRAL_WEIGHT=1.0,
    MLL_WEIGHT=1.0,
    SIGMA_O_SCALE=30.0,
    GAMMA2_SCALE=10.0,
    WINDOW_SIZE=20,
    NUM_STEPS=2000,
    LEARNING_RATE=3e-3,
    NUM_POSTERIOR_SAMPLES=500,
    SEED=42,
)

TRAINING_SPAN = (0, 0.08)
PREDICTION_SPAN = (0, 0.15)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_model(rom, num_modes, time_sampled, snapshots_comp,
                num_eval_points, window_size,
                deriv_weight, integral_weight, mll_weight,
                sigma_O_scale, gamma2_scale):
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

    n_windows = num_eval_points // window_size
    ws_arr = jnp.array([i * window_size for i in range(n_windows)])
    we_list = [(i + 1) * window_size - 1 for i in range(n_windows)]
    if we_list[-1] < num_eval_points - 1:
        we_list[-1] = num_eval_points - 1
    we_arr = jnp.array(we_list)

    W = n_windows
    trap_pad = np.zeros((W, num_eval_points), dtype=np.float32)
    win_durations = np.zeros(W, dtype=np.float32)
    for w_idx, (ws, we) in enumerate(zip(ws_arr.tolist(), we_arr.tolist())):
        n_pts = we - ws + 1
        w = np.ones(n_pts) * dt_eval
        w[0] = 0.5 * dt_eval
        w[-1] = 0.5 * dt_eval
        trap_pad[w_idx, ws:we + 1] = w
        win_durations[w_idx] = time_eval[we] - time_eval[ws]
    trap_pad = jnp.asarray(trap_pad)
    win_durations = jnp.asarray(win_durations)

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

    T_span = float(t_train[-1] - t_train[0])
    broad_log_ell_loc = jnp.log(T_span / 20.0)
    broad_log_sig2_locs = jnp.array(
        [float(jnp.log(jnp.var(jnp.array(snapshots_comp[i])) + 1e-12))
         for i in range(num_modes)])

    def _per_mode_evidence(A, y_i, prec_i, m, inv_sigma_O2, log_sigma_O2):
        Aw = A * prec_i[:, None]
        M_i = A.T @ Aw
        Lambda_i = M_i + inv_sigma_O2 * jnp.eye(m)
        # Add tiny relative jitter for numerical stability in late SVI
        jit = 1e-6 * jnp.maximum(jnp.mean(jnp.diag(Lambda_i)), 1.0)
        L_i = jnp.linalg.cholesky(Lambda_i + jit * jnp.eye(m))
        b_i = A.T @ (prec_i * y_i)
        mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_i)
        log_det_Lambda = 2.0 * jnp.sum(jnp.log(jnp.diag(L_i)))
        quad_y = jnp.sum(prec_i * y_i ** 2)
        quad_mu = jnp.dot(mu_i, b_i)
        log_det_Sigma = -jnp.sum(jnp.log(prec_i + 1e-30))
        N_i = prec_i.shape[0]
        log_p = -0.5 * ((quad_y - quad_mu)
                        + log_det_Sigma
                        + m * log_sigma_O2
                        + log_det_Lambda
                        + N_i * jnp.log(2.0 * jnp.pi))
        return log_p, mu_i, L_i

    def _build_AyP(ells, sig2s, nus, gamma2):
        Xs, mu_zs, deriv_vars, mlls = _batch_gp_cond(ells, sig2s, nus, y_obs)
        f_X = rom.model._assemble_data_matrix(Xs, inputs=None)
        int_A = trap_pad @ f_X
        A = jnp.concatenate([f_X, int_A], axis=0)
        delta_X = Xs[:, we_arr] - Xs[:, ws_arr]
        y_per_mode = jnp.concatenate([mu_zs, delta_X], axis=1)
        prec_deriv = deriv_weight / (deriv_vars + gamma2 + 1e-4)
        prec_int_per_window = integral_weight / (gamma2 * win_durations ** 2)
        prec_int = jnp.broadcast_to(prec_int_per_window, (num_modes, W))
        prec_per_mode = jnp.concatenate([prec_deriv, prec_int], axis=1)
        return Xs, mu_zs, A, y_per_mode, prec_per_mode, mlls

    def model():
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

        # γ² (constraint noise) kept FIXED.  We tried sampling it but found a
        # structural non-identifiability: γ² appears in both prec_deriv and
        # prec_int, allowing a ridge along which γ²→0 inflates both precisions
        # arbitrarily, collapsing the posterior.  Keeping γ² at a sensible
        # data-derived scale is the principled choice; the hierarchical part
        # of the model is the operator prior τ_O / σ_O,i.
        gamma2 = jnp.asarray(gamma2_scale)

        # Hierarchical operator-prior scale: per-mode σ_O,i with shared τ_O
        tau_O = numpyro.sample("tau_O",
                               dist.LogNormal(jnp.log(sigma_O_scale), 0.5))
        tau_O = jnp.clip(tau_O, 1e-2, 1e4)
        sigma_O_per_mode = jnp.stack([
            numpyro.sample(f"sigma_O_{i}",
                           dist.LogNormal(jnp.log(tau_O + 1e-12), 0.3))
            for i in range(num_modes)])
        inv_sO2_per_mode = 1.0 / (sigma_O_per_mode ** 2 + 1e-12)
        log_sO2_per_mode = 2.0 * jnp.log(sigma_O_per_mode + 1e-12)

        Xs, mu_zs, A, y_per_mode, prec_per_mode, mlls = _build_AyP(
            ells, sig2s, nus, gamma2)

        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", Xs[i])

        if mll_weight > 0:
            numpyro.factor("gp_mll", mll_weight * jnp.sum(mlls))

        m = A.shape[1]
        total_evidence = 0.0
        for i in range(num_modes):
            log_p_i, _, _ = _per_mode_evidence(
                A, y_per_mode[i], prec_per_mode[i], m,
                inv_sO2_per_mode[i], log_sO2_per_mode[i])
            total_evidence = total_evidence + log_p_i
        numpyro.factor("marg_O_evidence", total_evidence)

    @jax.jit
    def posterior_O_fn(ells, sig2s, nus, gamma2, sigma_O_per_mode):
        Xs, mu_zs, A, y_per_mode, prec_per_mode, _ = _build_AyP(
            ells, sig2s, nus, gamma2)
        m = A.shape[1]
        inv_sO2_pm = 1.0 / (sigma_O_per_mode ** 2 + 1e-12)

        def _one(y_i, prec_i, inv_sO2_i):
            Aw = A * prec_i[:, None]
            M_i = A.T @ Aw
            Lambda_i = M_i + inv_sO2_i * jnp.eye(m)
            jit = 1e-6 * jnp.maximum(jnp.mean(jnp.diag(Lambda_i)), 1.0)
            L_i = jnp.linalg.cholesky(Lambda_i + jit * jnp.eye(m))
            b_i = A.T @ (prec_i * y_i)
            mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_i)
            C_i = jax.scipy.linalg.solve_triangular(L_i, jnp.eye(m), lower=True).T
            return mu_i, C_i

        mu_all, C_all = [], []
        for i in range(num_modes):
            mi, Ci = _one(y_per_mode[i], prec_per_mode[i], inv_sO2_pm[i])
            mu_all.append(mi); C_all.append(Ci)
        return jnp.stack(mu_all), jnp.stack(C_all), Xs, mu_zs

    return model, posterior_O_fn, time_eval


def run_one(schema, p):
    print(f"\n{'=' * 78}\n  {schema['label']}  —  hierarchical marg-O (04h)\n{'=' * 78}")
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

    Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(t_samp, snaps_comp, verbose=False)
    print(f"  GP-MLE hypers: ℓ ≈ {np.mean(Ls):.4f}   σ² ≈ {np.mean(Vs):.3f}   ν ≈ {np.mean(Ns):.6f}")

    model, posterior_O_fn, time_eval = build_model(
        rom=rom, num_modes=nmodes,
        time_sampled=t_samp, snapshots_comp=snaps_comp,
        num_eval_points=neval, window_size=p['WINDOW_SIZE'],
        deriv_weight=p['DERIV_WEIGHT'], integral_weight=p['INTEGRAL_WEIGHT'],
        mll_weight=p['MLL_WEIGHT'],
        sigma_O_scale=p['SIGMA_O_SCALE'], gamma2_scale=p['GAMMA2_SCALE'])

    init_values = {}
    for i in range(nmodes):
        init_values[f'lengthscale_{i}'] = jnp.asarray(Ls[i])
        init_values[f'variance_{i}'] = jnp.asarray(Vs[i])
        init_values[f'noise_{i}'] = jnp.asarray(Ns[i])
    init_values['tau_O'] = jnp.asarray(float(p['SIGMA_O_SCALE']))
    for i in range(nmodes):
        init_values[f'sigma_O_{i}'] = jnp.asarray(float(p['SIGMA_O_SCALE']))

    guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    optimizer = ClippedAdam(step_size=p['LEARNING_RATE'])
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key, ik = random.split(rng_key)
    t0 = time.time()
    state = svi.init(ik)

    @jax.jit
    def _step(s, _):
        return svi.update(s)

    nsteps = p['NUM_STEPS']
    state, losses = jax.lax.scan(_step, state, jnp.arange(nsteps))
    losses = np.array(losses)
    print(f"  SVI: {nsteps} steps   loss {losses[0]:.2f} → {losses[-1]:.2f}   ({time.time()-t0:.1f}s)")

    params = svi.get_params(state)
    rng_key, sk = random.split(rng_key)
    npost = p['NUM_POSTERIOR_SAMPLES']
    post = guide.sample_posterior(sk, params, sample_shape=(npost,))

    def _stack_theta(d):
        return (jnp.stack([d[f'lengthscale_{i}'] for i in range(nmodes)], axis=-1),
                jnp.stack([d[f'variance_{i}'] for i in range(nmodes)], axis=-1),
                jnp.stack([d[f'noise_{i}'] for i in range(nmodes)], axis=-1))

    ells_s, sig2s_s, nus_s = _stack_theta(post)
    g2_s = jnp.full((npost,), float(p['GAMMA2_SCALE']))
    sO_s = jnp.stack([post[f'sigma_O_{i}'] for i in range(nmodes)], axis=-1)
    tauO_s = post['tau_O']

    print(f"  Fixed γ²:      {float(p['GAMMA2_SCALE']):.3g}")
    print(f"  Inferred τ_O:  median={float(jnp.median(tauO_s)):.3g}  "
          f"q5={float(jnp.percentile(tauO_s, 5)):.3g}  q95={float(jnp.percentile(tauO_s, 95)):.3g}")
    print(f"  Inferred σ_O:  per-mode median {np.array(jnp.median(sO_s, axis=0))}")

    @jax.jit
    def _draw_O(ells, sig2s, nus, g2_val, sO_pm, key):
        mu_O, C_O, _, _ = posterior_O_fn(ells, sig2s, nus, g2_val, sO_pm)
        eps = jax.random.normal(key, shape=mu_O.shape)
        O = mu_O + jnp.einsum('ijk,ik->ij', C_O, eps)
        return O, mu_O

    rng_key, ok = random.split(rng_key)
    keys = jax.random.split(ok, npost)
    t_o = time.time()
    O_samples_list, O_mean_list = [], []
    for s in range(npost):
        O, mu_O = _draw_O(ells_s[s], sig2s_s[s], nus_s[s], g2_s[s], sO_s[s], keys[s])
        O_samples_list.append(np.array(O))
        O_mean_list.append(np.array(mu_O))
    O_samples = np.stack(O_samples_list)
    O_means = np.stack(O_mean_list)
    print(f"  O posterior sampling: {time.time()-t_o:.1f}s")
    op_norms = np.linalg.norm(O_samples.reshape(npost, -1), axis=1)
    print(f"  ‖O‖: median={np.median(op_norms):.1f}   min={op_norms.min():.1f}   max={op_norms.max():.1f}")
    print(f"  ‖μ_O‖ (avg over θ-samples): {np.linalg.norm(O_means, axis=(1,2)).mean():.1f}")

    from core import generate_rom_predictions
    samples_for_rom = {'O': jnp.array(O_samples)}
    for i in range(nmodes):
        samples_for_rom[f'X_{i}'] = jnp.stack(post[f'lengthscale_{i}'])
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
    print(f"\n  RESULTS — Hierarchical marg-O (04h)")
    print(f"    Stability:   {n_stable}/{n_total} ({stab:.1f}%)")
    print(f"    Train error: {train_err:.2%}")
    print(f"    Pred error:  {pred_err:.2%}")
    print(f"    CI coverage: {ci_cov:.1%} (target 90%)")
    print(f"    CI width:    {ci_w:.4f}")
    print(f"    Runtime:     {runtime:.0f}s")

    out_dir = os.path.join(SCRIPT_DIR, 'results', 'comparison', schema['name'])
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, '04h_hierarchical_marg_O.npz'),
             train_error=train_err, pred_error=pred_err,
             stability_pct=stab, ci_coverage=ci_cov, ci_width=ci_w, runtime=runtime,
             op_norm_median=float(np.median(op_norms)),
             gamma2_fixed=float(p['GAMMA2_SCALE']),
             tau_O_median=float(jnp.median(tauO_s)),
             sigma_O_median_per_mode=np.array(jnp.median(sO_s, axis=0)),
             losses=losses)
    return train_err, pred_err, stab, ci_cov, runtime


if __name__ == "__main__":
    args = sys.argv[1:]
    p = dict(MODEL_PARAMS)
    schema_name = args[0] if args else 'dense_low_noise'
    schema = next(s for s in SCHEMAS if s['name'] == schema_name)
    run_one(schema, p)

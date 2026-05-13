"""
04_unified.py — Marginalised-O × Weak-Form Bayesian OpInf (Heat / multi-IC).

Adapts the single-IC burgers_2d/04_unified.py to the multi-IC, input-dependent
heat equation.  The operator O is shared across all ICs and marginalised
analytically; each IC contributes derivative-form rows + WSINDy-style
weak-form rows that are linear in O.  Stacking across ICs gives one big linear
system per ROM mode, solved in closed form via a single m×m Cholesky.

Usage
-----
    python 04_unified.py
    python 04_unified.py sparse_medium_noise
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
from config import (
    Basis, ReducedOrderModel, input_func_factory,
    input_parameters, test_parameters,
)
from step1_generate_data import TrajectorySampler
from core.bayesian_opinf import fit_gp_hyperparameters_mle, _find_operator_samples
from core.diagnostics import plot_trace
from core.plotting import plot_full_order_error
from heat_plotter import _generate_rom_solves
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

# ── Data regimes ─────────────────────────────────────────────────────────────
SCHEMAS = [
    {"name": "sparse_low_noise",    "label": "Sparse data, low noise",
     "NUM_SAMPLES": 20, "NOISE_LEVEL": 0.01, "NUM_EVAL_POINTS": 100},
    {"name": "sparse_medium_noise", "label": "Sparse data, medium noise",
     "NUM_SAMPLES": 20, "NOISE_LEVEL": 0.03, "NUM_EVAL_POINTS": 100},
    {"name": "sparse_high_noise",   "label": "Sparse data, high noise",
     "NUM_SAMPLES": 20, "NOISE_LEVEL": 0.05, "NUM_EVAL_POINTS": 100},
]

MODEL_PARAMS = dict(
    NUM_MODES=5,
    NUM_ICS=5,
    GAMMA2=0.5,
    DERIV_WEIGHT=1.0,
    WEAKFORM_WEIGHT=2.0,
    MLL_WEIGHT=0.1,
    SIGMA_O=3.0,
    WINDOW_SIZE=20,
    BUMP_P=6,
    NUM_TEST_FUNCS=None,
    BUMP_RADIUS_FRAC=None,
    NUM_STEPS=10000,
    LEARNING_RATE=3e-3,
    NUM_POSTERIOR_SAMPLES=500,
    REGULARIZER=1.0,
    GP_PRIOR_SCALE=0.1,
    SEED=42,
)

TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 2.0)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")

LAMBDA_JITTER = 1e-6


# =============================================================================
# Model builder
# =============================================================================
def build_model(
    rom, num_modes, num_ics,
    all_time_sampled, all_snapshots_comp, all_inputs_eval,
    all_mle_Ls, all_mle_Vs, all_mle_Ns,
    num_eval_points, window_size,
    deriv_weight, weakform_weight, mll_weight,
    sigma_O, bump_p, num_test_funcs, bump_radius_frac,
    gp_prior_scale=0.1,
):
    """Build multi-IC marginalised-O + weak-form Bayesian model."""

    # ── Per-IC precompute ────────────────────────────────────────────────
    all_ic_data = []
    for ic in range(num_ics):
        t_train = jnp.array(all_time_sampled[ic])
        n_train = len(t_train)
        y_obs = jnp.array(all_snapshots_comp[ic])
        inputs_eval = jnp.array(all_inputs_eval[ic])

        time_eval = np.linspace(float(t_train[0]), float(t_train[-1]),
                                num_eval_points)
        t_eval = jnp.array(time_eval)
        dt_eval = float(time_eval[1] - time_eval[0])

        sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
        sq_diffs_et = (t_eval[:, None] - t_train[None, :]) ** 2
        diffs_et = t_eval[:, None] - t_train[None, :]
        sq_diffs_ee = (t_eval[:, None] - t_eval[None, :]) ** 2
        I_train = jnp.eye(n_train)

        # ── Weak-form bump test functions ─────────────────────────────
        T_total = float(time_eval[-1] - time_eval[0])
        ntf = num_test_funcs
        if ntf is None:
            ntf = max(1, num_eval_points // window_size)
        if bump_radius_frac is None:
            radius = window_size * dt_eval
        else:
            radius = bump_radius_frac * T_total

        centres = np.linspace(time_eval[0] + radius, time_eval[-1] - radius, ntf)
        psi_list, psi_dot_list, int_psi_sq_list = [], [], []
        for tc in centres:
            tau = (time_eval - tc) / radius
            in_supp = np.abs(tau) < 1.0
            psi_vals = np.where(in_supp, (1.0 - tau ** 2) ** bump_p, 0.0)
            psi_d_vals = np.where(in_supp,
                -2.0 * bump_p * tau * (1.0 - tau ** 2) ** (bump_p - 1) / radius,
                0.0)
            psi_vals[~in_supp] = 0.0
            psi_d_vals[~in_supp] = 0.0
            psi_list.append(psi_vals.astype(np.float32))
            psi_dot_list.append(psi_d_vals.astype(np.float32))
            w = np.ones_like(time_eval) * dt_eval
            w[0] *= 0.5
            w[-1] *= 0.5
            int_psi_sq_list.append(float(np.sum(w * psi_vals ** 2)))

        psi_arr = jnp.asarray(np.stack(psi_list))
        psi_dot_arr = jnp.asarray(np.stack(psi_dot_list))
        int_psi_sq_arr = jnp.asarray(np.array(int_psi_sq_list, dtype=np.float32))

        trap_w = np.ones_like(time_eval) * dt_eval
        trap_w[0] *= 0.5
        trap_w[-1] *= 0.5
        trap_w_jnp = jnp.asarray(trap_w.astype(np.float32))
        wpsi = trap_w_jnp[None, :] * psi_arr
        wpsi_dot = trap_w_jnp[None, :] * psi_dot_arr

        mle_log_ells = jnp.array([float(np.log(all_mle_Ls[ic][j]))
                                  for j in range(num_modes)])
        mle_log_sig2s = jnp.array([float(np.log(all_mle_Vs[ic][j]))
                                   for j in range(num_modes)])
        mle_log_nus = jnp.array([float(np.log(all_mle_Ns[ic][j]))
                                 for j in range(num_modes)])

        all_ic_data.append(dict(
            t_train=t_train, n_train=n_train, y_obs=y_obs,
            inputs_eval=inputs_eval,
            t_eval=t_eval, time_eval=time_eval,
            sq_diff_tt=sq_diff_tt, sq_diffs_et=sq_diffs_et,
            diffs_et=diffs_et, sq_diffs_ee=sq_diffs_ee, I_train=I_train,
            psi_arr=psi_arr, psi_dot_arr=psi_dot_arr,
            int_psi_sq_arr=int_psi_sq_arr,
            wpsi=wpsi, wpsi_dot=wpsi_dot, K_test=wpsi.shape[0],
            mle_log_ells=mle_log_ells, mle_log_sig2s=mle_log_sig2s,
            mle_log_nus=mle_log_nus,
        ))

    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def _single_gp_conditional(ell, sig2, nu, y_i, d):
        ell2 = ell ** 2
        jitter = jnp.maximum(1e-5, sig2 * 1e-3)
        K_tt = _rbf_sq(ell, sig2, d['sq_diff_tt']) + (nu + jitter) * d['I_train']
        L = jnp.linalg.cholesky(K_tt)
        alpha = jax.scipy.linalg.cho_solve((L, True), y_i)
        K_et = _rbf_sq(ell, sig2, d['sq_diffs_et'])
        X_eval = K_et @ alpha
        K_zy = -(d['diffs_et'] / ell2) * K_et
        mu_z = K_zy @ alpha
        K_ee = _rbf_sq(ell, sig2, d['sq_diffs_ee'])
        K_zz = ((1.0 - d['sq_diffs_ee'] / ell2) / ell2) * K_ee
        V = jax.scipy.linalg.cho_solve((L, True), K_zy.T)
        deriv_var = jnp.maximum(jnp.diag(K_zz) - jnp.sum(K_zy * V.T, axis=1), 0.0)
        mll = -0.5 * (jnp.dot(y_i, alpha) +
                      2.0 * jnp.sum(jnp.log(jnp.diag(L))) +
                      d['n_train'] * jnp.log(2.0 * jnp.pi))
        return X_eval, mu_z, deriv_var, mll

    inv_sigma_O2_default = 1.0 / (sigma_O ** 2)
    log_sigma_O2_default = 2.0 * jnp.log(sigma_O)

    def _per_mode_evidence(A, y_i, prec_i, m, inv_sigma_O2, log_sigma_O2):
        Aw = A * prec_i[:, None]
        M_i = A.T @ Aw
        M_i = 0.5 * (M_i + M_i.T)
        ridge = (inv_sigma_O2 + LAMBDA_JITTER * jnp.maximum(
            jnp.trace(M_i) / m, 1.0))
        Lambda_i = M_i + ridge * jnp.eye(m)
        L_i = jnp.linalg.cholesky(Lambda_i)
        b_i = A.T @ (prec_i * y_i)
        mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_i)
        log_det_Lambda = 2.0 * jnp.sum(jnp.log(jnp.diag(L_i)))
        quad_y = jnp.sum(prec_i * y_i ** 2)
        quad_mu = jnp.dot(mu_i, b_i)
        log_det_Sigma = -jnp.sum(jnp.log(prec_i + 1e-30))
        N_i = prec_i.shape[0]
        log_p = -0.5 * ((quad_y - quad_mu) + log_det_Sigma +
                        m * log_sigma_O2 + log_det_Lambda +
                        N_i * jnp.log(2.0 * jnp.pi))
        return log_p, mu_i, L_i

    def _build_AyP_ic(ic, ells, sig2s, nus, gamma2):
        """Return (Xs, A_ic, y_per_mode_ic, prec_per_mode_ic, mll_total) for one IC."""
        d = all_ic_data[ic]
        Xs_list, muz_list, dv_list, mll_list = [], [], [], []
        for j in range(num_modes):
            Xj, muj, dvj, mllj = _single_gp_conditional(
                ells[j], sig2s[j], nus[j], d['y_obs'][j], d)
            Xs_list.append(Xj)
            muz_list.append(muj)
            dv_list.append(dvj)
            mll_list.append(mllj)
        Xs = jnp.stack(Xs_list)
        mu_zs = jnp.stack(muz_list)
        deriv_vars = jnp.stack(dv_list)
        mll_total = jnp.sum(jnp.stack(mll_list))

        f_X = rom.model._assemble_data_matrix(Xs, inputs=d['inputs_eval'])
        prec_deriv = deriv_weight / (deriv_vars + gamma2 + 1e-4)
        A_weak = d['wpsi'] @ f_X
        weak_obs = -(Xs @ d['wpsi_dot'].T)
        prec_weak_per_k = weakform_weight / (gamma2 * d['int_psi_sq_arr'] + 1e-12)
        prec_weak = jnp.broadcast_to(prec_weak_per_k, (num_modes, d['K_test']))

        A_ic = jnp.concatenate([f_X, A_weak], axis=0)
        y_ic = jnp.concatenate([mu_zs, weak_obs], axis=1)
        prec_ic = jnp.concatenate([prec_deriv, prec_weak], axis=1)
        return Xs, A_ic, y_ic, prec_ic, mll_total

    def _build_AyP_all(theta):
        """theta: dict of arrays (num_ics, num_modes) per hyper."""
        Xs_all, A_all, y_all, prec_all = [], [], [], []
        mll_total = 0.0
        for ic in range(num_ics):
            Xs, A_ic, y_ic, prec_ic, mll_ic = _build_AyP_ic(
                ic, theta['ells'][ic], theta['sig2s'][ic], theta['nus'][ic],
                theta['gamma2'])
            Xs_all.append(Xs)
            A_all.append(A_ic)
            y_all.append(y_ic)
            prec_all.append(prec_ic)
            mll_total = mll_total + mll_ic
        A = jnp.concatenate(A_all, axis=0)
        y_per_mode = jnp.concatenate(y_all, axis=1)
        prec_per_mode = jnp.concatenate(prec_all, axis=1)
        return Xs_all, A, y_per_mode, prec_per_mode, mll_total

    def model(gamma2=0.5):
        # Sample per-IC, per-mode GP hypers.
        ells_per = []
        sig2s_per = []
        nus_per = []
        for ic in range(num_ics):
            d = all_ic_data[ic]
            ells = jnp.stack([
                numpyro.sample(f"lengthscale_{ic}_{j}",
                    dist.LogNormal(d['mle_log_ells'][j], gp_prior_scale))
                for j in range(num_modes)])
            sig2s = jnp.stack([
                numpyro.sample(f"variance_{ic}_{j}",
                    dist.LogNormal(d['mle_log_sig2s'][j], gp_prior_scale))
                for j in range(num_modes)])
            nus = jnp.stack([
                numpyro.sample(f"noise_{ic}_{j}",
                    dist.LogNormal(d['mle_log_nus'][j], gp_prior_scale))
                for j in range(num_modes)])
            ells_per.append(ells)
            sig2s_per.append(sig2s)
            nus_per.append(nus)

        theta = dict(
            ells=jnp.stack(ells_per),
            sig2s=jnp.stack(sig2s_per),
            nus=jnp.stack(nus_per),
            gamma2=gamma2,
        )

        Xs_all, A, y_per_mode, prec_per_mode, mll_total = _build_AyP_all(theta)

        for ic in range(num_ics):
            for j in range(num_modes):
                numpyro.deterministic(f"X_{ic}_{j}", Xs_all[ic][j])

        if mll_weight > 0:
            numpyro.factor("gp_mll", mll_weight * mll_total)

        m = A.shape[1]
        total_evidence = 0.0
        for i in range(num_modes):
            log_p_i, _, _ = _per_mode_evidence(
                A, y_per_mode[i], prec_per_mode[i], m,
                inv_sigma_O2_default, log_sigma_O2_default)
            total_evidence = total_evidence + log_p_i
        numpyro.factor("marg_O_evidence", total_evidence)

    @jax.jit
    def posterior_O_fn(ells_stacked, sig2s_stacked, nus_stacked,
                       gamma2, sigma_O_val):
        """Closed-form O posterior given θ.

        ells_stacked etc.: (num_ics, num_modes)
        Returns (μ_O, C_O) where C_O Cᵀ_O = Σ_O for each mode.
        """
        inv_sO2 = 1.0 / (sigma_O_val ** 2 + 1e-12)
        theta = dict(ells=ells_stacked, sig2s=sig2s_stacked,
                     nus=nus_stacked, gamma2=gamma2)
        _, A, y_per_mode, prec_per_mode, _ = _build_AyP_all(theta)
        m = A.shape[1]

        def _one(y_i, prec_i):
            Aw = A * prec_i[:, None]
            M_i = A.T @ Aw
            M_i = 0.5 * (M_i + M_i.T)
            ridge = (inv_sO2 + LAMBDA_JITTER *
                     jnp.maximum(jnp.trace(M_i) / m, 1.0))
            Lambda_i = M_i + ridge * jnp.eye(m)
            L_i = jnp.linalg.cholesky(Lambda_i)
            b_i = A.T @ (prec_i * y_i)
            mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_i)
            C_i = jax.scipy.linalg.solve_triangular(
                L_i, jnp.eye(m), lower=True).T
            return mu_i, C_i

        mu_all, C_all = [], []
        for i in range(num_modes):
            mi, Ci = _one(y_per_mode[i], prec_per_mode[i])
            mu_all.append(mi)
            C_all.append(Ci)
        return jnp.stack(mu_all), jnp.stack(C_all)

    return model, posterior_O_fn


# =============================================================================
# Run one regime
# =============================================================================
def run_experiment(schema, p=None):
    if p is None:
        p = MODEL_PARAMS
    noise = schema['NOISE_LEVEL']
    nsamp = schema['NUM_SAMPLES']
    neval = schema['NUM_EVAL_POINTS']
    nmodes = p['NUM_MODES']
    nics = p['NUM_ICS']
    train_params = input_parameters[:nics]

    print(f"\n{'=' * 78}")
    print(f"  {schema['label']}  ({nsamp} samples, {noise:.0%} noise)"
          f"  —  marg-O × weak-form (heat, multi-IC)")
    print(f"{'=' * 78}")

    np.random.seed(p['SEED'])
    rng_key = random.PRNGKey(p['SEED'])

    # ── Data ─────────────────────────────────────────────────────────────
    sampler = TrajectorySampler(
        training_span=TRAINING_SPAN,
        num_samples=nsamp,
        noiselevel=noise,
        num_regression_points=neval,
        synced=False,
    )
    (all_true_states, all_time_sampled, all_snapshots,
     all_training_inputs) = sampler.multisample(train_params)

    snapshots_train = np.hstack(all_snapshots)
    basis = Basis(num_vectors=nmodes)
    basis.fit(snapshots_train)
    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    all_snapshots_comp = [basis.compress(s) for s in all_snapshots]
    all_true_comp = [basis.compress(s) for s in all_true_states]

    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(all_time_sampled[0]),
        model=ReducedOrderModel(),
    )
    first_input_func = input_func_factory(train_params[0])
    first_inputs = first_input_func(all_time_sampled[0])
    rom.fit(states=all_snapshots[0], inputs=first_inputs)
    print(f"  Operator shape: {rom.model.operator_matrix.shape}")

    # Per-IC inputs on eval grid
    all_inputs_eval = []
    for ic in range(nics):
        t_eval_ic = np.linspace(float(all_time_sampled[ic][0]),
                                float(all_time_sampled[ic][-1]), neval)
        in_func = input_func_factory(train_params[ic])
        all_inputs_eval.append(in_func(t_eval_ic))

    # GP-MLE per IC
    all_mle_Ls, all_mle_Vs, all_mle_Ns = [], [], []
    for ic in range(nics):
        Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(
            all_time_sampled[ic], all_snapshots_comp[ic], verbose=False)
        all_mle_Ls.append(Ls)
        all_mle_Vs.append(Vs)
        all_mle_Ns.append(Ns)
        print(f"  GP-MLE IC{ic} {train_params[ic]}: ℓ̄={np.mean(Ls):.4f} "
              f"σ̄²={np.mean(Vs):.3f} ν̄={np.mean(Ns):.6f}")

    # ── Build model ──────────────────────────────────────────────────────
    model, posterior_O_fn = build_model(
        rom=rom, num_modes=nmodes, num_ics=nics,
        all_time_sampled=all_time_sampled,
        all_snapshots_comp=all_snapshots_comp,
        all_inputs_eval=all_inputs_eval,
        all_mle_Ls=all_mle_Ls, all_mle_Vs=all_mle_Vs, all_mle_Ns=all_mle_Ns,
        num_eval_points=neval, window_size=p['WINDOW_SIZE'],
        deriv_weight=p['DERIV_WEIGHT'],
        weakform_weight=p['WEAKFORM_WEIGHT'],
        mll_weight=p['MLL_WEIGHT'], sigma_O=p['SIGMA_O'],
        bump_p=p['BUMP_P'], num_test_funcs=p['NUM_TEST_FUNCS'],
        bump_radius_frac=p['BUMP_RADIUS_FRAC'],
        gp_prior_scale=p['GP_PRIOR_SCALE'],
    )

    init_values = {}
    for ic in range(nics):
        for j in range(nmodes):
            init_values[f'lengthscale_{ic}_{j}'] = jnp.asarray(all_mle_Ls[ic][j])
            init_values[f'variance_{ic}_{j}'] = jnp.asarray(all_mle_Vs[ic][j])
            init_values[f'noise_{ic}_{j}'] = jnp.asarray(all_mle_Ns[ic][j])

    guide = autoguide.AutoNormal(model,
        init_loc_fn=init_to_value(values=init_values))
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
    seg_size = max(1, nsteps // 10)
    all_losses = []
    for seg in range(10):
        start = seg * seg_size
        end = min(start + seg_size, nsteps)
        if seg == 9:
            end = nsteps
        if start >= nsteps:
            break
        state, seg_losses = jax.lax.scan(_step, state, jnp.arange(end - start))
        seg_np = np.array(seg_losses)
        all_losses.extend(seg_np.tolist())
        print(f"    step {end:6d}/{nsteps}  loss={seg_np[-1]:10.2f}")
    losses = np.array(all_losses)

    params = svi.get_params(state)
    rng_key, sk = random.split(rng_key)
    npost = p['NUM_POSTERIOR_SAMPLES']
    post = guide.sample_posterior(sk, params, sample_shape=(npost,),
                                  **model_kwargs)

    # ── Draw O from closed-form conditional per θ-sample ──────────────────
    def _stack(d, key):
        return jnp.stack([
            jnp.stack([d[f'{key}_{ic}_{j}'] for j in range(nmodes)], axis=-1)
            for ic in range(nics)
        ], axis=-2)  # (npost, nics, nmodes)

    ells_s = _stack(post, 'lengthscale')
    sig2s_s = _stack(post, 'variance')
    nus_s = _stack(post, 'noise')

    sO_default = jnp.asarray(p['SIGMA_O'])
    rng_key, ok = random.split(rng_key)
    keys = jax.random.split(ok, npost)

    @jax.jit
    def _draw_O(ells, sig2s, nus, key):
        mu_O, C_O = posterior_O_fn(ells, sig2s, nus, p['GAMMA2'], sO_default)
        eps = jax.random.normal(key, shape=mu_O.shape)
        return mu_O + jnp.einsum('ijk,ik->ij', C_O, eps), mu_O

    t_o = time.time()
    O_samples_list, O_mean_list = [], []
    for s in range(npost):
        O, mu_O = _draw_O(ells_s[s], sig2s_s[s], nus_s[s], keys[s])
        O_samples_list.append(np.array(O))
        O_mean_list.append(np.array(mu_O))
    O_samples = np.stack(O_samples_list)
    O_means = np.stack(O_mean_list)
    print(f"  O posterior sampling: {time.time()-t_o:.1f}s")
    op_norms = np.linalg.norm(O_samples.reshape(npost, -1), axis=1)
    print(f"  ‖O‖: median={np.median(op_norms):.1f}  "
          f"min={op_norms.min():.1f}  max={op_norms.max():.1f}")

    # Build samples dict for heat_plotter / _find_operator_samples
    samples = dict(post)
    samples['O'] = jnp.array(O_samples)

    runtime = time.time() - t0

    # ── Test IC ──────────────────────────────────────────────────────────
    test_sampler = TrajectorySampler(
        training_span=TRAINING_SPAN, num_samples=nsamp,
        noiselevel=noise, num_regression_points=neval, synced=False)
    (test_true_list, test_t_list, test_snap_list,
     _) = test_sampler.multisample([test_parameters])
    test_true_comp = basis.compress(test_true_list[0])
    test_snaps_comp = basis.compress(test_snap_list[0])
    test_t_samp = test_t_list[0]

    eval_params = list(train_params) + [test_parameters]
    eval_snaps_comp = all_snapshots_comp + [test_snaps_comp]
    eval_true_comp = all_true_comp + [test_true_comp]
    eval_t_samp = all_time_sampled + [test_t_samp]
    eval_labels = [f"Train IC {i} {train_params[i]}" for i in range(nics)] + \
                  [f"Test IC {test_parameters}"]

    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    max_samp = min(200, npost)

    all_rom_solves, all_n_stable = [], []
    all_train_errors, all_pred_errors = [], []
    train_mask = t_pred <= TRAINING_SPAN[1]
    pred_mask = t_pred > TRAINING_SPAN[1]

    print(f"\n  Results ({runtime:.0f}s):")
    for ic_idx, (params_ic, true_c) in enumerate(zip(eval_params, eval_true_comp)):
        q0 = eval_snaps_comp[ic_idx][:, 0]
        _jax_input = input_func_factory(params_ic)
        ic_input_func = lambda t, _f=_jax_input: np.asarray(_f(t))
        ic_solves = _generate_rom_solves(
            operator_samples=O_samples, rom=rom, q0=q0,
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
        print(f"    {eval_labels[ic_idx]}: {all_n_stable[-1]}/{max_samp} stable, "
              f"train={te:.4%}, pred={pe:.4%}")

    train_ic_stable = sum(all_n_stable[:nics])
    train_ic_total = max_samp * nics
    tr_fin = [e for e in all_train_errors[:nics] if np.isfinite(e)]
    pr_fin = [e for e in all_pred_errors[:nics] if np.isfinite(e)]
    train_error = float(np.mean(tr_fin)) if tr_fin else float('inf')
    pred_error = float(np.mean(pr_fin)) if pr_fin else float('inf')
    stability_pct = train_ic_stable / max(train_ic_total, 1) * 100

    print(f"\n    Overall: {train_ic_stable}/{train_ic_total} ({stability_pct:.0f}%)")
    print(f"    Avg train: {train_error:.4%}  |  Avg pred: {pred_error:.4%}")
    print(f"    Test IC:   {all_n_stable[-1]}/{max_samp} stable, "
          f"train={all_train_errors[-1]:.4%}, pred={all_pred_errors[-1]:.4%}")

    ci_width = ci_coverage = float('nan')
    if train_ic_stable > 0:
        all_in_ci, all_widths = [], []
        for ic_idx in range(nics):
            if all_n_stable[ic_idx] > 0:
                ti = interp1d(config.time_domain, eval_true_comp[ic_idx],
                              kind='cubic', fill_value='extrapolate')
                ta = ti(t_pred)
                q05 = np.percentile(all_rom_solves[ic_idx], 5, axis=0)
                q95 = np.percentile(all_rom_solves[ic_idx], 95, axis=0)
                all_widths.append(np.mean(q95 - q05))
                all_in_ci.append(np.mean((ta >= q05) & (ta <= q95)))
        if all_widths:
            ci_width = float(np.mean(all_widths))
            ci_coverage = float(np.mean(all_in_ci))
            print(f"    CI coverage: {ci_coverage:.2%} (target: 90%)")

    # ── Persist (schema matches heat 04_conditional_integral) ────────────
    out_dir = os.path.join(SCRIPT_DIR, 'results', 'comparison', schema['name'])
    os.makedirs(out_dir, exist_ok=True)
    save_dict = {
        't_pred': t_pred,
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'runtime': runtime,
        'n_ics': len(all_rom_solves),
        'op_norm_median': float(np.median(op_norms)),
        'losses': losses,
    }
    for ic_idx, solves in enumerate(all_rom_solves):
        if len(solves) > 0:
            save_dict[f'rom_solves_{ic_idx}'] = np.array(solves)
        else:
            save_dict[f'rom_solves_{ic_idx}'] = np.empty(
                (0, p['NUM_MODES'], len(t_pred)))
    np.savez(os.path.join(out_dir, '04_unified.npz'), **save_dict)

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
        'all_rom_solves': all_rom_solves,
        'all_snaps_comp': eval_snaps_comp,
        'all_true_comp': eval_true_comp,
        'all_t_samp': eval_t_samp,
        'all_n_stable': all_n_stable,
        'eval_labels': eval_labels,
        't_full': config.time_domain,
        't_pred': t_pred,
        'training_span': TRAINING_SPAN,
        'num_modes': nmodes,
        'basis': basis,
        'all_true_states_full': all_true_states,
        'O_means': O_means,
    }


# =============================================================================
# Entry point
# =============================================================================
def main(schema_names=None):
    if schema_names is None or len(schema_names) == 0:
        schemas = SCHEMAS
    else:
        schemas = [s for s in SCHEMAS if s['name'] in schema_names]
        if not schemas:
            print(f"Unknown schema(s): {schema_names}")
            print(f"Available: {[s['name'] for s in SCHEMAS]}")
            return

    print("=" * 78)
    print("04_unified — Marginalised-O × Weak-Form Bayesian OpInf (Heat, multi-IC)")
    print("=" * 78)
    print(f"γ²={MODEL_PARAMS['GAMMA2']}  σ_O={MODEL_PARAMS['SIGMA_O']}  "
          f"bump_p={MODEL_PARAMS['BUMP_P']}  "
          f"lr={MODEL_PARAMS['LEARNING_RATE']}  steps={MODEL_PARAMS['NUM_STEPS']}  "
          f"ICs={MODEL_PARAMS['NUM_ICS']}")

    results = []
    for schema in schemas:
        r = run_experiment(schema)
        results.append(r)

    print(f"\n\n{'=' * 90}")
    print(f"SUMMARY — Marg-O × Weak-Form (Heat, multi-IC)")
    print(f"{'=' * 90}")
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

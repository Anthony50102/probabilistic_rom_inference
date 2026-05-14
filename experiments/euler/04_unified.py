"""
04_unified.py — Marginalised-O Bayesian OpInf with weak-form constraint.

Combines two ideas that compose cleanly because they are both LINEAR in the
operator O:

  (1) From 04g_marginal_O.py — analytically marginalise O.
      With O ~ N(0, σ_O² I) and any data terms linear in O, the conditional
      posterior p(O | θ, data) is Gaussian (conjugate) and the marginal
      likelihood p(data | θ) has a closed form.  SVI/NUTS therefore only
      need to explore the ~3 r-dimensional GP-hyperparameter space θ.

  (2) From 04b_weak_form.py — replace the ad-hoc indicator-window integral
      with a WSINDy-style weak form against smooth bump test functions
          ψ_k(t) = (1 - τ²)^p,  τ = (t - t_k) / r.
      Integration by parts gives  -∫ψ_k' X_i dt ≈ ∫ψ_k [f(X)Oᵀ]_i dt,
      which is *still linear in O* — so the marginalisation still applies.

The likelihood factors are therefore concatenated rows of one linear-in-O
system per ROM mode i:

    derivative rows (T_eval):
        μ_z_i(t_j)           ≈ [f(X) Oᵀ]_i (t_j)        prec = 1/(deriv_var_ij + γ²)
    weak-form rows (K):
        -∫ψ_k'(t) X_i(t) dt  ≈ ∫ψ_k(t) [f(X)Oᵀ]_i(t) dt  prec = 1/(γ² ∫ψ_k² dt)

`_per_mode_evidence` computes log p(y_i | θ) in closed form per mode; their
sum is the model factor.  `posterior_O_fn` returns (μ_O, C_O) where C_O Cᵀ_O
= Σ_O, used for posterior sampling of O for downstream ROM prediction.

Usage
-----
    python 04_unified.py                  # run all 3 regimes
    python 04_unified.py dense_low_noise  # run one regime
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
from numpyro.infer.initialization import init_to_median
from numpyro.optim import ClippedAdam
from jax import random
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis
from core import (
    generate_trajectory, JaxCompatibleModel, compute_gp_derivatives,
    generate_rom_predictions, rbf_eval,
)
from core.diagnostics import plot_trace
from core.plotting import plot_full_order_error
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

# ── Data regimes ─────────────────────────────────────────────────────────────
SCHEMAS = [
    {"name": "dense_low_noise",  "NUM_SAMPLES": 250, "NOISE_LEVEL": 0.01,
     "NUM_EVAL_POINTS": 400, "label": "Dense data, low noise"},
    {"name": "sparse_low_noise", "NUM_SAMPLES": 55,  "NOISE_LEVEL": 0.03,
     "NUM_EVAL_POINTS": 200, "label": "Sparse data, low noise"},
    {"name": "dense_high_noise", "NUM_SAMPLES": 250, "NOISE_LEVEL": 0.10,
     "NUM_EVAL_POINTS": 400, "label": "Dense data, high noise"},
]

# ── Shared model hyperparameters ─────────────────────────────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=6,
    GAMMA2=10.0,
    DERIV_WEIGHT=1.0,
    WEAKFORM_WEIGHT=1.0,
    MLL_WEIGHT=1.0,
    SIGMA_O=30.0,          # O ~ N(0, SIGMA_O² I)
    WINDOW_SIZE=20,
    # Weak-form (bump) test function settings
    BUMP_P=6,              # smoothness exponent for (1 - τ²)^p
    NUM_TEST_FUNCS=None,   # default: num_eval_points // WINDOW_SIZE
    BUMP_RADIUS_FRAC=None, # default: window_size * dt_eval (matches 04 / 04g)
    NUM_STEPS=8000,
    LEARNING_RATE=5e-3,
    NUM_POSTERIOR_SAMPLES=500,
    SEED=42,
)

TRAINING_SPAN = (0, 0.08)
PREDICTION_SPAN = (0, 0.15)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# Model builder
# =============================================================================
def build_model(rom, num_modes, time_sampled, snapshots_comp,
                num_eval_points, window_size,
                deriv_weight, weakform_weight, mll_weight,
                sigma_O, bump_p, num_test_funcs, bump_radius_frac):
    """Build marginalised-O + weak-form Bayesian model.

    Returns
    -------
    model : numpyro model over θ_GP only (O is marginalised analytically)
    posterior_O_fn : closed-form O posterior given θ
                     callable(ells, sig2s, nus, gamma2, sigma_O_val)
                     → (μ_O, C_O, Xs, mu_zs)  where C_O Cᵀ_O = Σ_O
    time_eval : np.ndarray of evaluation times
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

    # ── Precompute weak-form (bump) test functions ───────────────────────
    T_total = float(time_eval[-1] - time_eval[0])
    if num_test_funcs is None:
        num_test_funcs = max(1, num_eval_points // window_size)
    if bump_radius_frac is None:
        radius = window_size * dt_eval
    else:
        radius = bump_radius_frac * T_total

    centres = np.linspace(time_eval[0] + radius, time_eval[-1] - radius,
                          num_test_funcs)

    psi_list, psi_dot_list, int_psi_sq_list = [], [], []
    for tc in centres:
        tau = (time_eval - tc) / radius
        in_supp = np.abs(tau) < 1.0
        psi_vals = np.where(in_supp, (1.0 - tau ** 2) ** bump_p, 0.0)
        # dψ/dt = (dψ/dτ)(dτ/dt) = -2 p τ (1-τ²)^(p-1) / r
        psi_d_vals = np.where(in_supp,
            -2.0 * bump_p * tau * (1.0 - tau ** 2) ** (bump_p - 1) / radius,
            0.0)
        psi_vals[~in_supp] = 0.0
        psi_d_vals[~in_supp] = 0.0
        psi_list.append(psi_vals.astype(np.float32))
        psi_dot_list.append(psi_d_vals.astype(np.float32))
        # trapezoid weights for ∫ψ² dt
        w = np.ones_like(time_eval) * dt_eval
        w[0] *= 0.5
        w[-1] *= 0.5
        int_psi_sq_list.append(float(np.sum(w * psi_vals ** 2)))

    psi_arr = jnp.asarray(np.stack(psi_list))             # (K, T_eval)
    psi_dot_arr = jnp.asarray(np.stack(psi_dot_list))     # (K, T_eval)
    int_psi_sq_arr = jnp.asarray(np.array(int_psi_sq_list, dtype=np.float32))  # (K,)

    trap_w = np.ones_like(time_eval) * dt_eval
    trap_w[0] *= 0.5
    trap_w[-1] *= 0.5
    trap_w_jnp = jnp.asarray(trap_w.astype(np.float32))

    # Test-function design weights for fast matvecs:
    #   wpsi[k, t]     = trap_w[t] · ψ_k(t)
    #   wpsi_dot[k, t] = trap_w[t] · ψ_k'(t)
    wpsi = trap_w_jnp[None, :] * psi_arr                  # (K, T_eval)
    wpsi_dot = trap_w_jnp[None, :] * psi_dot_arr          # (K, T_eval)
    K_test = wpsi.shape[0]

    # ── GP conditional helpers (same as 04g / 04b) ───────────────────────
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

    # ── Broad GP-hyper priors (no MLE anchoring) ─────────────────────────
    # Once O is marginalised the optimisation landscape is much cleaner, so
    # we follow 04g and use broad priors anchored only to physical scales.
    T_span = float(t_train[-1] - t_train[0])
    broad_log_ell_loc = jnp.log(T_span / 20.0)
    broad_log_sig2_locs = jnp.array(
        [float(jnp.log(jnp.var(jnp.array(snapshots_comp[i])) + 1e-12))
         for i in range(num_modes)])

    inv_sigma_O2_default = 1.0 / (sigma_O ** 2)
    log_sigma_O2_default = 2.0 * jnp.log(sigma_O)

    # ── Closed-form per-mode marginal likelihood ─────────────────────────
    def _per_mode_evidence(A, y_i, prec_i, m, inv_sigma_O2, log_sigma_O2):
        """log p(y_i | θ) when y_i = A · O_i + ε, O_i ~ N(0, σ_O² I)."""
        Aw = A * prec_i[:, None]
        M_i = A.T @ Aw                                          # (m, m)
        Lambda_i = M_i + inv_sigma_O2 * jnp.eye(m)
        L_i = jnp.linalg.cholesky(Lambda_i)
        b_i = A.T @ (prec_i * y_i)                              # (m,)
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

    # ── Build (A, y, prec) per θ — concatenates derivative + weak-form ──
    def _build_AyP(ells, sig2s, nus, gamma2):
        """Per-θ pure-jax: build (A, y_per_mode, prec_per_mode).

        A             :  (T_eval + K, m)   stacked design matrix
        y_per_mode    :  (r, T_eval + K)   stacked observations per mode
        prec_per_mode :  (r, T_eval + K)   stacked precisions per mode
        """
        Xs, mu_zs, deriv_vars, mlls = _batch_gp_cond(ells, sig2s, nus, y_obs)
        # f(X) at eval times — (T_eval, m)
        f_X = rom.model._assemble_data_matrix(Xs, inputs=None)

        # Derivative-form block: A rows = f_X ; y = μ_z ; prec = 1/(deriv_var + γ²)
        prec_deriv = deriv_weight / (deriv_vars + gamma2 + 1e-4)  # (r, T_eval)

        # Weak-form block (WSINDy-style smooth bump test functions):
        #   A_weak[k, :]   = ∫ ψ_k(t) f(X(t)) dt           — (K, m)
        #   y_weak[i, k]   = -∫ ψ'_k(t) X_i(t) dt          — (r, K)
        #   prec_weak[k]   = 1 / (γ² ∫ ψ_k² dt)            — (K,)
        A_weak = wpsi @ f_X                                # (K, m)
        weak_obs = -(Xs @ wpsi_dot.T)                      # (r, K)
        prec_weak_per_k = weakform_weight / (gamma2 * int_psi_sq_arr + 1e-12)
        prec_weak = jnp.broadcast_to(prec_weak_per_k, (num_modes, K_test))

        A = jnp.concatenate([f_X, A_weak], axis=0)                 # (T_eval+K, m)
        y_per_mode = jnp.concatenate([mu_zs, weak_obs], axis=1)    # (r, T_eval+K)
        prec_per_mode = jnp.concatenate([prec_deriv, prec_weak], axis=1)
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
            log_p_i, _, _ = _per_mode_evidence(A, y_per_mode[i], prec_per_mode[i],
                                                m, inv_sO2, log_sO2)
            total_evidence = total_evidence + log_p_i
        numpyro.factor("marg_O_evidence", total_evidence)

    @jax.jit
    def posterior_O_fn(ells, sig2s, nus, gamma2, sigma_O_val):
        """Closed-form O posterior given θ: returns (μ_O, C_O, Xs, mu_zs)."""
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
            # Σ_O_i = Λ_i⁻¹; any factor C with C Cᵀ = Σ_O works for sampling.
            # L_i⁻ᵀ satisfies (L_i⁻ᵀ)(L_i⁻ᵀ)ᵀ = (L_i L_iᵀ)⁻¹ = Λ_i⁻¹.
            C_i = jax.scipy.linalg.solve_triangular(L_i, jnp.eye(m), lower=True).T
            return mu_i, C_i

        mu_all, C_all = [], []
        for i in range(num_modes):
            mi, Ci = _one(y_per_mode[i], prec_per_mode[i])
            mu_all.append(mi)
            C_all.append(Ci)
        return jnp.stack(mu_all), jnp.stack(C_all), Xs, mu_zs

    return model, posterior_O_fn, time_eval


# =============================================================================
# Run one regime
# =============================================================================
def run_experiment(schema, p=None):
    """Run one data regime. Returns results dict."""
    if p is None:
        p = MODEL_PARAMS
    noise = schema['NOISE_LEVEL']
    nsamp = schema['NUM_SAMPLES']
    neval = schema['NUM_EVAL_POINTS']
    nmodes = p['NUM_MODES']

    print(f"\n{'=' * 78}")
    print(f"  {schema['label']}  ({nsamp} samples, {noise:.0%} noise)"
          f"  —  marg-O × weak-form")
    print(f"{'=' * 78}")

    np.random.seed(p['SEED'])
    rng_key = random.PRNGKey(p['SEED'])

    fom, t_full, true_states, t_samp, snaps_samp = generate_trajectory(
        config, config.time_domain, TRAINING_SPAN, nsamp, noise)
    basis = Basis(num_vectors=nmodes)
    basis.fit(snaps_samp)
    snaps_comp = basis.compress(snaps_samp)
    true_comp = basis.compress(true_states)
    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
        model=JaxCompatibleModel(operators="cAH",
                                 solver=opinf.lstsq.L2Solver(regularizer=1e0)))
    rom.fit(states=snaps_samp)

    model, posterior_O_fn, time_eval = build_model(
        rom=rom, num_modes=nmodes,
        time_sampled=t_samp, snapshots_comp=snaps_comp,
        num_eval_points=neval, window_size=p['WINDOW_SIZE'],
        deriv_weight=p['DERIV_WEIGHT'],
        weakform_weight=p['WEAKFORM_WEIGHT'],
        mll_weight=p['MLL_WEIGHT'], sigma_O=p['SIGMA_O'],
        bump_p=p['BUMP_P'], num_test_funcs=p['NUM_TEST_FUNCS'],
        bump_radius_frac=p['BUMP_RADIUS_FRAC'])

    # No MLE warm-start: priors are already on physical scales and the
    # marginalised-O landscape is low-dimensional enough that SVI initialised
    # at the prior median converges reliably.
    guide = autoguide.AutoNormal(model, init_loc_fn=init_to_median)

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
    print(f"  SVI: {nsteps} steps   loss {losses[0]:.2f} → {losses[-1]:.2f}   "
          f"({time.time()-t0:.1f}s)")

    params = svi.get_params(state)
    rng_key, sk = random.split(rng_key)
    npost = p['NUM_POSTERIOR_SAMPLES']
    post = guide.sample_posterior(sk, params, sample_shape=(npost,),
                                   **model_kwargs)

    # ── Sample O from its closed-form conditional posterior per θ-sample ──
    rng_key, ok = random.split(rng_key)

    def _stack_theta(d):
        return (jnp.stack([d[f'lengthscale_{i}'] for i in range(nmodes)], axis=-1),
                jnp.stack([d[f'variance_{i}'] for i in range(nmodes)], axis=-1),
                jnp.stack([d[f'noise_{i}'] for i in range(nmodes)], axis=-1))

    ells_s, sig2s_s, nus_s = _stack_theta(post)
    sO_s = post.get('sigma_O', None)
    sO_default_jnp = jnp.asarray(p['SIGMA_O'])

    @jax.jit
    def _draw_O(ells, sig2s, nus, key, sO_val):
        mu_O, C_O, _, _ = posterior_O_fn(ells, sig2s, nus,
                                          p['GAMMA2'], sO_val)
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
    O_samples = np.stack(O_samples_list)
    O_means = np.stack(O_mean_list)
    print(f"  O posterior sampling: {time.time()-t_o:.1f}s")
    op_norms = np.linalg.norm(O_samples.reshape(npost, -1), axis=1)
    print(f"  ‖O‖: median={np.median(op_norms):.1f}  "
          f"min={op_norms.min():.1f}  max={op_norms.max():.1f}")

    # ── ROM predictions on the requested span ────────────────────────────
    samples_for_rom = {'O': jnp.array(O_samples)}
    for i in range(nmodes):
        # dummy X_i entries kept for downstream compatibility (unused)
        samples_for_rom[f'X_{i}'] = jnp.stack(post[f'lengthscale_{i}'])
    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    Os, Xs, rom_solves = generate_rom_predictions(
        samples=samples_for_rom, rom=rom, snapshots_compressed=snaps_comp,
        time_eval=t_pred, num_modes=nmodes, num_pulls=min(200, npost))
    n_stable = len(rom_solves)
    n_total = len(Os)
    stability_pct = n_stable / max(n_total, 1) * 100

    train_err = pred_err = float('inf')
    ci_cov = ci_w = float('nan')
    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        tm = t_pred <= TRAINING_SPAN[1]
        pm = t_pred > TRAINING_SPAN[1]
        ti = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)
        train_err = float(np.linalg.norm(rom_med[:, tm] - ta[:, tm]) /
                          np.linalg.norm(ta[:, tm]))
        pred_err = float(np.linalg.norm(rom_med[:, pm] - ta[:, pm]) /
                         np.linalg.norm(ta[:, pm]))
        q05 = np.percentile(rom_arr, 5, axis=0)
        q95 = np.percentile(rom_arr, 95, axis=0)
        ci_w = float(np.mean(q95 - q05))
        ci_cov = float(np.mean((ta >= q05) & (ta <= q95)))

    runtime = time.time() - t0
    print(f"\n  RESULTS — Marginalised-O + Weak-Form")
    print(f"    Stability:   {n_stable}/{n_total} ({stability_pct:.1f}%)")
    print(f"    Train error: {train_err:.2%}")
    print(f"    Pred error:  {pred_err:.2%}")
    print(f"    CI coverage: {ci_cov:.1%} (target 90%)")
    print(f"    CI width:    {ci_w:.4f}")
    print(f"    Runtime:     {runtime:.0f}s")

    # ── Persist results (schema matches 04_conditional_integral) ─────────
    out_dir = os.path.join(SCRIPT_DIR, 'results', 'comparison', schema['name'])
    os.makedirs(out_dir, exist_ok=True)
    rom_arr = np.array(rom_solves) if n_stable > 0 else np.empty((0, nmodes, len(t_pred)))
    np.savez(os.path.join(out_dir, '04_unified.npz'),
             rom_solves=rom_arr, t_pred=t_pred,
             train_error=train_err, pred_error=pred_err,
             stability_pct=stability_pct, ci_coverage=ci_cov,
             ci_width=ci_w, runtime=runtime,
             op_norm_median=float(np.median(op_norms)),
             losses=losses)

    return {
        'schema': schema,
        'train_error': train_err, 'pred_error': pred_err,
        'stability_pct': stability_pct,
        'n_stable': n_stable, 'n_total': n_total,
        'ci_coverage': ci_cov, 'ci_width': ci_w,
        'runtime': runtime, 'losses': losses,
        'O_samples': O_samples, 'O_means': O_means,
        'rom_solves': rom_solves,
        'snaps_comp': snaps_comp, 'snaps_noisy': snaps_samp,
        'true_comp': true_comp,
        't_full': t_full, 't_pred': t_pred, 't_samp': t_samp,
        'training_span': TRAINING_SPAN, 'num_modes': nmodes,
        'true_states': true_states, 'basis': basis,
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
    print("04_unified — Marginalised-O × Weak-Form Bayesian OpInf (Euler)")
    print("=" * 78)
    print(f"γ²={MODEL_PARAMS['GAMMA2']}  σ_O={MODEL_PARAMS['SIGMA_O']}  "
          f"bump_p={MODEL_PARAMS['BUMP_P']}  "
          f"lr={MODEL_PARAMS['LEARNING_RATE']}  steps={MODEL_PARAMS['NUM_STEPS']}")

    results = []
    for schema in schemas:
        r = run_experiment(schema)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*82}")
    print(f"SUMMARY — Marg-O × Weak-Form (Euler)")
    print(f"{'='*82}")
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

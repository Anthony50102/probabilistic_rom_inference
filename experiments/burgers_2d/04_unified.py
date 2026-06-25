"""
04_unified.py — Marginalised-O × Weak-Form Bayesian OpInf (Burgers-2D).
    2D Diffusion-Reaction (Burgers-style):  ∂u/∂t = κ∇²u − βu²

The operator O is analytically marginalised.  Because both derivative and
weak-form constraints are linear in O, the conditional posterior p(O | θ, data)
is Gaussian and SVI/NUTS only need to explore the GP hyperparameters θ.

For each ROM mode i, the likelihood uses two covariance-aware blocks:

    derivative rows (T_eval):
        μ_z_i(t_j)           ≈ [f(X) Oᵀ]_i(t_j)
        Σ_D,i                = Σ_z,i + γ² I

    weak-form rows (K):
        ∫ψ_k(t) μ_z_i(t) dt  ≈ ∫ψ_k(t) [f(X)Oᵀ]_i(t) dt
        Σ_W,i                = Ψ_w Σ_z,i Ψ_wᵀ + γ² diag(∫ψ_k² dt)

`_per_mode_evidence_dense` computes log p(y_i | θ) in closed form per mode.
`posterior_O_fn` returns (μ_O, C_O) where C_O Cᵀ_O = Σ_O, used for posterior
sampling of O for downstream ROM prediction.

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
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide, MCMC, NUTS
from numpyro.infer.initialization import init_to_value, init_to_median
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
from core.bayesian_opinf import fit_gp_hyperparameters_mle
from core.diagnostics import plot_trace
from core.plotting import plot_full_order_error
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

# ── Data regimes ─────────────────────────────────────────────────────────────
SCHEMAS = [
    {"name": "dense_medium_noise", "NUM_SAMPLES": 60, "NOISE_LEVEL": 0.03,
     "NUM_EVAL_POINTS": 200, "label": "Dense data, medium noise"},
]

# ── Shared model hyperparameters ─────────────────────────────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=3,
    GAMMA2=10.0,
    DERIV_WEIGHT=1.0,
    WEAKFORM_WEIGHT=1.0,
    MLL_WEIGHT=1.0,
    SIGMA_O=10.0,          # O ~ N(0, SIGMA_O² I)
    WINDOW_SIZE=10,
    # Weak-form (bump) test function settings
    BUMP_P=6,
    NUM_TEST_FUNCS=None,
    BUMP_RADIUS_FRAC=None,
    NUM_STEPS=8000,
    LEARNING_RATE=3e-3,
    NUM_POSTERIOR_SAMPLES=500,
    REGULARIZER=1.0,
    GP_PRIOR_SCALE=0.1,
    SEED=42,
)

TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 3.0)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# Model builder
# =============================================================================
def build_model(rom, num_modes, time_sampled, snapshots_comp,
                num_eval_points, window_size,
                deriv_weight, weakform_weight, mll_weight,
                sigma_O, bump_p, num_test_funcs, bump_radius_frac,
                mle_Ls=None, mle_Vs=None, mle_Ns=None, gp_prior_scale=0.1):
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

    # ── GP conditional helpers ───────────────────────────────────────────
    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def _single_gp_conditional(ell, sig2, nu, y_i):
        ell2 = ell ** 2
        jitter = jnp.maximum(1e-5, sig2 * 1e-3)
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
        # Full GP derivative posterior covariance: Σ_z = K_zz - K_zy K_yy^{-1} K_zy^T
        K_post_Z = K_zz - K_zy @ V
        K_post_Z = 0.5 * (K_post_Z + K_post_Z.T)
        # GP STATE posterior covariance (for integration-by-parts weak form):
        #   Σ_X = K_ee - K_et K_yy^{-1} K_et^T
        W = jax.scipy.linalg.cho_solve((L, True), K_et.T)
        K_post_X = K_ee - K_et @ W
        K_post_X = 0.5 * (K_post_X + K_post_X.T)
        mll = -0.5 * (jnp.dot(y_i, alpha)
                      + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
                      + n_train * jnp.log(2.0 * jnp.pi))
        return X_eval, mu_z, K_post_Z, K_post_X, mll

    _batch_gp_cond = jax.vmap(_single_gp_conditional)

    # WEAKFORM_MODE selects the weak-form representation:
    #   "ibp" (default): WSINDy integration-by-parts; w_i = -∫ψ'_k X_i dt uses the
    #       smooth GP STATE only (derivative moved onto the analytic ψ'), with
    #       Σ_W = Ψ' Σ_X Ψ'^T + slack. Noise-robust; matches the manuscript form.
    #   "deriv": w_i = ∫ψ_k μ_z dt — integrates the noisy GP derivative estimate,
    #       Σ_W = Ψ Σ_z Ψ^T + slack.
    WEAKFORM_MODE = os.environ.get("WEAKFORM_MODE", "ibp").lower()

    # ── GP-hyperparameter priors ─────────────────────────────────────────
    # If MLE values are supplied, anchor priors to them (tight); this is
    # essential for problems with very different data scales (e.g. burgers
    # σ²~10³).  Otherwise fall back to broad priors as in 04g.
    if mle_Ls is not None and mle_Vs is not None and mle_Ns is not None:
        log_ell_locs = jnp.array([float(jnp.log(L)) for L in mle_Ls])
        log_sig2_locs = jnp.array([float(jnp.log(V)) for V in mle_Vs])
        log_nu_locs = jnp.array([float(jnp.log(N)) for N in mle_Ns])
        prior_scales = (gp_prior_scale, gp_prior_scale, gp_prior_scale)
        log_ell_scales = jnp.full((num_modes,), prior_scales[0])
    else:
        T_span = float(t_train[-1] - t_train[0])
        # ── GP lengthscale prior: principled, parameter-free ──────────
        # LogNormal with median at the Nyquist limit Δt and 99th
        # percentile at the observation window T. ELL_PRIOR_MODE=legacy
        # recovers the ad-hoc T/20 prior.
        _dt_mean = T_span / max(int(n_train) - 1, 1)
        if os.environ.get("ELL_PRIOR_MODE", "principled") == "legacy":
            log_ell_locs = jnp.full((num_modes,), float(jnp.log(T_span / 20.0)))
            log_ell_scales = jnp.full((num_modes,), 1.0)
        else:
            log_ell_locs = jnp.full((num_modes,), float(jnp.log(_dt_mean)))
            log_ell_scales = jnp.full(
                (num_modes,),
                float(jnp.log(T_span / _dt_mean) / 2.3263))
        # Spectrum-anchored σ² and ν locations from POD energy
        log_sig2_locs = jnp.array(
            [float(jnp.log(jnp.var(jnp.array(snapshots_comp[i])) + 1e-12))
             for i in range(num_modes)])
        log_nu_locs = jnp.array(
            [float(jnp.log(0.01 * jnp.var(jnp.array(snapshots_comp[i])) + 1e-12))
             for i in range(num_modes)])
        prior_scales = (1.0, 1.0, 1.0)

    inv_sigma_O2_default = 1.0 / (sigma_O ** 2)
    log_sigma_O2_default = 2.0 * jnp.log(sigma_O)

    # Operator-block structure (cAH): const | linear(r) | quadratic(rest).
    n_const, n_lin = 1, num_modes
    m_total = 1 + num_modes + num_modes * (num_modes + 1) // 2
    n_quad = m_total - n_const - n_lin
    n_blocks = 3
    block_id = np.concatenate([np.zeros(n_const, int), np.ones(n_lin, int),
                               2 * np.ones(n_quad, int)])
    block_id_jnp = jnp.asarray(block_id)
    inv_prec_vec = jnp.full(m_total, 1.0 / (sigma_O ** 2))

    # ── Operator-prior modes ─────────────────────────────────────────────
    #   "fixed"      : O ~ N(0, σ_O² I)  (default; unchanged behaviour).
    #   "block_hier" : hierarchical/ARD per-block prior. Each block (const,
    #                  linear, quad) gets a LEARNED scale τ_b with a broad
    #                  hyperprior; O is still marginalised analytically given τ.
    #                  The marginal likelihood (Bayesian Occam) shrinks the
    #                  under-determined quadratic block automatically — no
    #                  hand-tuning. See euler/04_unified.py for the analysis.
    OP_PRIOR_MODE = os.environ.get("OP_PRIOR_MODE", "block_hier").lower()
    HIER_TAU0 = float(os.environ.get("HIER_TAU0", str(sigma_O)))
    HIER_TAU_SCALE = float(os.environ.get("HIER_TAU_SCALE", "3.0"))

    def _prior_prec_from_tau(tau_block):
        """Per-column prior precision + log|Σ_O| from per-block scales τ_block."""
        tau_col = tau_block[block_id_jnp]
        prior_prec = 1.0 / (tau_col ** 2 + 1e-12)
        log_prior_cov = 2.0 * jnp.sum(jnp.log(tau_col + 1e-12))
        return prior_prec, log_prior_cov

    # Lambda jitter to keep marginal-likelihood Cholesky SPD even when
    # the design matrix is ill-conditioned (e.g. quadratic terms with
    # large state magnitudes).
    LAMBDA_JITTER = 1e-6

    # ── Closed-form per-mode marginal likelihood ─────────────────────────

    def _dense_block_contrib(A_blk, y_blk, Sigma_blk):
        """Compute (M, b, quad_y, log_det_Sig, N) for one dense-covariance block."""
        N_blk = Sigma_blk.shape[0]
        L = jnp.linalg.cholesky(Sigma_blk + 1e-8 * jnp.eye(N_blk))
        Sinv_A = jax.scipy.linalg.cho_solve((L, True), A_blk)
        Sinv_y = jax.scipy.linalg.cho_solve((L, True), y_blk)
        M = A_blk.T @ Sinv_A
        b = A_blk.T @ Sinv_y
        log_det_Sig = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        quad_y = jnp.dot(y_blk, Sinv_y)
        return M, b, quad_y, log_det_Sig, N_blk

    def _per_mode_evidence_dense(A_D, y_D, Sigma_D,
                                  A_W, y_W, Sigma_W,
                                  prior_prec, log_prior_cov):
        """log p(y_i | θ) with two dense covariance blocks (derivative
        Σ_D = Σ_z + γ²I, weak-form Σ_W = Ψ Σ_z Ψ^T + slack) and a per-column
        operator-prior precision `prior_prec` (length m);
        `log_prior_cov` = log|Σ_O| = -Σ_j log(prior_prec_j)."""
        m = prior_prec.shape[0]
        M_D, b_D, qy_D, lds_D, N_D = _dense_block_contrib(A_D, y_D, Sigma_D)
        M_W, b_W, qy_W, lds_W, N_W = _dense_block_contrib(A_W, y_W, Sigma_W)

        M_i = M_D + M_W
        b_i = b_D + b_W
        M_i = 0.5 * (M_i + M_i.T)
        jitter = LAMBDA_JITTER * jnp.maximum(jnp.trace(M_i) / m, 1.0)
        Lambda_i = M_i + jnp.diag(prior_prec) + jitter * jnp.eye(m)
        L_i = jnp.linalg.cholesky(Lambda_i)
        mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_i)
        log_det_Lambda = 2.0 * jnp.sum(jnp.log(jnp.diag(L_i)))
        quad_y = qy_D + qy_W
        quad_mu = jnp.dot(mu_i, b_i)
        log_det_Sigma = lds_D + lds_W
        N_i = N_D + N_W
        log_p = -0.5 * ((quad_y - quad_mu)
                        + log_det_Sigma
                        + log_prior_cov
                        + log_det_Lambda
                        + N_i * jnp.log(2.0 * jnp.pi))
        return log_p, mu_i, L_i

    # ── Build design matrices + dense noise covariance per θ ─────────────
    def _build_AyP(ells, sig2s, nus, gamma2):
        """Per-θ build of design matrices + dense noise covariance.

        Both blocks use the full GP derivative posterior covariance Σ_z:
          - Derivative: Σ_D = (Σ_z + γ²I) / deriv_weight
          - Weak-form:  Σ_W = (Ψ Σ_z Ψ^T + γ² diag(∫ψ²)) / weakform_weight
        """
        Xs, mu_zs, K_posts_Z, K_posts_X, mlls = _batch_gp_cond(
            ells, sig2s, nus, y_obs)
        f_X = rom.model._assemble_data_matrix(Xs, inputs=None)

        n_eval = f_X.shape[0]
        I_eval = jnp.eye(n_eval)

        # Derivative block: diagonal Σ_D = (diag(Σ_z) + γ²I) / deriv_weight per mode.
        # Only the marginal derivative variances are used; the off-diagonal GP
        # correlations are dropped — they invert into a high-pass whitening filter
        # that overfits noise (see euler/04_unified.py for the analysis).
        Sigma_D = (jax.vmap(lambda K: jnp.diag(jnp.diag(K)))(K_posts_Z)
                   + gamma2 * I_eval[None, :, :]) / (deriv_weight + 1e-30)

        # Weak-form block (WEAKFORM_MODE): design matrix is the same in both modes.
        A_weak = wpsi @ f_X                                       # (K, m)
        diag_slack = gamma2 * jnp.diag(int_psi_sq_arr)            # (K, K)
        if WEAKFORM_MODE == "ibp":
            # Integration-by-parts: data uses the smooth GP state only.
            #   w_i = -∫ψ'_k X_i dt,  Σ_W = Ψ' Σ_X Ψ'^T + slack
            weak_obs = -(Xs @ wpsi_dot.T)                        # (r, K)
            def _sigma_w_one(K_post_X_i):
                return (wpsi_dot @ K_post_X_i @ wpsi_dot.T
                        + diag_slack) / (weakform_weight + 1e-30)
            Sigma_W = jax.vmap(_sigma_w_one)(K_posts_X)          # (r, K, K)
        else:
            # Derivative form: integrates the noisy GP derivative μ_z.
            weak_obs = mu_zs @ wpsi.T                            # (r, K)
            def _sigma_w_one(K_post_Z_i):
                return (wpsi @ K_post_Z_i @ wpsi.T
                        + diag_slack) / (weakform_weight + 1e-30)
            Sigma_W = jax.vmap(_sigma_w_one)(K_posts_Z)          # (r, K, K)
        # Diagonalize the weak-form covariance too, for consistency with the
        # derivative block (off-diagonals are negligible here — the weak block is
        # small and the derivative block dominates the fit).
        Sigma_W = jax.vmap(lambda S: jnp.diag(jnp.diag(S)))(Sigma_W)

        return (Xs, mu_zs, f_X, mu_zs, Sigma_D,
                A_weak, weak_obs, Sigma_W, mlls)

    HIERARCHICAL = bool(int(os.environ.get("HIER_SIGMA_O", "0")))

    def model(gamma2=10.0):
        ells = jnp.stack([
            numpyro.sample(f"lengthscale_{i}",
                           dist.LogNormal(log_ell_locs[i], log_ell_scales[i]))
            for i in range(num_modes)])
        sig2s = jnp.stack([
            numpyro.sample(f"variance_{i}",
                           dist.LogNormal(log_sig2_locs[i], prior_scales[1]))
            for i in range(num_modes)])
        nus = jnp.stack([
            numpyro.sample(f"noise_{i}",
                           dist.LogNormal(log_nu_locs[i], prior_scales[2]))
            for i in range(num_modes)])

        if OP_PRIOR_MODE == "block_hier":
            # Hierarchical/ARD per-block operator-prior scales (learned).
            log_tau = numpyro.sample(
                "log_tau_block",
                dist.Normal(jnp.log(HIER_TAU0) * jnp.ones(n_blocks),
                            HIER_TAU_SCALE))
            tau_block = jnp.exp(log_tau)
            prior_prec, log_prior_cov = _prior_prec_from_tau(tau_block)
        elif HIERARCHICAL:
            sO = numpyro.sample("sigma_O", dist.HalfCauchy(sigma_O))
            inv_sO2 = 1.0 / (sO ** 2 + 1e-12)
            prior_prec = inv_sO2 * jnp.ones(m_total)
            log_prior_cov = -m_total * jnp.log(inv_sO2)
        else:
            prior_prec = inv_prec_vec
            log_prior_cov = -jnp.sum(jnp.log(inv_prec_vec))

        (Xs, mu_zs,
         A_D, y_D, Sigma_D,
         A_W, y_W, Sigma_W,
         mlls) = _build_AyP(ells, sig2s, nus, gamma2)

        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", Xs[i])

        if mll_weight > 0:
            numpyro.factor("gp_mll", mll_weight * jnp.sum(mlls))

        total_evidence = 0.0
        for i in range(num_modes):
            log_p_i, _, _ = _per_mode_evidence_dense(
                A_D, y_D[i], Sigma_D[i],
                A_W, y_W[i], Sigma_W[i],
                prior_prec, log_prior_cov)
            total_evidence = total_evidence + log_p_i
        numpyro.factor("marg_O_evidence", total_evidence)

    @jax.jit
    def posterior_O_fn(ells, sig2s, nus, gamma2, sigma_O_val, tau_block=None):
        """Closed-form O posterior given θ (and optional per-block scales τ):
        returns (μ_O, C_O, Xs, mu_zs)."""
        inv_sO2 = 1.0 / (sigma_O_val ** 2 + 1e-12)
        if tau_block is None:
            prior_prec_vec = inv_sO2 * jnp.ones(m_total)
        else:
            prior_prec_vec, _ = _prior_prec_from_tau(tau_block)
        (Xs, mu_zs,
         A_D, y_D, Sigma_D,
         A_W, y_W, Sigma_W,
         _) = _build_AyP(ells, sig2s, nus, gamma2)
        m = A_D.shape[1]

        def _one_dense(y_Di, Sigma_Di, y_Wi, Sigma_Wi):
            M_D, b_D, _, _, _ = _dense_block_contrib(A_D, y_Di, Sigma_Di)
            M_W, b_W, _, _, _ = _dense_block_contrib(A_W, y_Wi, Sigma_Wi)
            M_i = M_D + M_W
            M_i = 0.5 * (M_i + M_i.T)
            jitter = LAMBDA_JITTER * jnp.maximum(jnp.trace(M_i) / m, 1.0)
            Lambda_i = M_i + jnp.diag(prior_prec_vec) + jitter * jnp.eye(m)
            L_i = jnp.linalg.cholesky(Lambda_i)
            mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_D + b_W)
            C_i = jax.scipy.linalg.solve_triangular(L_i, jnp.eye(m), lower=True).T
            return mu_i, C_i

        mu_all, C_all = [], []
        for i in range(num_modes):
            mi, Ci = _one_dense(y_D[i], Sigma_D[i], y_W[i], Sigma_W[i])
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
                                 solver=opinf.lstsq.L2Solver(regularizer=p['REGULARIZER'])))
    rom.fit(states=snaps_samp)

    model, posterior_O_fn, time_eval = build_model(
        rom=rom, num_modes=nmodes,
        time_sampled=t_samp, snapshots_comp=snaps_comp,
        num_eval_points=neval, window_size=p['WINDOW_SIZE'],
        deriv_weight=p['DERIV_WEIGHT'],
        weakform_weight=p['WEAKFORM_WEIGHT'],
        mll_weight=p['MLL_WEIGHT'], sigma_O=p['SIGMA_O'],
        bump_p=p['BUMP_P'], num_test_funcs=p['NUM_TEST_FUNCS'],
        bump_radius_frac=p['BUMP_RADIUS_FRAC'],
        mle_Ls=None, mle_Vs=None, mle_Ns=None,
        gp_prior_scale=p.get('GP_PRIOR_SCALE', 1.0))

    # ── Inference: SVI (default) or NUTS ─────────────────────────────────
    INFER = os.environ.get("INFER", "svi").lower()
    print(f"  INFER={INFER}")

    model_kwargs = dict(gamma2=p['GAMMA2'])
    rng_key, ik = random.split(rng_key)
    t0 = time.time()

    if INFER == "nuts":
        num_warmup = int(os.environ.get("NUTS_WARMUP", "500"))
        num_samples = int(os.environ.get("NUTS_SAMPLES", "500"))
        kernel = NUTS(model, init_strategy=init_to_median,
                      target_accept_prob=0.9)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=1, progress_bar=False)
        mcmc.run(ik, **model_kwargs)
        post = mcmc.get_samples()
        npost = num_samples
        losses = np.array([0.0])
        post = {k: np.asarray(v) for k, v in post.items()
                if not k.startswith("X_")}
        print(f"  NUTS: {num_warmup}+{num_samples} samples ({time.time()-t0:.1f}s)")
    else:
        # No MLE warm-start: spectrum-anchored priors are sufficient.
        guide = autoguide.AutoNormal(model, init_loc_fn=init_to_median)

        optimizer = ClippedAdam(step_size=p['LEARNING_RATE'])
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
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
    # Per-block operator scales τ (block_hier mode); None otherwise.
    tau_block_s = None
    if 'log_tau_block' in post:
        tau_block_s = jnp.exp(jnp.asarray(post['log_tau_block']))  # (npost, 3)
        _tb_mean = np.exp(np.asarray(post['log_tau_block']).mean(0))
        print(f"  learned τ_block (const,linear,quad): "
              f"{np.array2string(_tb_mean, precision=3)}")

    @jax.jit
    def _draw_O(ells, sig2s, nus, key, sO_val, tau_block):
        mu_O, C_O, _, _ = posterior_O_fn(ells, sig2s, nus,
                                          p['GAMMA2'], sO_val, tau_block)
        eps = jax.random.normal(key, shape=mu_O.shape)
        O = mu_O + jnp.einsum('ijk,ik->ij', C_O, eps)
        return O, mu_O

    keys = jax.random.split(ok, npost)
    t_o = time.time()
    O_samples_list, O_mean_list = [], []
    for s in range(npost):
        sO_val = sO_default_jnp if sO_s is None else sO_s[s]
        tb = None if tau_block_s is None else tau_block_s[s]
        O, mu_O = _draw_O(ells_s[s], sig2s_s[s], nus_s[s], keys[s], sO_val, tb)
        O_samples_list.append(np.array(O))
        O_mean_list.append(np.array(mu_O))
    O_samples = np.stack(O_samples_list)
    O_means = np.stack(O_mean_list)
    print(f"  O posterior sampling: {time.time()-t_o:.1f}s")
    op_norms = np.linalg.norm(O_samples.reshape(npost, -1), axis=1)
    print(f"  ‖O‖: median={np.median(op_norms):.1f}  "
          f"min={op_norms.min():.1f}  max={op_norms.max():.1f}")

    # ── ROM predictions on the requested span ────────────────────────────
    # Optional initial-condition uncertainty: draw each trajectory's start from
    # the GP state posterior at t0 instead of the single fixed data point. This
    # propagates IC uncertainty (which grows with the forecast horizon) into the
    # predictive band without touching the operator (stability-safe).
    IC_UNCERTAINTY = bool(int(os.environ.get("IC_UNCERTAINTY", "1")))
    IC_SCALE = float(os.environ.get("IC_SCALE", "1.0"))
    state0_samples = None
    if IC_UNCERTAINTY:
        t_tr = np.asarray(t_samp)
        n_tr = len(t_tr)
        ell_m = np.asarray(ells_s).mean(0)
        sig2_m = np.asarray(sig2s_s).mean(0)
        nu_m = np.asarray(nus_s).mean(0)
        sq_tt = (t_tr[:, None] - t_tr[None, :]) ** 2
        sq_0t = (t_tr[0] - t_tr) ** 2
        sig_ic = np.zeros(nmodes)
        for i in range(nmodes):
            ell2 = ell_m[i] ** 2
            K_tt = (sig2_m[i] * np.exp(-sq_tt / (2 * ell2))
                    + (nu_m[i] + max(1e-5, sig2_m[i] * 1e-4)) * np.eye(n_tr))
            k0 = sig2_m[i] * np.exp(-sq_0t / (2 * ell2))
            var = sig2_m[i] - k0 @ np.linalg.solve(K_tt, k0)
            sig_ic[i] = np.sqrt(max(float(var), 0.0))
        rng_ic = np.random.default_rng(p['SEED'])
        eps_ic = rng_ic.standard_normal((npost, nmodes))
        state0_samples = (snaps_comp[:, 0][None, :]
                          + IC_SCALE * sig_ic[None, :] * eps_ic)
        print(f"  IC uncertainty ON: σ_ic={np.array2string(sig_ic, precision=4)}"
              f"  scale={IC_SCALE}")

    samples_for_rom = {'O': jnp.array(O_samples)}
    for i in range(nmodes):
        # dummy X_i entries kept for downstream compatibility (unused)
        samples_for_rom[f'X_{i}'] = jnp.stack(post[f'lengthscale_{i}'])
    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    Os, Xs, rom_solves = generate_rom_predictions(
        samples=samples_for_rom, rom=rom, snapshots_compressed=snaps_comp,
        time_eval=t_pred, num_modes=nmodes, num_pulls=min(200, npost),
        state0_samples=state0_samples)
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

    # ── Persist results (schema matches 04_unified) ─────────
    out_dir = os.path.join(SCRIPT_DIR, 'results', 'comparison', schema['name'])
    os.makedirs(out_dir, exist_ok=True)
    rom_arr = np.array(rom_solves) if n_stable > 0 else np.empty((0, nmodes, len(t_pred)))
    np.savez(os.path.join(out_dir, f'04_unified{os.environ.get("OUTPUT_SUFFIX","")}.npz'),
             rom_solves=rom_arr, t_pred=t_pred,
             train_error=train_err, pred_error=pred_err,
             stability_pct=stability_pct, ci_coverage=ci_cov,
             ci_width=ci_w, runtime=runtime,
             op_norm_median=float(np.median(op_norms)),
             losses=losses,
             snaps_comp=snaps_comp, true_comp=true_comp,
             t_full=t_full, t_samp=t_samp,
             training_span=np.array(TRAINING_SPAN),
             num_modes=nmodes,
             O_samples=O_samples,
             basis_entries=np.asarray(basis.entries),
             true_states=true_states)

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
    print("04_unified — Marginalised-O × Weak-Form Bayesian OpInf (Burgers-2D)")
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
    print(f"SUMMARY — Marg-O × Weak-Form (Burgers-2D)")
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

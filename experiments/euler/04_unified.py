"""
04_unified.py — Marginalised-O × Weak-Form Bayesian OpInf (Euler).

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
from numpyro.infer.initialization import init_to_median, init_to_value
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
                mll_weight,
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
    _ntf_env = os.environ.get("NUM_TEST_FUNCS")
    if _ntf_env is not None:
        num_test_funcs = int(_ntf_env)
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
        # Full GP derivative posterior covariance:
        #   Σ_z = K_zz - K_zy K_yy^{-1} K_zy^T
        K_post_Z = K_zz - K_zy @ V                                # (T_eval, T_eval)
        K_post_Z = 0.5 * (K_post_Z + K_post_Z.T)                  # symmetrise
        # GP STATE posterior covariance (for integration-by-parts weak form):
        #   Σ_X = K_ee - K_et K_yy^{-1} K_et^T
        W = jax.scipy.linalg.cho_solve((L, True), K_et.T)
        K_post_X = K_ee - K_et @ W                                # (T_eval, T_eval)
        K_post_X = 0.5 * (K_post_X + K_post_X.T)                  # symmetrise
        mll = -0.5 * (jnp.dot(y_i, alpha)
                      + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
                      + n_train * jnp.log(2.0 * jnp.pi))
        return X_eval, mu_z, K_post_Z, K_post_X, mll

    _batch_gp_cond = jax.vmap(_single_gp_conditional)

    # ── Broad GP-hyper priors (no MLE anchoring) ─────────────────────────
    # Once O is marginalised the optimisation landscape is much cleaner, so
    # we follow 04g and use broad priors anchored only to physical scales.
    T_span = float(t_train[-1] - t_train[0])
    # ── GP lengthscale prior: principled, parameter-free derivation ──────
    # Identifiability pins ℓ to [Δt, T]:
    #   • Δt is the Nyquist limit (below which sampling cannot resolve),
    #   • T is the observation window (above which the GP is
    #     indistinguishable from a constant).
    # We use a LogNormal with median at Δt and 99th percentile at T:
    #     ℓ ~ LogNormal(log Δt, σ²),  σ = log(T/Δt) / z_{0.99}
    # with z_{0.99} ≈ 2.326 the standard-normal 99% quantile. No knobs.
    # Set ELL_PRIOR_MODE=legacy to recover the previous ad-hoc T/20 prior.
    _dt_mean = T_span / max(int(n_train) - 1, 1)
    if os.environ.get("ELL_PRIOR_MODE", "principled") == "legacy":
        broad_log_ell_loc = float(jnp.log(T_span / 20.0))
        broad_log_ell_scale = 1.0
    else:
        broad_log_ell_loc = float(jnp.log(_dt_mean))
        broad_log_ell_scale = float(jnp.log(T_span / _dt_mean) / 2.3263)
    broad_log_sig2_locs = jnp.array(
        [float(jnp.log(jnp.var(jnp.array(snapshots_comp[i])) + 1e-12))
         for i in range(num_modes)])

    # Optional fully uninformative priors (for prior-sensitivity tests).
    # PRIOR_MODE: "informative" (default), "wide" (data-scale anchored but
    # ~5x wider), or "uninformative" (no data anchoring).
    # Per-mode noise prior anchor: 1% of mode energy.  Like sig2, this uses
    # the POD singular value spectrum (a deterministic property of the basis)
    # to set per-mode scale — not empirical Bayes on the noise itself.
    broad_log_nu_locs = jnp.array(
        [float(jnp.log(0.01 * jnp.var(jnp.array(snapshots_comp[i])) + 1e-12))
         for i in range(num_modes)])

    PRIOR_MODE = os.environ.get("PRIOR_MODE", "informative")
    if PRIOR_MODE == "uninformative":
        ell_loc, ell_scale = 0.0, 3.0
        sig2_loc, sig2_scale = jnp.zeros(num_modes), 3.0
        nu_loc, nu_scale = jnp.zeros(num_modes), 3.0
    elif PRIOR_MODE == "weakly_physical":
        ell_loc, ell_scale = jnp.log(T_span / 10.0), 2.0
        sig2_loc, sig2_scale = jnp.zeros(num_modes), 3.0
        nu_loc, nu_scale = -8.0 * jnp.ones(num_modes), 1.5
    elif PRIOR_MODE == "wide":
        # Keep data-derived scale hints but widen dramatically
        ell_loc, ell_scale = broad_log_ell_loc, 2.5
        sig2_loc, sig2_scale = broad_log_sig2_locs, 2.0
        nu_loc, nu_scale = -8.0, 2.5
    else:  # "informative" (default)
        # Spectrum-anchored priors: every location comes from either the
        # observation window T (chosen) or the POD singular-value spectrum
        # (a deterministic property of the basis). No data values feed in.
        ell_loc, ell_scale = broad_log_ell_loc, broad_log_ell_scale
        sig2_loc, sig2_scale = broad_log_sig2_locs, 1.0
        nu_loc, nu_scale = broad_log_nu_locs, 1.0

    inv_sigma_O2_default = 1.0 / (sigma_O ** 2)
    log_sigma_O2_default = 2.0 * jnp.log(sigma_O)

    # WEAK_ONLY=1 drops the derivative residual entirely; only the weak-form
    # block constrains the operator. Useful for testing whether R^W alone
    # is sufficient (no need for R^D as well).
    WEAK_ONLY = bool(int(os.environ.get("WEAK_ONLY", "0")))

    # WEAKFORM_MODE selects the weak-form representation:
    #   "deriv" (default): w_i = ∫ψ_k μ_z dt          (uses noisy GP derivative)
    #                      Σ_W = Ψ Σ_z Ψ^T + slack
    #   "ibp": WSINDy integration-by-parts; w_i = -∫ψ'_k X_i dt (smooth GP state
    #          only, derivative moved onto the test function)
    #                      Σ_W = Ψ' Σ_X Ψ'^T + slack
    # The "ibp" form avoids differentiating the noisy data and is the
    # noise-robust constraint the weak form is meant to provide.
    WEAKFORM_MODE = os.environ.get("WEAKFORM_MODE", "deriv").lower()

    # DERIV_COV selects the derivative-block noise model:
    #   "diag" (default): Σ_D = diag(deriv_var + γ²) — independent per-time-point
    #       precision 1/(deriv_var_ij + γ²). This is the original (pre full-cov)
    #       form and is markedly more accurate + better-calibrated here.
    #   "full": Σ_D = Σ_z + γ²I — full dense GP derivative posterior covariance.
    #       Off-diagonal GP correlations de-regularise the operator fit, causing
    #       phase drift + under-coverage (regression introduced in f5dad02).
    DERIV_COV = os.environ.get("DERIV_COV", "diag").lower()
    deriv_is_diag = (DERIV_COV != "full")

    # WEAKFORM_COV selects the weak-form-block noise model:
    #   "full" (default): Σ_W = Ψ Σ_z Ψ^T + slack — full K×K covariance. The
    #       test-function projection makes this small + well-conditioned, so the
    #       off-diagonals are benign (no whitening pathology).
    #   "diag": keep only diag(Σ_W) — for consistency with the diagonal
    #       derivative block / to test whether the weak-form off-diagonals matter.
    WEAKFORM_COV = os.environ.get("WEAKFORM_COV", "diag").lower()
    weakform_is_diag = (WEAKFORM_COV == "diag")

    # ── Closed-form per-mode marginal likelihood ─────────────────────────

    def _diag_block_contrib(A_blk, y_blk, prec_vec):
        """(M, b, quad_y, log_det_Sig, N) for a diagonal-covariance block,
        given the precision vector prec_vec = 1/diag(Σ)."""
        Aw = A_blk * prec_vec[:, None]
        M = A_blk.T @ Aw
        b = A_blk.T @ (prec_vec * y_blk)
        quad_y = jnp.sum(prec_vec * y_blk ** 2)
        log_det_Sig = -jnp.sum(jnp.log(prec_vec + 1e-30))
        N_blk = prec_vec.shape[0]
        return M, b, quad_y, log_det_Sig, N_blk

    def _deriv_block_contrib(A_D, y_D, deriv_blk):
        """Derivative-block (M,b,quad_y,log_det,N): diagonal (precision vector)
        or dense (covariance matrix) depending on DERIV_COV."""
        if deriv_is_diag:
            return _diag_block_contrib(A_D, y_D, deriv_blk)
        return _dense_block_contrib(A_D, y_D, deriv_blk)

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
                                  m, inv_sigma_O2, log_sigma_O2):
        """log p(y_i | θ) with derivative block (diag or dense Σ_D) and dense
        weak-form Σ_W = Ψ Σ_z Ψ^T + slack."""
        M_D, b_D, qy_D, lds_D, N_D = _deriv_block_contrib(A_D, y_D, Sigma_D)
        M_W, b_W, qy_W, lds_W, N_W = _dense_block_contrib(A_W, y_W, Sigma_W)

        M_i = M_D + M_W
        b_i = b_D + b_W
        Lambda_i = M_i + inv_sigma_O2 * jnp.eye(m)
        L_i = jnp.linalg.cholesky(Lambda_i)
        mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_i)
        log_det_Lambda = 2.0 * jnp.sum(jnp.log(jnp.diag(L_i)))
        quad_y = qy_D + qy_W
        quad_mu = jnp.dot(mu_i, b_i)
        log_det_Sigma = lds_D + lds_W
        N_i = N_D + N_W
        log_p = -0.5 * ((quad_y - quad_mu)
                        + log_det_Sigma
                        + m * log_sigma_O2
                        + log_det_Lambda
                        + N_i * jnp.log(2.0 * jnp.pi))
        return log_p, mu_i, L_i

    def _per_mode_evidence_weak_only(A_W, y_W, Sigma_W, m, inv_sigma_O2, log_sigma_O2):
        """log p(y_i | θ) using only the weak-form block (no derivative term).
        y_W ~ N(A_W · O_i, Σ_W),   O_i ~ N(0, σ_O² I)."""
        M_W, b_W, qy_W, lds_W, N_W = _dense_block_contrib(A_W, y_W, Sigma_W)
        Lambda_i = M_W + inv_sigma_O2 * jnp.eye(m)
        L_i = jnp.linalg.cholesky(Lambda_i)
        mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_W)
        log_det_Lambda = 2.0 * jnp.sum(jnp.log(jnp.diag(L_i)))
        quad_mu = jnp.dot(mu_i, b_W)
        log_p = -0.5 * ((qy_W - quad_mu)
                        + lds_W
                        + m * log_sigma_O2
                        + log_det_Lambda
                        + N_W * jnp.log(2.0 * jnp.pi))
        return log_p, mu_i, L_i

    # ── Build design matrices + noise covariance per θ ───────────────────
    def _build_AyP(ells, sig2s, nus, gamma2):
        """Per-θ build of design matrices + noise covariance.

        Derivative block (DERIV_COV):
          - "diag" (default): Σ_D = diag(deriv_var + γ²) — returned as a
            per-mode precision vector (r, n_eval).
          - "full": Σ_D = Σ_z + γ²I — per-mode dense covariance (r, n, n).
        Weak-form block depends on WEAKFORM_MODE:
          - "deriv": data ∫ψ μ_z dt,  Σ_W = Ψ Σ_z Ψ^T + slack
          - "ibp":   data -∫ψ' X dt,  Σ_W = Ψ' Σ_X Ψ'^T + slack (state-based,
                     noise-robust WSINDy form)
        """
        Xs, mu_zs, K_posts_Z, K_posts_X, mlls = _batch_gp_cond(
            ells, sig2s, nus, y_obs)
        f_X = rom.model._assemble_data_matrix(Xs, inputs=None)

        n_eval = f_X.shape[0]
        I_eval = jnp.eye(n_eval)

        # Derivative block
        if deriv_is_diag:
            # Σ_D diagonal → return precision 1/(deriv_var + γ²) per mode.
            deriv_var = jnp.maximum(jax.vmap(jnp.diagonal)(K_posts_Z), 0.0)
            Sigma_D = 1.0 / (deriv_var + gamma2 + 1e-4)          # (r, n_eval)
        else:
            Sigma_D = K_posts_Z + gamma2 * I_eval[None, :, :]    # (r, n, n)

        # Weak-form block design matrix is the same in both modes:
        #   A_weak[k, :] = ∫ψ_k(t) f(X(t)) dt
        A_weak = wpsi @ f_X                                       # (K, m)
        diag_slack = gamma2 * jnp.diag(int_psi_sq_arr)            # (K, K)

        if WEAKFORM_MODE == "ibp":
            # WSINDy integration-by-parts: data uses only the smooth GP state X,
            # derivative moved onto the analytic test-function derivative ψ'.
            #   w_i = -∫ψ'_k(t) X_i(t) dt
            #   Σ_W = Ψ' Σ_X Ψ'^T + γ² diag(∫ψ²)
            weak_obs = -(Xs @ wpsi_dot.T)                        # (r, K)
            def _sigma_w_one(K_post_X_i):
                return wpsi_dot @ K_post_X_i @ wpsi_dot.T + diag_slack
            Sigma_W = jax.vmap(_sigma_w_one)(K_posts_X)          # (r, K, K)
        else:
            # Derivative form: data integrates the noisy GP derivative μ_z.
            #   w_i = ∫ψ_k(t) μ_z_i(t) dt
            #   Σ_W = Ψ Σ_z Ψ^T + γ² diag(∫ψ²)
            weak_obs = mu_zs @ wpsi.T                            # (r, K)
            def _sigma_w_one(K_post_Z_i):
                return wpsi @ K_post_Z_i @ wpsi.T + diag_slack
            Sigma_W = jax.vmap(_sigma_w_one)(K_posts_Z)          # (r, K, K)

        if weakform_is_diag:
            # Keep only the marginal variances of the weak-form covariance.
            Sigma_W = jax.vmap(lambda S: jnp.diag(jnp.diag(S)))(Sigma_W)

        return (Xs, mu_zs, f_X, mu_zs, Sigma_D,
                A_weak, weak_obs, Sigma_W, mlls)

    HIERARCHICAL = bool(int(os.environ.get("HIER_SIGMA_O", "0")))

    def model(gamma2=10.0):
        ells = jnp.stack([
            numpyro.sample(f"lengthscale_{i}",
                           dist.LogNormal(ell_loc, ell_scale))
            for i in range(num_modes)])
        sig2s = jnp.stack([
            numpyro.sample(f"variance_{i}",
                           dist.LogNormal(sig2_loc[i], sig2_scale))
            for i in range(num_modes)])
        nus = jnp.stack([
            numpyro.sample(f"noise_{i}",
                           dist.LogNormal(nu_loc[i], nu_scale))
            for i in range(num_modes)])

        if HIERARCHICAL:
            sO = numpyro.sample("sigma_O", dist.HalfCauchy(sigma_O))
            inv_sO2 = 1.0 / (sO ** 2 + 1e-12)
            log_sO2 = 2.0 * jnp.log(sO + 1e-12)
        else:
            inv_sO2 = inv_sigma_O2_default
            log_sO2 = log_sigma_O2_default

        (Xs, mu_zs,
         A_D, y_D, Sigma_D,
         A_W, y_W, Sigma_W,
         mlls) = _build_AyP(ells, sig2s, nus, gamma2)

        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", Xs[i])

        if mll_weight > 0:
            numpyro.factor("gp_mll", mll_weight * jnp.sum(mlls))

        m = A_D.shape[1]
        total_evidence = 0.0
        if WEAK_ONLY:
            for i in range(num_modes):
                log_p_i, _, _ = _per_mode_evidence_weak_only(
                    A_W, y_W[i], Sigma_W[i],
                    m, inv_sO2, log_sO2)
                total_evidence = total_evidence + log_p_i
        else:
            for i in range(num_modes):
                log_p_i, _, _ = _per_mode_evidence_dense(
                    A_D, y_D[i], Sigma_D[i],
                    A_W, y_W[i], Sigma_W[i],
                    m, inv_sO2, log_sO2)
                total_evidence = total_evidence + log_p_i
        numpyro.factor("marg_O_evidence", total_evidence)

    @jax.jit
    def posterior_O_fn(ells, sig2s, nus, gamma2, sigma_O_val):
        """Closed-form O posterior given θ: returns (μ_O, C_O, Xs, mu_zs)."""
        inv_sO2 = 1.0 / (sigma_O_val ** 2 + 1e-12)
        (Xs, mu_zs,
         A_D, y_D, Sigma_D,
         A_W, y_W, Sigma_W,
         _) = _build_AyP(ells, sig2s, nus, gamma2)
        m = A_D.shape[1]

        def _one_dense(y_Di, Sigma_Di, y_Wi, Sigma_Wi):
            M_D, b_D, _, _, _ = _deriv_block_contrib(A_D, y_Di, Sigma_Di)
            M_W, b_W, _, _, _ = _dense_block_contrib(A_W, y_Wi, Sigma_Wi)
            Lambda_i = M_D + M_W + inv_sO2 * jnp.eye(m)
            L_i = jnp.linalg.cholesky(Lambda_i)
            mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_D + b_W)
            C_i = jax.scipy.linalg.solve_triangular(L_i, jnp.eye(m), lower=True).T
            return mu_i, C_i

        def _one_weak(y_Wi, Sigma_Wi):
            M_W, b_W, _, _, _ = _dense_block_contrib(A_W, y_Wi, Sigma_Wi)
            Lambda_i = M_W + inv_sO2 * jnp.eye(m)
            L_i = jnp.linalg.cholesky(Lambda_i)
            mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_W)
            C_i = jax.scipy.linalg.solve_triangular(L_i, jnp.eye(m), lower=True).T
            return mu_i, C_i

        mu_all, C_all = [], []
        for i in range(num_modes):
            if WEAK_ONLY:
                mi, Ci = _one_weak(y_W[i], Sigma_W[i])
            else:
                mi, Ci = _one_dense(y_D[i], Sigma_D[i], y_W[i], Sigma_W[i])
            mu_all.append(mi)
            C_all.append(Ci)
        return jnp.stack(mu_all), jnp.stack(C_all), Xs, mu_zs

    return model, posterior_O_fn, time_eval, {
        'ell': float(jnp.exp(broad_log_ell_loc)),
        'sig2': [float(jnp.exp(broad_log_sig2_locs[i])) for i in range(num_modes)],
        'nu': [float(jnp.exp(broad_log_nu_locs[i])) for i in range(num_modes)],
    }


# =============================================================================
# Run one regime
# =============================================================================
def run_experiment(schema, p=None):
    """Run one data regime. Returns results dict."""
    if p is None:
        p = MODEL_PARAMS
    noise = schema['NOISE_LEVEL']
    nsamp = schema['NUM_SAMPLES']
    neval = int(os.environ.get("NUM_EVAL_POINTS", schema['NUM_EVAL_POINTS']))
    nmodes = int(os.environ.get("NUM_MODES", p['NUM_MODES']))
    gamma2_val = float(os.environ.get("GAMMA2", p['GAMMA2']))
    sigma_O_val = float(os.environ.get("SIGMA_O", p['SIGMA_O']))

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

    model, posterior_O_fn, time_eval, init_phys = build_model(
        rom=rom, num_modes=nmodes,
        time_sampled=t_samp, snapshots_comp=snaps_comp,
        num_eval_points=neval, window_size=p['WINDOW_SIZE'],
        mll_weight=p['MLL_WEIGHT'], sigma_O=sigma_O_val,
        bump_p=p['BUMP_P'], num_test_funcs=p['NUM_TEST_FUNCS'],
        bump_radius_frac=p['BUMP_RADIUS_FRAC'])

    # ── Inference: SVI (default) or NUTS, with selectable init ───────────
    INIT_MODE = os.environ.get("INIT_MODE", "median")
    INFER = os.environ.get("INFER", "svi").lower()
    print(f"  PRIOR_MODE={os.environ.get('PRIOR_MODE','informative')}  "
          f"INIT_MODE={INIT_MODE}  INFER={INFER}  γ²={gamma2_val:g}  σ_O={sigma_O_val:g}  "
          f"weakform={os.environ.get('WEAKFORM_MODE','deriv')}  "
          f"deriv_cov={os.environ.get('DERIV_COV','diag')}  "
          f"weakform_cov={os.environ.get('WEAKFORM_COV','diag')}")

    if INIT_MODE == "physical":
        init_values = {}
        for i in range(nmodes):
            init_values[f"lengthscale_{i}"] = init_phys['ell']
            init_values[f"variance_{i}"]    = init_phys['sig2'][i]
            init_values[f"noise_{i}"]       = init_phys['nu'][i]
        init_loc_fn = init_to_value(values=init_values)
    else:
        init_loc_fn = init_to_median

    model_kwargs = dict(gamma2=gamma2_val)
    rng_key, ik = random.split(rng_key)
    t0 = time.time()

    if INFER == "nuts":
        # MCMC: only ~3*nmodes latent vars (O is marginalised) — fast.
        num_warmup = int(os.environ.get("NUTS_WARMUP", "500"))
        num_samples = int(os.environ.get("NUTS_SAMPLES", "500"))
        kernel = NUTS(model, init_strategy=init_loc_fn,
                      target_accept_prob=0.9)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=1, progress_bar=True)
        mcmc.run(ik, **model_kwargs)
        post = mcmc.get_samples()
        npost = num_samples
        losses = np.array([0.0])  # placeholder
        # Trim deterministic X_i — they hold (n_samples, T_eval) but we don't
        # need them for downstream O sampling.
        post = {k: np.asarray(v) for k, v in post.items()
                if not k.startswith("X_")}
        print(f"  NUTS: {num_warmup}+{num_samples} samples ({time.time()-t0:.1f}s)")
    else:
        GUIDE = os.environ.get("GUIDE", "normal").lower()
        init_scale = float(os.environ.get("INIT_SCALE", "0.1"))
        if GUIDE == "mvn":
            guide = autoguide.AutoMultivariateNormal(
                model, init_loc_fn=init_loc_fn, init_scale=init_scale)
        elif GUIDE == "lowrank":
            guide = autoguide.AutoLowRankMultivariateNormal(
                model, init_loc_fn=init_loc_fn, init_scale=init_scale, rank=5)
        elif GUIDE == "dais":
            K_dais = int(os.environ.get("DAIS_K", "8"))
            eta_max = float(os.environ.get("DAIS_ETA_MAX", "0.1"))
            eta_init = float(os.environ.get("DAIS_ETA_INIT", "0.01"))
            guide = autoguide.AutoDAIS(
                model, K=K_dais, eta_max=eta_max, eta_init=eta_init,
                init_loc_fn=init_loc_fn, init_scale=init_scale)
        else:
            guide = autoguide.AutoNormal(
                model, init_loc_fn=init_loc_fn, init_scale=init_scale)
        print(f"  GUIDE={GUIDE}  init_scale={init_scale}"
              + (f"  DAIS_K={os.environ.get('DAIS_K','8')}" if GUIDE == 'dais' else ''))
        lr = float(os.environ.get("LR", p['LEARNING_RATE']))
        nsteps = int(os.environ.get("NUM_STEPS", p['NUM_STEPS']))
        optimizer = ClippedAdam(step_size=lr)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        state = svi.init(ik, **model_kwargs)

        @jax.jit
        def _step(s, _):
            return svi.update(s, **model_kwargs)

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
    sO_default_jnp = jnp.asarray(sigma_O_val)

    @jax.jit
    def _draw_O(ells, sig2s, nus, key, sO_val):
        mu_O, C_O, _, _ = posterior_O_fn(ells, sig2s, nus,
                                          gamma2_val, sO_val)
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

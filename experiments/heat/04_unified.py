"""
04_unified.py — Marginalised-O × Weak-Form Bayesian OpInf (Heat / multi-IC).

Adapts the single-IC burgers_2d/04_unified.py to the multi-IC, input-dependent
heat equation.  The operator O is shared across all ICs and marginalised
analytically; each IC contributes derivative-form rows + WSINDy-style
weak-form rows that are linear in O.  Stacking across ICs gives one big linear
system per ROM mode, solved in closed form via a single m×m Cholesky.

Both blocks use the full GP derivative posterior covariance:

    derivative rows:  Σ_D,i = Σ_z,i + γ² I
    weak-form rows:   Σ_W,i = Ψ_w Σ_z,i Ψ_wᵀ + γ² diag(∫ψ_k² dt)

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
from numpyro.infer import SVI, Trace_ELBO, autoguide, MCMC, NUTS
from numpyro.infer.initialization import init_to_value, init_to_median
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
from heat_rom import generate_rom_solves
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
    SIGMA_O=0.5,
    WINDOW_SIZE=20,
    BUMP_P=6,
    NUM_TEST_FUNCS=None,
    BUMP_RADIUS_FRAC=None,
    NUM_STEPS=10000,
    LEARNING_RATE=3e-3,
    NUM_POSTERIOR_SAMPLES=500,
    REGULARIZER=1.0,
    GP_PRIOR_SCALE=1.0,
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
    O_prior=None,
    sigma_O_vec=None,
):
    """Build multi-IC marginalised-O + weak-form Bayesian model.

    If ``O_prior`` is provided (shape (num_modes, m)), the operator prior is
    O ~ N(O_prior, sigma_O^2 I) instead of zero-mean. Centering on a
    pre-computed stable OpInf solution prevents the marginal likelihood from
    selecting trajectory-divergent operators.

    If ``sigma_O_vec`` is provided (length m), it overrides the scalar
    ``sigma_O`` with a per-entry prior std — useful for tightening on
    structural-zero blocks (e.g. quadratic terms in a linear PDE).
    """

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

        T_span = float(t_train[-1] - t_train[0])
        snaps_ic = np.asarray(all_snapshots_comp[ic])
        var_modes = np.array([float(np.var(snaps_ic[j]) + 1e-12)
                              for j in range(num_modes)])
        # ── GP lengthscale prior: principled, parameter-free ─────────────
        # LogNormal with median at the Nyquist limit Δt and 99th
        # percentile at the observation window T (the identifiable range
        # of ℓ). ELL_PRIOR_MODE=legacy recovers the ad-hoc T/20 prior.
        _dt_mean = T_span / max(int(n_train) - 1, 1)
        if os.environ.get("ELL_PRIOR_MODE", "principled") == "legacy":
            _ell_loc = float(np.log(T_span / 20.0))
            _ell_scale = None  # use shared gp_prior_scale
        else:
            _ell_loc = float(np.log(_dt_mean))
            _ell_scale = float(np.log(T_span / _dt_mean) / 2.3263)
        mle_log_ells = jnp.full((num_modes,), _ell_loc)
        mle_log_ell_scales = (
            None if _ell_scale is None
            else jnp.full((num_modes,), float(_ell_scale)))
        mle_log_sig2s = jnp.array([float(np.log(v)) for v in var_modes])
        # Spectrum-anchored noise prior: 1% of mode energy
        mle_log_nus = jnp.array([float(np.log(0.01 * v)) for v in var_modes])

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
            mle_log_ell_scales=mle_log_ell_scales,
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
        # Full GP derivative posterior covariance: Σ_z = K_zz - K_zy K_yy^{-1} K_zy^T
        K_post_Z = K_zz - K_zy @ V
        K_post_Z = 0.5 * (K_post_Z + K_post_Z.T)
        # GP STATE posterior covariance (for integration-by-parts weak form):
        #   Σ_X = K_ee - K_et K_yy^{-1} K_et^T
        W = jax.scipy.linalg.cho_solve((L, True), K_et.T)
        K_post_X = K_ee - K_et @ W
        K_post_X = 0.5 * (K_post_X + K_post_X.T)
        mll = -0.5 * (jnp.dot(y_i, alpha) +
                      2.0 * jnp.sum(jnp.log(jnp.diag(L))) +
                      d['n_train'] * jnp.log(2.0 * jnp.pi))
        return X_eval, mu_z, K_post_Z, K_post_X, mll

    # WEAKFORM_MODE selects the weak-form representation:
    #   "ibp" (default): WSINDy integration-by-parts; w_i = -∫ψ'_k X_i dt uses the
    #       smooth GP STATE only (derivative moved onto the analytic ψ'), with
    #       Σ_W = Ψ' Σ_X Ψ'^T + slack. Noise-robust; matches the manuscript form.
    #   "deriv": w_i = ∫ψ_k μ_z dt — integrates the noisy GP derivative estimate,
    #       Σ_W = Ψ Σ_z Ψ^T + slack.
    WEAKFORM_MODE = os.environ.get("WEAKFORM_MODE", "ibp").lower()

    inv_sigma_O2_default = 1.0 / (sigma_O ** 2)
    log_sigma_O2_default = 2.0 * jnp.log(sigma_O)

    # ── Operator-block structure (cAHBN) for hierarchical/ARD prior ───────
    # Column→block id from the fitted operators (const, linear, quad, input,
    # state-input). Used to give each operator block its own learned prior scale.
    _col_blocks = []
    for _bid, _op in enumerate(rom.model.operators):
        _e = _op.entries
        _ncols = _e.shape[1] if _e.ndim == 2 else 1
        _col_blocks.extend([_bid] * _ncols)
    block_id = np.asarray(_col_blocks, dtype=int)
    n_blocks = len(rom.model.operators)
    block_id_jnp = jnp.asarray(block_id)
    assert block_id.shape[0] == rom.model.operator_matrix.shape[1], \
        f"block_id {block_id.shape[0]} != m {rom.model.operator_matrix.shape[1]}"

    # OP_PRIOR_MODE "fixed" (default) or "block_hier": learn per-block operator
    # prior scales τ_b with a broad hyperprior log τ_b ~ Normal(log σ_O, s);
    # the marginal likelihood (Occam) shrinks under-determined blocks.
    OP_PRIOR_MODE = os.environ.get("OP_PRIOR_MODE", "block_hier").lower()
    HIER_TAU0 = float(os.environ.get("HIER_TAU0", str(sigma_O)))
    HIER_TAU_SCALE = float(os.environ.get("HIER_TAU_SCALE", "3.0"))

    def _prior_prec_from_tau(tau_block):
        """Per-column prior precision + log|Σ_O| from per-block scales τ_block."""
        tau_col = tau_block[block_id_jnp]
        prior_prec = 1.0 / (tau_col ** 2 + 1e-12)
        log_prior_cov = 2.0 * jnp.sum(jnp.log(tau_col + 1e-12))
        return prior_prec, log_prior_cov

    if O_prior is None:
        O_prior_jnp = jnp.zeros((num_modes, rom.model.operator_matrix.shape[1]))
    else:
        O_prior_jnp = jnp.asarray(O_prior)

    if sigma_O_vec is None:
        inv_sigma_O2_diag = None
        log_sigma_O2_sum = None
    else:
        _v = jnp.asarray(sigma_O_vec)
        inv_sigma_O2_diag = 1.0 / (_v ** 2)
        log_sigma_O2_sum = 2.0 * jnp.sum(jnp.log(_v))

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

    def _build_AyP_ic(ic, ells, sig2s, nus, gamma2):
        """Return per-IC blocks with full GP derivative posterior covariance.

        Both blocks use Σ_z (full GP derivative posterior covariance):
          - Derivative: Σ_D = (Σ_z + γ²I) / deriv_weight
          - Weak-form:  Σ_W = (Ψ Σ_z Ψ^T + γ² diag(∫ψ²)) / weakform_weight
        """
        d = all_ic_data[ic]
        Xs_list, muz_list, Kz_list, Kx_list, mll_list = [], [], [], [], []
        for j in range(num_modes):
            Xj, muj, Kzj, Kxj, mllj = _single_gp_conditional(
                ells[j], sig2s[j], nus[j], d['y_obs'][j], d)
            Xs_list.append(Xj)
            muz_list.append(muj)
            Kz_list.append(Kzj)
            Kx_list.append(Kxj)
            mll_list.append(mllj)
        Xs = jnp.stack(Xs_list)
        mu_zs = jnp.stack(muz_list)
        K_posts_Z = jnp.stack(Kz_list)
        K_posts_X = jnp.stack(Kx_list)
        mll_total = jnp.sum(jnp.stack(mll_list))

        f_X = rom.model._assemble_data_matrix(Xs, inputs=d['inputs_eval'])
        n_eval = f_X.shape[0]
        I_eval = jnp.eye(n_eval)

        # Derivative block: diagonal Σ_D = (diag(Σ_z) + γ²I) / deriv_weight.
        # Only the marginal derivative variances are used; the off-diagonal GP
        # correlations are dropped — they invert into a high-pass whitening filter
        # that overfits noise (see euler/04_unified.py for the analysis).
        Sigma_D = (jax.vmap(lambda K: jnp.diag(jnp.diag(K)))(K_posts_Z)
                   + gamma2 * I_eval[None, :, :]) / (deriv_weight + 1e-30)

        # Weak-form block (WEAKFORM_MODE): design matrix is the same in both modes.
        A_weak = d['wpsi'] @ f_X                                  # (K, m)
        diag_slack = gamma2 * jnp.diag(d['int_psi_sq_arr'])
        if WEAKFORM_MODE == "ibp":
            # Integration-by-parts: data uses the smooth GP state only.
            #   w_i = -∫ψ'_k X_i dt,  Σ_W = Ψ' Σ_X Ψ'^T + slack
            weak_obs = -(Xs @ d['wpsi_dot'].T)                   # (r, K)
            def _sigma_w_one(K_post_X_i):
                return (d['wpsi_dot'] @ K_post_X_i @ d['wpsi_dot'].T
                        + diag_slack) / (weakform_weight + 1e-30)
            Sigma_W = jax.vmap(_sigma_w_one)(K_posts_X)          # (r, K, K)
        else:
            # Derivative form: integrates the noisy GP derivative μ_z.
            weak_obs = mu_zs @ d['wpsi'].T                       # (r, K)
            def _sigma_w_one(K_post_Z_i):
                return (d['wpsi'] @ K_post_Z_i @ d['wpsi'].T
                        + diag_slack) / (weakform_weight + 1e-30)
            Sigma_W = jax.vmap(_sigma_w_one)(K_posts_Z)          # (r, K, K)
        # Diagonalize the weak-form covariance too, for consistency with the
        # derivative block (off-diagonals are negligible — the weak block is
        # small and the derivative block dominates the fit).
        Sigma_W = jax.vmap(lambda S: jnp.diag(jnp.diag(S)))(Sigma_W)

        return Xs, f_X, mu_zs, Sigma_D, A_weak, weak_obs, Sigma_W, mll_total

    def _build_AyP_all(theta):
        """theta: dict of arrays (num_ics, num_modes) per hyper.

        Returns per-IC blocks for dense covariance computation.
        """
        all_blocks = []
        mll_total = 0.0
        for ic in range(num_ics):
            blocks = _build_AyP_ic(
                ic, theta['ells'][ic], theta['sig2s'][ic], theta['nus'][ic],
                theta['gamma2'])
            all_blocks.append(blocks)
            mll_total = mll_total + blocks[7]  # mll_total
        return all_blocks, mll_total

    def _per_mode_evidence_multi_ic(all_blocks, mode_i, m,
                                     prior_prec_vec, log_det_prior, O_prior_i):
        """log p(y_i | θ) summing dense-block contributions across ICs.

        prior_prec_vec: per-column operator-prior precision (length m).
        O_prior_i: operator prior mean for this mode.
        """
        M_total = jnp.zeros((m, m))
        b_total = jnp.zeros(m)
        quad_y_total = 0.0
        log_det_Sig_total = 0.0
        N_total = 0

        for blocks in all_blocks:
            # blocks = (Xs, A_D, mu_zs, Sigma_D, A_W, weak_obs, Sigma_W, mll)
            A_D = blocks[1]         # (T_eval, m)
            y_D = blocks[2][mode_i] # mu_zs[i]
            Sigma_D_i = blocks[3][mode_i]
            A_W = blocks[4]         # (K, m)
            y_W = blocks[5][mode_i] # weak_obs[i]
            Sigma_W_i = blocks[6][mode_i]

            # Derivative block
            y_D_resid = y_D - A_D @ O_prior_i
            M_D, b_D, qy_D, lds_D, N_D = _dense_block_contrib(
                A_D, y_D_resid, Sigma_D_i)
            # Weak-form block
            y_W_resid = y_W - A_W @ O_prior_i
            M_W, b_W, qy_W, lds_W, N_W = _dense_block_contrib(
                A_W, y_W_resid, Sigma_W_i)

            M_total = M_total + M_D + M_W
            b_total = b_total + b_D + b_W
            quad_y_total = quad_y_total + qy_D + qy_W
            log_det_Sig_total = log_det_Sig_total + lds_D + lds_W
            N_total = N_total + N_D + N_W

        M_total = 0.5 * (M_total + M_total.T)
        jitter = LAMBDA_JITTER * jnp.maximum(jnp.trace(M_total) / m, 1.0)
        Lambda_i = M_total + jnp.diag(prior_prec_vec) + jitter * jnp.eye(m)
        L_i = jnp.linalg.cholesky(Lambda_i)
        mu_centered = jax.scipy.linalg.cho_solve((L_i, True), b_total)
        mu_i = mu_centered + O_prior_i
        log_det_Lambda = 2.0 * jnp.sum(jnp.log(jnp.diag(L_i)))
        quad_mu = jnp.dot(mu_centered, b_total)
        log_p = -0.5 * ((quad_y_total - quad_mu) + log_det_Sig_total +
                        log_det_prior + log_det_Lambda +
                        N_total * jnp.log(2.0 * jnp.pi))
        return log_p, mu_i, L_i

    def model(gamma2=0.5):
        # Sample per-IC, per-mode GP hypers.
        ells_per = []
        sig2s_per = []
        nus_per = []
        for ic in range(num_ics):
            d = all_ic_data[ic]
            _ell_sc = d.get('mle_log_ell_scales', None)
            if _ell_sc is None:
                ells = jnp.stack([
                    numpyro.sample(f"lengthscale_{ic}_{j}",
                        dist.LogNormal(d['mle_log_ells'][j], gp_prior_scale))
                    for j in range(num_modes)])
            else:
                ells = jnp.stack([
                    numpyro.sample(f"lengthscale_{ic}_{j}",
                        dist.LogNormal(d['mle_log_ells'][j], _ell_sc[j]))
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

        all_blocks, mll_total = _build_AyP_all(theta)

        for ic in range(num_ics):
            for j in range(num_modes):
                numpyro.deterministic(f"X_{ic}_{j}", all_blocks[ic][0][j])

        if mll_weight > 0:
            numpyro.factor("gp_mll", mll_weight * mll_total)

        m = all_blocks[0][1].shape[1]  # A_D.shape[1]

        # Operator-prior precision per column (fixed or hierarchical/learned).
        if OP_PRIOR_MODE == "block_hier":
            log_tau = numpyro.sample(
                "log_tau_block",
                dist.Normal(jnp.log(HIER_TAU0) * jnp.ones(n_blocks), HIER_TAU_SCALE))
            prior_prec_vec, log_det_prior = _prior_prec_from_tau(jnp.exp(log_tau))
        elif inv_sigma_O2_diag is None:
            prior_prec_vec = inv_sigma_O2_default * jnp.ones(m)
            log_det_prior = m * log_sigma_O2_default
        else:
            prior_prec_vec = inv_sigma_O2_diag
            log_det_prior = log_sigma_O2_sum

        total_evidence = 0.0
        for i in range(num_modes):
            log_p_i, _, _ = _per_mode_evidence_multi_ic(
                all_blocks, i, m,
                prior_prec_vec, log_det_prior,
                O_prior_jnp[i])
            total_evidence = total_evidence + log_p_i
        numpyro.factor("marg_O_evidence", total_evidence)

    @jax.jit
    def posterior_O_fn(ells_stacked, sig2s_stacked, nus_stacked,
                       gamma2, sigma_O_val, tau_block=None):
        """Closed-form O posterior given θ (and optional per-block scales τ).

        ells_stacked etc.: (num_ics, num_modes)
        Returns (μ_O, C_O) where C_O Cᵀ_O = Σ_O for each mode.
        """
        inv_sO2 = 1.0 / (sigma_O_val ** 2 + 1e-12)
        # Per-column prior precision: learned τ_block, fixed sigma_O_vec, or scalar.
        if tau_block is not None:
            prior_prec_vec, _ = _prior_prec_from_tau(tau_block)
        elif inv_sigma_O2_diag is not None:
            prior_prec_vec = inv_sigma_O2_diag
        else:
            prior_prec_vec = None  # scalar fallback handled below
        theta = dict(ells=ells_stacked, sig2s=sig2s_stacked,
                     nus=nus_stacked, gamma2=gamma2)
        all_blocks, _ = _build_AyP_all(theta)
        m = all_blocks[0][1].shape[1]  # A_D.shape[1]

        mu_all, C_all = [], []
        for i in range(num_modes):
            # Accumulate (M, b) across ICs
            M_total = jnp.zeros((m, m))
            b_total = jnp.zeros(m)
            for blocks in all_blocks:
                A_D = blocks[1]
                y_D_resid = blocks[2][i] - A_D @ O_prior_jnp[i]
                Sigma_D_i = blocks[3][i]
                A_W = blocks[4]
                y_W_resid = blocks[5][i] - A_W @ O_prior_jnp[i]
                Sigma_W_i = blocks[6][i]
                M_D, b_D, _, _, _ = _dense_block_contrib(A_D, y_D_resid, Sigma_D_i)
                M_W, b_W, _, _, _ = _dense_block_contrib(A_W, y_W_resid, Sigma_W_i)
                M_total = M_total + M_D + M_W
                b_total = b_total + b_D + b_W

            M_total = 0.5 * (M_total + M_total.T)
            jitter = LAMBDA_JITTER * jnp.maximum(jnp.trace(M_total) / m, 1.0)
            if prior_prec_vec is None:
                Lambda_i = M_total + (inv_sO2 + jitter) * jnp.eye(m)
            else:
                Lambda_i = (M_total + jnp.diag(prior_prec_vec) +
                            jitter * jnp.eye(m))
            L_i = jnp.linalg.cholesky(Lambda_i)
            mu_centered = jax.scipy.linalg.cho_solve((L_i, True), b_total)
            mu_i = mu_centered + O_prior_jnp[i]
            C_i = jax.scipy.linalg.solve_triangular(
                L_i, jnp.eye(m), lower=True).T
            mu_all.append(mu_i)
            C_all.append(C_i)
        return jnp.stack(mu_all), jnp.stack(C_all)

    return model, posterior_O_fn


# =============================================================================
# Run one regime
# =============================================================================
def run_experiment(schema, p=None):
    if p is None:
        p = MODEL_PARAMS
    # Allow quick env overrides of the operator-prior scale and ODE slack.
    p = dict(p)
    p['SIGMA_O'] = float(os.environ.get("SIGMA_O", p['SIGMA_O']))
    p['GAMMA2'] = float(os.environ.get("GAMMA2", p['GAMMA2']))
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

    # ── OpInf prior center via per-IC MLE GP fit + LS regression ─────────
    # The weak-form likelihood is locally satisfied by trajectory-divergent
    # operators (heat is stiff & decay-dominated). Centering the operator
    # prior on a stable OpInf solution prevents this failure mode.
    print("  Computing OpInf prior center via MLE GP fits...")
    from core import compute_gp_derivatives, rbf_eval
    all_mle_Ls, all_mle_Vs, all_mle_Ns = [], [], []
    for ic in range(nics):
        Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(
            all_time_sampled[ic], all_snapshots_comp[ic], verbose=False)
        all_mle_Ls.append(Ls)
        all_mle_Vs.append(Vs)
        all_mle_Ns.append(Ns)

    D_blocks, dXdt_blocks = [], []
    for ic in range(nics):
        t_eval_ic = np.linspace(
            float(all_time_sampled[ic][0]),
            float(all_time_sampled[ic][-1]), neval)
        X_mle = np.zeros((nmodes, neval))
        for j in range(nmodes):
            ell = all_mle_Ls[ic][j]
            sig2 = all_mle_Vs[ic][j]
            nu = all_mle_Ns[ic][j]
            K = rbf_eval(ell, sig2, all_time_sampled[ic],
                         all_time_sampled[ic]) \
                + (nu + 1e-5) * np.eye(len(all_time_sampled[ic]))
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
    n_cols = D_all.shape[1]
    # cAHBN block layout: c[0:1], A[1:1+r], H[1+r:1+r+r(r+1)/2],
    # B[..:..+m_in], N[last r*m_in cols]
    H_start = 1 + nmodes
    H_end = H_start + nmodes * (nmodes + 1) // 2
    n_input = (n_cols - H_end) // (nmodes + 1)
    B_start, B_end = H_end, H_end + n_input
    N_start, N_end = B_end, n_cols

    # Heat is a *linear* PDE: H (quadratic) and N (bilinear) blocks are 0 in
    # the true operator. LS regression on GP-smoothed derivatives overfits
    # them, yielding stable A but unstable trajectories. We penalise them
    # heavily in the LS solve and force them small in the SVI prior.
    ridge_diag = np.ones(n_cols)
    ridge_diag[H_start:H_end] = 1e3
    ridge_diag[N_start:N_end] = 1e3
    O_ls = np.linalg.solve(
        D_all.T @ D_all + np.diag(ridge_diag),
        D_all.T @ dXdt_all).T
    print(f"  OpInf prior O_ls: ‖O_ls‖={np.linalg.norm(O_ls):.2f}, "
          f"shape={O_ls.shape}  "
          f"‖H‖={np.linalg.norm(O_ls[:, H_start:H_end]):.3f}  "
          f"‖N‖={np.linalg.norm(O_ls[:, N_start:N_end]):.3f}")
    _A_ls = O_ls[:, 1:1+nmodes]
    _eigs = np.linalg.eigvals(_A_ls)
    print(f"  O_ls A-block: ‖A‖={np.linalg.norm(_A_ls):.2f}  "
          f"eig(A) real ∈ [{_eigs.real.min():.3f}, {_eigs.real.max():.3f}]")
    print(f"  O_ls ‖c‖={np.linalg.norm(O_ls[:, 0:1]):.2f}  "
          f"‖B‖={np.linalg.norm(O_ls[:, B_start:B_end]):.2f}")

    # ── Force A-block strictly stable ───────────────────────────────────
    # Heat is dissipative: eigenvalues of A should all be negative. GP-
    # smoothed LS regression yields a marginal A (max eig ≈ 0). Shift the
    # A diagonal so max real eigenvalue ≤ -DECAY_MARGIN.
    DECAY_MARGIN = 0.5
    max_eig = float(_eigs.real.max())
    if max_eig > -DECAY_MARGIN:
        shift = max_eig + DECAY_MARGIN
        O_ls[:, 1:1+nmodes] = _A_ls - shift * np.eye(nmodes)
        _eigs2 = np.linalg.eigvals(O_ls[:, 1:1+nmodes])
        print(f"  Stability shift applied: −{shift:.3f}·I  →  "
              f"eig(A) real ∈ [{_eigs2.real.min():.3f}, "
              f"{_eigs2.real.max():.3f}]")

    # Per-entry prior std vector: tight on H/N (linear PDE), moderate on
    # A (so SVI doesn't push it back into instability), looser on c/B.
    sigma_O_vec = np.full(n_cols, float(p['SIGMA_O']))
    sigma_O_vec[1:1+nmodes] = 0.1            # A block — tight around stable shift
    sigma_O_vec[H_start:H_end] = 0.02         # H block — heat is linear
    sigma_O_vec[N_start:N_end] = 0.02         # N block — heat is linear

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
        O_prior=O_ls,
        sigma_O_vec=sigma_O_vec,
    )

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
                if not k.startswith("X_") and not k.startswith("Xs_")}
        print(f"  NUTS: {num_warmup}+{num_samples} samples ({time.time()-t0:.1f}s)")
    else:
        # No MLE warm-start: spectrum-anchored priors are sufficient
        guide = autoguide.AutoNormal(model, init_loc_fn=init_to_median)
        optimizer = ClippedAdam(step_size=p['LEARNING_RATE'])
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
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
    # Per-block operator scales τ (block_hier mode); None otherwise.
    tau_block_s = None
    if 'log_tau_block' in post:
        tau_block_s = jnp.exp(jnp.asarray(post['log_tau_block']))  # (npost, n_blocks)
        _tb_mean = np.exp(np.asarray(post['log_tau_block']).mean(0))
        print(f"  learned τ_block (per operator c,A,H,B,N...): "
              f"{np.array2string(_tb_mean, precision=3)}")

    @jax.jit
    def _draw_O(ells, sig2s, nus, key, tau_block):
        mu_O, C_O = posterior_O_fn(ells, sig2s, nus, p['GAMMA2'], sO_default,
                                   tau_block)
        eps = jax.random.normal(key, shape=mu_O.shape)
        return mu_O + jnp.einsum('ijk,ik->ij', C_O, eps), mu_O

    t_o = time.time()
    O_samples_list, O_mean_list = [], []
    for s in range(npost):
        tb = None if tau_block_s is None else tau_block_s[s]
        O, mu_O = _draw_O(ells_s[s], sig2s_s[s], nus_s[s], keys[s], tb)
        O_samples_list.append(np.array(O))
        O_mean_list.append(np.array(mu_O))
    O_samples = np.stack(O_samples_list)
    O_means = np.stack(O_mean_list)
    print(f"  O posterior sampling: {time.time()-t_o:.1f}s")
    op_norms = np.linalg.norm(O_samples.reshape(npost, -1), axis=1)
    print(f"  ‖O‖: median={np.median(op_norms):.1f}  "
          f"min={op_norms.min():.1f}  max={op_norms.max():.1f}")

    # Build samples dict for _find_operator_samples
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

    # Optional initial-condition uncertainty (see euler/04_unified.py). Per eval
    # IC, draw each trajectory's start from the GP state posterior at t0.
    IC_UNCERTAINTY = bool(int(os.environ.get("IC_UNCERTAINTY", "1")))
    IC_SCALE = float(os.environ.get("IC_SCALE", "1.0"))

    def _ic_std(t_tr, ell_v, sig2_v, nu_v):
        t_tr = np.asarray(t_tr); n_tr = len(t_tr)
        sq_tt = (t_tr[:, None] - t_tr[None, :]) ** 2
        sq_0t = (t_tr[0] - t_tr) ** 2
        out = np.zeros(len(ell_v))
        for j in range(len(ell_v)):
            ell2 = float(ell_v[j]) ** 2
            K_tt = (sig2_v[j] * np.exp(-sq_tt / (2 * ell2))
                    + (nu_v[j] + max(1e-5, sig2_v[j] * 1e-4)) * np.eye(n_tr))
            k0 = sig2_v[j] * np.exp(-sq_0t / (2 * ell2))
            var = sig2_v[j] - k0 @ np.linalg.solve(K_tt, k0)
            out[j] = np.sqrt(max(float(var), 0.0))
        return out

    # Per-mode GP hypers averaged over posterior samples (per training IC).
    ells_m = np.asarray(ells_s).mean(0)   # (nics, nmodes)
    sig2s_m = np.asarray(sig2s_s).mean(0)
    nus_m = np.asarray(nus_s).mean(0)
    rng_ic = np.random.default_rng(p['SEED'])

    print(f"\n  Results ({runtime:.0f}s):")
    for ic_idx, (params_ic, true_c) in enumerate(zip(eval_params, eval_true_comp)):
        q0 = eval_snaps_comp[ic_idx][:, 0]
        _ic_input_func = input_func_factory(params_ic)
        ic_input_func = lambda t, f=_ic_input_func: np.asarray(f(t))
        q0_samples = None
        if IC_UNCERTAINTY:
            # Use this IC's fitted hypers; fall back to the train-IC mean for the
            # held-out test IC (which has no fitted GP).
            hp = ic_idx if ic_idx < ells_m.shape[0] else None
            if hp is None:
                ev, s2v, nuv = ells_m.mean(0), sig2s_m.mean(0), nus_m.mean(0)
            else:
                ev, s2v, nuv = ells_m[hp], sig2s_m[hp], nus_m[hp]
            sig_ic = _ic_std(eval_t_samp[ic_idx], ev, s2v, nuv)
            eps_ic = rng_ic.standard_normal((len(O_samples), len(q0)))
            q0_samples = q0[None, :] + IC_SCALE * sig_ic[None, :] * eps_ic
            if ic_idx == 0:
                print(f"  IC uncertainty ON: σ_ic[IC0]="
                      f"{np.array2string(sig_ic, precision=4)}  scale={IC_SCALE}")
        ic_solves = generate_rom_solves(
            operator_samples=O_samples, rom=rom, q0=q0,
            time_eval=t_pred, input_func=ic_input_func,
            max_samples=max_samp, q0_samples=q0_samples,
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

    print(f"\n    Stability — train ICs: {train_ic_stable}/{train_ic_total} "
          f"({stability_pct:.0f}%)   test IC: {all_n_stable[-1]}/{max_samp}")

    # ── CI coverage + width, computed PER eval-IC, then split into
    #    train-IC (averaged over the nics training ICs) and test-IC (held-out).
    #    Coverage is further split into the in-window (fit) and extrapolation
    #    horizons so the metric is unambiguous.
    def _ci_for_ic(ic_idx):
        if all_n_stable[ic_idx] == 0:
            return None
        ti = interp1d(config.time_domain, eval_true_comp[ic_idx],
                      kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)
        q05 = np.percentile(all_rom_solves[ic_idx], 5, axis=0)
        q95 = np.percentile(all_rom_solves[ic_idx], 95, axis=0)
        inside = (ta >= q05) & (ta <= q95)
        return dict(
            width=float(np.mean(q95 - q05)),
            cov=float(np.mean(inside)),
            cov_fit=float(np.mean(inside[:, train_mask])),
            cov_ext=float(np.mean(inside[:, pred_mask])),
        )

    ci_width = ci_coverage = float('nan')
    ci_cov_fit = ci_cov_ext = float('nan')
    test_ci_coverage = test_ci_cov_fit = test_ci_cov_ext = test_ci_width = float('nan')
    if train_ic_stable > 0:
        train_ci = [_ci_for_ic(i) for i in range(nics)]
        train_ci = [c for c in train_ci if c is not None]
        if train_ci:
            ci_width = float(np.mean([c['width'] for c in train_ci]))
            ci_coverage = float(np.mean([c['cov'] for c in train_ci]))
            ci_cov_fit = float(np.mean([c['cov_fit'] for c in train_ci]))
            ci_cov_ext = float(np.mean([c['cov_ext'] for c in train_ci]))
    test_ci = _ci_for_ic(nics)
    if test_ci is not None:
        test_ci_width = test_ci['width']
        test_ci_coverage = test_ci['cov']
        test_ci_cov_fit = test_ci['cov_fit']
        test_ci_cov_ext = test_ci['cov_ext']

    # ── Clear, disambiguated report ──────────────────────────────────────
    #   "fit"    = in-window (t ≤ train end), "ext" = temporal extrapolation.
    #   train-IC = averaged over training ICs; test-IC = held-out new IC.
    print(f"\n    ── Train ICs (n={nics}) ──")
    print(f"      fit err: {train_error:.4%}   extrap err: {pred_error:.4%}")
    print(f"      coverage: overall {ci_coverage:.2%}  "
          f"(fit {ci_cov_fit:.2%} / extrap {ci_cov_ext:.2%})   target 90%")
    print(f"    ── Test IC (held-out) ──")
    print(f"      fit err: {all_train_errors[-1]:.4%}   "
          f"extrap err: {all_pred_errors[-1]:.4%}")
    print(f"      coverage: overall {test_ci_coverage:.2%}  "
          f"(fit {test_ci_cov_fit:.2%} / extrap {test_ci_cov_ext:.2%})   target 90%")

    # ── Persist (schema matches heat 04_unified) ────────────
    out_dir = os.path.join(SCRIPT_DIR, 'results', 'comparison', schema['name'])
    os.makedirs(out_dir, exist_ok=True)
    save_dict = {
        't_pred': t_pred,
        't_full': np.asarray(config.time_domain),
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'ci_cov_fit': ci_cov_fit, 'ci_cov_ext': ci_cov_ext,
        'test_ci_coverage': test_ci_coverage, 'test_ci_width': test_ci_width,
        'test_ci_cov_fit': test_ci_cov_fit, 'test_ci_cov_ext': test_ci_cov_ext,
        'test_train_error': all_train_errors[-1],
        'test_pred_error': all_pred_errors[-1],
        'runtime': runtime,
        'n_ics': len(all_rom_solves),
        'op_norm_median': float(np.median(op_norms)),
        'losses': losses,
        'training_span': np.array(TRAINING_SPAN),
        'num_modes': nmodes,
        'eval_labels': np.array(eval_labels, dtype=object),
        'O_samples': O_samples,
        'basis_entries': np.asarray(basis.entries),
        'basis_shift': np.asarray(basis.shift_),
    }
    eval_true_states = list(all_true_states) + [test_true_list[0]]
    for ic_idx, solves in enumerate(all_rom_solves):
        if len(solves) > 0:
            save_dict[f'rom_solves_{ic_idx}'] = np.array(solves)
        else:
            save_dict[f'rom_solves_{ic_idx}'] = np.empty(
                (0, p['NUM_MODES'], len(t_pred)))
        save_dict[f'snaps_comp_{ic_idx}'] = eval_snaps_comp[ic_idx]
        save_dict[f'true_comp_{ic_idx}'] = eval_true_comp[ic_idx]
        save_dict[f'true_states_{ic_idx}'] = eval_true_states[ic_idx]
        save_dict[f't_samp_{ic_idx}'] = eval_t_samp[ic_idx]
    np.savez(os.path.join(out_dir, f'04_unified{os.environ.get("OUTPUT_SUFFIX","")}.npz'), **save_dict)

    return {
        'schema': schema,
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'n_stable': train_ic_stable, 'n_total': train_ic_total,
        'test_train_error': all_train_errors[-1],
        'test_pred_error': all_pred_errors[-1],
        'test_n_stable': all_n_stable[-1],
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'ci_cov_fit': ci_cov_fit, 'ci_cov_ext': ci_cov_ext,
        'test_ci_coverage': test_ci_coverage, 'test_ci_width': test_ci_width,
        'test_ci_cov_fit': test_ci_cov_fit, 'test_ci_cov_ext': test_ci_cov_ext,
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

    print(f"\n\n{'=' * 104}")
    print(f"SUMMARY — Marg-O × Weak-Form (Heat, multi-IC)")
    print(f"  'tr'=train ICs, 'te'=held-out test IC; "
          f"err split fit(in-window)/ext(extrapolation); CI vs 90% target")
    print(f"{'=' * 104}")
    print(f"{'Regime':<26s} {'Nois':>4s} {'Stab':>5s} "
          f"{'trFit':>7s} {'trExt':>7s} {'trCI':>6s} "
          f"{'teFit':>7s} {'teExt':>7s} {'teCI':>6s} {'Time':>6s}")
    print(f"{'-'*26} {'-'*4} {'-'*5} {'-'*7} {'-'*7} {'-'*6} "
          f"{'-'*7} {'-'*7} {'-'*6} {'-'*6}")
    for r in results:
        s = r['schema']
        print(f"{s['label']:<26s} {s['NOISE_LEVEL']:>3.0%} "
              f"{r['stability_pct']:>4.0f}% "
              f"{r['train_error']:>6.2%} {r['pred_error']:>6.2%} "
              f"{r['ci_coverage']:>5.0%} "
              f"{r['test_train_error']:>6.2%} {r['test_pred_error']:>6.2%} "
              f"{r['test_ci_coverage']:>5.0%} {r['runtime']:>5.0f}s")


if __name__ == "__main__":
    schema_names = sys.argv[1:] if len(sys.argv) > 1 else None
    main(schema_names)

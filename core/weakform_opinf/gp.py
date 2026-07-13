"""Gaussian-process conditional + spectrum-anchored hyperparameter priors.

The GP conditional returns, for a single mode given hyperparameters
(ℓ, σ², ν) and observations y_i, the quantities the weak-form OpInf model
needs:

    X_eval     GP posterior mean state on the eval grid
    mu_z       GP posterior mean derivative on the eval grid
    K_post_Z   full GP derivative posterior covariance  Σ_z
    K_post_X   full GP state posterior covariance        Σ_X (for the IBP weak form)
    mll        GP marginal log-likelihood

The prior locations are **spectrum-anchored** — derived from the observation
window T and the per-mode data variance (a deterministic property of the POD
basis). No MLE point estimates are ever used.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

_Z99 = 2.3263  # standard-normal 99% quantile


def make_gp_conditional(time_sampled):
    """Return a ``_single_gp_conditional(ell, sig2, nu, y_i)`` closure and its
    vmapped batch version, with kernel distance matrices baked in for the
    given training/eval grids.

    The eval grid is a dense uniform grid spanning the training window; it is
    returned so callers can share it with the weak-form construction.
    """
    t_train = jnp.asarray(time_sampled)
    n_train = len(t_train)
    I_train = jnp.eye(n_train)

    def _make(time_eval):
        t_eval = jnp.asarray(time_eval)
        sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
        sq_diffs_et = (t_eval[:, None] - t_train[None, :]) ** 2
        diffs_et = t_eval[:, None] - t_train[None, :]
        sq_diffs_ee = (t_eval[:, None] - t_eval[None, :]) ** 2

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
            # Full GP derivative posterior covariance Σ_z = K_zz - K_zy K_yy⁻¹ K_zyᵀ
            K_post_Z = K_zz - K_zy @ V
            K_post_Z = 0.5 * (K_post_Z + K_post_Z.T)
            # Full GP state posterior covariance Σ_X = K_ee - K_et K_yy⁻¹ K_etᵀ
            W = jax.scipy.linalg.cho_solve((L, True), K_et.T)
            K_post_X = K_ee - K_et @ W
            K_post_X = 0.5 * (K_post_X + K_post_X.T)
            mll = -0.5 * (jnp.dot(y_i, alpha)
                          + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
                          + n_train * jnp.log(2.0 * jnp.pi))
            return X_eval, mu_z, K_post_Z, K_post_X, mll

        return _single_gp_conditional, jax.vmap(_single_gp_conditional)

    return _make


def spectrum_anchored_prior_locs(snapshots_comp, time_sampled, num_modes, cfg):
    """Compute spectrum-anchored LogNormal prior locations for (ℓ, σ², ν).

    - ℓ : median at the Nyquist Δt, 99th percentile at the window T
          (``ell_prior_mode='principled'``), or the legacy T/20 anchor.
    - σ²: per-mode data variance (POD singular-value spectrum).
    - ν : 1% of per-mode energy.

    Returns
    -------
    dict with:
        log_ell_loc (float), log_ell_scale (float),
        log_sig2_locs (num_modes,), log_nu_locs (num_modes,)
    """
    t_train = np.asarray(time_sampled)
    n_train = len(t_train)
    T_span = float(t_train[-1] - t_train[0])
    dt_mean = T_span / max(int(n_train) - 1, 1)

    if cfg.ell_prior_mode == "legacy":
        log_ell_loc = float(np.log(T_span / 20.0))
        log_ell_scale = 1.0
    else:
        log_ell_loc = float(np.log(dt_mean))
        log_ell_scale = float(np.log(T_span / dt_mean) / _Z99)

    log_sig2_locs = jnp.array(
        [float(np.log(np.var(np.asarray(snapshots_comp[i])) + 1e-12))
         for i in range(num_modes)])
    log_nu_locs = jnp.array(
        [float(np.log(0.01 * np.var(np.asarray(snapshots_comp[i])) + 1e-12))
         for i in range(num_modes)])

    return dict(
        log_ell_loc=log_ell_loc,
        log_ell_scale=log_ell_scale,
        log_sig2_locs=log_sig2_locs,
        log_nu_locs=log_nu_locs,
    )

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


def make_gp_conditional(time_sampled, jitter_rel=1e-4, *,
                        feature_map=None, feature_variances=None):
    """Return a ``_single_gp_conditional(ell, sig2, nu, y_i)`` closure and its
    vmapped batch version, with kernel distance matrices baked in for the
    given training/eval grids.

    ``jitter_rel`` sets the relative kernel nugget
    ``max(1e-5, σ²·jitter_rel)`` added to the training-kernel diagonal.

    ``jitter_rel=None`` uses a dtype-aware roundoff nugget. ``feature_map``,
    when supplied, maps evaluation times to training/evaluation trend features
    and their evaluation derivatives. Their zero-mean Gaussian coefficients
    have per-mode variances ``feature_variances`` and are marginalized exactly.
    """
    t_train = jnp.asarray(time_sampled)
    n_train = len(t_train)
    I_train = jnp.eye(n_train)
    if (feature_map is None) != (feature_variances is None):
        raise ValueError("GP trend features require both a map and coefficient variances")
    if feature_map is not None:
        variances = np.asarray(feature_variances)
        if (not callable(feature_map) or variances.ndim != 1 or not len(variances)
                or not np.all(np.isfinite(variances)) or np.any(variances < 0)):
            raise ValueError("GP feature variances must be a finite nonnegative vector")
        feature_variances_jnp = jnp.asarray(variances)

    def _make(time_eval):
        t_eval = jnp.asarray(time_eval)
        sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
        sq_diffs_et = (t_eval[:, None] - t_train[None, :]) ** 2
        diffs_et = t_eval[:, None] - t_train[None, :]
        sq_diffs_ee = (t_eval[:, None] - t_eval[None, :]) ** 2
        if feature_map is not None:
            features = feature_map(np.asarray(time_eval))
            H_train, H_eval, dH_eval = (
                np.asarray(features[key]) for key in ("train", "eval", "derivative"))
            if (H_train.ndim != 2 or H_train.shape[0] != n_train
                    or H_eval.shape != (len(t_eval), H_train.shape[1])
                    or dH_eval.shape != H_eval.shape
                    or not all(np.all(np.isfinite(x)) for x in (H_train, H_eval, dH_eval))):
                raise ValueError("GP feature matrices have inconsistent dimensions or nonfinite values")
            H_train, H_eval, dH_eval = map(jnp.asarray, (H_train, H_eval, dH_eval))
            feature_tt = H_train @ H_train.T
            feature_et = H_eval @ H_train.T
            feature_zt = dH_eval @ H_train.T
            feature_ee = H_eval @ H_eval.T
            feature_zz = dH_eval @ dH_eval.T

        def _rbf_sq(ell, sig2, sq_diffs):
            return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

        def _single_gp_conditional(ell, sig2, nu, y_i, trend_variance=None):
            ell2 = ell ** 2
            prior_tt = _rbf_sq(ell, sig2, sq_diff_tt)
            if feature_map is not None:
                if trend_variance is None:
                    raise ValueError("Single-mode trend conditionals require a coefficient variance")
                prior_tt = prior_tt + trend_variance * feature_tt
            jitter = (jnp.finfo(prior_tt.dtype).eps * n_train
                      * jnp.maximum(jnp.max(jnp.diag(prior_tt)), 1.)
                      if jitter_rel is None else jnp.maximum(1e-5, sig2 * jitter_rel))
            K_tt = prior_tt + (nu + jitter) * I_train
            L = jnp.linalg.cholesky(K_tt)
            alpha = jax.scipy.linalg.cho_solve((L, True), y_i)
            K_et_rbf = _rbf_sq(ell, sig2, sq_diffs_et)
            K_et = (K_et_rbf if feature_map is None
                    else K_et_rbf + trend_variance * feature_et)
            X_eval = K_et @ alpha
            K_zy = -(diffs_et / ell2) * K_et_rbf
            if feature_map is not None:
                K_zy = K_zy + trend_variance * feature_zt
            mu_z = K_zy @ alpha
            K_ee = _rbf_sq(ell, sig2, sq_diffs_ee)
            K_zz = ((1.0 - sq_diffs_ee / ell2) / ell2) * K_ee
            if feature_map is not None:
                K_ee = K_ee + trend_variance * feature_ee
                K_zz = K_zz + trend_variance * feature_zz
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

        if feature_map is None:
            batch = jax.vmap(_single_gp_conditional)
        else:
            def batch(ells, sig2s, nus, observations):
                return jax.vmap(_single_gp_conditional)(
                    ells, sig2s, nus, observations, feature_variances_jnp)
        return _single_gp_conditional, batch

    return _make


def trajectory_gp_conditional(trajectory, cfg):
    """Construct a query-grid factory with the trajectory's optional input trend."""
    kwargs = {}
    if cfg.gp_input_trend:
        from .features import input_trend_features
        table = trajectory.get("input_table")
        if table is None:
            raise ValueError("Input-aware GP requires a scalar input_table with times and values")
        def feature_map(query):
            return input_trend_features(
                trajectory["t_sampled"], query, table["times"], table["values"])
        kwargs = dict(
            feature_map=feature_map,
            feature_variances=np.var(trajectory["snapshots_comp"], axis=1) + 1e-12)
    return make_gp_conditional(
        trajectory["t_sampled"], jitter_rel=cfg.gp_jitter_rel, **kwargs)


def spectrum_anchored_prior_locs(snapshots_comp, time_sampled, num_modes, cfg,
                                noise_variances=None):
    """Compute spectrum-anchored LogNormal prior locations for (ℓ, σ², ν).

    - ℓ : median at the Nyquist Δt, 99th percentile at the window T
          (``ell_prior_mode='principled'``), or the legacy T/20 anchor.
    - σ²: per-mode data variance (POD singular-value spectrum).
    - ν : supplied measurement variance, or legacy 1% of per-mode energy.

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
    if noise_variances is None:
        log_nu_locs = jnp.array(
            [float(np.log(0.01 * np.var(np.asarray(snapshots_comp[i])) + 1e-12))
             for i in range(num_modes)])
    else:
        variances = np.asarray(noise_variances)
        if (variances.shape != (num_modes,) or not np.all(np.isfinite(variances))
                or np.any(variances <= 0)):
            raise ValueError("GP measurement-noise prior requires one positive variance per mode")
        log_nu_locs = jnp.log(jnp.asarray(variances))

    return dict(
        log_ell_loc=log_ell_loc,
        log_ell_scale=log_ell_scale,
        log_sig2_locs=log_sig2_locs,
        log_nu_locs=log_nu_locs,
    )

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


def _rbf_sq(ell, sig2, sq_diffs):
    return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))


def gp_conditional_arrays(time_sampled, time_eval, feature_map=None):
    """One trajectory's constants of :func:`gp_conditional`: the kernel distance
    matrices on its training and evaluation grids and, with an input-trend
    ``feature_map``, the trend features' products. Trajectories with equal
    numbers of training times have equal shapes, so their arrays stack and the
    conditional vmaps over them."""
    t_train = jnp.asarray(time_sampled)
    n_train = len(t_train)
    t_eval = jnp.asarray(time_eval)
    arrays = dict(
        eye=jnp.eye(n_train),
        sq_diff_tt=(t_train[:, None] - t_train[None, :]) ** 2,
        sq_diffs_et=(t_eval[:, None] - t_train[None, :]) ** 2,
        diffs_et=t_eval[:, None] - t_train[None, :],
        sq_diffs_ee=(t_eval[:, None] - t_eval[None, :]) ** 2)
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
        arrays.update(
            feature_tt=H_train @ H_train.T,
            feature_et=H_eval @ H_train.T,
            feature_zt=dH_eval @ H_train.T,
            feature_ee=H_eval @ H_eval.T,
            feature_zz=dH_eval @ dH_eval.T)
    return arrays


def gp_conditional(arrays, ell, sig2, nu, y_i, trend_variance=None, *, jitter_rel=1e-4):
    """One mode's GP conditional (X_eval, mu_z, K_post_Z, K_post_X, mll) given
    hyperparameters (ℓ, σ², ν), observations y_i and its trajectory's
    :func:`gp_conditional_arrays`; with trend features, ``trend_variance`` is
    the variance of their coefficients. ``jitter_rel`` as for
    :func:`make_gp_conditional`."""
    features = "feature_tt" in arrays
    n_train = arrays["sq_diff_tt"].shape[0]
    ell2 = ell ** 2
    prior_tt = _rbf_sq(ell, sig2, arrays["sq_diff_tt"])
    if features:
        if trend_variance is None:
            raise ValueError("Single-mode trend conditionals require a coefficient variance")
        prior_tt = prior_tt + trend_variance * arrays["feature_tt"]
    # Round-off level of K_tt: max diag K_tt is σ² for the stationary RBF kernel.
    kernel_scale = sig2 if not features else jnp.max(jnp.diag(prior_tt))
    jitter = (jnp.finfo(prior_tt.dtype).eps * n_train * kernel_scale
              if jitter_rel is None else jnp.maximum(1e-5, sig2 * jitter_rel))
    K_tt = prior_tt + (nu + jitter) * arrays["eye"]
    L = jnp.linalg.cholesky(K_tt)
    alpha = jax.scipy.linalg.cho_solve((L, True), y_i)
    K_et_rbf = _rbf_sq(ell, sig2, arrays["sq_diffs_et"])
    K_et = (K_et_rbf if not features
            else K_et_rbf + trend_variance * arrays["feature_et"])
    X_eval = K_et @ alpha
    K_zy = -(arrays["diffs_et"] / ell2) * K_et_rbf
    if features:
        K_zy = K_zy + trend_variance * arrays["feature_zt"]
    mu_z = K_zy @ alpha
    K_ee = _rbf_sq(ell, sig2, arrays["sq_diffs_ee"])
    K_zz = ((1.0 - arrays["sq_diffs_ee"] / ell2) / ell2) * K_ee
    if features:
        K_ee = K_ee + trend_variance * arrays["feature_ee"]
        K_zz = K_zz + trend_variance * arrays["feature_zz"]
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


def gp_batch(arrays, ells, sig2s, nus, observations, trend_scales=None, *,
             feature_variances=None, jitter_rel=1e-4):
    """Every mode's :func:`gp_conditional` (vmapped over modes) for one
    trajectory. With trend features, ``feature_variances`` are the per-mode
    coefficient variances and ``trend_scales`` multiply the coefficient prior
    sds (the variances by their squares)."""
    def single(ell, sig2, nu, y_i, trend_variance=None):
        return gp_conditional(arrays, ell, sig2, nu, y_i, trend_variance, jitter_rel=jitter_rel)
    if feature_variances is None:
        return jax.vmap(single)(ells, sig2s, nus, observations)
    variances = (feature_variances if trend_scales is None
                 else feature_variances * trend_scales ** 2)
    return jax.vmap(single)(ells, sig2s, nus, observations, variances)


def make_gp_conditional(time_sampled, jitter_rel=1e-4, *,
                        feature_map=None, feature_variances=None):
    """Return a ``_single_gp_conditional(ell, sig2, nu, y_i)`` closure and its
    vmapped batch version, with kernel distance matrices baked in for the
    given training/eval grids.

    ``jitter_rel=None`` adds the round-off nugget ``n·ε·max diag K`` (ε the
    machine epsilon of the working dtype) to the training-kernel diagonal; a
    float selects the historical relative nugget ``max(1e-5, σ²·jitter_rel)``.

    ``feature_map``, when supplied, maps evaluation times to training/evaluation
    trend features and their evaluation derivatives. Their zero-mean Gaussian
    coefficients have per-mode variances ``feature_variances`` and are
    marginalized exactly.
    """
    if (feature_map is None) != (feature_variances is None):
        raise ValueError("GP trend features require both a map and coefficient variances")
    if feature_map is not None:
        variances = np.asarray(feature_variances)
        if (not callable(feature_map) or variances.ndim != 1 or not len(variances)
                or not np.all(np.isfinite(variances)) or np.any(variances < 0)):
            raise ValueError("GP feature variances must be a finite nonnegative vector")
        feature_variances_jnp = jnp.asarray(variances)

    def _make(time_eval):
        arrays = gp_conditional_arrays(time_sampled, time_eval, feature_map)

        def _single_gp_conditional(ell, sig2, nu, y_i, trend_variance=None):
            return gp_conditional(arrays, ell, sig2, nu, y_i, trend_variance, jitter_rel=jitter_rel)

        if feature_map is None:
            batch = jax.vmap(_single_gp_conditional)
        else:
            def batch(ells, sig2s, nus, observations, trend_scales=None):
                return gp_batch(arrays, ells, sig2s, nus, observations, trend_scales,
                                feature_variances=feature_variances_jnp, jitter_rel=jitter_rel)
        return _single_gp_conditional, batch

    return _make


def trajectory_trend(trajectory, cfg):
    """The trajectory's input-trend ``feature_map`` (query times → features) and
    per-mode coefficient variances, or ``(None, None)`` without the input-aware GP."""
    if not cfg.gp_input_trend:
        return None, None
    from .features import input_trend_features
    table = trajectory.get("input_table")
    if table is None:
        raise ValueError("Input-aware GP requires a scalar input_table with times and values")

    def feature_map(query):
        return input_trend_features(
            trajectory["t_sampled"], query, table["times"], table["values"])
    return feature_map, np.var(trajectory["snapshots_comp"], axis=1) + 1e-12


def trajectory_gp_conditional(trajectory, cfg):
    """Construct a query-grid factory with the trajectory's optional input trend."""
    feature_map, variances = trajectory_trend(trajectory, cfg)
    kwargs = {} if feature_map is None else dict(feature_map=feature_map, feature_variances=variances)
    return make_gp_conditional(
        trajectory["t_sampled"], jitter_rel=cfg.gp_jitter_rel, **kwargs)


def effective_dof(trajectory, cfg, ells, sig2s, nus, trend_scales=None):
    """Effective degrees of freedom df_i = tr K_i (K_i + ν_i I)⁻¹ of each mode's
    GP smoother at the training times: the number of independent pieces of
    information its posterior mean (and every derivative or weak-form row built
    from it) carries. K_i is the prior covariance of the observations, input
    trend included. Unit free: invariant to rescaling time and state."""
    t = np.asarray(trajectory["t_sampled"], dtype=float)
    sq = (t[:, None] - t[None, :]) ** 2
    trend = None
    if cfg.gp_input_trend:
        from .features import input_trend_features
        table = trajectory["input_table"]
        H = input_trend_features(t, t, table["times"], table["values"])["train"]
        variances = np.var(trajectory["snapshots_comp"], axis=1) + 1e-12
        if trend_scales is not None:
            variances = variances * np.asarray(trend_scales, dtype=float) ** 2
        trend = H @ H.T, variances
    out = []
    for i, (ell, sig2, nu) in enumerate(zip(*(np.asarray(x, dtype=float) for x in (ells, sig2s, nus)))):
        K = sig2 * np.exp(-sq / (2.0 * ell ** 2))
        if trend is not None:
            K = K + trend[1][i] * trend[0]
        lam = np.clip(np.linalg.eigvalsh(K), 0.0, None)
        out.append(float(np.sum(lam / (lam + nu))))
    return np.array(out)


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

"""Opt-in, training-scaled Gaussian operator priors; no stability constraints."""

import numpy as np

from .constraints import _features, shift_matrix
from .model import _block_id_from_rom, _column_degrees, nondimensional_column_scale


def training_scaled_prior(rom, trajectories, states, rates, *, normalization="rate",
                          center=False, scale=1., damping=0., correlation="none", ridge=.01,
                          mean_clock="training"):
    """Return a physical-coordinate mean, base root and a serializable record.

    In optionally centered feature coordinates, amplitude/rate priors give each
    block the same prior-predictive rate scale per output. Column j in block b
    has SD scale * rate_i / (RMS(feature_j) * sqrt(n_b)). ``amplitude`` uses
    within-trajectory observed SD / T; ``rate`` adds the data-only GP rate RMS
    in quadrature. ``dimensional`` is the original S,T,U rule; ``balanced`` adds
    only sqrt(n_b). Learned block multipliers are applied by build_model.

    Optional correlation uses a normalized feature Gram matrix, not its inverse:
    weakly excited directions receive less variance. Blockwise square roots
    preserve physical block-scale semantics. Full roots instead define latent
    groups; use fixed scales when comparing that experimental alternative.
    All statistics are fixed before joint inference, not uncertain latent data.
    """
    if normalization not in ("dimensional", "balanced", "amplitude", "rate"):
        raise ValueError(f"Unknown normalization: {normalization}")
    if correlation not in ("none", "block", "full"):
        raise ValueError(f"Unknown correlation: {correlation}")
    if mean_clock not in ("training", "rate"):
        raise ValueError(f"Unknown mean clock: {mean_clock}")
    if (not np.isfinite([scale, damping, ridge]).all() or scale <= 0 or damping < 0
            or not 0 < ridge <= 1):
        raise ValueError("Require finite scale>0, damping>=0, and 0<ridge<=1")
    if not trajectories or len(states) != len(trajectories) or len(rates) != len(trajectories):
        raise ValueError("One state/rate array is required per training trajectory")
    r = np.asarray(trajectories[0]["snapshots_comp"]).shape[0]
    blocks, m, nblocks = _block_id_from_rom(rom)
    column_scale, base = nondimensional_column_scale(rom, trajectories, r)
    observations = [np.asarray(tr["snapshots_comp"], dtype=float) for tr in trajectories]
    if any(q.ndim != 2 or q.shape[0] != r or not np.isfinite(q).all() for q in observations):
        raise ValueError("Observations must be finite mode-by-time matrices")
    offset = np.concatenate(observations, axis=1).mean(1) if center else np.zeros(r)
    inputs = [tr.get("inputs_eval") for tr in trajectories]
    if any(u is None for u in inputs) and any(u is not None for u in inputs):
        raise ValueError("Every trajectory or none must supply inputs")
    ninputs = 0 if inputs[0] is None else np.atleast_2d(inputs[0]).shape[0]
    G = shift_matrix(rom, r, offset, ninputs) if center else np.eye(m)
    features, derivative_arrays = [], []
    for q, dq, u in zip(states, rates, inputs):
        q, dq = np.asarray(q, dtype=float), np.asarray(dq, dtype=float)
        if q.ndim != 2 or q.shape[0] != r or dq.shape != q.shape or not np.isfinite([q, dq]).all():
            raise ValueError("GP states and rates must be matching finite mode-by-time matrices")
        features.append(_features(rom, q - offset[:, None], u))
        derivative_arrays.append(dq)
    F = np.concatenate(features)
    feature_rms = np.sqrt(np.mean(F ** 2, axis=0))
    amplitude = np.sqrt(np.mean([np.var(q, axis=1) for q in observations], axis=0))
    rate_rms = np.sqrt(np.mean(np.concatenate(derivative_arrays, axis=1) ** 2, axis=1))
    rate_scale = np.sqrt(rate_rms ** 2 + (amplitude / base["T"]) ** 2)
    if not np.isfinite(F).all() or np.any(feature_rms <= 0) or np.any(amplitude <= 0):
        raise ValueError("Prior scaling needs finite excited features and positive modal fluctuations")
    counts = np.bincount(blocks, minlength=nblocks)
    if normalization in ("dimensional", "balanced"):
        sd = np.broadcast_to(column_scale, (r, m)).copy()
        if normalization == "balanced":
            sd /= np.sqrt(counts[blocks])
    else:
        output_scale = amplitude / base["T"] if normalization == "amplitude" else rate_scale
        sd = output_scale[:, None] / (feature_rms * np.sqrt(counts[blocks]))[None]
    sd *= scale

    Croot = np.eye(m)
    eigenspectra = []
    if correlation != "none":
        normalized = F / feature_rms
        groups = [np.arange(m)] if correlation == "full" else [np.flatnonzero(blocks == b) for b in range(nblocks)]
        for indices in groups:
            X = normalized[:, indices]
            gram = X.T @ X / len(X)
            values, vectors = np.linalg.eigh((1 - ridge) * gram + ridge * np.eye(len(indices)))
            if values.min() <= 0:
                raise ValueError("Regularized feature covariance is not positive definite")
            Croot[np.ix_(indices, indices)] = (vectors * np.sqrt(values)) @ vectors.T
            eigenspectra.append(values.tolist())
    root = np.einsum("ab,ibc->iac", G, sd[:, :, None] * Croot[None])
    mean = np.zeros((r, m))
    if damping:
        degree_q, degree_u = _column_degrees(rom, r, ninputs)
        linear = np.flatnonzero((degree_q == 1) & (degree_u == 0))
        if len(linear) != r:
            raise ValueError("A damping mean requires a complete linear state block")
        u0 = None if not ninputs else np.zeros((ninputs, r))
        linear_features = _features(rom, np.eye(r), u0)[:, linear]
        clock = np.full(r, 1 / base["T"]) if mean_clock == "training" else rate_scale / amplitude
        mean[:, linear] = -damping * clock[:, None] * np.linalg.inv(linear_features).T
        mean = mean @ G.T
    predictive_rms = np.sqrt(np.mean([
        np.sum((F @ (sd[i, :, None] * Croot)) ** 2, axis=1) for i in range(r)], axis=1))
    info = dict(normalization=normalization, center=bool(center), scale=float(scale), damping=float(damping),
                mean_clock=mean_clock,
                correlation=correlation, ridge=float(ridge), offset=offset.tolist(), base=base,
                block_counts=counts.tolist(), feature_rms=feature_rms.tolist(), amplitude=amplitude.tolist(),
                gp_rate_rms=rate_rms.tolist(), rate_scale=rate_scale.tolist(),
                prior_predictive_rate_rms=predictive_rms.tolist(), eigenvalues=eigenspectra,
                statistics="training observations and data-only GP MAP, frozen before joint inference")
    return mean, root, info

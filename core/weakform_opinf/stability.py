"""Training-local Gaussian contraction priors for linear-in-operator ROMs."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import qr, solve_triangular


def local_contraction_prior(rom, states, rates, inputs, state_scales, column_scales,
                            time_scale, damping=20., width=3., anchors=12):
    """Favor T D^-1 J D P = -damping P at fixed training anchors.

    P removes the scaled trajectory tangent. The residual is Gaussian with
    dimensionless width, normalized by sqrt(anchor count). Multiplying it by
    the diagonal base prior gives independent Gaussian operator rows with full
    within-row covariance. Geometry is fixed, not inferred by this function.
    """
    amp, scale = np.asarray(state_scales, float), np.asarray(column_scales, float)
    if (amp.ndim != 1 or scale.ndim != 1 or not np.all(np.isfinite(amp))
            or not np.all(np.isfinite(scale)) or np.any(amp <= 0) or np.any(scale <= 0)):
        raise ValueError("State and operator scales must be finite positive vectors")
    if (not np.isfinite(time_scale) or time_scale <= 0 or not np.isfinite(width) or width <= 0
            or not np.isfinite(damping) or damping < 0 or not isinstance(anchors, int) or anchors < 1):
        raise ValueError("Positive time scale, width and anchor count, and nonnegative damping are required")
    if not len(states) == len(rates) == len(inputs) or not states:
        raise ValueError("Provide aligned, nonempty training states, rates and inputs")
    r, m = len(amp), len(scale)
    if rom.model.operator_matrix.shape != (r, m):
        raise ValueError("Scales do not match the ROM operator shape")

    def feature(q, u):
        return rom.model._assemble_data_matrix(
            q[:, None], inputs=None if u is None else u[:, None])[0]

    jacobian = jax.jit(jax.jacfwd(feature, argnums=0))
    blocks, targets, used = [], [], []
    skipped = 0
    for k, (X, dz, u) in enumerate(zip(states, rates, inputs)):
        X, dz = np.asarray(X, float), np.asarray(dz, float)
        if (X.ndim != 2 or X.shape[0] != r or dz.shape != X.shape or X.shape[1] < 5
                or not np.isfinite(X).all() or not np.isfinite(dz).all()):
            raise ValueError("Each training grid needs at least five finite, aligned state/rate columns")
        if u is not None:
            u = np.atleast_2d(np.asarray(u, float))
            if u.shape[1] != X.shape[1] or not np.isfinite(u).all():
                raise ValueError("Inputs must be finite and aligned with the training grid")
        indices = np.unique(np.linspace(2, X.shape[1] - 3, min(anchors, X.shape[1] - 4), dtype=int))
        for j in indices:
            tangent = dz[:, j] / amp
            speed = np.linalg.norm(tangent)
            if time_scale * speed <= np.sqrt(np.finfo(float).eps):
                skipped += 1
                continue
            tangent /= speed
            P = np.eye(r) - np.outer(tangent, tangent)
            Fq = np.asarray(jacobian(jnp.asarray(X[:, j]),
                                    None if u is None else jnp.asarray(u[:, j])), float)
            if Fq.shape != (m, r) or not np.isfinite(Fq).all():
                raise ValueError("The ROM feature Jacobian must be finite with shape (features, modes)")
            blocks.append(time_scale * (Fq * amp[None]) @ P)
            targets.append(-damping * P)
            used.append([k, int(j)])
    if not blocks:
        raise ValueError("No resolved training tangents; a transverse-contraction prior is undefined")
    normalizer = width * np.sqrt(len(blocks))
    A, y = np.hstack(blocks).T / normalizer, np.hstack(targets) / normalizer
    means, roots = [], []
    for i in range(r):
        design = np.vstack([np.eye(m), A * scale[None] / amp[i]])
        rhs = np.r_[np.zeros(m), y[i]]
        Q, R = qr(design, mode="economic")
        means.append(scale * solve_triangular(R, Q.T @ rhs))
        roots.append(scale[:, None] * solve_triangular(R, np.eye(m)))
    info = dict(rule="training-local transverse Gaussian contraction", damping=float(damping),
                width=float(width), time_scale=float(time_scale), anchors=used,
                skipped_stationary_anchors=skipped, requested_anchors_per_trajectory=anchors,
                state_scales=amp.tolist(), column_scales=scale.tolist(),
                geometry_uncertainty_propagated=False, global_stability_guarantee=False)
    return np.asarray(means), np.asarray(roots), info

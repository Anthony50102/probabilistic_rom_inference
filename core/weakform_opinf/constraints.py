"""Energy-preserving quadratic operators and shifted operator features.

Energy preservation (Goyal, Pontes Duff & Benner 2023, arXiv 2308.13819; Schlegel & Noack 2015): the quadratic
term conserves ½‖q‖², i.e. Σ_i q_i H_i(q ⊗ q) = 0 for every q. In the compressed (symmetric) Kronecker layout this is
one linear equation per multiset {a ≤ b ≤ c} of mode indices — the coefficient of q_a q_b q_c, summed over the ways of
choosing which index is the row — so it is a linear constraint C vec(O) = 0 on the operator, and conditioning the
Gaussian operator posterior on it is exact and closed form. It is invariant under a shift of the state, since the
quadratic coefficients do not change.

Shifted features: the operator is fitted to features of z = q − q̄ (q̄ fixed); for a polynomial library without
inputs, features(q − q̄) = G features(q) for a constant G, so the operator in q-coordinates is O_q = O_z Gᵀ exactly.
"""

from __future__ import annotations

import itertools

import numpy as np
import jax
import jax.experimental
import jax.numpy as jnp


def _features(rom, states):
    """The ROM's own feature assembly at probe states, in float64 whatever the caller's precision."""
    with jax.experimental.enable_x64(True):
        return np.asarray(rom.model._assemble_data_matrix(jnp.asarray(states), inputs=None), dtype=float)


def _probe_states(num_modes, n):
    return np.random.default_rng(0).uniform(-1.5, 1.5, size=(num_modes, n))


def quadratic_columns(rom, num_modes):
    """{(j, k): column} for the pure quadratic columns q_j q_k (j ≤ k), read off the ROM's own feature assembly.

    Raises unless every pair appears exactly once."""
    Q = _probe_states(num_modes, 4 * num_modes ** 2 + 20)
    F = _features(rom, Q)
    found = {}
    for j, k in itertools.combinations_with_replacement(range(num_modes), 2):
        target = Q[j] * Q[k]
        match = [col for col in range(F.shape[1]) if np.allclose(F[:, col], target, rtol=1e-12, atol=1e-12)]
        if len(match) != 1:
            raise ValueError(f"Energy preservation needs exactly one feature column q_{j} q_{k}; found {len(match)}")
        found[(j, k)] = match[0]
    return found


def energy_preserving_matrix(rom, num_modes):
    """C (C(r+2, 3) × r·m), acting on vec(O) row-major (index i·m + column): C vec(O) = 0 ⇔ Σ_i q_i H_i(q⊗q) ≡ 0."""
    cols = quadratic_columns(rom, num_modes)
    m = rom.model.operator_matrix.shape[1]
    rows = []
    for triple in itertools.combinations_with_replacement(range(num_modes), 3):
        row = np.zeros(num_modes * m)
        for i in sorted(set(triple)):
            rest = list(triple)
            rest.remove(i)
            row[i * m + cols[tuple(sorted(rest))]] += 1.0
        rows.append(row)
    return np.asarray(rows)


def shift_matrix(rom, num_modes, shift):
    """G (m × m) with features(q − shift) = features(q) G (row-feature convention), so O_q = O_z Gᵀ.

    Fitted on probe states and checked to round-off; raises if the library is not closed under shifts."""
    shift = np.asarray(shift, dtype=float).reshape(num_modes, 1)
    m = rom.model.operator_matrix.shape[1]
    Q = _probe_states(num_modes, 4 * m + 20) + shift
    F, F_shift = _features(rom, Q), _features(rom, Q - shift)
    G = np.linalg.lstsq(F, F_shift, rcond=None)[0]
    misfit = np.linalg.norm(F @ G - F_shift) / np.linalg.norm(F_shift)
    if not misfit < 1e-9:
        raise ValueError(f"The operator features are not closed under a state shift (misfit {misfit:.2e})")
    G[np.abs(G) < 1e-12 * np.abs(G).max()] = 0.0
    return G


def constraint_log_correction(C, means, lowers, prior_prec):
    """log p(y | C vec(O) = 0) − log p(y) = log N(0; C μ, C Σ Cᵀ) − log N(0; 0, C Σ₀ Cᵀ).

    means (r, m) posterior means; lowers (r, m, m) with posterior precision Λ_i = L_i L_iᵀ; prior_prec (m,) the
    (row-shared) diagonal prior precision. Σ is block diagonal over rows (rows independent before conditioning)."""
    r, m = means.shape
    Cr = C.reshape(C.shape[0], r, m)
    W = jax.vmap(lambda L, Ci: jax.scipy.linalg.solve_triangular(L, Ci.T, lower=True))(lowers, jnp.swapaxes(Cr, 0, 1))
    S = jnp.einsum("imn,imp->np", W, W)
    S0 = jnp.einsum("nim,m,pim->np", Cr, 1.0 / prior_prec, Cr)
    Cmu = jnp.einsum("nim,im->n", Cr, means)
    return _log_normal_zero(Cmu, S) - _log_normal_zero(jnp.zeros_like(Cmu), S0)


def _log_normal_zero(mean, cov):
    """log N(0; mean, cov) with a round-off jitter on cov."""
    n = cov.shape[0]
    cov = 0.5 * (cov + cov.T)
    jitter = n * jnp.finfo(cov.dtype).eps * jnp.trace(cov) / n
    L = jnp.linalg.cholesky(cov + jitter * jnp.eye(n))
    alpha = jax.scipy.linalg.solve_triangular(L, mean, lower=True)
    return -0.5 * (alpha @ alpha + n * jnp.log(2 * jnp.pi)) - jnp.sum(jnp.log(jnp.diag(L)))


def condition_on_constraint(x, roots, C):
    """Matheron's rule: x − Σ Cᵀ (C Σ Cᵀ)⁻¹ C x for x (r, m) drawn from (or equal to the mean of) the row-wise
    Gaussian with covariance blocks Σ_i = R_i R_iᵀ (roots (r, m, m)); exact draw (mean) of the conditioned Gaussian."""
    r, m = x.shape
    Cr = C.reshape(C.shape[0], r, m)
    CR = jnp.einsum("nim,imk->nik", Cr, roots)               # C_i R_i
    S = jnp.einsum("nik,pik->np", CR, CR)
    S = 0.5 * (S + S.T)
    S = S + S.shape[0] * jnp.finfo(S.dtype).eps * jnp.trace(S) / S.shape[0] * jnp.eye(S.shape[0])
    lam = jnp.linalg.solve(S, jnp.einsum("nim,im->n", Cr, x))
    return x - jnp.einsum("imk,nik,n->im", roots, CR, lam)    # Σ_i C_iᵀ λ = R_i (C_i R_i)ᵀ λ

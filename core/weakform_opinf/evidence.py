"""Closed-form marginalised-operator evidence and posterior.

Because both the derivative and weak-form constraints are linear in the
operator O, the conditional posterior p(O_i | θ, data) is Gaussian and both the
per-mode marginal likelihood and the posterior (μ_O, Σ_O) are available in
closed form. SVI/NUTS therefore only explore the GP hyperparameters θ.

The evidence accumulates block contributions **across a list of trajectories**
(a single operator shared across initial conditions). Single-trajectory
experiments are simply the ``len(trajectories) == 1`` case.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

LAMBDA_JITTER = 1e-6


def diag_block_contrib(A_blk, y_blk, prec_vec):
    """(M, b, quad_y, log_det_Σ, N) for a diagonal-covariance block, given the
    precision vector ``prec_vec = 1/diag(Σ)``."""
    Aw = A_blk * prec_vec[:, None]
    M = A_blk.T @ Aw
    b = A_blk.T @ (prec_vec * y_blk)
    quad_y = jnp.sum(prec_vec * y_blk ** 2)
    log_det_Sig = -jnp.sum(jnp.log(prec_vec + 1e-30))
    return M, b, quad_y, log_det_Sig, prec_vec.shape[0]


def dense_block_contrib(A_blk, y_blk, Sigma_blk):
    """(M, b, quad_y, log_det_Σ, N) for a dense-covariance block."""
    N_blk = Sigma_blk.shape[0]
    L = jnp.linalg.cholesky(Sigma_blk + 1e-8 * jnp.eye(N_blk))
    Sinv_A = jax.scipy.linalg.cho_solve((L, True), A_blk)
    Sinv_y = jax.scipy.linalg.cho_solve((L, True), y_blk)
    M = A_blk.T @ Sinv_A
    b = A_blk.T @ Sinv_y
    log_det_Sig = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    quad_y = jnp.dot(y_blk, Sinv_y)
    return M, b, quad_y, log_det_Sig, N_blk


def deriv_block_contrib(A_D, y_D, deriv_blk, deriv_is_diag):
    """Derivative-block contribution: diagonal (precision vector) or dense
    (covariance matrix) depending on ``deriv_is_diag``."""
    if deriv_is_diag:
        return diag_block_contrib(A_D, y_D, deriv_blk)
    return dense_block_contrib(A_D, y_D, deriv_blk)


def make_prior_prec_from_tau(block_id_jnp):
    """Return ``prior_prec_from_tau(tau_block)`` → (per-column precision,
    log|Σ_O|) mapping per-block ARD scales τ_b onto operator columns."""
    def _prior_prec_from_tau(tau_block):
        tau_col = tau_block[block_id_jnp]
        prior_prec = 1.0 / (tau_col ** 2 + 1e-12)
        log_prior_cov = 2.0 * jnp.sum(jnp.log(tau_col + 1e-12))
        return prior_prec, log_prior_cov
    return _prior_prec_from_tau


def per_mode_evidence(traj_blocks, mode_i, m, prior_prec, log_prior_cov,
                      deriv_is_diag):
    """log p(y_i | θ) for mode i, summing derivative + weak-form block
    contributions over all trajectories (shared operator).

    Parameters
    ----------
    traj_blocks : list of tuples
        Per trajectory: (A_D, y_D, Sigma_D, A_W, y_W, Sigma_W) where the y_*/
        Sigma_* are per-mode stacks indexed by ``mode_i``.
    mode_i : int
    m : int
        Operator column count.
    prior_prec : (m,) per-column operator prior precision.
    log_prior_cov : scalar  log|Σ_O| = -Σ_j log(prior_prec_j).
    deriv_is_diag : bool
    """
    M_total = jnp.zeros((m, m))
    b_total = jnp.zeros(m)
    quad_y_total = 0.0
    log_det_Sig_total = 0.0
    N_total = 0

    for (A_D, y_D, Sigma_D, A_W, y_W, Sigma_W) in traj_blocks:
        M_D, b_D, qy_D, lds_D, N_D = deriv_block_contrib(
            A_D, y_D[mode_i], Sigma_D[mode_i], deriv_is_diag)
        M_W, b_W, qy_W, lds_W, N_W = dense_block_contrib(
            A_W, y_W[mode_i], Sigma_W[mode_i])
        M_total = M_total + M_D + M_W
        b_total = b_total + b_D + b_W
        quad_y_total = quad_y_total + qy_D + qy_W
        log_det_Sig_total = log_det_Sig_total + lds_D + lds_W
        N_total = N_total + N_D + N_W

    M_total = 0.5 * (M_total + M_total.T)
    jitter = LAMBDA_JITTER * jnp.maximum(jnp.trace(M_total) / m, 1.0)
    Lambda_i = M_total + jnp.diag(prior_prec) + jitter * jnp.eye(m)
    L_i = jnp.linalg.cholesky(Lambda_i)
    mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_total)
    log_det_Lambda = 2.0 * jnp.sum(jnp.log(jnp.diag(L_i)))
    quad_mu = jnp.dot(mu_i, b_total)
    log_p = -0.5 * ((quad_y_total - quad_mu)
                    + log_det_Sig_total
                    + log_prior_cov
                    + log_det_Lambda
                    + N_total * jnp.log(2.0 * jnp.pi))
    return log_p, mu_i, L_i


def per_mode_posterior(traj_blocks, mode_i, m, prior_prec, deriv_is_diag):
    """Closed-form operator posterior (μ_i, C_i) for mode i with C_i C_iᵀ = Σ_O,i,
    summing block contributions over all trajectories."""
    M_total = jnp.zeros((m, m))
    b_total = jnp.zeros(m)
    for (A_D, y_D, Sigma_D, A_W, y_W, Sigma_W) in traj_blocks:
        M_D, b_D, _, _, _ = deriv_block_contrib(
            A_D, y_D[mode_i], Sigma_D[mode_i], deriv_is_diag)
        M_W, b_W, _, _, _ = dense_block_contrib(
            A_W, y_W[mode_i], Sigma_W[mode_i])
        M_total = M_total + M_D + M_W
        b_total = b_total + b_D + b_W
    M_total = 0.5 * (M_total + M_total.T)
    jitter = LAMBDA_JITTER * jnp.maximum(jnp.trace(M_total) / m, 1.0)
    Lambda_i = M_total + jnp.diag(prior_prec) + jitter * jnp.eye(m)
    L_i = jnp.linalg.cholesky(Lambda_i)
    mu_i = jax.scipy.linalg.cho_solve((L_i, True), b_total)
    C_i = jax.scipy.linalg.solve_triangular(L_i, jnp.eye(m), lower=True).T
    return mu_i, C_i


def _whiten_block(A, y, covariance, diagonal_precision=False):
    if diagonal_precision:
        root_precision = jnp.sqrt(covariance)
        return (A * root_precision[:, None], y * root_precision,
                -jnp.sum(jnp.log(covariance)))
    L = jnp.linalg.cholesky(covariance + 1e-8 * jnp.eye(len(y)))
    return (jax.scipy.linalg.solve_triangular(L, A, lower=True),
            jax.scipy.linalg.solve_triangular(L, y, lower=True),
            2 * jnp.sum(jnp.log(jnp.diag(L))))


def _qr_operator_system(traj_blocks, mode_i, m, prior_prec, deriv_is_diag):
    designs, observations = [], []
    log_det_Sigma = 0.0
    for A_D, y_D, Sigma_D, A_W, y_W, Sigma_W in traj_blocks:
        for A, y, covariance, diagonal in (
                (A_D, y_D[mode_i], Sigma_D[mode_i], deriv_is_diag),
                (A_W, y_W[mode_i], Sigma_W[mode_i], False)):
            whitened, observed, log_det = _whiten_block(
                A, y, covariance, diagonal)
            designs.append(whitened)
            observations.append(observed)
            log_det_Sigma = log_det_Sigma + log_det
    n = sum(len(y) for y in observations)
    augmented = jnp.vstack([*designs, jnp.diag(jnp.sqrt(prior_prec))])
    rhs = jnp.concatenate([*observations, jnp.zeros(m)])
    Q, R = jnp.linalg.qr(augmented, mode="reduced")
    signs = jnp.where(jnp.diag(R) < 0, -1.0, 1.0)
    R = signs[:, None] * R
    mu = jax.scipy.linalg.solve_triangular(
        R, signs * (Q.T @ rhs), lower=False)
    # Avoid subtracting two large quadratic forms in the marginal likelihood.
    residual = rhs - augmented @ mu
    return mu, R, jnp.dot(residual, residual), log_det_Sigma, n


def per_mode_evidence_qr(traj_blocks, mode_i, m, prior_prec, log_prior_cov,
                         deriv_is_diag):
    """Exact Gaussian evidence via whitened QR, without precision-matrix jitter."""
    mu, R, quadratic, log_det_Sigma, n = _qr_operator_system(
        traj_blocks, mode_i, m, prior_prec, deriv_is_diag)
    log_det_precision = 2 * jnp.sum(jnp.log(jnp.diag(R)))
    log_p = -0.5 * (quadratic + log_det_Sigma + log_prior_cov
                    + log_det_precision + n * jnp.log(2 * jnp.pi))
    return log_p, mu, R.T


def per_mode_posterior_qr(traj_blocks, mode_i, m, prior_prec, deriv_is_diag):
    """Return the Gaussian mean and covariance factor from the same QR system."""
    mu, R, _, _, _ = _qr_operator_system(
        traj_blocks, mode_i, m, prior_prec, deriv_is_diag)
    factor = jax.scipy.linalg.solve_triangular(R, jnp.eye(m), lower=False)
    return mu, factor


# -----------------------------------------------------------------------------
# Partial pooling, blocked: O_k = M + δ_k, eliminating each δ_k by its own QR
# -----------------------------------------------------------------------------
def _pooled_qr_system(traj_blocks, mode_i, m, prec_M, prec_delta, deriv_is_diag):
    """Mode i's partial-pooling system, trajectory k's whitened rows W_k (M + δ_k) ≈ z_k with the priors
    √prec_M M ≈ 0 and √prec_δ δ_k ≈ 0, factored with the unknowns ordered [δ_0, …, δ_K-1, M]: one QR of
    [[W_k, W_k, z_k], [√prec_δ, 0, 0]] per trajectory gives δ_k's triangle (R_kk, its coupling R_kM to M and
    right-hand side c_k), M's rows [T_k, t_k] and the residual ρ_k left once δ_k and M are free; one QR of all
    M rows with M's prior gives (R_MM, d_M, ρ_M). The full triangular factor is block upper triangular with
    diagonal blocks R_kk and R_MM, so this is the stacked system [M, δ_0, …] reordered (same Gaussian, same
    evidence) at a cost linear, not cubic, in the number of trajectories."""
    root_delta = jnp.diag(jnp.sqrt(prec_delta))
    local, rows_M = [], []
    quadratic = log_det_Sigma = log_det_precision = 0.0
    n = 0
    for A_D, y_D, Sigma_D, A_W, y_W, Sigma_W in traj_blocks:
        designs, observations = [], []
        for A, y, covariance, diagonal in (
                (A_D, y_D[mode_i], Sigma_D[mode_i], deriv_is_diag),
                (A_W, y_W[mode_i], Sigma_W[mode_i], False)):
            whitened, observed, log_det = _whiten_block(A, y, covariance, diagonal)
            designs.append(whitened)
            observations.append(observed)
            log_det_Sigma = log_det_Sigma + log_det
            n = n + len(observed)
        W, z = jnp.vstack(designs), jnp.concatenate(observations)
        system = jnp.vstack([jnp.hstack([W, W, z[:, None]]),
                             jnp.hstack([root_delta, jnp.zeros((m, m + 1), dtype=W.dtype)])])
        R = jnp.linalg.qr(system, mode="r")
        local.append((R[:m, :m], R[:m, m:2 * m], R[:m, 2 * m]))
        rows_M.append(R[m:2 * m, m:])
        # The last row is [0, 0, ρ_k]: the part of z_k that no choice of δ_k and M fits.
        quadratic = quadratic + R[2 * m, 2 * m] ** 2
        log_det_precision = log_det_precision + 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(R[:m, :m]))))
    rows_M.append(jnp.hstack([jnp.diag(jnp.sqrt(prec_M)), jnp.zeros((m, 1), dtype=root_delta.dtype)]))
    R = jnp.linalg.qr(jnp.vstack(rows_M), mode="r")
    log_det_precision = log_det_precision + 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(R[:m, :m]))))
    quadratic = quadratic + R[m, m] ** 2
    return local, (R[:m, :m], R[:m, m]), quadratic, log_det_precision, log_det_Sigma, n


def pooled_evidence_qr(traj_blocks, mode_i, m, prec_M, prec_delta, log_prior_cov, deriv_is_diag):
    """log p(y_i | θ, κ_M, κ_δ) of partial pooling for mode i (``traj_blocks`` as for the shared operator, one
    m-column design per trajectory; ``log_prior_cov`` = log|Σ_M| + K log|Σ_δ|), by the blocked QR."""
    _, _, quadratic, log_det_precision, log_det_Sigma, n = _pooled_qr_system(
        traj_blocks, mode_i, m, prec_M, prec_delta, deriv_is_diag)
    return -0.5 * (quadratic + log_det_Sigma + log_prior_cov + log_det_precision + n * jnp.log(2 * jnp.pi))


def pooled_posterior_qr(traj_blocks, mode_i, m, prec_M, prec_delta, deriv_is_diag, which):
    """Posterior mean and square (m × m) covariance factor of M (``which="mean"``) or of trajectory ``which``'s
    operator M + δ_which (an integer, possibly traced), from the blocked QR: μ_M = R_MM⁻¹ d_M,
    μ_δ = R_kk⁻¹ (c_k − R_kM μ_M), and the factor [R_kk⁻¹, (I − R_kk⁻¹ R_kM) R_MM⁻¹] of M + δ_k reduced to a
    square root by QR."""
    local, (R_MM, d_M), *_ = _pooled_qr_system(traj_blocks, mode_i, m, prec_M, prec_delta, deriv_is_diag)
    solve = jax.scipy.linalg.solve_triangular
    eye = jnp.eye(m, dtype=R_MM.dtype)
    mu_M = solve(R_MM, d_M, lower=False)
    inverse_M = solve(R_MM, eye, lower=False)
    if isinstance(which, str):
        mean, factor = mu_M, inverse_M
    else:
        R_kk, R_kM, c_k = (jnp.stack(part)[which] for part in zip(*local))
        inverse_k = solve(R_kk, eye, lower=False)
        mean = mu_M + solve(R_kk, c_k - R_kM @ mu_M, lower=False)
        factor = jnp.hstack([inverse_k, inverse_M - inverse_k @ (R_kM @ inverse_M)])
    _, R = jnp.linalg.qr(factor.T, mode="reduced")
    return mean, R.T

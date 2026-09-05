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

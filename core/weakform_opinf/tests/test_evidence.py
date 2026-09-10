"""Gaussian operator identities, including weak and badly scaled directions."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

from core.weakform_opinf.evidence import (
    per_mode_evidence, per_mode_evidence_qr, per_mode_posterior_qr,
)


class EvidenceTests(unittest.TestCase):
    def check_gaussian(self, diagonal, scales, trajectories=1):
        rng = np.random.default_rng(42)
        m = len(scales)
        prior = np.array([1., 1e-4, 1e-4])
        blocks, designs, observations, covariances = [], [], [], []
        for _ in range(trajectories):
            A = rng.normal(size=(24, m)) * scales
            y = A @ np.array([.001, 2., 10.]) + rng.normal(scale=.1, size=24)
            W = rng.normal(size=(5, m)) * scales
            w = rng.normal(size=5)
            raw = rng.normal(size=(24, 24))
            D = np.eye(24) if diagonal else raw @ raw.T + np.eye(24)
            raw = rng.normal(size=(5, 5))
            S = raw @ raw.T + np.eye(5)
            blocks.append(tuple(jnp.asarray(x) for x in (
                A, y[None], np.ones((1, 24)) if diagonal else D[None],
                W, w[None], S[None])))
            designs.extend([A, W])
            observations.extend([y, w])
            covariances.extend([
                D if diagonal else D + 1e-8 * np.eye(24),
                S + 1e-8 * np.eye(5)])
        A = np.vstack(designs)
        y = np.concatenate(observations)
        noise = block_diag(*covariances)
        cf = cho_factor(noise, lower=True)
        precision = np.diag(prior) + A.T @ cho_solve(cf, A)
        expected_cov = np.linalg.inv(precision)
        expected_mu = expected_cov @ A.T @ cho_solve(cf, y)
        marginal = noise + (A / prior) @ A.T
        cf_marginal = cho_factor(marginal, lower=True)
        expected_logp = -.5 * (
            y @ cho_solve(cf_marginal, y)
            + 2 * np.log(np.diag(cf_marginal[0])).sum()
            + len(y) * np.log(2 * np.pi))
        args = (blocks, 0, m, jnp.asarray(prior))
        logp, mu, _ = per_mode_evidence_qr(
            *args, -jnp.log(prior).sum(), diagonal)
        posterior_mu, factor = per_mode_posterior_qr(*args, diagonal)
        np.testing.assert_allclose(mu, expected_mu, rtol=2e-7, atol=1e-8)
        np.testing.assert_allclose(posterior_mu, mu, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(factor @ factor.T, expected_cov,
                                   rtol=2e-7, atol=1e-8)
        np.testing.assert_allclose(logp, expected_logp, rtol=2e-7, atol=1e-6)
        return blocks, prior, float(logp)

    def test_diagonal_and_dense_multitrajectory_gaussians(self):
        with jax.experimental.enable_x64():
            for diagonal in (True, False):
                for trajectories in (1, 2):
                    with self.subTest(diagonal=diagonal, trajectories=trajectories):
                        self.check_gaussian(diagonal, np.array([1., 2., .5]),
                                            trajectories)

    def test_scaled_columns_do_not_add_an_operator_prior(self):
        with jax.experimental.enable_x64():
            blocks, prior, exact = self.check_gaussian(True, np.array([1000., 1., .01]))
            historical, _, _ = per_mode_evidence(
                blocks, 0, 3, jnp.asarray(prior), -jnp.log(prior).sum(), True)
            self.assertGreater(abs(float(historical) - exact), 1.)

    def test_qr_evidence_gradient(self):
        with jax.experimental.enable_x64():
            blocks, prior, _ = self.check_gaussian(True, np.array([10., 1., .1]))
            def objective(log_scale):
                precision = jnp.asarray(prior) * jnp.exp(-2 * log_scale)
                return per_mode_evidence_qr(
                    blocks, 0, 3, precision, -jnp.log(precision).sum(), True)[0]
            epsilon = 1e-4
            finite_difference = (objective(epsilon) - objective(-epsilon)) / (2 * epsilon)
            np.testing.assert_allclose(jax.grad(objective)(0.), finite_difference,
                                       rtol=1e-6, atol=1e-6)

    def test_single_precision_scaled_columns(self):
        with jax.experimental.enable_x64():
            blocks, prior, expected_logp = self.check_gaussian(
                True, np.array([1000., 1., .01]))
            expected_mu, expected_factor = per_mode_posterior_qr(
                blocks, 0, 3, jnp.asarray(prior), True)
            expected_mu = np.asarray(expected_mu)
            expected_cov = np.asarray(expected_factor @ expected_factor.T)
        with jax.experimental.enable_x64(False):
            blocks = [tuple(jnp.asarray(x, dtype=jnp.float32) for x in block)
                      for block in blocks]
            prior = jnp.asarray(prior, dtype=jnp.float32)
            logp, mu, _ = per_mode_evidence_qr(
                blocks, 0, 3, prior, -jnp.log(prior).sum(), True)
            _, factor = per_mode_posterior_qr(blocks, 0, 3, prior, True)
            self.assertEqual(mu.dtype, jnp.float32)
            np.testing.assert_allclose(mu, expected_mu, rtol=3e-4, atol=1e-5)
            np.testing.assert_allclose(factor @ factor.T, expected_cov,
                                       rtol=3e-4, atol=1e-5)
            np.testing.assert_allclose(logp, expected_logp, rtol=3e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()

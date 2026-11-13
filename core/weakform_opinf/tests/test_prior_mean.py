"""A non-zero operator prior mean M: O_i ~ N(M_i, Σ_0) instead of N(0, Σ_0)."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from core.weakform_opinf import pipeline
from core.weakform_opinf.config import WeakFormConfig
from core.weakform_opinf.model import _block_id_from_rom, build_model, nondimensional_column_scale


def _rom():
    """Two-mode cAH ROM (6 columns in three blocks)."""
    def assemble(states, inputs=None):
        q1, q2 = states
        return jnp.column_stack([jnp.ones(states.shape[1]), q1, q2, q1 * q1, q1 * q2, q2 * q2])
    operators = [SimpleNamespace(entries=np.zeros(shape)) for shape in ((2,), (2, 2), (2, 3))]
    return SimpleNamespace(model=SimpleNamespace(
        operators=operators, operator_matrix=np.zeros((2, 6)), _assemble_data_matrix=assemble))


class PriorMeanTests(unittest.TestCase):
    def setUp(self):
        t = np.linspace(2., 6., 15)
        self.trajectory = dict(t_sampled=t, snapshots_comp=np.stack([np.sin(t) + 2., np.cos(t)]))
        self.cfg = WeakFormConfig(num_modes=2, num_eval_points=40, window_size=5, precision="float64")
        self.theta = np.array([.7, .9]), np.array([1.5, .8]), np.array([1e-3, 2e-3])
        self.tau = np.array([.5, 2., 1.5])
        self.mean = np.random.default_rng(3).normal(size=(2, 6))

    def _evidence(self, model):
        values = {f"{kind}_0_{i}": self.theta[j][i]
                  for j, kind in enumerate(("lengthscale", "variance", "noise")) for i in range(2)}
        values["log_tau_block"] = jnp.log(jnp.asarray(self.tau))
        trace = handlers.trace(handlers.substitute(model, data=values)).get_trace()
        return float(trace["marg_O_evidence"]["fn"].log_factor)

    def _posterior(self, posterior):
        hypers = tuple(jnp.asarray(x)[None] for x in self.theta)
        mean, root = posterior(hypers, None, None, jnp.asarray(self.tau))
        return np.asarray(mean), np.asarray(root)

    def test_zero_mean_reproduces_the_default(self):
        with pipeline.precision_context(self.cfg):
            default = build_model(_rom(), [self.trajectory], self.cfg)
            zero = build_model(_rom(), [self.trajectory], self.cfg, prior_mean=np.zeros((2, 6)))
            self.assertEqual(self._evidence(zero[0]), self._evidence(default[0]))
            for a, b in zip(self._posterior(zero[1]), self._posterior(default[1])):
                np.testing.assert_array_equal(a, b)
            self.assertNotIn("mean", default[3]["operator_prior"])
            self.assertEqual(zero[3]["operator_prior"]["mean"]["rule"], "given")

    def test_posterior_and_evidence_are_the_shifted_gaussians(self):
        """With prior precision P and zero-mean posterior N(mu0, L^-1): mu_M = mu0 + L^-1 P M and
        log p_M(y) - log p_0(y) = M'P mu0 - (M'PM - M'P L^-1 P M)/2 per mode (Woodbury)."""
        with pipeline.precision_context(self.cfg):
            zero = build_model(_rom(), [self.trajectory], self.cfg)
            given = build_model(_rom(), [self.trajectory], self.cfg, prior_mean=self.mean)
            mu0, root0 = self._posterior(zero[1])
            mu, root = self._posterior(given[1])
            difference = self._evidence(given[0]) - self._evidence(zero[0])
        np.testing.assert_allclose(root, root0, rtol=1e-12, atol=1e-14)
        block_id, _, _ = _block_id_from_rom(_rom())
        scale, _ = nondimensional_column_scale(_rom(), [self.trajectory], 2)
        P = 1. / (self.tau[block_id] * scale) ** 2
        expected = 0.
        for i in range(2):
            covariance = root0[i] @ root0[i].T
            shift = covariance @ (P * self.mean[i])
            np.testing.assert_allclose(mu[i], mu0[i] + shift, rtol=1e-8, atol=1e-10)
            M = self.mean[i]
            expected += M @ (P * mu0[i]) - .5 * (M @ (P * M) - (P * M) @ covariance @ (P * M))
        self.assertAlmostEqual(difference, expected, delta=1e-7 * max(1., abs(expected)))

    def test_normal_equations_agree(self):
        """The normal-equation solver (with its small precision jitter) applies the same shift."""
        cfg = replace(self.cfg, operator_solver="normal")
        with pipeline.precision_context(cfg):
            normal = self._posterior(build_model(_rom(), [self.trajectory], cfg, prior_mean=self.mean)[1])
            qr = self._posterior(build_model(_rom(), [self.trajectory], self.cfg, prior_mean=self.mean)[1])
        np.testing.assert_allclose(normal[0], qr[0], rtol=2e-3)

    def test_invalid_means_are_rejected(self):
        with pipeline.precision_context(self.cfg):
            for mean in (np.zeros((2, 5)), np.full((2, 6), np.nan)):
                with self.assertRaises(ValueError):
                    build_model(_rom(), [self.trajectory], self.cfg, prior_mean=mean)
            with self.assertRaises(ValueError):
                build_model(_rom(), [self.trajectory], self.cfg, row_maps=np.stack([np.eye(6)] * 2),
                            prior_mean=np.zeros((2, 6)))

    def test_given_covariance_posterior_and_evidence(self):
        cfg = replace(self.cfg, closure="slack", op_prior_mode="fixed")
        roots = np.random.default_rng(4).normal(size=(2, 6, 6)) * .2 + np.eye(6)[None]
        hypers = tuple(jnp.asarray(x)[None] for x in self.theta)
        with pipeline.precision_context(cfg):
            reference = build_model(_rom(), [self.trajectory], replace(cfg, sigma_O=1.))
            given = build_model(_rom(), [self.trajectory], cfg, prior_mean=self.mean, prior_root=roots)
            mu0, R0 = map(np.asarray, reference[1](hypers, None, 1.))
            mu, R = map(np.asarray, given[1](hypers, None, None))
            difference = self._evidence(given[0]) - self._evidence(reference[0])
        expected = 0.
        for i in range(2):
            L0 = np.linalg.inv(R0[i] @ R0[i].T)
            P = np.linalg.inv(roots[i] @ roots[i].T)
            L = L0 - np.eye(6) + P
            cov = np.linalg.inv(L)
            mean = cov @ (L0 @ mu0[i] + P @ self.mean[i])
            np.testing.assert_allclose(mu[i], mean, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(R[i] @ R[i].T, cov, rtol=1e-8, atol=1e-10)
            expected += .5 * (mean @ L @ mean - mu0[i] @ L0 @ mu0[i]
                              - self.mean[i] @ P @ self.mean[i] - np.linalg.slogdet(L)[1]
                              + np.linalg.slogdet(L0)[1] + np.linalg.slogdet(P)[1])
        self.assertAlmostEqual(difference, expected, delta=1e-7 * max(1., abs(expected)))

    def test_given_covariance_rejects_singular_or_incompatible_inputs(self):
        cfg = replace(self.cfg, closure="slack", op_prior_mode="fixed")
        with pipeline.precision_context(cfg):
            for root in (np.zeros((2, 6, 6)), np.ones((2, 6, 5)), np.full((2, 6, 6), np.nan)):
                with self.assertRaisesRegex(ValueError, "prior_root"):
                    build_model(_rom(), [self.trajectory], cfg, prior_root=root)
            with self.assertRaisesRegex(ValueError, "prior_root"):
                build_model(_rom(), [self.trajectory], replace(cfg, sigma_O=1.),
                            prior_root=np.stack([np.eye(6)] * 2))

    def test_hierarchical_root_scales_latent_columns_in_evidence_and_posterior(self):
        cfg = replace(self.cfg, closure="slack")
        roots = np.random.default_rng(4).normal(size=(2, 6, 6)) * .2 + np.eye(6)[None]
        blocks, _, _ = _block_id_from_rom(_rom())
        hypers = tuple(jnp.asarray(x)[None] for x in self.theta)
        with pipeline.precision_context(cfg):
            hierarchical = build_model(_rom(), [self.trajectory], cfg,
                                       prior_mean=self.mean, prior_root=roots)
            for tau in (np.ones(3), self.tau, 2 * self.tau):
                self.tau = tau
                fixed = build_model(_rom(), [self.trajectory], replace(cfg, op_prior_mode="fixed"),
                                    prior_mean=self.mean, prior_root=roots * tau[blocks][None, None, :])
                self.assertAlmostEqual(self._evidence(hierarchical[0]), self._evidence(fixed[0]), places=8)
                mu, root = map(np.asarray, hierarchical[1](hypers, None, None, jnp.asarray(tau)))
                expected, expected_root = map(np.asarray, fixed[1](hypers, None, None))
                np.testing.assert_allclose(mu, expected, rtol=1e-8, atol=1e-10)
                np.testing.assert_allclose(root @ root.transpose(0, 2, 1),
                                           expected_root @ expected_root.transpose(0, 2, 1),
                                           rtol=1e-8, atol=1e-10)

    def test_diagonal_root_reproduces_native_hierarchical_scales(self):
        cfg = replace(self.cfg, closure="slack")
        scales, _ = nondimensional_column_scale(_rom(), [self.trajectory], 2)
        with pipeline.precision_context(cfg):
            native = build_model(_rom(), [self.trajectory], cfg, prior_mean=self.mean)
            supplied = build_model(_rom(), [self.trajectory], cfg, prior_mean=self.mean,
                                   prior_root=np.stack([np.diag(scales)] * 2))
            self.assertAlmostEqual(self._evidence(native[0]), self._evidence(supplied[0]), places=8)
            (mu, root), (expected, expected_root) = self._posterior(supplied[1]), self._posterior(native[1])
            np.testing.assert_allclose(mu, expected, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(root @ root.transpose(0, 2, 1),
                                       expected_root @ expected_root.transpose(0, 2, 1),
                                       rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

"""Generic contraction geometry, units, and Gaussian algebra."""

import unittest
from types import SimpleNamespace

import jax.experimental
import jax.numpy as jnp
import numpy as np

from core.weakform_opinf.stability import local_contraction_prior


def rom(with_inputs=False):
    def features(q, inputs=None):
        x, y = q
        columns = [jnp.ones(len(x)), x, y, x * x, x * y, y * y]
        if with_inputs:
            a, b = inputs
            columns += [a, b, a * x, a * y, b * x, b * y]
        return jnp.column_stack(columns)
    return SimpleNamespace(model=SimpleNamespace(
        operator_matrix=np.zeros((2, 12 if with_inputs else 6)), _assemble_data_matrix=features))


class LocalContractionTests(unittest.TestCase):
    def setUp(self):
        context = jax.experimental.enable_x64(True)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        t = np.linspace(0, 1, 20)
        self.X = np.array([np.sin(t), np.cos(t)])
        self.dz = np.array([np.cos(t), -np.sin(t)])
        self.amp = np.array([.3, .7])

    def test_multi_input_moments_match_independent_dense_gaussian(self):
        model = rom(True)
        u = np.array([np.linspace(.2, 1, 20), np.linspace(-1, .4, 20)])
        scale = np.linspace(.4, 1.3, 12)
        mean, roots, info = local_contraction_prior(
            model, [self.X], [self.dz], [u], self.amp, scale, 2., damping=5., width=3.)
        blocks, targets = [], []
        for _, j in info["anchors"]:
            x, y = self.X[:, j]
            a, b = u[:, j]
            Fq = np.array([[0, 0], [1, 0], [0, 1], [2*x, 0], [y, x], [0, 2*y],
                           [0, 0], [0, 0], [a, 0], [0, a], [b, 0], [0, b]])
            v = self.dz[:, j] / self.amp
            v /= np.linalg.norm(v)
            P = np.eye(2) - np.outer(v, v)
            blocks.append(2 * (Fq * self.amp) @ P)
            targets.append(-5 * P)
        A, targets = np.hstack(blocks).T / (3*np.sqrt(len(blocks))), np.hstack(targets) / (3*np.sqrt(len(blocks)))
        for i in range(2):
            B = A / self.amp[i]
            covariance = np.linalg.inv(np.diag(scale ** -2) + B.T @ B)
            np.testing.assert_allclose(mean[i], covariance @ B.T @ targets[i], atol=1e-12)
            np.testing.assert_allclose(roots[i] @ roots[i].T, covariance, atol=1e-12)

    def test_time_units_and_duplicate_anchors_preserve_prior(self):
        args = (rom(), [self.X], [self.dz], [None], self.amp, np.ones(6), 1.)
        mean, roots, _ = local_contraction_prior(*args)
        slower = local_contraction_prior(rom(), [self.X], [self.dz / 7], [None],
                                         self.amp, np.ones(6) / 7, 7.)
        duplicate = local_contraction_prior(rom(), [self.X]*2, [self.dz]*2, [None]*2,
                                            self.amp, np.ones(6), 1.)
        np.testing.assert_allclose(slower[0] * 7, mean, atol=1e-12)
        np.testing.assert_allclose(duplicate[0], mean, atol=1e-12)
        for i in range(2):
            covariance = roots[i] @ roots[i].T
            np.testing.assert_allclose(slower[1][i] @ slower[1][i].T * 49, covariance, atol=1e-12)
            np.testing.assert_allclose(duplicate[1][i] @ duplicate[1][i].T, covariance, atol=1e-12)

    def test_stationary_anchors_are_reported_not_given_an_arbitrary_direction(self):
        dz = self.dz.copy()
        dz[:, :10] = 0
        _, _, info = local_contraction_prior(rom(), [self.X], [dz], [None], self.amp, np.ones(6), 1.)
        self.assertGreater(info["skipped_stationary_anchors"], 0)
        self.assertTrue(all(j >= 10 for _, j in info["anchors"]))
        with self.assertRaisesRegex(ValueError, "No resolved training tangents"):
            local_contraction_prior(rom(), [self.X], [0*dz], [None], self.amp, np.ones(6), 1.)

    def test_invalid_scales_and_inputs_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "finite positive"):
            local_contraction_prior(rom(), [self.X], [self.dz], [None], [0, 1], np.ones(6), 1.)
        with self.assertRaisesRegex(ValueError, "aligned"):
            local_contraction_prior(rom(True), [self.X], [self.dz], [np.ones((2, 19))],
                                     self.amp, np.ones(12), 1.)


if __name__ == "__main__":
    unittest.main()

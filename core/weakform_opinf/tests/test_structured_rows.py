"""Known structure of the operator rows: O_i = P_i θ_i with the prior and posterior over θ_i."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from core.weakform_opinf import pipeline
from core.weakform_opinf.config import WeakFormConfig
from core.weakform_opinf.model import build_model, data_scales


def _rom(constant=True, with_inputs=False):
    """Two-mode cAH (or AH) ROM, plus an N block (q * u) when with_inputs."""
    def assemble(states, inputs=None):
        q1, q2 = states
        columns = ([jnp.ones(states.shape[1])] if constant else []) + [q1, q2, q1 * q1, q1 * q2, q2 * q2]
        if with_inputs:
            columns += [q1 * inputs[0], q2 * inputs[0]]
        return jnp.column_stack(columns)
    shapes = ([(2,)] if constant else []) + [(2, 2), (2, 3)] + ([(2, 2)] if with_inputs else [])
    operators = [SimpleNamespace(entries=np.zeros(shape)) for shape in shapes]
    width = sum(1 if len(s) == 1 else s[1] for s in shapes)
    return SimpleNamespace(model=SimpleNamespace(
        operators=operators, operator_matrix=np.zeros((2, width)), _assemble_data_matrix=assemble))


def _select(columns, width=6):
    P = np.zeros((width, len(columns)))
    P[columns, np.arange(len(columns))] = 1.
    return P


class StructuredRowTests(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(2., 6., 15)
        self.y = np.stack([np.sin(self.t) + 2., np.cos(self.t)])
        self.trajectory = dict(t_sampled=self.t, snapshots_comp=self.y)
        self.cfg = WeakFormConfig(num_modes=2, num_eval_points=40, window_size=5, precision="float64")
        self.theta = np.array([.7, .9]), np.array([1.5, .8]), np.array([1e-3, 2e-3])

    def _evidence(self, model, tau):
        values = {f"{kind}_0_{i}": self.theta[j][i]
                  for j, kind in enumerate(("lengthscale", "variance", "noise")) for i in range(2)}
        values["log_tau_block"] = jnp.log(jnp.asarray(tau))
        trace = handlers.trace(handlers.substitute(model, data=values)).get_trace()
        return float(trace["marg_O_evidence"]["fn"].log_factor)

    def _posterior(self, posterior, tau):
        hypers = tuple(jnp.asarray(x)[None] for x in self.theta)
        mean, root = posterior(hypers, None, None, jnp.asarray(tau))
        return np.asarray(mean), np.asarray(root)

    def test_identity_maps_reproduce_the_free_operator(self):
        tau = [.5, 2., 1.5]
        with pipeline.precision_context(self.cfg):
            free = build_model(_rom(), [self.trajectory], self.cfg)
            same = build_model(_rom(), [self.trajectory], self.cfg, row_maps=np.stack([np.eye(6)] * 2))
            self.assertAlmostEqual(self._evidence(same[0], tau), self._evidence(free[0], tau), places=8)
            for a, b in zip(self._posterior(same[1], tau), self._posterior(free[1], tau)):
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)

    def test_structural_zeros_match_the_reduced_operator(self):
        """Dropping the constant column by a row map is the AH model with the same column priors."""
        tau = [3., .5, 2.]
        with pipeline.precision_context(self.cfg):
            masked = build_model(_rom(), [self.trajectory], self.cfg,
                                 row_maps=np.stack([_select([1, 2, 3, 4, 5])] * 2))
            reduced = build_model(_rom(constant=False), [self.trajectory], self.cfg)
            self.assertAlmostEqual(self._evidence(masked[0], tau), self._evidence(reduced[0], tau[1:]), places=8)
            mean, root = self._posterior(masked[1], tau)
            mean_r, root_r = self._posterior(reduced[1], tau[1:])
            self.assertTrue(np.all(mean[:, 0] == 0.) and np.all(root[:, 0] == 0.))
            np.testing.assert_allclose(mean[:, 1:], mean_r, rtol=1e-9)
            covariance = np.einsum("ijk,ilk->ijl", root, root)[:, 1:, 1:]
            np.testing.assert_allclose(covariance, np.einsum("ijk,ilk->ijl", root_r, root_r), rtol=1e-8)

    def test_combined_column_has_a_fixed_shape_and_its_own_scale(self):
        weights = np.array([[1., -2., .5], [.3, 1., -1.]])
        maps = np.zeros((2, 6, 3))
        for i in range(2):
            maps[i, 1:3, :2] = np.eye(2)
            maps[i, 3:, 2] = weights[i]
        with pipeline.precision_context(self.cfg):
            model, posterior, _, priors = build_model(_rom(), [self.trajectory], self.cfg, row_maps=maps)
            info = priors["operator_prior"]["structured_rows"]
            T, S = data_scales([self.trajectory])
            Q = np.stack([self.y[0] ** 2, self.y[0] * self.y[1], self.y[1] ** 2], axis=1)
            rms = np.sqrt(np.mean([np.mean((Q @ w) ** 2) for w in weights]))
            np.testing.assert_allclose(np.asarray(info["scale"])[:, 2], S / (T * rms), rtol=1e-12)
            np.testing.assert_allclose(np.asarray(info["scale"])[:, :2], 1 / T, rtol=1e-12)
            self.assertEqual(info["block"], [[1, 1, 2], [1, 1, 2]])
            mean, root = self._posterior(posterior, [1., 1., 1.])
            draw = mean + np.einsum("ijk,ik->ij", root, np.random.default_rng(0).standard_normal((2, 6)))
            for O in (mean, draw):
                self.assertTrue(np.all(O[:, 0] == 0.))
                rate = O[:, 3] / weights[:, 0]
                np.testing.assert_allclose(O[:, 3:], rate[:, None] * weights, rtol=1e-10)

    def test_invalid_maps_are_rejected(self):
        with pipeline.precision_context(self.cfg):
            cases = {
                "shape": np.zeros((2, 5, 2)),
                "full column rank": np.stack([_select([1, 1])] * 2),
                "several operator blocks": np.stack([_select([1]) + _select([3])] * 2),
            }
            for message, maps in cases.items():
                with self.assertRaisesRegex(ValueError, message):
                    build_model(_rom(), [self.trajectory], self.cfg, row_maps=maps)
            inputs = dict(self.trajectory, inputs_eval=np.ones((1, 40)))
            mixed = np.zeros((2, 8, 1))
            mixed[:, 6:, 0] = 1.
            with self.assertRaisesRegex(ValueError, "state-only"):
                build_model(_rom(with_inputs=True), [inputs], self.cfg, row_maps=mixed)
            with self.assertRaisesRegex(ValueError, "nondimensional"):
                build_model(_rom(), [self.trajectory], replace(self.cfg, sigma_O=1.),
                            row_maps=np.stack([np.eye(6)] * 2))


if __name__ == "__main__":
    unittest.main()

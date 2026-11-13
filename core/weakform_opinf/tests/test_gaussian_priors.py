"""Scaling, coordinate transforms and feature-support Gaussian priors."""

import unittest

import jax.experimental
import numpy as np

from core.weakform_opinf.constraints import _features
from core.weakform_opinf.gaussian_priors import training_scaled_prior
from core.weakform_opinf.tests.test_stability_prior import _cahbn_rom


class TrainingScaledPriorTests(unittest.TestCase):
    def setUp(self):
        context = jax.experimental.enable_x64(True)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.rom = _cahbn_rom(2)
        t = np.linspace(0, 3, 31)
        self.q = np.stack([2 + np.sin(t), .2 * np.cos(2*t)])
        self.dq = np.stack([np.cos(t), -.4 * np.sin(2*t)])
        self.u = np.sin(t)[None] + 1
        self.tr = dict(t_sampled=t, snapshots_comp=self.q, inputs_eval=self.u)

    def prior(self, **kwargs):
        return training_scaled_prior(self.rom, [self.tr], [self.q], [self.dq], **kwargs)

    def test_rate_prior_balances_each_block_at_training_states(self):
        _, root, info = self.prior()
        F = _features(self.rom, self.q, self.u)
        start = 0
        for count in info["block_counts"]:
            for i in range(2):
                contribution = F[:, start:start+count] @ root[i, start:start+count, start:start+count]
                self.assertAlmostEqual(np.mean(np.sum(contribution**2, axis=1)),
                                       info["rate_scale"][i]**2, places=12)
            start += count

    def test_centered_prior_and_damping_are_exact_coordinate_transforms(self):
        mean, root, info = self.prior(center=True, damping=1.5)
        offset = np.asarray(info["offset"])
        shifted = dict(self.tr, snapshots_comp=self.q-offset[:, None])
        m0, R0, _ = training_scaled_prior(self.rom, [shifted], [self.q-offset[:, None]],
                                         [self.dq], damping=1.5)
        F = _features(self.rom, self.q, self.u)
        F0 = _features(self.rom, self.q-offset[:, None], self.u)
        np.testing.assert_allclose(F @ mean.T, -(self.q-offset[:, None]).T * .5, atol=1e-12)
        np.testing.assert_allclose(F @ mean.T, F0 @ m0.T, atol=1e-12)
        for i in range(2):
            np.testing.assert_allclose(F @ root[i], F0 @ R0[i], atol=1e-12)

    def test_state_input_and_time_unit_changes_preserve_prior_predictive_law(self):
        units, input_unit, clock = np.array([3., .2]), 7., 4.
        tr = dict(t_sampled=self.tr["t_sampled"] * clock,
                  snapshots_comp=self.q * units[:, None], inputs_eval=self.u * input_unit)
        for correlation in ("none", "block", "full"):
            mean, root, _ = self.prior(center=True, damping=1., correlation=correlation)
            changed, R, _ = training_scaled_prior(
                self.rom, [tr], [self.q * units[:, None]], [self.dq * units[:, None] / clock],
                center=True, damping=1., correlation=correlation)
            F = _features(self.rom, self.q, self.u)
            G = _features(self.rom, tr["snapshots_comp"], tr["inputs_eval"])
            np.testing.assert_allclose(G @ changed.T, (F @ mean.T) * units / clock, atol=1e-10)
            for i in range(2):
                np.testing.assert_allclose(G @ R[i], (F @ root[i]) * units[i] / clock, atol=1e-9)

    def test_full_support_covariance_shrinks_an_unexcited_combination(self):
        q = np.stack([self.q[0], self.q[0]])
        dq = np.stack([self.dq[0], self.dq[0]])
        tr = dict(self.tr, snapshots_comp=q)
        _, plain, _ = training_scaled_prior(self.rom, [tr], [q], [dq])
        _, supported, _ = training_scaled_prior(self.rom, [tr], [q], [dq],
                                                 correlation="full", ridge=.01)
        direction = np.zeros(9)
        direction[1:3] = [1., -1.]
        for i in range(2):
            ratio = np.sum((direction @ supported[i])**2) / np.sum((direction @ plain[i])**2)
            self.assertAlmostEqual(ratio, .01, places=10)

    def test_rate_clock_mean_uses_each_modes_observed_rate_scale(self):
        mean, _, info = self.prior(center=True, damping=.3, mean_clock="rate")
        expected = -.3 * (np.asarray(info["rate_scale"]) / info["amplitude"])[:, None] * (
            self.q - np.asarray(info["offset"])[:, None])
        np.testing.assert_allclose(_features(self.rom, self.q, self.u) @ mean.T, expected.T, atol=1e-12)
        scaled = dict(self.tr, t_sampled=self.tr["t_sampled"] * 4)
        changed, _, _ = training_scaled_prior(
            self.rom, [scaled], [self.q], [self.dq / 4], center=True, damping=.3, mean_clock="rate")
        np.testing.assert_allclose(changed, mean / 4, atol=1e-12)

    def test_invalid_or_unexcited_inputs_raise(self):
        for args in (dict(scale=0), dict(ridge=0), dict(damping=-1), dict(normalization="unknown")):
            with self.assertRaises(ValueError):
                self.prior(**args)
        with self.assertRaises(ValueError):
            training_scaled_prior(self.rom, [dict(self.tr, inputs_eval=np.zeros_like(self.u))],
                                  [self.q], [self.dq])


if __name__ == "__main__":
    unittest.main()

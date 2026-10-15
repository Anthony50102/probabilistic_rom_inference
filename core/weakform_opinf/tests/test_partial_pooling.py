"""Partial pooling: trajectory k's operator O_k = M + δ_k, M ~ N(0, (κ_M s)²), δ_k ~ N(0, (κ_δ s)²)."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from core.weakform_opinf import pipeline
from core.weakform_opinf.config import WeakFormConfig
from core.weakform_opinf.model import _block_id_from_rom, build_model, nondimensional_column_scale

KINDS = ("lengthscale", "variance", "noise")


def _rom():
    """Two-mode cAH ROM (6 columns in three blocks)."""
    def assemble(states, inputs=None):
        q1, q2 = states
        return jnp.column_stack([jnp.ones(states.shape[1]), q1, q2, q1 * q1, q1 * q2, q2 * q2])
    operators = [SimpleNamespace(entries=np.zeros(shape)) for shape in ((2,), (2, 2), (2, 3))]
    return SimpleNamespace(model=SimpleNamespace(
        operators=operators, operator_matrix=np.zeros((2, 6)), _assemble_data_matrix=assemble))


def _trajectories():
    t0, t1 = np.linspace(2., 6., 15), np.linspace(1., 7., 18)
    return [dict(t_sampled=t0, snapshots_comp=np.stack([np.sin(t0) + 2., np.cos(t0)])),
            dict(t_sampled=t1, snapshots_comp=np.stack([1.5 * np.cos(t1) + 2.5, .8 * np.sin(1.3 * t1)]))]


THETA = [(np.array([.7, .9]), np.array([1.5, .8]), np.array([1e-3, 2e-3])),
         (np.array([1.1, .6]), np.array([2.5, .4]), np.array([3e-3, 1e-3]))]


def _evidence(model, theta, tau):
    values = {f"{kind}_{ic}_{i}": th[j][i] for ic, th in enumerate(theta)
              for j, kind in enumerate(KINDS) for i in range(2)}
    values["log_tau_block"] = jnp.log(jnp.asarray(tau))
    trace = handlers.trace(handlers.substitute(model, data=values)).get_trace()
    return float(trace["marg_O_evidence"]["fn"].log_factor), trace["log_tau_block"]["value"].shape


def _posterior(posterior, theta, tau, **kwargs):
    hypers = tuple(jnp.stack([jnp.asarray(th[j]) for th in theta]) for j in range(3))
    mean, root = posterior(hypers, None, None, jnp.asarray(tau), **kwargs)
    return np.asarray(mean), np.asarray(root)


def _covariance(root):
    return np.einsum("ijk,ilk->ijl", root, root)


class PartialPoolingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = WeakFormConfig(num_modes=2, num_eval_points=40, window_size=5, precision="float64")
        self.kappa_M, self.kappa_d = np.array([.8, 1.7, .6]), np.array([.3, .5, 1.2])
        self.tau = np.r_[self.kappa_M, self.kappa_d]

    def test_one_trajectory_has_the_summed_prior(self):
        """With one trajectory O = M + δ ~ N(0, (κ_M² + κ_δ²) s²): the single fit with κ = √(κ_M² + κ_δ²)."""
        trajectory = _trajectories()[:1]
        with pipeline.precision_context(self.cfg):
            single = build_model(_rom(), trajectory, self.cfg)
            pooled = build_model(_rom(), trajectory, self.cfg, pooled=True)
            kappa = np.hypot(self.kappa_M, self.kappa_d)
            expected, shape = _evidence(single[0], THETA[:1], kappa)
            value, pooled_shape = _evidence(pooled[0], THETA[:1], self.tau)
            mean, root = _posterior(single[1], THETA[:1], kappa)
            pooled_mean, pooled_root = _posterior(pooled[1], THETA[:1], self.tau)
        self.assertEqual((shape, pooled_shape), ((3,), (6,)))
        self.assertAlmostEqual(value, expected, delta=1e-8 * abs(expected))
        np.testing.assert_allclose(pooled_mean, mean, rtol=1e-8, atol=1e-10)
        self.assertEqual(pooled_root.shape, root.shape)
        np.testing.assert_allclose(_covariance(pooled_root), _covariance(root), rtol=1e-8, atol=1e-12)
        self.assertEqual(pooled[3]["operator_prior"]["pooled"]["trajectories"], 1)

    def test_matches_the_brute_force_gaussian(self):
        """Two trajectories: each one's Gaussian likelihood in O (precision H_k, information b_k, constant c_k) is
        read off its own single-trajectory posterior and evidence; the pooled evidence and posteriors are then the
        Gaussian integrals over [M, δ_0, δ_1] computed directly (fixed slack, so the rows do not depend on which
        trajectories are fitted together)."""
        cfg = replace(self.cfg, closure="slack", gamma2=1e-3)
        trajectories = _trajectories()
        rom = _rom()
        block_id, m, _ = _block_id_from_rom(rom)
        H, b, constant = [], [], 0.
        with pipeline.precision_context(cfg):
            for k, trajectory in enumerate(trajectories):
                model, posterior, _, _ = build_model(rom, [trajectory], cfg)
                evidence, _ = _evidence(model, THETA[k:k + 1], np.ones(3))
                mean, root = _posterior(posterior, THETA[k:k + 1], np.ones(3))
                prior_precision = 1. / nondimensional_column_scale(rom, [trajectory], 2)[0] ** 2
                constant += evidence
                H.append([]), b.append([])
                for i in range(2):
                    inverse_root = np.linalg.inv(root[i])
                    precision = inverse_root.T @ inverse_root
                    H[k].append(precision - np.diag(prior_precision))
                    b[k].append(precision @ mean[i])
                    constant -= (.5 * np.sum(np.log(prior_precision)) - .5 * np.linalg.slogdet(precision)[1]
                                 + .5 * mean[i] @ precision @ mean[i])
            model, posterior, _, info = build_model(rom, trajectories, cfg, pooled=True)
            value, _ = _evidence(model, THETA, self.tau)
            pooled = {which: _posterior(posterior, THETA, self.tau, which=which) for which in ("mean", 0, 1)}
            np.testing.assert_array_equal(_posterior(posterior, THETA, self.tau)[0], pooled[0][0])
        scale = nondimensional_column_scale(rom, trajectories, 2)[0]
        prior_variance = np.concatenate([(self.kappa_M[block_id] * scale) ** 2]
                                        + [(self.kappa_d[block_id] * scale) ** 2] * 2)
        placement = [np.hstack([np.eye(m), np.eye(m) * (k == 0), np.eye(m) * (k == 1)]) for k in range(2)]
        selector = {"mean": np.hstack([np.eye(m), np.zeros((m, 2 * m))]), 0: placement[0], 1: placement[1]}
        expected = constant
        for i in range(2):
            precision = np.diag(1. / prior_variance) + sum(E.T @ H[k][i] @ E for k, E in enumerate(placement))
            information = sum(E.T @ b[k][i] for k, E in enumerate(placement))
            covariance = np.linalg.inv(precision)
            mean = covariance @ information
            expected += (-.5 * np.sum(np.log(prior_variance)) - .5 * np.linalg.slogdet(precision)[1]
                         + .5 * information @ mean)
            for which, S in selector.items():
                with self.subTest(mode=i, operator=which):
                    np.testing.assert_allclose(pooled[which][0][i], S @ mean, rtol=1e-7, atol=1e-9)
                    np.testing.assert_allclose(_covariance(pooled[which][1])[i], S @ covariance @ S.T,
                                               rtol=1e-6, atol=1e-10)
        self.assertAlmostEqual(value, expected, delta=1e-7 * max(1., abs(expected)))
        self.assertEqual(info["operator_prior"]["pooled"]["trajectories"], 2)
        self.assertEqual(info["num_traj"], 2)

    def test_no_spread_is_one_shared_operator(self):
        """κ_δ → 0 forces O_k = M: the shared-operator fit of both trajectories with κ = κ_M."""
        trajectories = _trajectories()
        tau = np.r_[self.kappa_M, np.full(3, 1e-6)]
        with pipeline.precision_context(self.cfg):
            shared = build_model(_rom(), trajectories, self.cfg)
            pooled = build_model(_rom(), trajectories, self.cfg, pooled=True)
            expected, _ = _evidence(shared[0], THETA, self.kappa_M)
            value, _ = _evidence(pooled[0], THETA, tau)
            mean, root = _posterior(shared[1], THETA, self.kappa_M)
            for which in ("mean", 0, 1):
                pooled_mean, pooled_root = _posterior(pooled[1], THETA, tau, which=which)
                np.testing.assert_allclose(pooled_mean, mean, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(_covariance(pooled_root), _covariance(root), rtol=1e-4, atol=1e-9)
        self.assertAlmostEqual(value, expected, delta=1e-5 * max(1., abs(expected)))

    def test_unsupported_combinations_are_rejected(self):
        trajectories = _trajectories()
        with pipeline.precision_context(self.cfg):
            for kwargs in (dict(prior_mean=np.zeros((2, 6))), dict(row_maps=np.stack([np.eye(6)] * 2))):
                with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                    build_model(_rom(), trajectories, self.cfg, pooled=True, **kwargs)
            for cfg in (replace(self.cfg, sigma_O=5.), replace(self.cfg, op_prior_mode="fixed")):
                with self.assertRaises(ValueError):
                    build_model(_rom(), trajectories, cfg, pooled=True)
            posterior = build_model(_rom(), trajectories, self.cfg, pooled=True)[1]
            for which in (2, -1, "target", True):
                with self.subTest(which=which), self.assertRaises(ValueError):
                    _posterior(posterior, THETA, self.tau, which=which)


if __name__ == "__main__":
    unittest.main()

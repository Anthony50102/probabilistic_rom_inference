"""Energy-preserving quadratic constraint and shifted operator features (quad_constraint / feature_shift)."""

import itertools
import unittest
from dataclasses import replace
from types import SimpleNamespace

import jax.experimental
import jax.numpy as jnp
import numpy as np
import opinf
from numpyro import handlers

from core.weakform_opinf import constraints, pipeline
from core.weakform_opinf.config import WeakFormConfig
from core.weakform_opinf.model import build_model


def _cah_rom(r, shift=None):
    """cAH ROM in opinf's column layout [1 | q | ckron(q)] (traceable); optionally fitted to features of q − shift."""
    pairs = [(j, i) for i in range(r) for j in range(i + 1)]
    m = 1 + r + len(pairs)

    def assemble(states, inputs=None):
        q = states if shift is None else states - jnp.asarray(shift)[:, None]
        return jnp.column_stack([jnp.ones(q.shape[1]), *q, *(q[j] * q[i] for j, i in pairs)])
    operators = [SimpleNamespace(entries=np.zeros(shape)) for shape in ((r,), (r, r), (r, len(pairs)))]
    return SimpleNamespace(model=SimpleNamespace(
        operators=operators, operator_matrix=np.zeros((r, m)), _assemble_data_matrix=assemble))


class _Float64:
    def setUp(self):
        context = jax.experimental.enable_x64(True)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)


def _dense_condition(mean, cov, C):
    S = C @ cov @ C.T
    gain = cov @ C.T @ np.linalg.inv(S)
    return mean - gain @ (C @ mean), cov - gain @ C @ cov


class EnergyPreservingMatrixTests(_Float64, unittest.TestCase):
    def test_null_space_conserves_energy_with_opinf_ckron(self):
        r = 4
        rom = _cah_rom(r)
        C = constraints.energy_preserving_matrix(rom, r)
        m = rom.model.operator_matrix.shape[1]
        self.assertEqual(C.shape, (len(list(itertools.combinations_with_replacement(range(r), 3))), r * m))
        self.assertEqual(np.linalg.matrix_rank(C), C.shape[0])
        rng = np.random.default_rng(0)
        x = rng.normal(size=r * m)
        x -= C.T @ np.linalg.solve(C @ C.T, C @ x)
        O = x.reshape(r, m)
        H = O[:, 1 + r:]
        free = rng.normal(size=(r, m))[:, 1 + r:]
        for q in rng.normal(size=(5, r)):
            kq = opinf.operators.QuadraticOperator.ckron(q)
            self.assertLess(abs(q @ H @ kq), 1e-12 * np.linalg.norm(q) ** 3 * np.abs(H).max())
            self.assertGreater(abs(q @ free @ kq), 1e-3)

    def test_shift_matrix_is_exact(self):
        r, shift = 3, np.array([1.3, -0.4, 0.7])
        rom = _cah_rom(r)
        G = constraints.shift_matrix(rom, r, shift)
        q = np.random.default_rng(1).normal(size=(r, 7))
        F = constraints._features(rom, q)
        F_shift = constraints._features(rom, q - shift[:, None])
        np.testing.assert_allclose(F @ G, F_shift, atol=1e-12)
        # quadratic columns map to themselves, so the constraint is the same in either coordinate system
        quad = 1 + r + np.arange(r * (r + 1) // 2)
        np.testing.assert_allclose(G[np.ix_(quad, quad)], np.eye(len(quad)), atol=1e-12)
        np.testing.assert_allclose(G[np.ix_(quad, np.arange(1 + r))], 0.0, atol=1e-12)


class ConditioningTests(_Float64, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.r = 3
        self.rom = _cah_rom(self.r)
        self.C = constraints.energy_preserving_matrix(self.rom, self.r)
        self.m = self.rom.model.operator_matrix.shape[1]
        rng = np.random.default_rng(2)
        self.roots = rng.normal(size=(self.r, self.m, self.m)) / self.m
        self.mean = rng.normal(size=(self.r, self.m))

    def test_matheron_matches_the_dense_conditional(self):
        cov = np.zeros((self.r * self.m,) * 2)
        for i in range(self.r):
            cov[i * self.m:(i + 1) * self.m, i * self.m:(i + 1) * self.m] = self.roots[i] @ self.roots[i].T
        mean_c, cov_c = _dense_condition(self.mean.ravel(), cov, self.C)
        got = np.asarray(constraints.condition_on_constraint(
            jnp.asarray(self.mean), jnp.asarray(self.roots), jnp.asarray(self.C)))
        np.testing.assert_allclose(got.ravel(), mean_c, rtol=1e-9, atol=1e-10)
        self.assertLess(np.abs(self.C @ got.ravel()).max(), 1e-10)
        # the map x -> x - Σ Cᵀ S⁻¹ C x applied to a draw gives the conditioned covariance (linear in x)
        basis = np.eye(self.r * self.m)
        P = np.stack([np.asarray(constraints.condition_on_constraint(
            jnp.asarray(e.reshape(self.r, self.m)), jnp.asarray(self.roots), jnp.asarray(self.C))).ravel()
            for e in basis], axis=1)
        np.testing.assert_allclose(P @ cov @ P.T, cov_c, atol=1e-10)

    def test_evidence_correction_matches_the_constrained_prior(self):
        """Per-row linear Gaussian y_i = A x_i + e; brute force: evidence under the prior conditioned on Cx = 0."""
        rng = np.random.default_rng(3)
        n, prec = 9, rng.uniform(0.5, 2.0, self.m)
        A, noise = rng.normal(size=(n, self.m)), 0.3
        Y = rng.normal(size=(self.r, n))
        log_p, means, lowers = 0.0, [], []
        for i in range(self.r):
            K = A @ np.diag(1 / prec) @ A.T + noise * np.eye(n)
            log_p += -0.5 * (Y[i] @ np.linalg.solve(K, Y[i]) + np.linalg.slogdet(2 * np.pi * K)[1])
            Lam = A.T @ A / noise + np.diag(prec)
            means.append(np.linalg.solve(Lam, A.T @ Y[i] / noise))
            lowers.append(np.linalg.cholesky(Lam))
        corr = float(constraints.constraint_log_correction(
            jnp.asarray(self.C), jnp.asarray(np.stack(means)), jnp.asarray(np.stack(lowers)), jnp.asarray(prec)))
        prior = np.kron(np.eye(self.r), np.diag(1 / prec))
        _, prior_c = _dense_condition(np.zeros(self.r * self.m), prior, self.C)
        A_full = np.kron(np.eye(self.r), A)
        K = A_full @ prior_c @ A_full.T + noise * np.eye(self.r * n)
        brute = -0.5 * (Y.ravel() @ np.linalg.solve(K, Y.ravel()) + np.linalg.slogdet(2 * np.pi * K)[1])
        self.assertAlmostEqual(log_p + corr, brute, places=7)


class ModelTests(unittest.TestCase):
    def setUp(self):
        t = np.linspace(2., 6., 15)
        self.trajectory = dict(t_sampled=t, snapshots_comp=np.stack([np.sin(t) + 2., np.cos(t)]))
        self.cfg = WeakFormConfig(num_modes=2, num_eval_points=40, window_size=5, precision="float64")
        self.theta = np.array([.7, .9]), np.array([1.5, .8]), np.array([1e-3, 2e-3])
        self.tau = np.array([.5, 2., 1.5])

    def _evidence(self, model):
        values = {f"{kind}_0_{i}": self.theta[j][i]
                  for j, kind in enumerate(("lengthscale", "variance", "noise")) for i in range(2)}
        values["log_tau_block"] = jnp.log(jnp.asarray(self.tau))
        trace = handlers.trace(handlers.substitute(model, data=values)).get_trace()
        return float(trace["marg_O_evidence"]["fn"].log_factor)

    def _posterior(self, posterior, tau=None):
        hypers = tuple(jnp.asarray(x)[None] for x in self.theta)
        mean, root = posterior(hypers, None, None, jnp.asarray(self.tau if tau is None else tau))
        return np.asarray(mean), np.asarray(root)

    def test_constrained_evidence_adds_the_correction(self):
        with pipeline.precision_context(self.cfg):
            rom = _cah_rom(2)
            free = build_model(rom, [self.trajectory], self.cfg)
            ep = build_model(rom, [self.trajectory], replace(self.cfg, quad_constraint="energy_preserving"))
            np.testing.assert_array_equal(self._posterior(ep[1])[0], self._posterior(free[1])[0])
            mean, root = self._posterior(free[1])
            C = constraints.energy_preserving_matrix(rom, 2)
            lowers = np.stack([np.linalg.cholesky(np.linalg.inv(R @ R.T)) for R in root])
            prec = 1 / (self.tau[np.array([0, 1, 1, 2, 2, 2])] ** 2
                        * np.asarray(ep[3]["operator_prior"]["block_scale"])[np.array([0, 1, 1, 2, 2, 2])] ** 2)
            corr = float(constraints.constraint_log_correction(
                jnp.asarray(C), jnp.asarray(mean), jnp.asarray(lowers), jnp.asarray(prec)))
            self.assertAlmostEqual(self._evidence(ep[0]) - self._evidence(free[0]), corr, places=6)
            self.assertEqual(ep[3]["operator_prior"]["quad_constraint"]["n_constraints"], 4)

    def test_shift_is_the_fit_on_shifted_features_mapped_back(self):
        cfg = replace(self.cfg, sigma_O=1.0)
        shift = self.trajectory["snapshots_comp"].mean(axis=1)
        with pipeline.precision_context(cfg):
            ours = build_model(_cah_rom(2), [self.trajectory], replace(cfg, feature_shift="training_mean"))
            manual = build_model(_cah_rom(2, shift), [self.trajectory], cfg)
            self.assertAlmostEqual(self._evidence(ours[0]), self._evidence(manual[0]), places=9)
            G = constraints.shift_matrix(_cah_rom(2), 2, shift)
            (mean_q, root_q), (mean_z, root_z) = self._posterior(ours[1]), self._posterior(manual[1])
            np.testing.assert_allclose(mean_q, mean_z @ G.T, rtol=1e-10, atol=1e-12)
            for i in range(2):
                np.testing.assert_allclose(root_q[i] @ root_q[i].T, G @ root_z[i] @ root_z[i].T @ G.T,
                                           rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(ours[3]["operator_prior"]["feature_shift"]["shift"], shift)

    def test_unsupported_settings_raise(self):
        with self.assertRaises(ValueError):
            WeakFormConfig(num_modes=2, quad_constraint="trapping")
        cfg = replace(self.cfg, quad_constraint="energy_preserving")
        with pipeline.precision_context(cfg):
            with self.assertRaises(ValueError):
                build_model(_cah_rom(2), [self.trajectory], cfg, prior_mean=np.zeros((2, 6)))
            with self.assertRaises(ValueError):
                build_model(_cah_rom(2), [dict(self.trajectory, inputs_eval=np.ones((1, 40)))], cfg)


if __name__ == "__main__":
    unittest.main()

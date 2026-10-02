"""Tempered pseudo-likelihood: effective degrees of freedom, exponents and evidence."""

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from core.weakform_opinf import gp as _gp
from core.weakform_opinf import pipeline
from core.weakform_opinf import weakform as _wf
from core.weakform_opinf.config import WeakFormConfig
from core.weakform_opinf.model import (
    build_model,
    nondimensional_column_scale,
    power_likelihood_correction,
)


def _rom():
    """Two-mode autonomous cAH ROM (state degrees 0, 1, 1, 2, 2, 2)."""
    def assemble(states, inputs=None):
        q1, q2 = states
        return jnp.column_stack([jnp.ones(states.shape[1]), q1, q2, q1 * q1, q1 * q2, q2 * q2])
    operators = [SimpleNamespace(entries=np.zeros(shape)) for shape in ((2,), (2, 2), (2, 3))]
    return SimpleNamespace(model=SimpleNamespace(
        operators=operators, operator_matrix=np.zeros((2, 6)), _assemble_data_matrix=assemble))


def _log_normal(r, var):
    return -.5 * (np.log(2 * np.pi * var) + r ** 2 / var)


class TemperingTests(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(2., 6., 15)
        self.y = np.stack([np.sin(self.t) + 2., np.cos(self.t)])
        self.trajectory = dict(t_sampled=self.t, snapshots_comp=self.y)
        self.cfg = WeakFormConfig(num_modes=2, num_eval_points=40, window_size=5, precision="float64")

    def test_effective_dof_limits_and_units(self):
        cfg = WeakFormConfig(num_modes=2)
        # A lengthscale far below the sampling interval leaves K = σ² I, so df = n σ²/(σ² + ν).
        df = _gp.effective_dof(self.trajectory, cfg, [1e-3, 1e-3], [1., 3.], [1., 1e9])
        np.testing.assert_allclose(df, [7.5, 15 * 3 / (3 + 1e9)], rtol=1e-12)
        theta = np.array([.7, 1.3]), np.array([1.5, .8]), np.array([1e-3, 2e-2])
        base = _gp.effective_dof(self.trajectory, cfg, *theta)
        self.assertTrue(np.all((0 < base) & (base < len(self.t))))
        a, b = 1e-2, 30.
        rescaled = _gp.effective_dof(dict(t_sampled=a * self.t, snapshots_comp=b * self.y), cfg,
                                     a * theta[0], b ** 2 * theta[1], b ** 2 * theta[2])
        np.testing.assert_allclose(rescaled, base, rtol=1e-9)

    def test_input_trend_adds_its_features(self):
        t = np.linspace(0., 1., 12)
        table = dict(times=np.linspace(0., 2., 21), values=np.linspace(0., 2., 21) ** 2)
        trajectory = dict(t_sampled=t, snapshots_comp=np.stack([np.sin(3 * t) + 1.]), input_table=table)
        cfg = WeakFormConfig(num_modes=1, gp_input_trend=True)
        # With a negligible stationary part the smoother keeps the three trend features.
        self.assertAlmostEqual(float(_gp.effective_dof(trajectory, cfg, [.1], [1e-20], [1e-9])[0]), 3., places=4)
        plain = WeakFormConfig(num_modes=1)
        self.assertGreater(float(_gp.effective_dof(trajectory, cfg, [.1], [1.], [1e-2])[0]),
                           float(_gp.effective_dof(trajectory, plain, [.1], [1.], [1e-2])[0]))

    def test_power_likelihood_correction(self):
        rng = np.random.default_rng(0)
        u, r = rng.uniform(.1, 2., (2, 7)), rng.normal(size=(2, 7))
        alpha = np.array([[.3], [.8]])
        expected = np.sum(alpha * _log_normal(r, u) - _log_normal(r, u / alpha))
        with jax.experimental.enable_x64():
            got = float(power_likelihood_correction(jnp.asarray(u), jnp.asarray(alpha)))
        self.assertAlmostEqual(got, expected, places=10)

    def test_settings_are_validated(self):
        self.assertEqual((WeakFormConfig().closure, WeakFormConfig().gamma2), ("tempered", None))
        for kwargs in (dict(closure="radius"), dict(gamma2=1.), dict(deriv_cov="full"),
                       dict(weakform_cov="full"), dict(weakform_mode="deriv")):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                WeakFormConfig(**kwargs)
        WeakFormConfig(closure="slack", gamma2=1., deriv_cov="full")

    def test_tempered_model_records_its_exponents_and_evidence(self):
        cfg = self.cfg
        with pipeline.precision_context(cfg):
            model, _, time_evals, priors = build_model(_rom(), [self.trajectory], cfg)
            info = priors["closure"]
            self.assertEqual((info["rule"], priors["gamma2"]), ("tempered", 0.))
            tf = _wf.build_test_functions(time_evals[0], cfg)
            self.assertEqual(info["rows"], [40 + tf["wpsi"].shape[0]])
            fitted = tuple(np.asarray(info["data_only_gp"][k][0]) for k in ("ell", "sig2", "nu"))
            df = _gp.effective_dof(self.trajectory, cfg, *fitted)
            np.testing.assert_allclose(info["df"][0], df, rtol=1e-12)
            alpha = np.minimum(1., df / info["rows"][0])
            np.testing.assert_allclose(info["alpha"][0], alpha, rtol=1e-12)
            self.assertTrue(np.all((alpha > 0) & (alpha < 1)))

            theta = np.array([.7, .9]), np.array([1.5, .8]), np.array([1e-3, 2e-3])
            values = {f"{kind}_0_{i}": theta[j][i] for j, kind in enumerate(("lengthscale", "variance", "noise"))
                      for i in range(2)}
            values["log_tau_block"] = jnp.zeros(3)
            trace = handlers.trace(handlers.substitute(model, data=values)).get_trace()
            evidence = float(trace["marg_O_evidence"]["fn"].log_factor)

            # Reference: the power likelihood's operator evidence, densely in numpy.
            _, batch = _gp.trajectory_gp_conditional(self.trajectory, cfg)(time_evals[0])
            X, mz, KZ, KX, _ = (np.asarray(a) for a in batch(
                *(jnp.asarray(x) for x in theta), jnp.asarray(self.y)))
            F = np.asarray(_rom().model._assemble_data_matrix(jnp.asarray(X)))
            wpsi, wpsid = np.asarray(tf["wpsi"]), np.asarray(tf["wpsi_dot"])
            scale, _ = nondimensional_column_scale(_rom(), [self.trajectory], 2)
            eps, n = np.finfo(float).eps, len(self.t)
            design = np.vstack([F, wpsi @ F])
            expected = 0.
            for i in range(2):
                v_D = np.diag(KZ[i]) + n * eps * theta[1][i] / theta[0][i] ** 2
                v_W = (np.diag(wpsid @ KX[i] @ wpsid.T)
                       + n * eps * theta[1][i] * np.abs(wpsid).sum(axis=1) ** 2 + 1e-8 * alpha[i])
                u = np.concatenate([v_D, v_W])
                rows = np.concatenate([mz[i], -(wpsid @ X[i])])
                marginal = np.diag(u / alpha[i]) + (design * scale ** 2) @ design.T
                sign, logdet = np.linalg.slogdet(marginal)
                self.assertEqual(sign, 1.)
                gaussian = -.5 * (rows @ np.linalg.solve(marginal, rows) + logdet + len(rows) * np.log(2 * np.pi))
                expected += gaussian - .5 * np.sum((alpha[i] - 1) * np.log(2 * np.pi * u) + np.log(alpha[i]))
            self.assertAlmostEqual(evidence, expected, delta=1e-7 * abs(expected))

    def test_tempered_operator_posterior_is_unit_invariant(self):
        theta = (np.array([[.7, .9]]), np.array([[1.5, .8]]), np.array([[1e-4, 2e-4]]))
        degree = np.array([0, 1, 1, 2, 2, 2])

        def nondimensional_mean(a, b):
            with pipeline.precision_context(self.cfg):
                trajectory = dict(t_sampled=a * self.t, snapshots_comp=b * self.y)
                _, posterior, _, priors = build_model(_rom(), [trajectory], self.cfg)
                hypers = tuple(jnp.asarray(x) for x in (a * theta[0], b ** 2 * theta[1], b ** 2 * theta[2]))
                mean, _ = posterior(hypers, None, None, jnp.ones(3))
                return np.asarray(mean) / (b ** (1. - degree) / a)[None], np.asarray(priors["closure"]["alpha"])

        base, alpha = nondimensional_mean(1., 1.)
        rescaled, alpha_rescaled = nondimensional_mean(1e-2, 30.)
        np.testing.assert_allclose(alpha_rescaled, alpha, rtol=1e-5)
        np.testing.assert_allclose(rescaled, base, rtol=1e-4, atol=1e-6 * np.abs(base).max())


if __name__ == "__main__":
    unittest.main()

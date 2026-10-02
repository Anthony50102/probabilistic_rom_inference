import unittest
from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from core.weakform_opinf import pipeline
from core.weakform_opinf.config import WeakFormConfig
from core.weakform_opinf.model import build_model, closure_slack
from core.weakform_opinf.weakform import build_test_functions


def _rom():
    """Two-mode autonomous cAH ROM (state degrees 0, 1, 1, 2, 2, 2)."""
    def assemble(states, inputs=None):
        q1, q2 = states
        return jnp.column_stack([jnp.ones(states.shape[1]), q1, q2, q1 * q1, q1 * q2, q2 * q2])
    operators = [SimpleNamespace(entries=np.zeros(shape)) for shape in ((2,), (2, 2), (2, 3))]
    return SimpleNamespace(model=SimpleNamespace(
        operators=operators, operator_matrix=np.zeros((2, 6)), _assemble_data_matrix=assemble))


class ClosureSlackTests(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(2., 6., 15)
        self.y = np.stack([np.sin(self.t) + 2., np.cos(self.t)])

    def test_production_defaults(self):
        cfg = WeakFormConfig()
        self.assertEqual((cfg.gamma2, cfg.weak_slack, cfg.operator_solver), (None, "grid", "qr"))
        self.assertEqual((cfg.mll_weight, cfg.deriv_weight, cfg.weakform_weight), (1., 1., 1.))

    def test_rule_scales_with_the_reference_rate(self):
        cfg = WeakFormConfig(num_modes=2, gamma2_nd=.3)
        gamma2, info = closure_slack([dict(t_sampled=self.t, snapshots_comp=self.y)], cfg)
        S = np.sqrt(np.mean(self.y ** 2))
        self.assertAlmostEqual(gamma2, .3 * (S / 4.) ** 2)
        self.assertEqual((info["rule"], info["T"], info["weak_slack"]), ("nondimensional", 4., "grid"))
        fixed, info = closure_slack([dict(t_sampled=self.t, snapshots_comp=self.y)], replace(cfg, gamma2=2.))
        self.assertEqual((fixed, info["rule"]), (2., "fixed"))

    def test_slack_settings_are_validated(self):
        for kwargs in (dict(gamma2=0.), dict(gamma2_nd=0.), dict(gamma2_nd=-1.), dict(weak_slack="radius")):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                WeakFormConfig(**kwargs)

    def test_grid_slack_is_the_quadrature_variance(self):
        tf = build_test_functions(np.linspace(0., 3., 61), WeakFormConfig(window_size=5))
        wpsi = np.asarray(tf["wpsi"], dtype=float)
        np.testing.assert_allclose(tf["quad_psi_sq"], np.sum(wpsi ** 2, axis=1), rtol=1e-5)
        np.testing.assert_allclose(tf["int_psi"], np.sum(wpsi, axis=1), rtol=1e-5)
        # Bumps vanish at the grid ends, so the quadrature variance is Δt ∫ψ².
        np.testing.assert_allclose(tf["quad_psi_sq"], .05 * np.asarray(tf["int_psi_sq"]), rtol=1e-5)

    def test_default_operator_posterior_is_unit_invariant(self):
        """Grid and support slack with the default QR solve are invariant; the legacy slack and the
        ridge of the normal-equation solver are not."""
        theta = (np.array([[.7, .9]]), np.array([[1.5, .8]]), np.array([[1e-4, 2e-4]]))
        degree = np.array([0, 1, 1, 2, 2, 2])

        def nondimensional_mean(slack, solver, a, b):
            """Posterior operator mean in units of the rescaled data, mapped back to the original units."""
            cfg = WeakFormConfig(num_modes=2, num_eval_points=40, window_size=5, weak_slack=slack,
                                 precision="float64", **({} if solver is None else {"operator_solver": solver}))
            with pipeline.precision_context(cfg):
                trajectory = dict(t_sampled=a * self.t, snapshots_comp=b * self.y)
                _, posterior, _, priors = build_model(_rom(), [trajectory], cfg)
                hypers = tuple(jnp.asarray(x) for x in (a * theta[0], b ** 2 * theta[1], b ** 2 * theta[2]))
                mean, _ = posterior(hypers, None, None, jnp.ones(3))
                self.assertAlmostEqual(priors["gamma2"] / (b / a) ** 2,
                                       cfg.gamma2_nd * np.mean(self.y ** 2) / 16., places=12)
                return np.asarray(mean) / (b ** (1. - degree) / a)[None]

        for slack, solver, invariant in (("grid", None, True), ("support", None, True),
                                         ("legacy", None, False), ("grid", "normal", False)):
            with self.subTest(slack=slack, solver=solver):
                base = nondimensional_mean(slack, solver, 1., 1.)
                rescaled = nondimensional_mean(slack, solver, 1e-2, 30.)
                close = np.allclose(rescaled, base, rtol=1e-4, atol=1e-6 * np.abs(base).max())
                self.assertEqual(close, invariant)


if __name__ == "__main__":
    unittest.main()

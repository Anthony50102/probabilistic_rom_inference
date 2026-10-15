"""Learned input-trend scale: the GP trend's coefficient prior sd as a per-mode hyperparameter."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from core.weakform_opinf import pipeline
from core.weakform_opinf.config import WeakFormConfig
from core.weakform_opinf.features import input_trend_features
from core.weakform_opinf.gp import make_gp_conditional
from core.weakform_opinf.model import build_model


def _rom():
    """Two-mode autonomous cA ROM."""
    def assemble(states, inputs=None):
        q1, q2 = states
        return jnp.column_stack([jnp.ones(states.shape[1]), q1, q2])
    operators = [SimpleNamespace(entries=np.zeros(shape)) for shape in ((2,), (2, 2))]
    return SimpleNamespace(model=SimpleNamespace(
        operators=operators, operator_matrix=np.zeros((2, 3)), _assemble_data_matrix=assemble))


class TrendScaleTests(unittest.TestCase):
    def setUp(self):
        t = np.linspace(1., 9., 9)
        table = dict(times=np.linspace(0., 10., 101), values=(np.linspace(0., 10., 101) > 4.).astype(float))
        self.trajectory = dict(t_sampled=t, snapshots_comp=np.stack([np.sin(t) + .3 * t, np.cos(t)]),
                               input_table=table)

    def test_settings_are_validated(self):
        self.assertEqual(WeakFormConfig().gp_trend_scale, "fixed")
        WeakFormConfig(gp_input_trend=True, gp_trend_scale="learned")
        for kwargs in (dict(gp_trend_scale="learned"), dict(gp_input_trend=True, gp_trend_scale="free")):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                WeakFormConfig(**kwargs)

    def test_batch_scales_multiply_the_coefficient_sds(self):
        with jax.experimental.enable_x64():
            train, query = np.array([.1, .5, 1.4, 2.]), np.array([.2, 1.1, 1.7])
            def features(times):
                return input_trend_features(train, times, [0., 1., 2.], [0., 2., 2.])
            single, batch = make_gp_conditional(
                train, jitter_rel=None, feature_map=features, feature_variances=[2., 3.])(query)
            args = jnp.array([.7, .8]), jnp.array([1.2, 1.5]), jnp.array([.03, .02])
            y = jnp.array([[.4, -.1, .7, .2], [.3, -.4, .8, .1]])
            kappa = jnp.array([1.7, .4])
            result = batch(*args, y, kappa)
            for mode in range(2):
                expected = single(*(a[mode] for a in args), y[mode], [2., 3.][mode] * float(kappa[mode]) ** 2)
                for values, reference in zip(result, expected):
                    np.testing.assert_allclose(values[mode], reference, rtol=1e-10, atol=1e-10)
            for values, reference in zip(batch(*args, y, jnp.ones(2)), batch(*args, y)):
                np.testing.assert_allclose(values, reference, rtol=1e-12, atol=1e-12)

    def test_unit_scales_reproduce_the_fixed_model(self):
        common = dict(num_modes=2, num_eval_points=40, window_size=5, precision="float64", gp_input_trend=True,
                      closure="slack", gamma2=1e-3)
        fixed_cfg = WeakFormConfig(**common)
        learned_cfg = WeakFormConfig(**common, gp_trend_scale="learned")
        theta = np.array([.9, 1.1]), np.array([1.5, .8]), np.array([1e-3, 2e-3])
        values = {f"{kind}_0_{i}": theta[j][i] for j, kind in enumerate(("lengthscale", "variance", "noise"))
                  for i in range(2)}
        values["log_tau_block"] = jnp.zeros(2)

        def evidence(cfg, extra):
            with pipeline.precision_context(cfg):
                model, posterior, _, priors = build_model(_rom(), [self.trajectory], cfg)
                trace = handlers.trace(handlers.substitute(model, data={**values, **extra})).get_trace()
                return float(trace["marg_O_evidence"]["fn"].log_factor), trace, posterior, priors

        fixed, fixed_trace, fixed_posterior, fixed_priors = evidence(fixed_cfg, {})
        self.assertNotIn("trend_scale", fixed_priors)
        self.assertNotIn("trend_scale_0_0", fixed_trace)
        with pipeline.precision_context(learned_cfg):
            model, posterior, _, priors = build_model(_rom(), [self.trajectory], learned_cfg)
            self.assertEqual(priors["trend_scale"]["log_kappa_prior"], ["normal", 0., learned_cfg.hier_tau_scale])
            ones = {f"trend_scale_0_{i}": 1. for i in range(2)}
            trace = handlers.trace(handlers.substitute(model, data={**values, **ones})).get_trace()
            self.assertAlmostEqual(float(trace["marg_O_evidence"]["fn"].log_factor), fixed, delta=1e-9 * abs(fixed))
            wider = {f"trend_scale_0_{i}": 3. for i in range(2)}
            trace = handlers.trace(handlers.substitute(model, data={**values, **wider})).get_trace()
            self.assertGreater(abs(float(trace["marg_O_evidence"]["fn"].log_factor) - fixed), 1e-6)
            stacked = tuple(jnp.asarray(x)[None] for x in theta)
            mean_learned, root_learned = posterior(stacked + (jnp.ones((1, 2)),), None, None, None)
        with pipeline.precision_context(fixed_cfg):
            mean_fixed, root_fixed = fixed_posterior(stacked, None, None, None)
        np.testing.assert_allclose(mean_learned, mean_fixed, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(root_learned, root_fixed, rtol=1e-9, atol=1e-12)

    def test_tempering_fits_the_scale_with_the_data_only_gp(self):
        cfg = WeakFormConfig(num_modes=2, num_eval_points=40, window_size=5, precision="float64",
                             gp_input_trend=True, gp_trend_scale="learned")
        with pipeline.precision_context(cfg):
            _, _, _, priors = build_model(_rom(), [self.trajectory], cfg)
        fitted = priors["closure"]["data_only_gp"]
        self.assertEqual(len(fitted["trend_scale"][0]), 2)
        self.assertTrue(np.all(np.asarray(fitted["trend_scale"][0]) > 0))

    def test_initial_state_uncertainty_uses_the_learned_scale(self):
        cfg = WeakFormConfig(num_modes=2, gp_input_trend=True, gp_trend_scale="learned", precision="float64",
                             ic_uncertainty=True)
        target = SimpleNamespace(state0_comp=np.zeros(2), training_index=0, t_pred=np.array([1., 5.]),
                                 t_sampled=self.trajectory["t_sampled"], snapshots_comp=None, input_func=None)
        prepared = SimpleNamespace(eval_targets=[target], trajectories=[self.trajectory], rom=None,
                                   training_span=(1., 9.))
        hypers = np.array([[.9, 1.1]]), np.array([[1.5, .8]]), np.array([[1e-3, 2e-3]])
        draws = np.zeros((6, 2, 3))

        def initial_states(config, *kappa):
            with patch.object(pipeline, "generate_rom_predictions", return_value=(None, None, [])), \
                    patch.object(pipeline, "_score", side_effect=lambda *args: {}), \
                    pipeline.precision_context(config):
                return pipeline._predict_targets(prepared, config, draws, hypers + kappa)[0]["state0_samples"]

        fixed = initial_states(replace(cfg, gp_trend_scale="fixed"))
        np.testing.assert_allclose(initial_states(cfg, np.ones((1, 2))), fixed, rtol=1e-12)
        self.assertGreater(np.abs(initial_states(cfg, np.full((1, 2), 3.)) - fixed).max(), 1e-6)

    def test_stack_theta_carries_learned_scales(self):
        post = {f"{kind}_0_{i}": jnp.full(5, float(i + 1)) for kind in ("lengthscale", "variance", "noise")
                for i in range(2)}
        self.assertEqual(len(pipeline._stack_theta(post, 1, 2)), 3)
        post.update({f"trend_scale_0_{i}": jnp.full(5, 2. + i) for i in range(2)})
        stacked = pipeline._stack_theta(post, 1, 2)
        self.assertEqual(len(stacked), 4)
        np.testing.assert_allclose(stacked[3][0, 0], [2., 3.])


if __name__ == "__main__":
    unittest.main()

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from core.weakform_opinf.config import WeakFormConfig
from core.weakform_opinf.model import build_model
from core.weakform_opinf import pipeline


class InputModelTests(unittest.TestCase):
    def test_input_trend_and_measurement_priors_reach_operator_posterior(self):
        cfg = WeakFormConfig(
            operators="cABN", num_modes=2, num_eval_points=8, window_size=2,
            gp_input_trend=True, gp_noise_prior="measurement", gp_jitter_rel=None,
            operator_solver="qr", precision="float64")
        t = np.linspace(5., 7., 8)
        table_t = np.linspace(5., 10., 11)
        table_u = np.maximum(table_t - 6., 0.) * .3
        observations = np.stack([np.sin(t), np.cos(t)])
        trajectory = dict(
            t_sampled=t, snapshots_comp=observations,
            inputs_eval=np.interp(t, table_t, table_u)[None],
            input_table=dict(times=table_t, values=table_u),
            noise_variances=np.array([.001, .002]))
        def assemble(states, inputs):
            return jnp.column_stack([
                jnp.ones(states.shape[1]), states.T, inputs.T, (states * inputs).T])
        model = SimpleNamespace(
            operators=[SimpleNamespace(entries=np.zeros(shape))
                       for shape in ((2,), (2, 2), (2, 1), (2, 2))],
            operator_matrix=np.zeros((2, 6)), _assemble_data_matrix=assemble)
        rom = SimpleNamespace(model=model)
        with pipeline.precision_context(cfg):
            _, posterior, _, priors = build_model(rom, [trajectory], cfg)
            theta = tuple(jnp.asarray(x) for x in (
                [[.6, .8]], [[1., 2.]], [[.001, .002]]))
            mean, factor = posterior(theta, cfg.gamma2, cfg.sigma_O, jnp.ones(4))
            self.assertEqual(mean.shape, (2, 6))
            self.assertEqual(factor.shape, (2, 6, 6))
            self.assertEqual(mean.dtype, jnp.float64)
            self.assertTrue(np.isfinite(mean).all() and np.isfinite(factor).all())
            np.testing.assert_allclose(priors["nu"], trajectory["noise_variances"])
            _, baseline, _, _ = build_model(
                rom, [trajectory], replace(cfg, gp_input_trend=False))
            baseline_mean, _ = baseline(theta, cfg.gamma2, cfg.sigma_O, jnp.ones(4))
            self.assertFalse(np.allclose(mean, baseline_mean))
        incomplete = {key: value for key, value in trajectory.items() if key != "input_table"}
        with self.assertRaisesRegex(ValueError, "input_table"):
            build_model(rom, [incomplete], cfg)

    def test_precision_does_not_leak_into_other_methods(self):
        with jax.experimental.enable_x64(False):
            cfg = WeakFormConfig(precision="float64")
            with patch.object(pipeline, "_run_experiment",
                              side_effect=lambda *args: jax.config.x64_enabled):
                self.assertTrue(pipeline.run_experiment(None, cfg, {}, "."))
                self.assertFalse(jax.config.x64_enabled)
                self.assertFalse(pipeline.run_experiment(
                    None, replace(cfg, precision="default"), {}, "."))
            with patch.object(pipeline, "_run_experiment", side_effect=ValueError("failure")):
                with self.assertRaises(ValueError):
                    pipeline.run_experiment(None, cfg, {}, ".")
            self.assertFalse(jax.config.x64_enabled)


if __name__ == "__main__":
    unittest.main()

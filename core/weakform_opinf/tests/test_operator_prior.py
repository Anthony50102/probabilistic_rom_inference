import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from core.weakform_opinf.config import WeakFormConfig
from core.weakform_opinf.gp import make_gp_conditional
from core.weakform_opinf.model import nondimensional_column_scale


def _rom(with_inputs):
    """Two-mode cAH (+ B, N when with_inputs) data-matrix assembly."""
    def assemble(states, inputs=None):
        q1, q2 = states
        columns = [jnp.ones(states.shape[1]), q1, q2, q1 * q1, q1 * q2, q2 * q2]
        if with_inputs:
            columns += [inputs[0], q1 * inputs[0], q2 * inputs[0]]
        return jnp.column_stack(columns)
    return SimpleNamespace(model=SimpleNamespace(_assemble_data_matrix=assemble))


class NondimensionalPriorTests(unittest.TestCase):
    def setUp(self):
        t = np.linspace(2., 6., 9)
        self.trajectory = dict(t_sampled=t, snapshots_comp=np.stack([np.sin(t) + 2., np.cos(t)]),
                               inputs_eval=(.5 + .1 * t)[None])

    def test_column_scale_follows_each_column_degree(self):
        scale, info = nondimensional_column_scale(_rom(True), [self.trajectory], 2)
        self.assertEqual(info["state_degree"], [0, 1, 1, 2, 2, 2, 0, 1, 1])
        self.assertEqual(info["input_degree"], [0, 0, 0, 0, 0, 0, 1, 1, 1])
        S = np.sqrt(np.mean(self.trajectory["snapshots_comp"] ** 2))
        U = np.sqrt(np.mean(self.trajectory["inputs_eval"] ** 2))
        self.assertAlmostEqual(info["T"], 4.)
        np.testing.assert_allclose(scale[[0, 1, 3, 6, 7]],
                                   [S / 4., 1 / 4., 1 / (4. * S), S / (4. * U), 1 / (4. * U)])

    def test_column_scale_carries_the_data_units(self):
        base, _ = nondimensional_column_scale(_rom(True), [self.trajectory], 2)
        c, a, b = 30., 1e-2, 7.
        rescaled = dict(t_sampled=a * self.trajectory["t_sampled"],
                        snapshots_comp=c * self.trajectory["snapshots_comp"],
                        inputs_eval=b * self.trajectory["inputs_eval"])
        scale, info = nondimensional_column_scale(_rom(True), [rescaled], 2)
        p, q = np.asarray(info["state_degree"]), np.asarray(info["input_degree"], dtype=float)
        np.testing.assert_allclose(scale, base * c ** (1. - p) * b ** (-q) / a, rtol=1e-12)

    def test_autonomous_and_degenerate_scales(self):
        autonomous = {k: v for k, v in self.trajectory.items() if k != "inputs_eval"}
        _, info = nondimensional_column_scale(_rom(False), [autonomous], 2)
        self.assertEqual(info["U"], 1.)
        silent = dict(self.trajectory, inputs_eval=0. * self.trajectory["inputs_eval"])
        with self.assertRaisesRegex(ValueError, "positive scales"):
            nondimensional_column_scale(_rom(True), [silent], 2)
        with self.assertRaisesRegex(ValueError, "every trajectory or none"):
            nondimensional_column_scale(_rom(True), [self.trajectory, autonomous], 2)

    def test_scale_constants_must_be_positive(self):
        for field in ("sigma_O", "gp_jitter_rel"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                WeakFormConfig(**{field: 0.})


class RoundoffNuggetTests(unittest.TestCase):
    def test_roundoff_nugget_is_scale_invariant(self):
        with jax.experimental.enable_x64():
            t = np.linspace(0., 1., 7)
            y = 1e-3 * np.sin(3. * t)
            sig2, nu, c = 1.3e-6, 2e-8, 1e3

            def rescaled_outputs(jitter_rel):
                single, _ = make_gp_conditional(t, jitter_rel)(np.linspace(0., 1., 5))
                small = single(.3, sig2, nu, jnp.asarray(y))
                large = single(.3, c ** 2 * sig2, c ** 2 * nu, jnp.asarray(c * y))
                return [(np.asarray(s), np.asarray(g) / c ** k)
                        for s, g, k in zip(small[:4], large[:4], (1, 1, 2, 2))]

            for small, large in rescaled_outputs(None):
                np.testing.assert_allclose(large, small, rtol=1e-6, atol=1e-9 * np.abs(small).max())
            # The historical relative nugget floors at 1e-5, which dominates this small-scale kernel.
            self.assertFalse(all(np.allclose(large, small, rtol=1e-3)
                                 for small, large in rescaled_outputs(1e-3)))


if __name__ == "__main__":
    unittest.main()

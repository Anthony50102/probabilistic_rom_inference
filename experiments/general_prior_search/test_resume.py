"""Checkpoint continuation must preserve the full-joint inference trajectory."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from core.weakform_opinf.config import WeakFormConfig
from core.weakform_opinf.pipeline import precision_context
from core.weakform_opinf.tests.test_prior_mean import _rom
from experiments.general_prior_search.run import CANDIDATES, fit


class CheckpointTests(unittest.TestCase):
    def test_resumed_fit_matches_uninterrupted_fit(self):
        t = np.linspace(2., 6., 21)
        prepared = SimpleNamespace(rom=_rom(), trajectories=[
            dict(t_sampled=t, snapshots_comp=np.stack([2 + np.sin(t), .3 * np.cos(t)]))])
        cfg = WeakFormConfig(num_modes=2, num_eval_points=40, window_size=5, precision="float64",
                             num_steps=200, num_posterior_samples=4, seed=45, hier_tau_scale=1.)
        candidate = CANDIDATES["rate-clock-d1"]
        with TemporaryDirectory() as tmp, precision_context(cfg):
            direct, resumed = Path(tmp)/"direct", Path(tmp)/"resumed"
            direct.mkdir()
            resumed.mkdir()
            expected = fit(prepared, cfg, candidate, direct)
            fit(prepared, replace(cfg, num_steps=100), candidate, resumed)
            actual = fit(prepared, cfg, candidate, resumed)
        for key in ("losses", "O_samples", "O_conditional_means", "gp_lengthscale_samples",
                    "gp_variance_samples", "gp_noise_samples", "op_log_tau_samples"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-9, atol=1e-10, err_msg=key)


if __name__ == "__main__":
    unittest.main()

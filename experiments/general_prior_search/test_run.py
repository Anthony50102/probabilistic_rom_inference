"""The shared evaluator retains the cylinder scoring and guard conventions."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.stability_transfer.run import evaluate
from experiments.stability_transfer.test_run import LinearModel


class CylinderEvaluationTests(unittest.TestCase):
    def test_held_out_interval_cap_and_full_state_error(self):
        t = np.linspace(0, 3, 31)
        observation_times = np.linspace(0, 1, 11)
        observations = np.exp(-observation_times)[None]
        targets = [SimpleNamespace(
            t_pred=t, t_full=t, true_comp=np.exp(-t)[None], true_states=np.exp(-t)[None],
            t_sampled=observation_times, snapshots_comp=observations, state0_comp=np.ones(1),
            training_index=idx, input_func=None, label=str(idx)) for idx in (0, None)]
        basis = SimpleNamespace(entries=np.eye(1), decompress=lambda x: x)
        prepared = SimpleNamespace(
            trajectories=[dict(t_sampled=observation_times, snapshots_comp=observations)],
            training_span=(0., 1.), basis=basis, rom=SimpleNamespace(model=LinearModel()), eval_targets=targets)
        arrays = dict(O_samples=-np.ones((2, 1, 1)), O_mean_rao_blackwell=-np.ones((1, 1)),
                      gp_lengthscale_samples=np.ones((2, 1, 1)), gp_variance_samples=np.ones((2, 1, 1)),
                      gp_noise_samples=np.ones((2, 1, 1)), losses=np.zeros(1000), fit_seconds=np.array(1.))
        cfg = SimpleNamespace(seed=42, num_modes=1, gp_jitter_rel=None, ic_scale=1.)
        with TemporaryDirectory() as tmp, patch(
                "experiments.stability_transfer.run._ic_sigma", return_value=np.zeros(1)):
            evaluate(prepared, cfg, arrays, None, "cylinder", Path(tmp), 1, short_draws=2)
            targets = json.loads((Path(tmp) / "summary.json").read_text())["targets"]
        self.assertEqual(targets[0]["score_interval"], [1., 3.])
        self.assertEqual(targets[1]["score_interval"], [0., 3.])
        for target in targets:
            self.assertEqual(target["cap"], 2.)
            self.assertEqual(target["survival"], 1.)
            self.assertEqual(target["long_survival"], 1.)
            self.assertLess(target["mean"]["errors"]["full_relative"], 1e-6)


if __name__ == "__main__":
    unittest.main()

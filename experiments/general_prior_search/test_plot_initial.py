"""The initial figures must retain failures without requiring every raw archive."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.general_prior_search.plot_initial import CANDIDATES, main


class InitialPlotTests(unittest.TestCase):
    def test_failed_means_and_metadata_only_covariance_case(self):
        with TemporaryDirectory() as tmp:
            root, output = Path(tmp)/"inputs", Path(tmp)/"report"
            for name in CANDIDATES:
                folder = root/name
                folder.mkdir(parents=True)
                request = dict(system="cylinder", candidate=name, config=dict(num_modes=25, seed=42),
                               schema=dict(window=[16., 1., 3.]))
                targets, arrays = [], {}
                for i, label in enumerate(("Re 150 (train)", "Re 300 (train)", "Re 200 (held out)")):
                    complete = name != "original"
                    errors = dict(fluct_relative=.1) if complete else None
                    targets.append(dict(label=label, held_out=i == 2, mean=dict(complete=complete, errors=errors),
                                        projected_reconstruction=dict(fluct_relative=.02), survival=0.,
                                        short=[dict(complete=False)]*2, long=[dict(complete=False)]))
                    time = np.linspace(0, 3, 10)
                    truth = np.arange(1, 26)[:, None]*np.sin(time)
                    arrays.update({f"t_{i}": time, f"truth_{i}": truth,
                                   f"t_sampled_{i}": time[:4], f"observations_{i}": truth[:, :4]})
                    if complete:
                        arrays[f"mean_{i}"] = truth
                (folder/"request.json").write_text(json.dumps(request))
                (folder/"summary.json").write_text(json.dumps(dict(targets=targets)))
                if name != "full-support-0.0001":
                    np.savez(folder/"plotdata.npz", **arrays)
            annotations = []

            def capture_figure(fig, *_args, **_kwargs):
                annotations.extend(text.get_text() for ax in fig.axes for text in ax.texts)

            with patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=capture_figure):
                main(root, output)
            with (output/"cylinder_targets.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            original = [row for row in rows if row["candidate"] == "original"]
            self.assertEqual(len(original), 3)
            self.assertTrue(all(row["mean_fluctuation_error"] == "" for row in original))
            self.assertTrue(all(row["mean_completed"] == "False" for row in original))
            self.assertIn("mean failed", annotations)
            self.assertIn("All three priors: 0/2 sampled forecasts survive", annotations)


if __name__ == "__main__":
    unittest.main()

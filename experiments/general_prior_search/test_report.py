"""Reports must preserve failures and reject unmatched comparisons."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from experiments.general_prior_search.report import collect, main


class ReportTests(unittest.TestCase):
    def write_fit(self, root, name, mean_error, data_hash="same"):
        folder = root / name
        folder.mkdir()
        request = dict(system="euler", schema=dict(name="dense_low_noise"),
                       config=dict(num_modes=6, seed=42), candidate=name, training_hash=data_hash)
        errors = dict(full_relative=.1, fluct_relative=.2, reduced_relative=.3)
        target = dict(held_out=False, survival=1. if mean_error is not None else 0.,
                      long_survival=0., long_mean=dict(complete=False), coverage=None, relative_band_width=None,
                      mean=dict(errors=None if mean_error is None else dict(errors, full_relative=mean_error)),
                      median_errors=None if mean_error is None else errors, projected_reconstruction=errors,
                      short=[dict(complete=False, reason="amplitude")])
        summary = dict(targets=[target], fit_seconds=1., loss_block_change=1., final_loss_sd=2.,
                       gp_rate_relative_shift=[[.5]])
        for filename, value in (("job.json", dict(returncode=0)), ("request.json", request),
                                ("summary.json", summary)):
            (folder / filename).write_text(json.dumps(value))

    def test_failed_forecast_and_failed_fit_are_not_zero_errors(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fit(root, "original", .1)
            self.write_fit(root, "good", .2)
            self.write_fit(root, "escaped", None)
            failed = root / "failed"
            failed.mkdir()
            (failed / "job.json").write_text(json.dumps(dict(returncode=1)))
            rows, failures = collect(root)
        indexed = {row["candidate"]: row for row in rows}
        self.assertEqual(len(failures), 1)
        self.assertEqual(indexed["good"]["mean_full_relative_ratio"], 2.)
        self.assertIsNone(indexed["escaped"]["mean_full_relative"])
        self.assertIsNone(indexed["escaped"]["mean_full_relative_ratio"])
        self.assertEqual(indexed["escaped"]["short_failures"], {"amplitude": 1})

    def test_data_mismatch_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fit(root, "original", .1)
            self.write_fit(root, "different", .2, data_hash="different")
            with self.assertRaisesRegex(ValueError, "Unmatched data"):
                collect(root)

    def test_published_paths_do_not_expose_local_workspace(self):
        with TemporaryDirectory() as tmp:
            root, output = Path(tmp)/"inputs", Path(tmp)/"report"
            root.mkdir()
            self.write_fit(root, "original", .1)
            main(SimpleNamespace(results=root, output=output, compare=""))
            text = (output/"summary.json").read_text()
            self.assertNotIn(tmp, text)
            self.assertEqual(json.loads(text)["records"][0]["path"], "original")


if __name__ == "__main__":
    unittest.main()

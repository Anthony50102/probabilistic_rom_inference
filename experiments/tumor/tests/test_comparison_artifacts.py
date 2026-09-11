import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chemo_evaluation import plot_comparison
from chemo_protocol import PROTOCOL_ID


class ComparisonArtifactTests(unittest.TestCase):
    def make_fixture(self, root):
        protocol = dict(
            data_fingerprint="data", protocol_id=PROTOCOL_ID,
            schema=dict(name="dense_low_noise", NOISE_LEVEL=.01))
        (root / "protocol.json").write_text(json.dumps(protocol))
        for method, fit_name, profile in (
                ("04_unified_chemo", "bayesian_fit", "input-aware"),
                ("05_neural_ode_chemo", "neural_fit", None)):
            fit = dict(model_id=method, data_fingerprint="data")
            if profile is not None:
                fit["profile"] = profile
            (root / f"{fit_name}.json").write_text(json.dumps(fit))
            for dose in (.8, 1., 1.2):
                tag = f"{dose:g}".replace(".", "p")
                stem = root / f"{method}_dose{tag}"
                row = dict(
                    model_id=method, data_fingerprint="data", method=method,
                    dose_scale=dose, n_stable=1, protocol_id=PROTOCOL_ID,
                    schema="dense_low_noise", noise=.01)
                stem.with_suffix(".json").write_text(json.dumps(row))
                np.savez(stem.with_suffix(".npz"), t_pred=[5., 70., 110.],
                         true_volume=np.ones(3), median_volume=np.ones(3),
                         lower_volume=np.full(3, .9), upper_volume=np.full(3, 1.1),
                         field_error=np.full(3, .1))

    def test_profiles_are_derived_from_verified_fits(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            with patch("core.plotting.style.save_figure", side_effect=lambda fig, _: plt.close(fig)):
                rows = plot_comparison(root)
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertEqual(row["inference_profile"],
                                 "input-aware" if row["method"].startswith("04") else "neural-ensemble")
            self.assertEqual(json.loads((root / "comparison.json").read_text()), rows)

    def test_stale_predictions_fail_before_plotting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            path = root / "04_unified_chemo_dose0p8.json"
            original = json.loads(path.read_text())
            for key, value in (("model_id", "stale"), ("data_fingerprint", "other"),
                               ("schema", "wrong"), ("noise", .05)):
                with self.subTest(key=key):
                    path.write_text(json.dumps(dict(original, **{key: value})))
                    with patch("matplotlib.pyplot.subplots") as create:
                        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                            plot_comparison(root)
                        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()

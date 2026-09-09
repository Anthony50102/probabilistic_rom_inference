"""Run with prob_rom Python's built-in unittest; no training or fixture downloads."""

import importlib.util
import argparse
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

import nibabel as nib
import numpy as np
from scipy.interpolate import interp1d


MODULE = Path(__file__).resolve().parents[2] / "export_chemo_niivue.py"
SPEC = importlib.util.spec_from_file_location("export_chemo_niivue", MODULE)
exporter = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(MODULE.parent), *sys.path]):
    SPEC.loader.exec_module(exporter)


class ExportTests(unittest.TestCase):
    def test_zero_stable_partial_export_and_provenance_mismatch(self):
        root = Path(__file__).resolve().parents[1] / f".test-data-{uuid.uuid4().hex}"
        schema = root / "results/dense_low_noise"
        data = root / "data"
        output = root / "output"
        schema.mkdir(parents=True)
        data.mkdir()
        try:
            times = np.linspace(5, 110, 400)
            entries = np.arange(96, dtype=float).reshape(24, 4) / 100
            shift = np.zeros(24)
            metadata = {"protocol_id": exporter.PROTOCOL, "schema": "dense_low_noise",
                        "method": "04_unified_chemo", "dose_scale": 1., "model_id": "model",
                        "data_fingerprint": "data", "noise": .01, "n_stable": 0, "n_total": 2}
            stem = schema / "04_unified_chemo_dose1"
            stem.with_suffix(".json").write_text(json.dumps(metadata))
            np.savez(stem.with_suffix(".npz"), t_pred=times, rom_solves=np.empty((0, 4, 400)))
            fit = schema / "bayesian_fit"
            fit.with_suffix(".json").write_text(json.dumps({"model_id": "model", "data_fingerprint": "data"}))
            np.savez(fit.with_suffix(".npz"), basis_entries=entries, basis_shift=shift, t_pred=times)
            (schema / "protocol.json").write_text(json.dumps({
                "protocol_id": exporter.PROTOCOL, "data_fingerprint": "data", "training_span": [5., 70.],
                "schema": {"name": "dense_low_noise", "NOISE_LEVEL": .01}}))
            fom_times = np.linspace(0, 120, 241)
            np.savez(data / "TNBC_demo_001_fom_chemo_sparse5_sens0p5.npz",
                     snapshots=np.ones((24, 241), dtype=np.float32), times_days=fom_times,
                     grid_shape=[2, 3, 4], spacing=[1., 2., 3.], dose_scale=1.)
            args = argparse.Namespace(results_root=root / "results", output_dir=output,
                                      dose=[1.], schema=["dense_low_noise"],
                                      method=list(exporter.METHODS), times=[5., 70., 110.],
                                      chunk_voxels=3, strict=False)
            with patch.object(exporter, "HERE", root):
                exporter.export(args)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["cases"]), 1)
            case = manifest["cases"][0]
            self.assertEqual(case["status"], "no_stable_solves")
            self.assertEqual(list(case["volumes"]), ["truth"])
            self.assertEqual(len(manifest["warnings"]), 1)
            self.assertIn("Incomplete result", manifest["warnings"][0])
            fit.with_suffix(".json").write_text(json.dumps({"model_id": "wrong", "data_fingerprint": "data"}))
            with self.assertRaisesRegex(ValueError, "model_id mismatch"):
                exporter.load_case(root / "results", "dense_low_noise", "04_unified_chemo", 1.)
            args.strict = True
            previous = (output / "manifest.json").read_bytes()
            with patch.object(exporter, "HERE", root), self.assertRaises(ValueError):
                exporter.export(args)
            self.assertEqual(previous, (output / "manifest.json").read_bytes())
            fit.with_suffix(".json").unlink()
            np.savez(fit.with_suffix(".npz"), basis_entries=entries, basis_shift=shift,
                     t_pred=times, model_id="model", data_fingerprint="data")
            loaded = exporter.load_case(root / "results", "dense_low_noise", "04_unified_chemo", 1.)
            self.assertNotIn("basis_metadata", loaded[1])
            self.assertEqual(loaded[0]["model_id"], "model")
        finally:
            # Only remove files in this test's uniquely owned, known fixture directories.
            for directory in (output, data, schema):
                if directory.exists():
                    for file in directory.iterdir():
                        file.unlink()
                    directory.rmdir()
            (root / "results").rmdir()
            root.rmdir()

    def test_actual_selected_times_and_rejection(self):
        times = np.linspace(5, 110, 400)
        indices = exporter.selected_frames(times, [5, 70, 70, 110])
        np.testing.assert_array_equal(indices, [0, 247, 399])
        for invalid in ([0], [111], [np.nan]):
            with self.assertRaises(ValueError):
                exporter.selected_frames(times, invalid)

    def test_reconstructed_voxel_quantiles_not_reduced_quantiles(self):
        basis = np.array([[1., -1.], [2., 3.], [-1., .5]])
        shift = np.array([4., -2., 0.])
        solves = np.array([[[0., 1.], [0., 2.]], [[1., 2.], [2., 4.]],
                           [[2., 3.], [4., 6.]], [[3., 4.], [6., 8.]]])
        median = np.median(solves, axis=0)
        prediction, width = exporter.reconstruct(basis, shift, median, solves, [0, 1], 2)
        fields = np.einsum("vr,srt->vst", basis, solves) + shift[:, None, None]
        quantiles = np.percentile(fields, [5, 95], axis=1)
        np.testing.assert_allclose(prediction, basis @ median + shift[:, None])
        np.testing.assert_allclose(width, quantiles[1] - quantiles[0], rtol=1e-6)
        reduced = np.percentile(solves, [5, 95], axis=0)
        self.assertFalse(np.allclose(width, basis @ (reduced[1] - reduced[0])))

    def test_cubic_interpolation_matches_evaluator(self):
        times = np.linspace(0, 120, 241)
        snapshots = np.asarray(np.arange(1, 10)[:, None] * np.sin(times[None, :] / 20), dtype=np.float32)
        selected = np.array([5., 20.263, 70., 110.])
        truth = exporter.interpolate_truth(snapshots, times, selected, 2)
        expected = interp1d(times, snapshots, axis=1, kind="cubic", bounds_error=True, copy=False)(selected)
        np.testing.assert_allclose(truth, expected, rtol=1e-6)

    def test_nifti_shape_order_affine_and_non_destructive_repeat(self):
        output = Path(__file__).resolve().parents[1] / f".test-data-{uuid.uuid4().hex}"
        output.mkdir()
        files = []
        try:
            exporter.prepare_output(output)
            files.append(output / ".chemo-niivue-export.json")
            field = np.arange(48, dtype=np.float32).reshape(24, 2)
            info = exporter.save_volume(output, field, (2, 3, 4), [1., 2., 3.], "truth")
            files.append(output / info["file"])
            image = nib.load(files[-1])
            self.assertEqual(image.shape, (2, 3, 4, 2))
            np.testing.assert_array_equal(image.affine, np.diag([1., 2., 3., 1.]))
            np.testing.assert_array_equal(np.asanyarray(image.dataobj).reshape(24, 2), field)
            self.assertIn(b"not registered", image.header["descrip"].tobytes())
            self.assertEqual(image.header.get_xyzt_units(), ("mm", "unknown"))
            self.assertEqual(info, exporter.save_volume(output, field, (2, 3, 4), [1., 2., 3.], "truth"))
        finally:
            for file in files:
                file.unlink(missing_ok=True)
            output.rmdir()


if __name__ == "__main__":
    unittest.main()

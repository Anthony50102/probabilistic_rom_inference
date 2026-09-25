"""Fast checks of the tumor benchmark runners (no fits, no reference solves)."""
import argparse
from contextlib import redirect_stderr
from dataclasses import replace
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import benchmark_data as bd  # noqa: E402
import benchmark_environment  # noqa: E402
import benchmark_evaluation as be  # noqa: E402
import benchmark_models as bm  # noqa: E402


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(value):
    return {"physical_percent": value, "burden_percent": value / 2, "reduced_percent": 2 * value,
            "projection_floor_percent": 1., "status": "complete"}


def effect(value, gain):
    return {**score(value), "final_burden_gain_predicted_over_true": gain,
            "true_final_change_percent_of_control": 10.}


def write_evaluations(root, case, seed, production_value, node_value, *, node=True):
    grid, arms = bd.grid(case), bd.arms(case)
    control = next(iter(arms))
    production = {"arms": {}}
    neural = {"arms": {}, "filter": {"kept_indices": list(range(18))}}
    for arm in arms:
        production["arms"][arm] = {
            "point": {"scores": {window: score(production_value) for window in grid.windows}},
            "draws": {"complete": 64, "members": 64, "complete_field_le25": 60,
                      "burden_truth_in_complete_band": 100, "queries": 152}}
        neural["arms"][arm] = {
            "centers": {center: {window: score(node_value) for window in grid.windows}
                        for center in ("filtered_median", "all20_median")},
            "members": {"complete": 20, "members": 20, "complete_field_le25": 10}}
        if arm != control:
            production["arms"][arm]["effect"] = {
                "point": effect(production_value, 1.),
                "draws": {"complete": 64, "members": 64, "truth_in_complete_band_queries": 50, "effect_queries": 114}}
            neural["arms"][arm]["effect"] = {center: effect(90., 0.) for center in ("filtered_median", "all20_median")}
    folder = bd.seed_directory(case, seed, root=root) / "evaluation"
    bd.write_json(folder / "production.json", production)
    if node:
        bd.write_json(folder / "neural_ode.json", neural)


class EnvironmentTests(unittest.TestCase):
    def test_reported_threads_fill_only_missing_settings(self):
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "4"}, clear=True):
            applied = benchmark_environment.apply()
            self.assertEqual(applied["OMP_NUM_THREADS"], "4")
            self.assertEqual(applied["VECLIB_MAXIMUM_THREADS"], "1")
            self.assertIn("intra_op_parallelism_threads=1", applied["XLA_FLAGS"])
            self.assertFalse(benchmark_environment.matches_reported())
        with patch.dict(os.environ, {}, clear=True):
            benchmark_environment.apply()
            self.assertTrue(benchmark_environment.matches_reported())


class FitBookkeepingTests(unittest.TestCase):
    def test_refuses_to_reuse_a_fit_started_with_other_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fit_start.json"
            bm._same_request(path, {"updates": 6000})
            bd.write_json(path, {"members": 20, "updates": 6000})
            bm._same_request(path, {"members": 20, "updates": 6000})
            with self.assertRaisesRegex(ValueError, "different settings"):
                bm._same_request(path, {"members": 4, "updates": 6000})

    def test_loss_filter_matches_the_original_chemo_rule(self):
        selection = bm._filter([1., 1.2, .9, 4.])
        self.assertTrue(selection["policy_defined"])
        self.assertEqual(selection["kept_indices"], [0, 1, 2])
        self.assertAlmostEqual(selection["cutoff"], 3 * 1.1)
        undefined = bm._filter([1., np.nan])
        self.assertFalse(undefined["policy_defined"])
        self.assertIsNone(undefined["kept_indices"])

    def test_step_overrides_cannot_exceed_the_recipe(self):
        self.assertEqual(bm._checked_steps(None, 6000), 6000)
        self.assertEqual(bm._checked_steps(200, 6000), 200)
        for steps in (0, 6001):
            with self.assertRaises(ValueError):
                bm._checked_steps(steps, 6000)


class MetricTests(unittest.TestCase):
    def setUp(self):
        truth = np.array([[1., 2., 3.], [0., 1., 0.]])
        self.residual2 = np.array([.1, .2, .3])
        self.target = be.Target(np.arange(3.), truth, np.eye(2), truth, self.residual2,
                                np.sum(truth**2, axis=0) + self.residual2, np.ones(2), 0., truth.sum(axis=0))
        self.mask = np.ones(3, dtype=bool)

    def test_exact_reduced_forecast_scores_the_projection_floor(self):
        report = be.point_window(self.target.reduced_truth, self.target, self.mask)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["reduced_percent"], 0.)
        self.assertEqual(report["burden_percent"], 0.)
        self.assertAlmostEqual(report["physical_percent"], report["projection_floor_percent"])
        self.assertAlmostEqual(report["physical_percent"],
                               100 * np.sqrt(self.residual2.sum() / self.target.physical_norm2.sum()))

    def test_nonfinite_forecast_is_incomplete_not_truncated(self):
        prediction = self.target.reduced_truth.copy()
        prediction[0, 2] = np.nan
        report = be.point_window(prediction, self.target, self.mask)
        self.assertEqual((report["status"], report["available_points"]), ("incomplete", 2))
        self.assertIsNone(report["physical_percent"])

    def test_strict_median_needs_every_member(self):
        center = be.strict_median(np.array([[[1., 1.]], [[2., np.nan]], [[4., 3.]]]))
        self.assertEqual(center[0, 0], 2.)
        self.assertTrue(np.isnan(center[0, 1]))

    def test_headline_center_keeps_each_baseline_recipe(self):
        self.assertEqual(be.headline_node_center(bd.get_case("untreated-growth")), "all20_median")
        for name in ("single-dose-chemo", "multi-dose-chemo"):
            self.assertEqual(be.headline_node_center(bd.get_case(name)), "filtered_median")


class CachedEvaluationTests(unittest.TestCase):
    def test_evaluations_are_reused_only_for_the_same_fits(self):
        case = bd.get_case("multi-dose-chemo")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit = bm.production_directory(case, 1, root=root) / "result.json"
            evaluation = bd.seed_directory(case, 1, root=root) / "evaluation"
            bd.write_json(fit, {"operators_sha256": "a"})
            bd.write_json(evaluation / "production.json",
                          {"operators_sha256": "a", "draws_evaluated": 0, "marker": True})
            self.assertTrue(be.evaluate_production(case, 1, root=root, draws=False)["marker"])
            with self.assertRaises(FileNotFoundError):
                be.evaluate_production(case, 1, root=root)
            bd.write_json(fit, {"operators_sha256": "b"})
            with self.assertRaises(FileNotFoundError):
                be.evaluate_production(case, 1, root=root, draws=False)
            summary = bm.node_directory(case, 1, root=root) / "summary.json"
            bd.write_json(summary, {"source": "first"})
            bd.write_json(evaluation / "neural_ode.json",
                          {"node_summary_sha256": bd.digest(summary), "marker": True})
            self.assertTrue(be.evaluate_node(case, 1, root=root)["marker"])
            bd.write_json(summary, {"source": "second"})
            with self.assertRaises(FileNotFoundError):
                be.evaluate_node(case, 1, root=root)


class CommandLineTests(unittest.TestCase):
    def test_reported_seeds_and_POD_tags(self):
        self.assertEqual(bd.DEFAULT_SEEDS, {"untreated-growth": (42, 43, 44), "single-dose-chemo": (51, 52, 53),
                                            "multi-dose-chemo": (48, 49, 50)})
        self.assertEqual(bd.LEGACY_SEEDS, {"untreated-growth": (42, 43, 44), "single-dose-chemo": (45, 46, 47),
                                           "multi-dose-chemo": (48, 49, 50)})
        self.assertEqual({name: bd.acquisition_tag(bd.get_case(name)) for name in bd.DEFAULT_SEEDS},
                         {"untreated-growth": "segmented_observed_mean_r3",
                          "single-dose-chemo": "segmented_observed_none_r4",
                          "multi-dose-chemo": "segmented_observed_none_r4"})
        self.assertEqual({name: bd.acquisition_tag(bd.get_case(name, "oracle_masked")) for name in bd.DEFAULT_SEEDS},
                         {"untreated-growth": "observed_mean_r4", "single-dose-chemo": "nominal_mean_r4",
                          "multi-dose-chemo": "matched_none_r4"})
        self.assertEqual(list(bd.arms(bd.get_case("multi-dose-chemo"))),
                         ["strength0p50", "strength0p25", "strength0p75", "strength1p00"])

    def test_POD_options_override_only_what_is_given(self):
        parser = bd.add_common_arguments(argparse.ArgumentParser())
        case, pod, seeds = bd.resolve(parser.parse_args(["multi-dose-chemo", "--pod-rank", "6"]))
        self.assertEqual(pod.rank, 6)
        self.assertEqual((pod.source, pod.centering), (case.pod.source, case.pod.centering))
        self.assertEqual(seeds, (48, 49, 50))
        _, _, seeds = bd.resolve(parser.parse_args(["single-dose-chemo", "--seeds", "46"]))
        self.assertEqual(seeds, (46,))

    def test_earlier_design_keeps_its_output_paths(self):
        parser = bd.add_common_arguments(argparse.ArgumentParser())
        self.assertEqual(bd.resolve(parser.parse_args(["single-dose-chemo"]))[2], (51, 52, 53))
        self.assertEqual(bd.resolve(parser.parse_args(["single-dose-chemo", "--observation", "oracle_masked"]))[2],
                         (45, 46, 47))
        case, pod, _ = bd.resolve(parser.parse_args(["multi-dose-chemo", "--observation", "oracle_masked"]))
        self.assertEqual((case.observation, bd.pod_tag(pod)), ("oracle_masked", "matched_none_r4"))
        self.assertEqual(bd.seed_directory(case, 48, pod, "/r"), Path("/r/multi-dose-chemo/matched_none_r4/seed48"))
        case, pod, _ = bd.resolve(parser.parse_args(["multi-dose-chemo", "--pod-centering", "none"]))
        self.assertEqual(bd.seed_directory(case, 48, pod, "/r"),
                         Path("/r/multi-dose-chemo/segmented_observed_none_r4/seed48"))
        with self.assertRaisesRegex(ValueError, "observed_training"):
            bd.resolve(parser.parse_args(["multi-dose-chemo", "--pod-source", "matched_training"]))

    def test_case_specific_options_need_a_case(self):
        scripts = {"benchmark_04": "04_unified_benchmark.py", "benchmark_05": "05_neural_ode_benchmark.py",
                   "benchmark_06": "06_compare_benchmark.py"}
        for name, filename in scripts.items():
            module = load_script(name, filename)
            with self.subTest(script=filename):
                self.assertIsNone(module.parse([]).case)
                with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                    module.parse(["--seeds", "48"])
        compare = load_script("benchmark_06", "06_compare_benchmark.py")
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            compare.parse(["single-dose-chemo", "--paper-figure", "figure.png"])
        self.assertEqual(compare.parse(["multi-dose-chemo", "--figure-seed", "49"]).figure_seed, 49)


class SegmentedScanTests(unittest.TestCase):
    shape = (20, 20, 20)

    def setUp(self):
        self.tissue = np.ones(self.shape, dtype=bool)
        self.tissue[:, :, :3] = False
        self.tissue = self.tissue.reshape(-1)
        self.rule = bd.detection_rule(self.tissue, (1., 1., 1.), .01)

    def test_threshold_controls_false_positives_on_empty_scans(self):
        from scipy.stats import norm
        self.assertAlmostEqual(self.rule["z"], norm.isf(bd.DETECTION_ALPHA / self.tissue.sum()))
        self.assertAlmostEqual(self.rule["threshold"], self.rule["z"] * .01 * self.rule["smoothed_noise_gain"])
        self.assertLess(self.rule["smoothed_noise_gain"], .2)
        rng, empty = np.random.default_rng(0), np.zeros(self.tissue.size)
        scans_with_detections = sum(
            bool(bd.cellularity_map(bd.noisy_scan(empty, rng.standard_normal(self.tissue.sum()), self.tissue, .01),
                                    self.tissue, self.shape, self.rule)[1].any()) for _ in range(300))
        self.assertLessEqual(scans_with_detections / 300, 2 * bd.DETECTION_ALPHA)

    def test_cellularity_is_reported_only_inside_the_detected_lesion(self):
        grid = np.indices(self.shape) - 10.
        clean = .9 * np.exp(-np.sum(grid**2, axis=0) / 8.).reshape(-1) * self.tissue
        raw = bd.noisy_scan(clean, np.random.default_rng(1).standard_normal(self.tissue.sum()), self.tissue, .05)
        observed, lesion = bd.cellularity_map(raw, self.tissue, self.shape, bd.detection_rule(
            self.tissue, (1., 1., 1.), .05))
        self.assertTrue(lesion[np.argmax(clean)])
        self.assertFalse(np.any(lesion & ~self.tissue))
        np.testing.assert_array_equal(observed[~lesion], 0.)
        np.testing.assert_array_equal(observed[lesion], np.clip(raw[lesion], 0., bd.THETA))
        self.assertTrue(np.any(raw[~lesion & self.tissue] != 0.))

    def test_observed_POD_is_orthonormal_signed_and_refuses_unresolved_ranks(self):
        scans = np.random.default_rng(2).standard_normal((6, 50))
        for centering in ("mean", "none"):
            D, shift, values = bd.observed_pod(scans, bd.PODSettings(3, "observed_training", centering))
            np.testing.assert_allclose(D.T @ D, np.eye(3), atol=1e-12)
            self.assertTrue(np.all(D[np.argmax(abs(D), axis=0), np.arange(3)] > 0))
            np.testing.assert_allclose(shift, scans.mean(axis=0) if centering == "mean" else 0.)
            self.assertTrue(np.all(np.diff(values) <= 0))
        with self.assertRaises(ValueError):
            bd.observed_pod(scans, bd.PODSettings(6, "observed_training", "mean"))

    def test_noise_threshold_separates_signal_modes_from_the_noise_bulk(self):
        self.assertAlmostEqual(bd.noise_threshold(1., [50] * 50), 4. / np.sqrt(3.) * np.sqrt(50.))
        rng = np.random.default_rng(4)
        voxels, scans, noise_sd = 3000, 40, .02
        pattern = np.linalg.qr(rng.standard_normal((voxels, 2)))[0]
        signal = pattern @ (np.array([[60.], [.5]]) * rng.standard_normal((2, scans)) / np.sqrt(scans))
        values = np.linalg.svd(signal + noise_sd * rng.standard_normal((voxels, scans)), compute_uv=False)
        threshold = bd.noise_threshold(noise_sd, [voxels] * scans)
        self.assertEqual(int(np.sum(values > threshold)), 1)
        pure = np.linalg.svd(noise_sd * rng.standard_normal((voxels, scans)), compute_uv=False)
        self.assertLess(pure.max(), threshold)

    def test_acquisition_is_deterministic_noisy_from_the_first_scan_and_uses_no_future_knots(self):
        case = replace(bd.get_case("multi-dose-chemo"), observations=6)
        grid = np.indices(self.shape).reshape(3, -1) - 10.
        radius = 2. + bd.SOURCE_KNOTS / 20.
        source = .8 * np.exp(-np.sum(grid**2, axis=0)[:, None] / radius[None]**2) * self.tissue[:, None]
        geometry = {"breast_mask": self.tissue.reshape(self.shape), "grid_shape": np.array(self.shape),
                    "spacing": np.ones(3)}
        pod = bd.PODSettings(2, "observed_training", "mean")
        with tempfile.TemporaryDirectory() as tmp:
            first = bd._segmented_acquisition(case, 3, pod, source, geometry, Path(tmp) / "a")
            second = bd._segmented_acquisition(case, 3, pod, source, geometry, Path(tmp) / "b")
            future = source.copy()
            future[:, bd.SOURCE_KNOTS > case.training_span[1]] = 7.
            third = bd._segmented_acquisition(case, 3, pod, future, geometry, Path(tmp) / "c")
            self.assertTrue((Path(tmp) / "a" / "basis.npz").exists())
            with self.assertRaisesRegex(ValueError, "scan-noise threshold"):
                bd._segmented_acquisition(case, 3, replace(pod, rank=5), source, geometry, Path(tmp) / "d")
            self.assertFalse((Path(tmp) / "d").exists())
        times, y, clean_q, D, shift, info = first
        self.assertEqual((times[0], times[-1], y.shape), (5., 70., (2, 6)))
        for index, array in enumerate(first[:5]):
            np.testing.assert_array_equal(array, second[index])
            np.testing.assert_array_equal(array, third[index])
        self.assertGreater(abs(y[:, 0] - clean_q[:, 0]).max(), 0.)
        self.assertEqual(info["observation_model"], "segmented")
        self.assertEqual(len(info["lesion_voxels_per_scan"]), 6)
        self.assertLessEqual(info["latest_source_day_used"], case.training_span[1])


class ComparisonTableTests(unittest.TestCase):
    def test_medians_over_seeds_and_wins_against_the_baseline_recipe(self):
        compare = load_script("benchmark_06", "06_compare_benchmark.py")
        case = bd.get_case("multi-dose-chemo")
        with tempfile.TemporaryDirectory() as tmp:
            for seed, (production, node) in zip((1, 2, 3), ((4., 9.), (6., 5.), (5., 12.))):
                write_evaluations(Path(tmp), case, seed, production, node)
            table, rows = compare.summarize(case, case.pod, (1, 2, 3), Path(tmp))
        arm = table["arms"]["strength0p25"]
        headline = arm["windows"]["forecast_70_110"]
        self.assertEqual(table["node_headline_center"], "filtered_median")
        self.assertEqual(headline["production"]["median"], 5.)
        self.assertEqual(headline["production"]["per_seed"], {"1": 4., "2": 6., "3": 5.})
        self.assertEqual(headline["neural_ode"]["filtered_median"]["median"], 9.)
        self.assertEqual(arm["production_wins_over_headline_node"], 2)
        self.assertEqual(arm["effect"]["production"]["final_gain"]["median"], 1.)
        self.assertEqual(arm["neural_ode_members"]["1"]["kept_by_loss_filter"], 18)
        self.assertNotIn("effect", table["arms"]["strength0p50"])
        windows, arms, seeds, methods = len(bd.grid(case).windows), 4, 3, 3
        self.assertEqual(len(rows), (windows * arms + (arms - 1)) * seeds * methods)

    def test_missing_evaluation_names_the_runner_to_use(self):
        compare = load_script("benchmark_06", "06_compare_benchmark.py")
        case = bd.get_case("single-dose-chemo")
        with tempfile.TemporaryDirectory() as tmp:
            write_evaluations(Path(tmp), case, 1, 5., 9., node=False)
            with self.assertRaisesRegex(FileNotFoundError, "05_neural_ode_benchmark.py"):
                compare.summarize(case, case.pod, (1,), Path(tmp))


class NiivueFigureTests(unittest.TestCase):
    def setUp(self):
        self.figure = load_script("niivue_benchmark_figure", "niivue_benchmark_figure.py")

    def test_days_must_be_saved_evaluation_times(self):
        times = np.linspace(5., 110., 400)
        self.assertEqual(self.figure.time_index(times, 110.), 399)
        self.assertEqual(self.figure.time_index(times, 100.), 361)
        with self.assertRaisesRegex(ValueError, "not a saved evaluation time; the nearest are 100.263 and 100.526"):
            self.figure.time_index(times, 100.4)

    def test_crop_covers_every_shown_field_with_a_clipped_margin(self):
        shape = (8, 9, 10)
        first, second = np.zeros(shape), np.zeros(shape)
        first[2, 3, 4], second[6, 1, 9] = 1., -.5
        crop = self.figure.lesion_crop([first.ravel(), second.ravel()], shape, margin=2)
        self.assertEqual([(part.start, part.stop) for part in crop], [(0, 8), (0, 6), (2, 10)])

    def test_volume_keeps_grid_values_and_a_display_only_affine(self):
        import nibabel as nib
        shape, spacing = (4, 5, 6), [1., 2., 3.]
        field = np.arange(np.prod(shape), dtype=float)
        crop = (slice(1, 3), slice(2, 5), slice(0, 6))
        with tempfile.TemporaryDirectory() as tmp:
            record = self.figure.save_volume(Path(tmp) / "panel.nii.gz", field, shape, spacing, crop)
            image = nib.load(Path(tmp) / "panel.nii.gz")
            np.testing.assert_array_equal(image.get_fdata(), field.reshape(shape)[crop])
            np.testing.assert_allclose(image.affine[:3, 3], [1., 4., 0.])
            np.testing.assert_allclose(np.diag(image.affine)[:3], spacing)
            self.assertEqual((int(image.header["qform_code"]), int(image.header["sform_code"])), (0, 2))
        self.assertEqual(record["max"], field.reshape(shape)[crop].max())


if __name__ == "__main__":
    unittest.main()

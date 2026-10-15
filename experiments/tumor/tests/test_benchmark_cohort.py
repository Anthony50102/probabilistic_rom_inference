import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark_cohort as bc  # noqa: E402
import benchmark_data as bd  # noqa: E402
from benchmark_cases import CASES, COHORT_CASES, COHORT_DESIGN, PATIENT_DOSE_DAYS, TUTORIAL_TWIN  # noqa: E402


class CohortDesignTests(unittest.TestCase):
    def test_dose_modifications(self):
        np.testing.assert_array_equal(bc.dose_factors(0), np.ones(12))
        np.testing.assert_array_equal(bc.dose_factors(1), [1.] * 7 + [0.] + [.75] * 4)
        np.testing.assert_array_equal(bc.dose_factors(4), [1.] * 5 + [0.] + [.75] * 6)
        np.testing.assert_array_equal(bc.dose_factors(6), [1.] * 6 + [0.] + [.5] * 5)
        modified = [m for m in range(COHORT_DESIGN["members"]) if np.any(bc.dose_factors(m) != 1.)]
        self.assertEqual(modified, [1, 4, 6])

    def test_scan_days(self):
        np.testing.assert_array_equal(bc.scan_days(0, "weekly"), 5. + 7. * np.arange(17))
        np.testing.assert_array_equal(bc.scan_days(7, "weekly"), bc.scan_days(0, "weekly"))
        self.assertEqual(bc.scan_days(6, "weekly")[0], 11.)
        self.assertEqual(bc.scan_days(7, "every8")[0], 12.)
        for member in range(COHORT_DESIGN["members"]):
            for protocol, stride in COHORT_DESIGN["strides"].items():
                days = bc.scan_days(member, protocol)
                self.assertTrue(np.all(np.diff(days) == stride))
                self.assertEqual(days[0], 5. + member % int(stride))
                self.assertLessEqual(days[-1], 119.)
                self.assertGreater(days[-1] + stride, 119.)
        # Weekly scans fall at every phase of the weekly dose cycle across the cohort.
        phases = {(bc.scan_days(m, "weekly")[0] - PATIENT_DOSE_DAYS[0]) % 7 for m in range(8)}
        self.assertEqual(phases, set(np.arange(7.)))

    def test_member_parameters(self):
        first, second = bc.member_parameters(), bc.member_parameters()
        np.testing.assert_array_equal(first["values"], second["values"])
        twin = np.log([TUTORIAL_TWIN[name] for name in bc.PARAMETERS])
        np.testing.assert_allclose(np.log(first["values"]) - twin,
                                   COHORT_DESIGN["log_sd"] * (first["centre_z"] + first["member_z"]), atol=1e-12)
        np.testing.assert_array_equal(first["target_z"], -first["centre_z"])
        self.assertEqual(first["values"].shape, (COHORT_DESIGN["members"], 3))

    def test_member_inputs_are_in_the_targets_units(self):
        times, values = bc.input_table(bc.dose_factors(1))
        self.assertEqual((times[0], times[-1], len(times)), (5., 119., bd.INPUT_POINTS))
        expected = sum(f * np.where(times >= d, np.exp(-bd.DECAY_RATE * (times - d)), 0.)
                       for d, f in zip(PATIENT_DOSE_DAYS, bc.dose_factors(1)))
        np.testing.assert_allclose(values, TUTORIAL_TWIN["sensitivity"] * expected, rtol=1e-14)
        held = (times >= 84.) & (times < 91.)
        np.testing.assert_allclose(values[held], bc.input_table(np.r_[np.ones(7), np.zeros(5)])[1][held])

    def test_targets_and_fit_directories(self):
        for name in COHORT_CASES:
            bc.check_target(CASES[name])
        with self.assertRaises(ValueError):
            bc.check_target(CASES["midcourse-reduced-trend-chemo-weekly-scans-learned-trend"])
        weekly = CASES["midcourse-reduced-trend-chemo-weekly-scans-learned-trend-cohort-prior"]
        folders = {bc.fit_directory(CASES[name]) for name in COHORT_CASES}
        self.assertEqual(len(folders), 4)
        self.assertEqual(bc.fit_directory(weekly), bc.fit_directory(
            CASES["midcourse-reduced-trend-chemo-weekly-scans-learned-trend-cohort-basis"]))
        self.assertIn("weekly_r4_", bc.fit_directory(weekly).name)
        self.assertEqual(len(bc.design_fingerprint()), 64)

    def test_prior_mean_refuses_other_bases_and_short_fits(self):
        case = CASES["reduced-regimen-trend-chemo-every8-scans-learned-trend-cohort-prior"]
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                bc.prior_record(case, root=root)
            folder = bc.fit_directory(case, root=root)
            folder.mkdir(parents=True)
            M = np.arange(21.).reshape(3, 7)
            bd.write_npz(folder / "operators.npz", O_point_fit=M)
            result = {"status": "complete", "operators_sha256": bd.digest(folder / "operators.npz"), "steps": 20,
                      "reduced_steps": True, "cohort_basis_sha256": "abc", "members_sha256": "def"}
            bd.write_json(folder / "result.json", result)
            record = bc.prior_record(case, root=root)
            metadata = {"sources": {"basis": {"sha256": "abc"}}}
            np.testing.assert_array_equal(bc.prior_mean(case, case.pod, root, metadata, record, full_run=False), M)
            with self.assertRaises(ValueError):
                bc.prior_mean(case, case.pod, root, metadata, record, full_run=True)
            with self.assertRaises(ValueError):
                bc.prior_mean(case, case.pod, root, {"sources": {"basis": {"sha256": "xyz"}}}, record, full_run=False)
            bd.write_npz(folder / "operators.npz", O_point_fit=2 * M)
            with self.assertRaises(RuntimeError):
                bc.prior_mean(case, case.pod, root, metadata, record, full_run=False)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from unittest import mock
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark_cohort as bc  # noqa: E402
import benchmark_data as bd  # noqa: E402
from benchmark_cases import (CASES, COHORT_CASES, COHORT_DESIGN, COHORT_SWEEP_CASES, COHORT_SWEEP_DESIGN,  # noqa: E402
                             PATIENT_DOSE_DAYS, TUTORIAL_TWIN, PODSettings, describe, production_config)


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

    def test_pooled_members_join_an_acquisition_on_the_same_basis(self):
        case = CASES["midcourse-reduced-trend-chemo-weekly-scans-learned-trend-cohort-pooled"]
        built = {"members": [{}] * 8, "files": {"members": {"sha256": "m"}, "weekly_basis": {"sha256": "abc"},
                                                 "every8_basis": {"sha256": "xyz"}}}
        times = np.linspace(5., 119., 60)
        members = [{"t_sampled": 5. + 7. * np.arange(17), "y": np.full((4, 17), float(k)), "input_times": times,
                    "input_values": np.exp(-times / (k + 10.))} for k in range(8)]
        metadata = {"sources": {"basis": {"sha256": "abc"}}}
        cfg = production_config(case)
        with mock.patch.object(bc, "_built", return_value=built), \
                mock.patch.object(bc, "load_members", return_value=members) as load:
            record = bc.pooled_record(case)
            self.assertEqual((record["protocol"], record["rank"], record["members"], record["members_sha256"],
                              record["cohort_basis_sha256"]), ("weekly", 4, 8, "m", "abc"))
            with self.assertRaises(ValueError):
                bc.pooled_record(CASES[case.name.replace("-cohort-pooled", "-cohort-prior")])
            trajectories = bc.pooled_trajectories(case, case.pod, cfg, metadata, record)
            load.assert_called_once_with("weekly", 4)
            self.assertEqual(len(trajectories), 8)
            for k, trajectory in enumerate(trajectories):
                np.testing.assert_array_equal(trajectory["snapshots_comp"], members[k]["y"])
                self.assertEqual(trajectory["inputs_eval"].shape, (1, cfg.num_eval_points))
                np.testing.assert_array_equal(trajectory["input_table"]["values"], members[k]["input_values"])
            with self.assertRaises(ValueError):
                bc.pooled_trajectories(case, case.pod, cfg, {"sources": {"basis": {"sha256": "xyz"}}}, record)
            with self.assertRaises(RuntimeError):
                bc.pooled_trajectories(case, case.pod, cfg, metadata, {**record, "members_sha256": "old"})

    def test_pooled_export_draws_the_acquisitions_operator(self):
        """Trajectory 0 (the acquisition) gives the operator draws and the initial-state rule; the population
        operator and every trajectory's point are stored in the canonical layout beside them."""
        from types import SimpleNamespace
        import jax
        import jax.numpy as jnp
        import benchmark_models as bm
        from core.weakform_opinf.pipeline import _ic_sigma
        r, trajectories, draws = 2, 3, bm.PRODUCTION_DRAWS
        rng = np.random.default_rng(0)
        posterior = {f"{kind}_{ic}_{i}": np.exp(rng.normal(size=draws))
                     for kind in ("lengthscale", "variance", "noise") for ic in range(trajectories) for i in range(r)}
        posterior["log_tau_block"] = rng.normal(size=(draws, 6))
        guide = SimpleNamespace(sample_posterior=lambda key, params, sample_shape, gamma2: posterior)
        svi = SimpleNamespace(get_params=lambda state: None)
        cfg = SimpleNamespace(num_modes=r, operators="cAN", sigma_O=None, seed=42, gp_jitter_rel=None,
                              ic_uncertainty=True, ic_scale=1.)

        def conditional(theta, gamma2, sigma_O, tau, which=0):
            level = -1. if which == "mean" else 10. * (which + 1)
            return (jnp.full((r, 1 + 2 * r), level) + 0. * theta[0][0, 0] + 0. * tau[0],
                    jnp.broadcast_to(1e-3 * jnp.eye(1 + 2 * r), (r, 1 + 2 * r, 1 + 2 * r)))

        t = np.linspace(5., 57., 9)
        observations = {"t_sampled": t, "q0": np.array([3., -1.])}
        key = jax.random.PRNGKey(0)
        with tempfile.TemporaryDirectory() as folder:
            arrays = bm._export_production(Path(folder), cfg, 0., observations, guide, svi, None, conditional, key,
                                           key, trajectories, True)
            initial = bd.read_npz(Path(folder) / "initial_conditions.npz")
        data = [0, 1, 2, 4, 5]   # canonical cABN columns of the fitted cAN layout; column 3 is the zero B column
        self.assertEqual(arrays["O_samples"].shape, (draws, r, 6))
        np.testing.assert_allclose(arrays["O_samples"][..., data], 10., atol=.02)
        np.testing.assert_array_equal(arrays["population_point"][:, data], -1.)
        np.testing.assert_array_equal(arrays["population_point"][:, 3], 0.)
        np.testing.assert_array_equal(arrays["population_point_fit"], -1.)
        self.assertEqual(arrays["trajectory_points"].shape, (trajectories, r, 6))
        for k in range(trajectories):
            np.testing.assert_array_equal(arrays["trajectory_points"][k][:, data], 10. * (k + 1))
        self.assertEqual(arrays["population_root"].shape, (r, 6, 6))
        self.assertEqual(arrays["gp_lengthscale_samples"].shape, (draws, trajectories, r))
        self.assertEqual(arrays["tau_block_samples"].shape, (draws, 6))
        mean = [np.stack([[posterior[f"{kind}_{ic}_{i}"].mean() for i in range(r)] for ic in range(trajectories)])
                for kind in ("lengthscale", "variance", "noise")]
        np.testing.assert_allclose(arrays["mean_lengthscale"], mean[0], rtol=1e-6)
        expected = _ic_sigma(t, *(value[0] for value in mean), r, roundoff_nugget=True)
        np.testing.assert_allclose(initial["sigma"], expected, rtol=1e-5)


SWEEP_SOURCE = "midcourse-reduced-trend-chemo-weekly-scans-learned-trend"


class CohortSweepTests(unittest.TestCase):
    def test_sweep_continues_the_declared_cohort(self):
        """Members 0-7 of the sweep are the declared cohort's; later members repeat its dose modifications."""
        declared, sweep = bc.member_parameters(), bc.member_parameters(COHORT_SWEEP_DESIGN)
        self.assertEqual(sweep["values"].shape, (64, 3))
        np.testing.assert_array_equal(sweep["values"][:8], declared["values"])
        np.testing.assert_array_equal(sweep["centre_z"], declared["centre_z"])
        for member in range(64):
            np.testing.assert_array_equal(bc.dose_factors(member, COHORT_SWEEP_DESIGN), bc.dose_factors(member % 8))
            np.testing.assert_array_equal(bc.scan_days(member, "weekly", COHORT_SWEEP_DESIGN),
                                          bc.scan_days(member % 7, "weekly"))
        modified = {size: sum(np.any(bc.dose_factors(m, COHORT_SWEEP_DESIGN) != 1.) for m in range(size))
                    for size in COHORT_SWEEP_DESIGN["sizes"]}
        self.assertEqual(modified, {4: 1, 8: 3, 16: 6, 32: 12, 64: 24})
        self.assertNotEqual(bc.sweep_fingerprint(), bc.design_fingerprint())
        self.assertEqual(bc.sweep_design()["cohort_design"], COHORT_SWEEP_DESIGN)
        self.assertIn("cohort_sweep_", bc.sweep_basis_path(16).name)
        self.assertTrue(bc.sweep_basis_path(16).name.endswith("_K16_basis.npz"))

    def test_sweep_cases(self):
        self.assertEqual(len(COHORT_SWEEP_CASES), 2 * len(COHORT_SWEEP_DESIGN["sizes"]))
        fingerprints = set()
        for name in COHORT_SWEEP_CASES:
            case = CASES[name]
            bc.check_target(case)
            self.assertEqual((case.cohort, case.pod.source, case.pod.rank, case.pod.centering),
                             ("weekly", "cohort_training", 4, "none"))
            self.assertEqual(case.operator_prior_mean, "pooled" if name.endswith("-pooled") else "zero")
            self.assertIn(f"-cohort{case.cohort_size}-", name)
            recipe = describe(case)
            self.assertEqual((recipe["cohort_size"], recipe["cohort_design"]), (case.cohort_size, COHORT_SWEEP_DESIGN))
            fingerprints.add(recipe["recipe_fingerprint"])
        self.assertEqual(len(fingerprints), len(COHORT_SWEEP_CASES))
        self.assertNotIn("cohort_size", describe(CASES[SWEEP_SOURCE + "-cohort-pooled"]))
        from dataclasses import replace
        pooled = CASES[SWEEP_SOURCE + "-cohort16-pooled"]
        for change in (dict(cohort_size=5), dict(cohort_size=True), dict(cohort="every8"),
                       dict(operator_prior_mean="cohort"), dict(pod=replace(pooled.pod, rank=5))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(pooled, **change)

    def test_gram_basis_is_observed_pod(self):
        """The method of snapshots on sparse scans gives observed_pod's uncentred decoder, also for the first rows
        of a larger scan matrix (a smaller cohort)."""
        import scipy.sparse
        rng = np.random.default_rng(3)
        dense = rng.normal(size=(30, 400)) * (rng.uniform(size=(30, 400)) < .3)
        dense[:, :40] = 0.
        dense += np.outer(np.linspace(1., 3., 30), np.r_[np.zeros(40), np.sin(np.arange(360) / 9.)])
        scans = scipy.sparse.csr_matrix(dense)
        for count in (12, 30):
            gram = dense[:count] @ dense[:count].T
            eigenvalues, vectors = np.linalg.eigh(gram)
            order = np.argsort(eigenvalues)[::-1]
            D = bc._gram_basis(scans, eigenvalues[order], vectors[:, order], 4)
            expected, shift, _ = bd.observed_pod(dense[:count], PODSettings(4, "cohort_training", "none"))
            np.testing.assert_allclose(D, expected, atol=1e-10)
            np.testing.assert_array_equal(shift, 0.)
        with self.assertRaises(ValueError):
            bc._gram_basis(scans, np.r_[1., 0., 0.], np.eye(3), 2)

    def test_sweep_cases_use_their_own_cohort(self):
        case = CASES[SWEEP_SOURCE + "-cohort16-pooled"]
        built = {"design_fingerprint": "f" * 64, "files": {"members": {"sha256": "m", "path": "members.npz"},
                                                            "K16_basis": {"sha256": "abc", "path": "K16.npz"},
                                                            "K8_basis": {"sha256": "xyz", "path": "K8.npz"}},
                 "sizes": {"16": {"scans": 263, "modes_above_threshold": 9, "singular_value_threshold": 1.5}}}
        times = np.linspace(5., 119., 60)
        members = [{"t_sampled": 5. + 7. * np.arange(17), "y": np.full((4, 17), float(k)), "input_times": times,
                    "input_values": np.exp(-times / (k + 10.))} for k in range(16)]
        cfg = production_config(case)
        stored = np.arange(40.).reshape(10, 4)
        with mock.patch.object(bc, "_sweep_built", return_value=built), \
                mock.patch.object(bc, "_built", side_effect=AssertionError("the declared cohort was used")), \
                mock.patch.object(bc, "load_sweep_members", return_value=members) as load, \
                mock.patch.object(bd, "read_npz", return_value={"physical_matrix": stored}) as read:
            record = bc.pooled_record(case)
            self.assertEqual((record["members"], record["cohort_size"], record["cohort_basis_sha256"],
                              record["solver"], record["members_sha256"]), (16, 16, "abc", "blocked", "m"))
            trajectories = bc.pooled_trajectories(case, case.pod, cfg, {"sources": {"basis": {"sha256": "abc"}}},
                                                  record)
            load.assert_called_once_with(16, 4)
            self.assertEqual(len(trajectories), 16)
            D, shift, basis = bc.cohort_basis(case, case.pod)
            read.assert_called_once_with(bc.sweep_basis_path(16))
            np.testing.assert_array_equal(D, stored)
            np.testing.assert_array_equal(shift, 0.)
            self.assertEqual((basis["path"], basis["cohort_size"], basis["pooled_scans"]), ("K16.npz", 16, 263))
        # The declared cohort's pooled record keeps its keys (its fits' recorded requests still match).
        declared = {"members": [{}] * 8, "files": {"members": {"sha256": "m"}, "weekly_basis": {"sha256": "abc"}}}
        with mock.patch.object(bc, "_built", return_value=declared):
            record = bc.pooled_record(CASES[SWEEP_SOURCE + "-cohort-pooled"])
        self.assertEqual(set(record), {"protocol", "rank", "members", "members_sha256", "cohort_basis_sha256",
                                       "trajectory_order"})


if __name__ == "__main__":
    unittest.main()

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
                             PATIENT_DOSE_DAYS, TUTORIAL_TWIN, VARIANT_ARMS, VARIANT_CASES, VARIANT_PATIENTS,
                             VARIANT_SIZE_ARMS, VARIANT_SOURCES, VIRTUAL_PATIENT_DESIGN, VIRTUAL_PATIENTS, PODSettings,
                             _variant_case, describe, production_config, variant_name)


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


class VariantRoundTests(unittest.TestCase):
    """The variant round's new patients and cases (declared before any fit)."""

    def test_virtual_patients_are_the_declared_draw(self):
        drawn = bc.virtual_patient_parameters()
        self.assertEqual(drawn["rounded"], VIRTUAL_PATIENTS)
        self.assertEqual(drawn["names"], tuple(VIRTUAL_PATIENTS))
        self.assertEqual(drawn["names"], tuple(f"vp{i}" for i in range(1, 10)))
        twin = np.log([TUTORIAL_TWIN[name] for name in bc.PARAMETERS])
        # Around the declared cohort's centre, with the cohort's spread; only the patients' own draw is new.
        np.testing.assert_allclose(np.log(drawn["values"]) - twin, VIRTUAL_PATIENT_DESIGN["log_sd"] * (
            bc.member_parameters()["centre_z"] + drawn["z"]), atol=1e-12)
        np.testing.assert_array_equal(drawn["z"], np.random.default_rng(202).standard_normal((9, 3)))
        rounded = np.array([[VIRTUAL_PATIENTS[name][key] for key in bc.PARAMETERS] for name in drawn["names"]])
        np.testing.assert_allclose(rounded, drawn["values"], rtol=5e-4)

    def test_variant_cases(self):
        from dataclasses import replace
        self.assertEqual(len(VARIANT_CASES), len(VARIANT_SOURCES) * len(VARIANT_PATIENTS) * 7)
        self.assertEqual(len(set(VARIANT_CASES)), len(VARIANT_CASES))
        shared = [name for name in VARIANT_CASES if name in COHORT_SWEEP_CASES]
        self.assertEqual(shared, [variant_name(SWEEP_SOURCE, 8, arm) for arm in ("basis", "pooled")])
        for name in shared:
            # The sweep's own cases (tutorial twin, weekly to day 84) are the variant definitions; only the
            # description differs, so their recorded fingerprints stand.
            variant = _variant_case(SWEEP_SOURCE, 8, name.rsplit("-", 1)[1], "")
            self.assertEqual(replace(variant, description=CASES[name].description), CASES[name])
        fingerprints = set()
        for source in VARIANT_SOURCES:
            for patient in VARIANT_PATIENTS:
                for size, arms in VARIANT_SIZE_ARMS.items():
                    for arm in arms:
                        case = CASES[variant_name(source, size, arm, patient)]
                        structure, mean, drug = VARIANT_ARMS[arm]
                        with self.subTest(case=case.name):
                            bc.check_target(case)
                            self.assertEqual((case.structure, case.operator_prior_mean, case.drug_factor, case.cohort,
                                              case.cohort_size, case.patient), (structure, mean, drug, "weekly", size,
                                                                                patient))
                            self.assertEqual(case.pod, replace(CASES[source].pod, source="cohort_training"))
                            self.assertEqual({key: getattr(case, key) for key in bc.PARAMETERS},
                                             VIRTUAL_PATIENTS[patient] if patient else TUTORIAL_TWIN)
                            self.assertEqual(bd.input_sensitivity(case), COHORT_SWEEP_DESIGN["input_sensitivity"])
                            # Everything else is the source case's.
                            self.assertEqual(replace(
                                case, name=source, description=CASES[source].description, cohort="", cohort_size=0,
                                operator_prior_mean="zero", structure="free", drug_factor=False,
                                pod=CASES[source].pod, patient="", input_sensitivity=0., **TUTORIAL_TWIN),
                                CASES[source])
                            recipe = describe(case)
                            self.assertEqual(recipe.get("virtual_patient_design"),
                                             VIRTUAL_PATIENT_DESIGN if patient else None)
                            self.assertEqual(recipe.get("drug_factor", False), drug)
                            self.assertEqual(recipe["production_config"]["operators"],
                                             "AHN" if structure != "free" else "cAN")
                            self.assertEqual(bd.DEFAULT_SEEDS[case.name], (48,))
                            fingerprints.add(recipe["recipe_fingerprint"])
        self.assertEqual(len(fingerprints), len(VARIANT_CASES))
        self.assertEqual({CASES[name].pod.rank for name in VARIANT_CASES if "4weekly" in name}, {3})
        self.assertEqual({CASES[name].pod.rank for name in VARIANT_CASES if name.startswith("reduced-")}, {3})

    def test_variant_rejections(self):
        from dataclasses import replace
        vp = CASES[variant_name(SWEEP_SOURCE, 8, "pooled-drug", "vp2")]
        twin = CASES[variant_name(SWEEP_SOURCE, 8, "pooled-drug")]
        changes = ((vp, dict(growth_rate=.05)), (vp, dict(sensitivity=.23)), (vp, dict(patient="vp10")),
                   (vp, dict(input_sensitivity=0.)), (vp, dict(input_sensitivity=.4333)),
                   (vp, dict(cohort_size=0, drug_factor=False)), (twin, dict(input_sensitivity=.23)),
                   (twin, dict(drug_factor=1)), (twin, dict(operator_prior_mean="zero")), (twin, dict(cohort_size=0)),
                   (CASES[variant_name(SWEEP_SOURCE, 8, "projected-pooled")], dict(cohort_size=0)),
                   (CASES[SWEEP_SOURCE + "-cohort-pooled"], dict(drug_factor=True)))
        for case, change in changes:
            with self.subTest(case=case.name, change=change), self.assertRaises(ValueError):
                replace(case, **change)
        # A new patient the cohort was not built for (here vp2's parameters with inputs in its own units).
        stranger = replace(vp, patient="", input_sensitivity=0.)
        with self.assertRaises(ValueError):
            bc.check_target(stranger)
        bc.check_target(replace(stranger, **TUTORIAL_TWIN))

    @unittest.skipUnless((bd.TUMORTWIN / "input_files" / bd.PATIENT).exists(), "TumorTwin patient files")
    def test_virtual_patient_inputs_are_in_the_cohorts_units(self):
        """A virtual patient's ROM exposure is the tutorial twin's (sensitivity 0.23), whatever its own sensitivity;
        its full-order pulses and simulations are its own."""
        twin = CASES[variant_name(SWEEP_SOURCE, 8, "pooled")]
        vp = CASES[variant_name(SWEEP_SOURCE, 8, "pooled", "vp2")]
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(bd, "CACHE", Path(folder)):
            self.assertNotEqual(bd._training_inputs_path(vp), bd._training_inputs_path(twin))
            ours, theirs = bd._training_inputs(vp), bd._training_inputs(twin)
            for key in ("input_times", "input_values", "native_inputs_eval"):
                np.testing.assert_array_equal(ours[key], theirs[key])
            for strength in vp.future_strengths:
                np.testing.assert_array_equal(bd.chemo_inputs(vp, strength)["input_values"],
                                              bd.chemo_inputs(twin, strength)["input_values"])
        self.assertEqual(vp.pulse_coefficients(.5), tuple(.4333 * (1. if day <= 84. else .5)
                                                          for day in PATIENT_DOSE_DAYS))
        self.assertNotEqual(bd.chemo_history_path(vp), bd.chemo_history_path(twin))
        self.assertNotEqual(bd.chemo_branch_path(vp, .5), bd.chemo_branch_path(twin, .5))

    def test_pooled_record_names_the_drug_factor(self):
        built = {"design_fingerprint": "f" * 64, "files": {"members": {"sha256": "m", "path": "members.npz"},
                                                            "K8_basis": {"sha256": "abc", "path": "K8.npz"}},
                 "sizes": {"8": {"scans": 132, "modes_above_threshold": 8, "singular_value_threshold": 1.5}}}
        with mock.patch.object(bc, "_sweep_built", return_value=built):
            plain = bc.pooled_record(CASES[variant_name(SWEEP_SOURCE, 8, "pooled")])
            for arm in ("pooled-drug", "projected-pooled-drug"):
                record = bc.pooled_record(CASES[variant_name(SWEEP_SOURCE, 8, arm, "vp1")])
                self.assertEqual(set(record) - set(plain), {"drug_factor"})
            record = bc.pooled_record(CASES[variant_name(SWEEP_SOURCE, 8, "projected-pooled", "vp1")])
        # Without the factor the record is the sweep's (resuming the sweep's fits still matches).
        self.assertEqual(record, plain)
        self.assertEqual(set(plain), {"protocol", "rank", "members", "cohort_size", "design_fingerprint",
                                      "members_sha256", "cohort_basis_sha256", "solver", "trajectory_order"})

    def test_structured_row_maps_from_the_cohort_basis(self):
        from dataclasses import replace
        import benchmark_models as bm
        r = 3
        case = CASES[variant_name(SWEEP_SOURCE, 8, "projected-pooled-drug", "vp1")]
        cfg = replace(production_config(case), num_modes=r)
        rng = np.random.default_rng(1)
        times = np.linspace(5., 84., 30)
        observations = {"t_sampled": times, "y": 1. + .1 * rng.standard_normal((r, len(times))),
                        "input_times": times, "input_values": rng.uniform(0., .3, len(times))}
        rom = bm._production_structure(case, cfg, observations)
        V, _ = np.linalg.qr(rng.uniform(0., 1., (50, r)))
        data = {"metadata": {"sources": {"basis": {"sha256": "abc"}}}, "folder": Path("/no/such/folder")}
        with mock.patch.object(bc, "cohort_basis", return_value=(V, np.zeros(50), {"sha256": "abc"})) as basis:
            maps = bm.structured_row_maps(case, case.pod, cfg, rom, data)
            basis.assert_called_once_with(case, case.pod)
            np.testing.assert_array_equal(maps, bm.galerkin_row_maps(case.structure, rom, V, r))
            with self.assertRaises(RuntimeError):
                bm.structured_row_maps(case, case.pod, cfg, rom,
                                       {**data, "metadata": {"sources": {"basis": {"sha256": "xyz"}}}})
        with mock.patch.object(bc, "cohort_basis", return_value=(V, np.ones(50), {"sha256": "abc"})), \
                self.assertRaises(ValueError):
            bm.structured_row_maps(case, case.pod, cfg, rom, data)
        free = CASES[variant_name(SWEEP_SOURCE, 8, "pooled-drug", "vp1")]
        self.assertIsNone(bm.structured_row_maps(free, free.pod, cfg, rom, data))

    def test_export_with_learned_drug_factors(self):
        """Each operator draw uses its own draw of the factors eta; the point and the pooled diagnostics their mean;
        the factors are stored."""
        from types import SimpleNamespace
        import jax
        import jax.numpy as jnp
        import benchmark_models as bm
        r, trajectories, draws = 2, 3, bm.PRODUCTION_DRAWS
        rng = np.random.default_rng(0)
        posterior = {f"{kind}_{ic}_{i}": np.exp(rng.normal(size=draws))
                     for kind in ("lengthscale", "variance", "noise") for ic in range(trajectories) for i in range(r)}
        posterior["log_tau_block"] = rng.normal(size=(draws, 6))
        posterior["input_scale_sd"] = np.exp(rng.normal(size=draws))
        posterior["input_scale_z"] = rng.normal(size=(draws, trajectories))
        guide = SimpleNamespace(sample_posterior=lambda key, params, sample_shape, gamma2: posterior)
        svi = SimpleNamespace(get_params=lambda state: None)
        cfg = SimpleNamespace(num_modes=r, operators="cAN", sigma_O=None, seed=42, gp_jitter_rel=None,
                              ic_uncertainty=True, ic_scale=1.)

        def conditional(theta, gamma2, sigma_O, tau, which=0, log_input_scale=None):
            if log_input_scale is None:
                raise AssertionError("every posterior call needs the factors")
            level = -1. + 0. * log_input_scale[0] if which == "mean" else 10. * (which + 1) + log_input_scale[which]
            return (jnp.full((r, 1 + 2 * r), level) + 0. * theta[0][0, 0] + 0. * tau[0],
                    jnp.broadcast_to(1e-3 * jnp.eye(1 + 2 * r), (r, 1 + 2 * r, 1 + 2 * r)))

        observations = {"t_sampled": np.linspace(5., 57., 9), "q0": np.array([3., -1.])}
        key = jax.random.PRNGKey(0)
        with tempfile.TemporaryDirectory() as folder:
            arrays = bm._export_production(Path(folder), cfg, 0., observations, guide, svi, None, conditional, key,
                                           key, trajectories, True, True)
        eta = posterior["input_scale_sd"][:, None] * posterior["input_scale_z"]
        data = [0, 1, 2, 4, 5]
        np.testing.assert_allclose(arrays["O_samples"][..., data],
                                   np.broadcast_to(10. + eta[:, :1, None], (draws, r, len(data))), atol=.02)
        np.testing.assert_array_equal(arrays["log_input_scale_samples"], eta)
        np.testing.assert_array_equal(arrays["input_scale_sd_samples"], posterior["input_scale_sd"])
        np.testing.assert_array_equal(arrays["mean_log_input_scale"], eta.mean(0))
        np.testing.assert_allclose(arrays["O_point"][:, data], 10. + eta.mean(0)[0], rtol=1e-6)
        np.testing.assert_array_equal(arrays["population_point"][:, data], -1.)
        for k in range(trajectories):
            np.testing.assert_allclose(arrays["trajectory_points"][k][:, data], 10. * (k + 1) + eta.mean(0)[k],
                                       rtol=1e-6)
        # A fit without learned factors stores none (and the posterior is called without them).
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(AssertionError):
            bm._export_production(Path(folder), cfg, 0., observations, guide, svi, None, conditional, key, key,
                                  trajectories, True)


if __name__ == "__main__":
    unittest.main()

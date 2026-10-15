import unittest
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace

from benchmark_cases import (CASES, COHORT_CASES, COHORT_DESIGN, LEARNED_TREND_CASES, LEGACY_PODS, PATIENT_DOSE_DAYS,
                             POOLED_CASES, REPORTED_CASES, SCAN_SCHEDULE_CASES, TUTORIAL_TWIN,
                             SPARSE_FIX_CASES, PODSettings, describe, galerkin_operators, get_case, production_config,
                             regular_scan_days, stride_scan_days, _adapter)
from core.weakform_opinf import WeakFormConfig


class BenchmarkCaseTests(unittest.TestCase):
    def test_three_explicit_cases(self):
        self.assertEqual(REPORTED_CASES, ("untreated-growth", "single-dose-chemo", "multi-dose-chemo"))
        self.assertEqual(set(CASES), set(REPORTED_CASES) | {"standard-regimen-chemo", "alternative-regimen-chemo",
                                                            "alternative-regimen-logistic-trend-chemo",
                                                            "standard-regimen-trend-chemo", "reduced-regimen-trend-chemo",
                                                            "midcourse-reduced-trend-chemo",
                                                            "midcourse-regimen-chemo", "midcourse-galerkin-chemo",
                                                            "midcourse-logistic-chemo", "midcourse-regimen-trend-chemo",
                                                            "midcourse-logistic-trend-chemo"} | set(SCAN_SCHEDULE_CASES)
                         | set(SPARSE_FIX_CASES) | set(LEARNED_TREND_CASES) | set(COHORT_CASES))
        self.assertEqual(CASES["untreated-growth"].dose_days, ())
        self.assertEqual(CASES["untreated-growth"].prediction_end, 90.)
        single, multi = CASES["single-dose-chemo"], CASES["multi-dose-chemo"]
        self.assertEqual(single.dose_days, multi.dose_days)
        self.assertEqual(len(single.dose_days), 5)
        self.assertEqual(single.future_strengths, (.5,))
        self.assertEqual(multi.future_strengths, (.25, .5, .75, 1.))

    def test_tutorial_schedule_cases(self):
        standard, alternative = CASES["standard-regimen-chemo"], CASES["alternative-regimen-chemo"]
        self.assertEqual(PATIENT_DOSE_DAYS, (35., 42., 49., 56., 63., 70., 77., 84., 91., 98., 105., 112.))
        for case in (standard, alternative):
            self.assertEqual((case.training_span, case.prediction_end), ((5., 57.), 119.))
            self.assertEqual((case.growth_rate, case.diffusion, case.sensitivity), (.0521, .0965, .23))
            self.assertEqual((case.dose_days, case.training_strength), (PATIENT_DOSE_DAYS, 1.))
        self.assertEqual(standard.future_strengths, (1.,))
        self.assertEqual(alternative.future_strengths, (.5, 1., 1.5, 2.))
        coefficients = alternative.pulse_coefficients(2.)
        self.assertEqual(coefficients[:4], (.23,) * 4)
        self.assertEqual(coefficients[4:], (.46,) * 8)
        with self.assertRaises(ValueError):
            get_case("standard-regimen-chemo", "oracle_masked")

    def test_only_future_pulses_change(self):
        case = CASES["multi-dose-chemo"]
        for strength in case.future_strengths:
            coefficients = case.pulse_coefficients(strength)
            self.assertEqual(coefficients[:3], (.25, .25, .25))
            self.assertEqual(coefficients[3:], (.5 * strength,) * 2)
        with self.assertRaises(ValueError):
            case.pulse_coefficients(2.)

    def test_production_algorithm_is_not_replaced(self):
        for name, case in CASES.items():
            with self.subTest(case=name):
                cfg = production_config(case)
                self.assertEqual(cfg.gp_input_trend, case.input_aware_gp)
                self.assertEqual(cfg.gp_noise_prior, case.gp_noise_prior)
                self.assertEqual(case.gp_noise_prior == "measurement",
                                 name in SPARSE_FIX_CASES and "known-noise" in name)
                self.assertEqual(cfg.gp_trend_scale, case.gp_trend_scale)
                self.assertEqual(case.gp_trend_scale == "learned", name in LEARNED_TREND_CASES + COHORT_CASES)
                self.assertEqual(cfg.operator_solver, "qr")
                self.assertEqual(cfg.precision, "default")
                self.assertEqual(cfg.num_steps, 12000)
                self.assertEqual(cfg.num_posterior_samples, 500)
                shared = WeakFormConfig()
                for field in ("closure", "gamma2", "mll_weight", "weakform_weight",
                              "deriv_weight", "num_eval_points", "window_size", "learning_rate",
                              "operator_solver"):
                    self.assertEqual(getattr(cfg, field), getattr(shared, field), field)
                if name in LEGACY_PODS:
                    legacy = production_config(get_case(name, "oracle_masked"))
                    self.assertEqual((legacy.closure, legacy.gamma2, legacy.weak_slack, legacy.mll_weight,
                                      legacy.weakform_weight, legacy.operator_solver),
                                     ("slack", .035, "legacy", .1, 8., "normal"))
                modified_pod = PODSettings(6, "matched_training")
                if case.structure != "free":
                    with self.assertRaisesRegex(ValueError, "uncentred"):
                        production_config(case, modified_pod)
                    modified_pod = replace(modified_pod, centering="none")
                modified = production_config(case, modified_pod)
                expected = asdict(cfg)
                expected.update(num_modes=6, operators=cfg.operators if case.structure != "free" else case.operators)
                self.assertEqual(asdict(modified), expected)
        schema = {"NUM_EVAL_POINTS": 200}
        self.assertEqual(
            asdict(production_config(CASES["single-dose-chemo"])),
            asdict(replace(_adapter(True).make_config(schema, profile="historical"), operators="cAN")))

    def test_direct_input_block_only_with_a_centred_basis(self):
        for name in ("single-dose-chemo", "multi-dose-chemo"):
            case = CASES[name]
            with self.subTest(case=name):
                self.assertEqual(case.pod.centering, "none")
                self.assertEqual(galerkin_operators(case, case.pod), "cAN")
                self.assertEqual(production_config(case).operators, "cAN")
                centred = PODSettings(4, "observed_training", "mean")
                self.assertEqual(production_config(case, centred).operators, "cABN")
                self.assertEqual(production_config(get_case(name, "oracle_masked")).operators, "cABN")
        untreated = CASES["untreated-growth"]
        self.assertEqual(production_config(untreated).operators, "cA")
        self.assertEqual(galerkin_operators(untreated, PODSettings(3, "observed_training", "none")), "cA")

    def test_omitted_input_block_is_a_zero_column_of_the_canonical_layout(self):
        import numpy as np
        from benchmark_models import canonical_layout
        r, rng = 3, np.random.default_rng(0)
        fitted = {"O_point": rng.normal(size=(r, 2 * r + 1)), "O_samples": rng.normal(size=(5, r, 2 * r + 1)),
                  "conditional_root": rng.normal(size=(r, 2 * r + 1, 2 * r + 1)), "mean_tau_block": np.ones(3)}
        out = canonical_layout(fitted, "cAN", r)
        self.assertEqual(out["O_point"].shape, (r, 2 * r + 2))
        self.assertEqual(out["O_samples"].shape, (5, r, 2 * r + 2))
        self.assertEqual(out["conditional_root"].shape, (r, 2 * r + 2, 2 * r + 2))
        keep = np.r_[0:r + 1, r + 2:2 * r + 2]
        np.testing.assert_array_equal(out["O_point"][:, r + 1], 0.)
        np.testing.assert_array_equal(out["O_point"][:, keep], fitted["O_point"])
        np.testing.assert_array_equal(out["O_samples"][..., keep], fitted["O_samples"])
        np.testing.assert_array_equal(out["conditional_root"][:, keep][:, :, keep], fitted["conditional_root"])
        np.testing.assert_array_equal(out["conditional_root"][:, r + 1], 0.)
        np.testing.assert_array_equal(out["conditional_root"][:, :, r + 1], 0.)
        self.assertIs(canonical_layout(fitted, "cABN", r), fitted)
        with self.assertRaises(ValueError):
            canonical_layout(fitted, "cAHN", r)

    def test_POD_identity_is_not_a_checkpoint_alias(self):
        case = get_case("multi-dose-chemo", "oracle_masked")
        recipes = [
            describe(case, PODSettings(4, "nominal_training", "mean")),
            describe(case, PODSettings(2, "matched_training", "mean")),
            describe(case, PODSettings(4, "matched_training", "mean")),
            describe(case, PODSettings(4, "matched_training", "none")),
        ]
        self.assertEqual(len({row["output_namespace"] for row in recipes}), 4)
        self.assertEqual(recipes[3], describe(case))
        self.assertEqual([row["uses_default_POD"] for row in recipes], [False, False, False, True])
        self.assertTrue(all(row["POD_fitting_may_not_use_future_fields"] for row in recipes))

    def test_default_POD_per_case(self):
        reported = {name: PODSettings(4, "observed_training", "none")
                    for name in ("single-dose-chemo", "multi-dose-chemo")}
        reported["untreated-growth"] = PODSettings(3, "observed_training", "mean")
        legacy = {
            "untreated-growth": PODSettings(4, "observed_training", "mean"),
            "single-dose-chemo": PODSettings(4, "nominal_training", "mean"),
            "multi-dose-chemo": PODSettings(4, "matched_training", "none"),
        }
        for observation, expected in (("segmented", reported), ("oracle_masked", legacy)):
            for name, pod in expected.items():
                with self.subTest(case=name, observation=observation):
                    case = get_case(name, observation)
                    recipe = describe(case)
                    self.assertEqual((case.pod, case.observation), (pod, observation))
                    self.assertEqual(recipe["pod"], asdict(pod))
                    self.assertEqual(recipe["observation"], observation)
                    self.assertTrue(recipe["uses_default_POD"])
                    self.assertTrue(recipe["default_POD_provenance"])
        self.assertTrue(all(case.observation == "segmented" for case in CASES.values()))
        self.assertIn("TUMOR_BENCHMARK_POD_COMPARISON.md",
                      describe(get_case("multi-dose-chemo", "oracle_masked"))["default_POD_provenance"])

    def test_earlier_design_keeps_its_recorded_fingerprints(self):
        recorded = {"untreated-growth": "712f1f331f330eed", "single-dose-chemo": "122f5bcc0b6161ee",
                    "multi-dose-chemo": "bd342ef7bd246912"}
        for name, prefix in recorded.items():
            with self.subTest(case=name):
                self.assertEqual(describe(get_case(name, "oracle_masked"))["recipe_fingerprint"][:16], prefix)
                self.assertNotEqual(describe(get_case(name))["recipe_fingerprint"][:16], prefix)

    def test_fixed_scan_schedules(self):
        self.assertEqual(regular_scan_days((5., 84.), spacing=7.),
                         (5., 12., 19., 27., 34., 41., 48., 55., 62., 70., 77., 84.))
        self.assertEqual(regular_scan_days((5., 57.), spacing=14.), (5., 18., 31., 44., 57.))
        self.assertEqual(regular_scan_days((5., 57.), count=2), (5., 57.))
        for name in SCAN_SCHEDULE_CASES:
            with self.subTest(case=name):
                case = CASES[name]
                base = CASES[name.rsplit("-", 2)[0]]
                self.assertEqual(len(case.scan_days), case.observations)
                self.assertEqual(replace(case, name=base.name, description=base.description, observations=120,
                                         scan_days=(), pod=base.pod), base)
                self.assertIn("scan_days", describe(case))
                self.assertLessEqual(case.pod.rank, min(4, case.observations))
        self.assertNotIn("scan_days", describe(CASES["midcourse-reduced-trend-chemo"]))
        base = CASES["midcourse-reduced-trend-chemo"]
        for count, days in ((3, (5., 84.)), (2, (6., 84.)), (2, (5., 90.)), (3, (5., 40., 40.))):
            with self.assertRaises(ValueError):
                replace(base, observations=count, scan_days=days)
        with self.assertRaises(ValueError):
            regular_scan_days((5., 84.), count=1)

    def test_sparse_fix_cases_change_only_the_declared_settings(self):
        self.assertEqual(stride_scan_days((5., 84.), 8.),
                         (5., 13., 21., 29., 37., 45., 53., 61., 69., 77., 84.))
        self.assertEqual(stride_scan_days((5., 57.), 8.), (5., 13., 21., 29., 37., 45., 53., 57.))
        self.assertEqual(stride_scan_days((5., 61.), 8.), (5., 13., 21., 29., 37., 45., 53., 61.))
        for span, stride in (((5., 84.), 0.), ((5., 5.), 8.)):
            with self.assertRaises(ValueError):
                stride_scan_days(span, stride)
        self.assertEqual(len(SPARSE_FIX_CASES), 9)
        for name in SPARSE_FIX_CASES:
            with self.subTest(case=name):
                case = CASES[name]
                base = CASES[next(b for b in ("midcourse-reduced-trend-chemo", "reduced-regimen-trend-chemo")
                                  if name.startswith(b + "-"))]
                every8 = "-every8-" in name
                source = base if every8 else CASES[name.removesuffix("-known-noise")]
                days = stride_scan_days(base.training_span, 8.) if every8 else source.scan_days
                self.assertEqual(case.scan_days, days)
                self.assertEqual(case.gp_noise_prior, "measurement" if "known-noise" in name else "spectrum")
                self.assertEqual(case.pod.rank, 2 if name.endswith("-r2") else
                                 CASES[f"{base.name}-weekly-scans"].pod.rank)
                self.assertEqual(replace(case, name=source.name, description=source.description, pod=source.pod,
                                         observations=source.observations, scan_days=source.scan_days,
                                         gp_noise_prior="spectrum"), source)
                self.assertEqual(production_config(case).gp_noise_prior, case.gp_noise_prior)
                self.assertEqual("gp_noise_prior" in describe(case), "known-noise" in name)
                if "-regular-" not in name:
                    self.assertLessEqual(case.observations, CASES[f"{base.name}-weekly-scans"].observations)
        self.assertEqual(CASES["midcourse-reduced-trend-chemo-regular-scans-known-noise"].observations, 120)
        with self.assertRaises(ValueError):
            replace(CASES["midcourse-reduced-trend-chemo"], gp_noise_prior="oracle")
        with self.assertRaises(ValueError):
            replace(get_case("multi-dose-chemo", "oracle_masked"), gp_noise_prior="measurement")

    def test_learned_trend_cases_change_only_the_trend_scale(self):
        self.assertEqual(len(LEARNED_TREND_CASES), 5)
        for name in LEARNED_TREND_CASES:
            with self.subTest(case=name):
                case, source = CASES[name], CASES[name.removesuffix("-learned-trend")]
                self.assertEqual(case.gp_trend_scale, "learned")
                self.assertEqual(source.gp_noise_prior, "spectrum")
                self.assertEqual(replace(case, name=source.name, description=source.description,
                                         gp_trend_scale="fixed"), source)
                self.assertEqual(production_config(case).gp_trend_scale, "learned")
                self.assertEqual(describe(case)["gp_trend_scale"], "learned")
                self.assertNotIn("gp_trend_scale", describe(source))
        for value in ("free", "learned"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(CASES["multi-dose-chemo"], gp_trend_scale=value)

    def test_cohort_cases_change_only_the_basis_and_prior_centre(self):
        self.assertEqual(len(COHORT_CASES), 12)
        self.assertEqual(POOLED_CASES, tuple(name for name in COHORT_CASES if name.endswith("-cohort-pooled")))
        self.assertEqual(len(POOLED_CASES), 4)
        means = {"basis": "zero", "prior": "cohort", "pooled": "pooled"}
        for name in COHORT_CASES:
            with self.subTest(case=name):
                case = CASES[name]
                arm = name.rsplit("-", 1)[1]
                source = CASES[name.removesuffix(f"-cohort-{arm}")]
                self.assertIn(source.name, LEARNED_TREND_CASES)
                self.assertEqual(case.cohort, "every8" if "-every8-" in name else "weekly")
                self.assertEqual(case.operator_prior_mean, means[arm])
                self.assertEqual(case.pod, replace(source.pod, source="cohort_training"))
                self.assertEqual(replace(case, name=source.name, description=source.description, pod=source.pod,
                                         cohort="", operator_prior_mean="zero"), source)
                self.assertEqual(production_config(case), production_config(source))
                recipe = describe(case)
                self.assertEqual(recipe["cohort_design"], COHORT_DESIGN)
                self.assertEqual("operator_prior_mean" in recipe, arm != "basis")
                self.assertNotEqual(recipe["recipe_fingerprint"], describe(source)["recipe_fingerprint"])
                self.assertNotIn("cohort", describe(source))
                self.assertNotIn("operator_prior_mean", describe(source))
                self.assertEqual({key: getattr(case, key) for key in TUTORIAL_TWIN}, TUTORIAL_TWIN)
        self.assertEqual(COHORT_DESIGN["input_sensitivity"], TUTORIAL_TWIN["sensitivity"])
        weekly = CASES["midcourse-reduced-trend-chemo-weekly-scans-learned-trend-cohort-prior"]
        source = CASES["midcourse-reduced-trend-chemo-weekly-scans-learned-trend"]
        for change in (dict(cohort="monthly"), dict(cohort="weekly"), dict(pod=replace(weekly.pod, rank=5)),
                       dict(pod=replace(weekly.pod, centering="mean")), dict(operator_prior_mean="median"),
                       dict(structure="galerkin"), dict(pod=replace(weekly.pod, source="observed_training"))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(weekly if "cohort" not in change or change["cohort"] != "weekly" else source, **change)
        for mean in ("cohort", "pooled"):
            with self.subTest(mean=mean), self.assertRaises(ValueError):
                replace(source, operator_prior_mean=mean)
        pooled = CASES["midcourse-reduced-trend-chemo-weekly-scans-learned-trend-cohort-pooled"]
        for change in (dict(gp_noise_prior="measurement"), dict(structure="galerkin")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(pooled, **change)
        fingerprints = {describe(CASES[name])["recipe_fingerprint"] for name in COHORT_CASES}
        self.assertEqual(len(fingerprints), 12)

    def test_segmented_scans_fit_their_own_POD(self):
        case = CASES["multi-dose-chemo"]
        for source in ("nominal_training", "matched_training"):
            with self.assertRaisesRegex(ValueError, "observed_training"):
                replace(case, pod=PODSettings(4, source, "mean"))
        with self.assertRaises(ValueError):
            replace(case, observation="clean")
        with self.assertRaises(ValueError):
            get_case("multi-dose-chemo", "clean")

    def test_invalid_recipes_are_explicit_errors(self):
        for rank in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                PODSettings(rank)
        with self.assertRaises(ValueError):
            PODSettings(source="future_truth")
        with self.assertRaises(ValueError):
            PODSettings(centering="invalid")
        with self.assertRaises(ValueError):
            get_case("unknown")


if __name__ == "__main__":
    unittest.main()

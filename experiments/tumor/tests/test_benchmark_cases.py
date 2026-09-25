import unittest
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace

from benchmark_cases import CASES, PODSettings, describe, get_case, production_config, _adapter


class BenchmarkCaseTests(unittest.TestCase):
    def test_three_explicit_cases(self):
        self.assertEqual(set(CASES), {"untreated-growth", "single-dose-chemo", "multi-dose-chemo"})
        self.assertEqual(CASES["untreated-growth"].dose_days, ())
        self.assertEqual(CASES["untreated-growth"].prediction_end, 90.)
        single, multi = CASES["single-dose-chemo"], CASES["multi-dose-chemo"]
        self.assertEqual(single.dose_days, multi.dose_days)
        self.assertEqual(len(single.dose_days), 5)
        self.assertEqual(single.future_strengths, (.5,))
        self.assertEqual(multi.future_strengths, (.25, .5, .75, 1.))

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
                self.assertFalse(cfg.gp_input_trend)
                self.assertEqual(cfg.gp_noise_prior, "spectrum")
                self.assertEqual(cfg.operator_solver, "normal")
                self.assertEqual(cfg.precision, "default")
                self.assertEqual(cfg.num_steps, 12000)
                self.assertEqual(cfg.num_posterior_samples, 500)
                modified = production_config(case, PODSettings(6, "matched_training"))
                expected = asdict(cfg)
                expected["num_modes"] = 6
                self.assertEqual(asdict(modified), expected)
        schema = {"NUM_EVAL_POINTS": 200}
        self.assertEqual(
            asdict(production_config(CASES["single-dose-chemo"])),
            asdict(_adapter(True).make_config(schema, profile="historical")))

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
        reported = {name: PODSettings(4, "observed_training", "mean")
                    for name in ("untreated-growth", "single-dose-chemo", "multi-dose-chemo")}
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

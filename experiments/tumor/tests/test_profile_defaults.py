import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from chemo_protocol import HISTORICAL_OUTPUT_ROOT, INPUT_AWARE_OUTPUT_ROOT, OUTPUT_ROOT, SCHEMAS


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProfileDefaultTests(unittest.TestCase):
    def test_original_statistical_model_is_default(self):
        module = load_script("profile_defaults_bayes", "04_unified_chemo.py")
        cfg = module.make_config(SCHEMAS[0])
        self.assertFalse(cfg.gp_input_trend)
        self.assertEqual(cfg.gp_noise_prior, "spectrum")
        self.assertIsNone(cfg.gp_jitter_rel)
        self.assertIsNone(cfg.sigma_O)
        self.assertIsNone(cfg.gamma2)
        self.assertEqual(cfg.gamma2_nd, .1)
        self.assertEqual(cfg.weak_slack, "grid")
        self.assertEqual((cfg.mll_weight, cfg.weakform_weight, cfg.deriv_weight), (1., 1., 1.))
        self.assertEqual(cfg.operator_solver, "qr")
        self.assertEqual(cfg.precision, "default")
        self.assertEqual(OUTPUT_ROOT, HISTORICAL_OUTPUT_ROOT)
        experimental = module.make_config(SCHEMAS[0], profile="input-aware")
        self.assertTrue(experimental.gp_input_trend)
        self.assertEqual(experimental.gp_noise_prior, "measurement")
        self.assertEqual(experimental.operator_solver, "qr")
        self.assertEqual(experimental.sigma_O, 5.)
        self.assertEqual((experimental.gamma2, experimental.weak_slack, experimental.weakform_weight),
                         (.035, "legacy", 8.))

    def test_report_defaults_and_explicit_experimental_directory(self):
        module = load_script("profile_defaults_compare", "06_compare_chemo.py")
        for arguments, expected in (
                ([], HISTORICAL_OUTPUT_ROOT),
                (["--bayes-profile", "input-aware"], INPUT_AWARE_OUTPUT_ROOT)):
            with self.subTest(arguments=arguments):
                with patch.object(sys, "argv", ["06_compare_chemo.py", "--method", "report", *arguments]):
                    with patch.object(module, "report") as report:
                        module.main()
                        report.assert_called_once_with(SCHEMAS, Path(expected))


if __name__ == "__main__":
    unittest.main()

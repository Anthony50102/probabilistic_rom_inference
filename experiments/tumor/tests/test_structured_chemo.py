"""Mid-course cases and the Galerkin-structured chemo ROM (row maps, canonical layout, forecast field)."""
import unittest
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_data as bd  # noqa: E402
from benchmark_cases import CASES, PATIENT_DOSE_DAYS, PODSettings, production_config  # noqa: E402
from benchmark_evaluation import ProductionField  # noqa: E402
from benchmark_models import _production_structure, canonical_layout, galerkin_row_maps  # noqa: E402

MIDCOURSE = {"midcourse-regimen-chemo": "free", "midcourse-galerkin-chemo": "galerkin",
             "midcourse-logistic-chemo": "galerkin_logistic"}


def _rom(case, r=3):
    rng = np.random.default_rng(1)
    times = np.linspace(5., 84., 30)
    observations = {"t_sampled": times, "y": 1. + .1 * rng.standard_normal((r, len(times))),
                    "input_times": times, "input_values": rng.uniform(0., .3, len(times))}
    return _production_structure(case, replace(production_config(case), num_modes=r), observations)


class MidcourseCaseTests(unittest.TestCase):
    def test_cases_differ_only_in_structure(self):
        base = CASES["alternative-regimen-chemo"]
        for name, structure in MIDCOURSE.items():
            case = CASES[name]
            with self.subTest(case=name):
                self.assertEqual((case.training_span, case.prediction_end, case.structure), ((5., 84.), 119., structure))
                self.assertEqual(replace(case, name=base.name, description=base.description, training_span=(5., 57.),
                                         structure="free"), base)
                self.assertEqual(bd.first_future_dose(case), 91.)
                self.assertEqual(case.pulse_coefficients(2.), (.23,) * 8 + (.46,) * 4)
                self.assertEqual(case.dose_days, PATIENT_DOSE_DAYS)
                # Same training history as the other tutorial cases, but its own branches and exposure table.
                self.assertEqual(bd._history_stem(case), bd._history_stem(base))
                self.assertNotEqual(bd.chemo_branch_path(case, 2.), bd.chemo_branch_path(base, 2.))
                self.assertNotEqual(bd._training_inputs_path(case), bd._training_inputs_path(base))
        self.assertEqual([production_config(CASES[name]).operators for name in MIDCOURSE], ["cAN", "AN", "AHN"])

    def test_structure_needs_an_uncentred_chemo_basis(self):
        case = CASES["midcourse-galerkin-chemo"]
        with self.assertRaisesRegex(ValueError, "uncentred"):
            replace(case, pod=PODSettings(4, "observed_training", "mean"))
        with self.assertRaisesRegex(ValueError, "uncentred"):
            replace(CASES["untreated-growth"], structure="galerkin")
        with self.assertRaisesRegex(ValueError, "Unknown reduced-model structure"):
            replace(case, structure="hybrid")


class GalerkinStructureTests(unittest.TestCase):
    def setUp(self):
        self.r = 3
        V, _ = np.linalg.qr(np.random.default_rng(0).uniform(0., 1., (50, self.r)))
        self.V = V

    def test_row_maps_impose_the_projected_form(self):
        r, V = self.r, self.V
        T = np.einsum("xi,xj,xk->ijk", V, V, V)
        rng = np.random.default_rng(2)
        q, alpha = rng.uniform(-1., 1., r), .7
        for name, width in (("midcourse-galerkin-chemo", r + 1), ("midcourse-logistic-chemo", r + 2)):
            case = CASES[name]
            with self.subTest(case=name):
                rom = _rom(case, r)
                maps = galerkin_row_maps(case.structure, rom, V, r)
                self.assertEqual(maps.shape, (r, rom.model.operator_matrix.shape[1], width))
                theta = rng.standard_normal((r, width))
                O = np.einsum("imk,ik->im", maps, theta)
                features = np.asarray(rom.model._assemble_data_matrix(q[:, None], inputs=np.array([[alpha]])),
                                      dtype=float)[0]
                crowding = np.einsum("ijk,j,k->i", T, q, q) * theta[:, r] if width == r + 2 else 0.
                expected = theta[:, :r] @ q + crowding + alpha * theta[:, -1] * q
                np.testing.assert_allclose(O @ features, expected, rtol=1e-5)
        with self.assertRaisesRegex(ValueError, "needs a AN ROM"):
            galerkin_row_maps("galerkin", _rom(CASES["midcourse-logistic-chemo"], r), V, r)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            galerkin_row_maps("galerkin", _rom(CASES["midcourse-galerkin-chemo"], r), 2. * V, r)

    def test_structured_layouts_and_quadratic_forecast_field(self):
        r, rng = self.r, np.random.default_rng(3)
        h = r * (r + 1) // 2
        q, alpha = rng.uniform(-1., 1., r), .4
        ckron = np.concatenate([q[a] * q[:a + 1] for a in range(r)])
        for operators, width in (("AN", 2 * r), ("AHN", 2 * r + h)):
            with self.subTest(operators=operators):
                fitted = {"O_point": rng.normal(size=(r, width)), "O_samples": rng.normal(size=(4, r, width)),
                          "conditional_root": rng.normal(size=(r, width, width))}
                out = canonical_layout(fitted, operators, r)
                total = 2 * r + 2 + (h if "H" in operators else 0)
                self.assertEqual(out["O_samples"].shape, (4, r, total))
                kept = np.r_[1:r + 1, r + 2:total]
                np.testing.assert_array_equal(out["O_point"][:, [0, r + 1]], 0.)
                np.testing.assert_array_equal(out["conditional_root"][:, [0, r + 1]], 0.)
                np.testing.assert_array_equal(out["conditional_root"][:, :, [0, r + 1]], 0.)
                O = fitted["O_point"]
                A, rest = O[:, :r], O[:, r:]
                H, N = (rest[:, :h], rest[:, h:]) if "H" in operators else (np.zeros((r, h)), rest)
                np.testing.assert_array_equal(out["O_point"][:, kept], np.hstack([A, N] + ([H] if "H" in operators
                                                                                           else [])))
                field = ProductionField(out["O_point"], lambda t: alpha)
                np.testing.assert_allclose(field.rhs(0., q), A @ q + H @ ckron + alpha * N @ q, rtol=1e-12)
        with self.assertRaises(ValueError):
            ProductionField(np.zeros((r, 2 * r + 3)), lambda t: alpha)


if __name__ == "__main__":
    unittest.main()

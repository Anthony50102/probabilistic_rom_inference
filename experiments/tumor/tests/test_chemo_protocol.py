import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chemo_protocol import projected_noise_variances


class NoiseProjectionTests(unittest.TestCase):
    def test_declared_masked_noise_and_exact_initial_state(self):
        clean = np.array([[2., 2., 2.], [4., 0., 4.]])
        entries = np.eye(2)
        np.testing.assert_allclose(
            projected_noise_variances(entries, clean, .1), [.16, .08])
        rotation = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
        expected = np.diag(rotation.T @ np.diag([.16, .08]) @ rotation)
        np.testing.assert_allclose(
            projected_noise_variances(rotation, clean, .1), expected)

    def test_variances_scale_with_measurement_units_and_noise_amplitude(self):
        clean = np.array([[2., 2., 2.], [4., 0., 4.]])
        entries = np.eye(2)
        variance = projected_noise_variances(entries, clean, .1)
        np.testing.assert_allclose(
            projected_noise_variances(entries, clean * 10, .1), variance * 100)
        np.testing.assert_allclose(
            projected_noise_variances(entries, clean, .3), variance * 9)
        np.testing.assert_array_equal(
            projected_noise_variances(entries, clean, 0), np.zeros(2))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from core.weakform_opinf.features import integrate_linear_input, input_trend_features


class InputFeatureTests(unittest.TestCase):
    def test_exact_piecewise_linear_primitive(self):
        times = np.array([0., .25, 1., 1.5, 2.])
        np.testing.assert_allclose(
            integrate_linear_input([0., 1., 2.], [0., 2., 2.], times),
            [0., .0625, 1., 2., 3.])
        self.assertEqual(integrate_linear_input([0., 1., 2.], [0., 2., 2.], 2.), 3.)
        for invalid in ([-.01], [2.01], [np.nan]):
            with self.assertRaises(ValueError):
                integrate_linear_input([0., 1., 2.], [0., 2., 2.], invalid)

    def test_feature_derivatives_and_training_only_normalization(self):
        train = np.array([.2, .4, .9, 1.2])
        query = np.array([.3, .7, 1.5])
        grid, values = [0., 1., 2.], [0., 2., 2.]
        features = input_trend_features(train, query, grid, values)
        np.testing.assert_allclose(features["train"][:, 1:].mean(0), 0., atol=1e-14)
        np.testing.assert_allclose(features["train"][:, 1:].std(0), 1.)
        epsilon = 1e-5
        plus = input_trend_features(train, query + epsilon, grid, values)
        minus = input_trend_features(train, query - epsilon, grid, values)
        np.testing.assert_allclose(
            (plus["eval"] - minus["eval"]) / (2 * epsilon), features["derivative"],
            rtol=1e-9, atol=1e-9)
        extended = input_trend_features(train, np.r_[query, 2.], grid, values)
        np.testing.assert_array_equal(features["center"], extended["center"])
        np.testing.assert_array_equal(features["scale"], extended["scale"])
        np.testing.assert_array_equal(features["eval"], extended["eval"][:len(query)])

    def test_input_scaling_changes_exposure_not_training_units(self):
        train, query = [.2, .7, 1.2], [.3, 1.5]
        first = input_trend_features(train, query, [0., 1., 2.], [0., 2., 2.])
        scaled = input_trend_features(train, query, [0., 1., 2.], [0., 4., 4.])
        np.testing.assert_allclose(scaled["scale"][1], first["scale"][1] * 2)
        np.testing.assert_allclose(scaled["eval"], first["eval"])
        np.testing.assert_allclose(scaled["derivative"], first["derivative"])
        with self.assertRaises(ValueError):
            input_trend_features(train, query, [0., 1., 2.], [0., 0., 0.])


if __name__ == "__main__":
    unittest.main()

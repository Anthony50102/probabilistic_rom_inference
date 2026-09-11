import unittest

import jax
import jax.numpy as jnp
import numpy as np

from core.weakform_opinf.features import input_trend_features
from core.weakform_opinf.gp import make_gp_conditional


class GPFeatureTests(unittest.TestCase):
    def test_feature_conditional_matches_joint_gaussian_and_derivative(self):
        with jax.experimental.enable_x64():
            train = np.array([.1, .5, 1.4])
            query = np.array([.2, 1.1])
            observations = np.array([.4, -.1, .7])
            def features(times):
                return input_trend_features(train, times, [0., 1., 2.], [0., 2., 2.])
            make = make_gp_conditional(
                train, feature_map=features, feature_variances=[2.])
            ell, variance, noise = .7, 1.2, .03
            result = make(query)[0](ell, variance, noise, observations, 2.)
            Ht, He = features(query)["train"], features(query)["eval"]
            kernel_tt = variance * np.exp(-(train[:, None] - train) ** 2 / (2 * ell ** 2))
            kernel_tt += 2 * Ht @ Ht.T + (noise + variance * 1e-4) * np.eye(3)
            kernel_et = variance * np.exp(-(query[:, None] - train) ** 2 / (2 * ell ** 2))
            kernel_et += 2 * He @ Ht.T
            expected = kernel_et @ np.linalg.solve(kernel_tt, observations)
            np.testing.assert_allclose(result[0], expected, rtol=1e-11, atol=1e-11)
            epsilon = 1e-5
            plus = make(query + epsilon)[0](ell, variance, noise, observations, 2.)[0]
            minus = make(query - epsilon)[0](ell, variance, noise, observations, 2.)[0]
            np.testing.assert_allclose((plus - minus) / (2 * epsilon), result[1],
                                       rtol=1e-8, atol=1e-8)
            def kernel(left, right):
                return (variance * np.exp(-(left[:, None] - right) ** 2 / (2 * ell ** 2))
                        + 2 * features(left)["eval"] @ features(right)["eval"].T)
            def conditional_covariance(left, right):
                return (kernel(left, right)
                        - kernel(left, train) @ np.linalg.solve(kernel_tt, kernel(train, right)))
            np.testing.assert_allclose(result[3], conditional_covariance(query, query),
                                       rtol=1e-10, atol=1e-10)
            step = 1e-4
            derivative_covariance = (
                conditional_covariance(query + step, query + step)
                - conditional_covariance(query + step, query - step)
                - conditional_covariance(query - step, query + step)
                + conditional_covariance(query - step, query - step)) / (4 * step ** 2)
            np.testing.assert_allclose(result[2], derivative_covariance,
                                       rtol=1e-5, atol=1e-6)
            for covariance in result[2:4]:
                self.assertGreaterEqual(float(np.linalg.eigvalsh(covariance).min()), -1e-10)

    def test_batched_modes_and_roundoff_nugget(self):
        with jax.experimental.enable_x64():
            train = np.array([.1, .5, 1.4])
            query = np.array([.2, 1.1])
            def features(times):
                return input_trend_features(train, times, [0., 1., 2.], [0., 2., 2.])
            single, batch = make_gp_conditional(
                train, jitter_rel=None, feature_map=features, feature_variances=[2., 3.])(query)
            ell = jnp.array([.7, .8])
            variance = jnp.array([1.2, 1.5])
            noise = jnp.array([.03, .02])
            observations = jnp.array([[.4, -.1, .7], [.3, -.4, .8]])
            result = batch(ell, variance, noise, observations)
            for mode in range(2):
                expected = single(ell[mode], variance[mode], noise[mode],
                                  observations[mode], [2., 3.][mode])
                for values, reference in zip(result, expected):
                    np.testing.assert_allclose(values[mode], reference, rtol=1e-10, atol=1e-10)

    def test_legacy_rbf_without_features(self):
        with jax.experimental.enable_x64():
            train = np.array([.1, .5, 1.4])
            observations = np.array([.4, -.1, .7])
            single, batch = make_gp_conditional(train)(train)
            expected = single(.7, 1.2, .03, observations)
            result = batch(jnp.array([.7]), jnp.array([1.2]), jnp.array([.03]),
                           observations[None])
            for values, reference in zip(result, expected):
                np.testing.assert_allclose(values[0], reference, rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            make_gp_conditional([0., 1.], feature_variances=[1.])


if __name__ == "__main__":
    unittest.main()

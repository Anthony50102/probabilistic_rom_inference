import numpy as np

from experiments.cylinder.after_the_fact.classical_baselines import (
    features, physical_operator, ridge_coefficients, sparse_coefficients, training_score,
)


def regression_fixture():
    rng = np.random.default_rng(42)
    q = np.array([.2, 2., 5.])[:, None] * rng.normal(size=(3, 300))
    design = np.vstack((features(q[:, :150], 4/3), features(q[:, 150:], 2/3)))
    operator = np.zeros((3, design.shape[1]))
    operator[0, 1], operator[1, 2], operator[2, 3] = -.7, -1.2, -.3
    operator[0, 7], operator[2, -1] = .2, .4
    response = design @ operator.T
    fs, ys = np.linalg.norm(design, axis=0), np.linalg.norm(response, axis=0)
    return design/fs, response/ys, fs, ys, operator


def test_ridge_recovers_operator_after_feature_and_output_scaling():
    X, Y, fs, ys, expected = regression_fixture()
    fitted = physical_operator(ridge_coefficients(X, Y, 1e-9), fs, ys)
    np.testing.assert_allclose(fitted, expected, atol=1e-10)


def test_sindy_recovers_sparse_operator_with_standard_unbiasing():
    X, Y, fs, ys, expected = regression_fixture()
    fitted = physical_operator(sparse_coefficients(X, Y, 1e-4, .01), fs, ys)
    np.testing.assert_allclose(fitted, expected, atol=1e-10)
    np.testing.assert_array_equal(fitted != 0, expected != 0)


def test_sindy_retained_ridge_matches_dense_ridge_at_zero_threshold():
    X, Y, _, _, _ = regression_fixture()
    np.testing.assert_allclose(sparse_coefficients(X, Y, .1, 0., unbias=False),
                               ridge_coefficients(X, Y, .1), atol=1e-10)


def test_training_score_does_not_drop_a_failed_reynolds_case():
    t = np.linspace(0, 1, 51)
    training = [dict(t_obs=t, obs=np.array([np.exp(-2*t)])) for _ in range(2)]
    operator = np.zeros((1, 5))
    operator[0, 1], operator[0, -1] = -6., 6.
    error, statuses = training_score(operator, training)
    assert error is None
    assert statuses[0]["reason"] == "cap"
    assert not statuses[0]["bounded"]
    assert statuses[1]["bounded"]

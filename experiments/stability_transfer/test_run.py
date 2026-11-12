"""The study's integration guards must distinguish boundedness and work limits."""

import unittest

import numpy as np

from experiments.stability_transfer.run import WorkBudgetExceeded, integrate


class LinearModel:
    def _extract_operators(self, operator):
        self.operator = operator

    def rhs(self, t, state, inputs):
        return self.operator @ state + (0 if inputs is None else inputs(t))


class IntegrationTests(unittest.TestCase):
    def test_autonomous_and_time_varying_input(self):
        times = np.linspace(0, 2, 101)
        status, sol = integrate(LinearModel(), -np.eye(2), np.array([1., 2.]),
                                None, times, 100., 1.)
        self.assertTrue(status["complete"])
        np.testing.assert_allclose(sol, np.array([1., 2.])[:, None] * np.exp(-times),
                                   rtol=2e-7, atol=2e-9)
        status, sol = integrate(LinearModel(), np.zeros((2, 2)), np.zeros(2),
                                lambda t: np.array([np.sin(t), np.cos(t)]), times, 100., 1.)
        self.assertTrue(status["complete"])
        np.testing.assert_allclose(sol, [1-np.cos(times), np.sin(times)], atol=1e-6)

    def test_amplitude_and_work_censoring_are_distinct(self):
        status, sol = integrate(LinearModel(), 10*np.eye(2), np.ones(2),
                                None, np.linspace(0, 2, 101), 2., 1.)
        self.assertEqual(status["reason"], "amplitude")
        self.assertIsNone(sol)

        class ExhaustedModel(LinearModel):
            def rhs(self, t, state, inputs):
                raise WorkBudgetExceeded

        status, sol = integrate(ExhaustedModel(), np.eye(2), np.ones(2),
                                None, np.linspace(0, 2, 101), 100., 1.)
        self.assertEqual(status["reason"], "work budget")
        self.assertIsNone(sol)


if __name__ == "__main__":
    unittest.main()

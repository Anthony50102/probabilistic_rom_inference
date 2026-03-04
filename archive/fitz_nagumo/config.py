# config.py
"""Configuration for FitzHugh-Nagumo experiments.

This experiment reduces the lifted variables (q1, q2, q3=q1^2) jointly and
learns a ROM with the quadratic structure:
    dq/dt = c + Aq + H[q x q] + B[u] + N[u x q]
"""

__all__ = [
    "spatial_domain",
    "time_domain",
    "initial_conditions",
    "FullOrderModel",
    "Basis",
    "ReducedOrderModel",
    "CONSTANT_VALUE_BOUNDS",
    "LENGTH_SCALE_BOUNDS",
    "NOISE_LEVEL_BOUNDS",
    "N_RESTARTS_OPTIMIZER",
]

import os
import sys
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import gammaln
from typing import override

import opinf

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core import pde_models as pdes


# =============================================================================
# Simulation specifications
# =============================================================================
spatial_domain = np.linspace(0, 1, 512)  # Spatial domain x
time_domain = np.linspace(0, 4, 401)     # Temporal domain t
initial_conditions = None
a = 50000.0  # First parameter for Neumann BC
b = 15.0     # Second parameter for Neumann BC


# =============================================================================
# Full-Order Model
# =============================================================================
class FullOrderModel(pdes.FitzHughNagumo):
    """Full-order model for FitzHugh-Nagumo equations."""

    def __init__(self):
        super().__init__(spatial_domain, a=a, b=b)


# =============================================================================
# POD Basis with Quadratic Lifting
# =============================================================================
class Basis(opinf.basis.PODBasis):
    """Basis for states of the form (q1, q2, q1^2).
    A separate POD basis is used for each state variable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_vectors = kwargs['num_vectors']

    def fit(self, states):
        """Construct the bases.

        Only fits once; subsequent calls are a no-op so that
        opinf.ROM.fit() does not silently overwrite the basis.
        Call refit() to force re-fitting.
        """
        if self.entries is not None:
            return self
        q1, q2 = np.split(states, 2, axis=0)
        return super().fit(np.concatenate((q1, q2, q1**2)))

    def refit(self, states):
        """Force re-fit (clears existing basis first)."""
        self._LinearBasis__entries = None
        return self.fit(states)

    @override
    def compress(self, state: np.ndarray) -> np.ndarray:
        """Map high-dimensional states to low-dimensional coordinates."""
        q1, q2 = np.split(state, 2, axis=0)
        return super().compress(np.concatenate((q1, q2, q1**2)))

    @override
    def decompress(self, states_compressed: np.ndarray, locs=None) -> np.ndarray:
        """Map low-dimensional coordinates to high-dimensional states."""
        q = super().decompress(states_compressed)
        q1, q2, _ = np.split(q, 3, axis=0)
        return np.concatenate((q1, q2))


# =============================================================================
# JAX-compatible operator utilities
# =============================================================================
def binom(x, y):
    return jnp.exp(gammaln(x + 1) - gammaln(y + 1) - gammaln(x - y + 1))


def Quadraticckron(state):
    return jnp.concatenate(
        [state[i] * state[: i + 1] for i in range(state.shape[0])],
        axis=0,
    )


def khatri_rao(a, b):
    return jnp.vstack([jnp.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T


# =============================================================================
# Reduced-Order Model
# =============================================================================
class ReducedOrderModel(opinf.models.ContinuousModel):
    """Reduced-order model for FitzHugh-Nagumo."""

    ivp_method = "Radau"
    input_dimension = 1

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('operators', "cAHBN")
        super().__init__(*args, **kwargs)

    @staticmethod
    def input_func(t):
        return FullOrderModel.left_neumann_condition(t, a, b)

    @staticmethod
    def input_func_jax(t):
        return FullOrderModel.left_neumann_condition_jax(t, a, b)

    def _assemble_data_matrix(self, states, inputs):
        """Assemble the data matrix for operator inference."""
        blocks = []
        for i in self._indices_of_operators_to_infer:
            op = self.operators[i]
            if isinstance(op, opinf.operators.ConstantOperator):
                block = jnp.ones((1, jnp.atleast_1d(states).shape[-1]))
            elif isinstance(op, opinf.operators.LinearOperator):
                block = jnp.atleast_2d(states)
            elif isinstance(op, opinf.operators.QuadraticOperator):
                block = Quadraticckron(jnp.atleast_2d(states))
            elif isinstance(op, opinf.operators.InputOperator):
                block = jnp.atleast_2d(inputs)
            elif isinstance(op, opinf.operators.StateInputOperator):
                block = khatri_rao(jnp.atleast_2d(inputs), jnp.atleast_2d(states))
            else:
                raise ValueError(f"Unknown operator type: {type(op)}")
            blocks.append(block.T)
        return jnp.hstack(blocks)


# =============================================================================
# GP Kernel Hyperparameters
# =============================================================================
CONSTANT_VALUE_BOUNDS = (1e-5, 1e5)
LENGTH_SCALE_BOUNDS = (1e-5, 1e2)
NOISE_LEVEL_BOUNDS = (1e-16, 1e2)
N_RESTARTS_OPTIMIZER = 100

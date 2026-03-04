# config.py
"""Configuration for viscous Burgers equation experiments.

This experiment learns a ROM with the quadratic structure:
    dq/dt = c + Aq + H[q x q]
No input operators — the Burgers equation is autonomous.
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

import opinf

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core import pde_models as pdes


# =============================================================================
# Simulation specifications
# =============================================================================
spatial_domain = np.linspace(0, 1, 256, endpoint=False)  # Periodic domain
time_domain = np.linspace(0, 1.0, 401)
initial_conditions = pdes.Burgers.initial_conditions(spatial_domain)


# =============================================================================
# Full-Order Model
# =============================================================================
class FullOrderModel(pdes.Burgers):
    """Full-order model for the viscous Burgers equation."""

    def __init__(self):
        super().__init__(spatial_domain, nu=0.01)


# =============================================================================
# POD Basis
# =============================================================================
class Basis(opinf.basis.PODBasis):
    """Simple POD basis with fit-once pattern."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, states):
        """Fit once; subsequent calls are a no-op so that
        opinf.ROM.fit() does not silently overwrite the basis.
        Call refit() to force re-fitting.
        """
        if self.entries is not None:
            return self
        return super().fit(states)

    def refit(self, states):
        """Force re-fit (clears existing basis first)."""
        self._LinearBasis__entries = None
        return self.fit(states)


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
    """Reduced-order model for the viscous Burgers equation."""

    OPERATORS = "cAH"
    ivp_method = "RK45"
    input_dimension = 0
    input_func = None

    def __init__(self):
        super().__init__(self.OPERATORS)

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

# config.py
"""Configuration for 2D diffusion-reaction equation experiments.

PDE:  ∂u/∂t = κ ∇²u − β u²   on [0,1]² with Dirichlet BCs.

This experiment learns a ROM with quadratic structure:
    dq/dt = c + Aq + H[q ⊗ q]

The quadratic reaction term −βu² maps naturally to cAH operators
(no lifting or nondimensionalization needed).
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
import diffrax

import opinf
from opinf.operators import (
    ConstantOperator,
    LinearOperator,
    QuadraticOperator,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.pde_models import DiffusionReaction2D


# =============================================================================
# Simulation specifications
# =============================================================================
spatial_domain = np.linspace(0, 1, 64)          # for compatibility
time_domain = np.linspace(0, 3.0, 301)          # full temporal domain

_fom_tmp = DiffusionReaction2D(nx=64, ny=64, kappa=0.01, beta=1.0)
initial_conditions = _fom_tmp.initial_conditions('multimode')
del _fom_tmp


# =============================================================================
# Full-Order Model
# =============================================================================
class FullOrderModel(DiffusionReaction2D):
    """Full-order model for the 2D diffusion-reaction equation."""

    def __init__(self):
        super().__init__(nx=64, ny=64, kappa=0.01, beta=1.0)


# =============================================================================
# POD Basis
# =============================================================================
class Basis(opinf.basis.PODBasis):
    """Standard POD basis — no lifting or scaling needed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, states):
        if self.entries is not None:
            return self
        return super().fit(states)

    def refit(self, states):
        self._LinearBasis__entries = None
        return self.fit(states)


# =============================================================================
# JAX-compatible operator utilities
# =============================================================================
def khatri_rao(a, b):
    return jnp.vstack([jnp.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T


def constant_apply(state, input_, entries):
    if entries.shape[0] == 1:
        if state is None or np.isscalar(state):
            return entries[0]
        return np.full_like(state, entries[0])
    if np.ndim(state) == 2:
        return np.outer(entries, np.ones(state.shape[-1]))
    return entries


def linear_apply(state, entries):
    if entries.shape[0] == 1:
        return entries[0, 0] * state
    return entries @ state


def quadratic_apply(state, entries, _mask):
    if entries.shape[0] == 1:
        return entries[0, 0] * state**2
    return entries @ jnp.prod(state[_mask], axis=1)


# =============================================================================
# Reduced-Order Model
# =============================================================================
class ReducedOrderModel(opinf.models.ContinuousModel):
    """Reduced-order model for 2D diffusion-reaction with JAX support."""

    ivp_method = "RK45"
    input_dimension = 0
    input_func = None

    def __init__(self):
        super().__init__("cAH")

    def rhs(self, state, input_):
        out = jnp.zeros_like(state)
        for op in self.operators:
            if isinstance(op, ConstantOperator):
                out += constant_apply(state, input_, op.entries)
            elif isinstance(op, LinearOperator):
                out += linear_apply(state, op.entries)
            elif isinstance(op, QuadraticOperator):
                out += quadratic_apply(state, op.entries, op._mask)
            else:
                raise RuntimeError(f"Unknown operator: {type(op)}")
        return out

    def predict(self, state0, t):
        def f(t, y, args):
            input_func, = args
            return self.rhs(t, y, self.input_func)

        solver = diffrax.Tsit5()
        rtol, atol = 1e-6, 1e-9
        saveat = diffrax.SaveAt(ts=t)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f),
            solver=solver,
            t0=float(t[0]),
            t1=float(t[-1]),
            dt0=(t[1] - t[0]),
            y0=state0,
            args=(self.input_func,),
            rtol=rtol,
            atol=atol,
            saveat=saveat,
        )
        return sol.ys.T


# =============================================================================
# GP Kernel Hyperparameters
# =============================================================================
CONSTANT_VALUE_BOUNDS = (1e-5, 1e5)
LENGTH_SCALE_BOUNDS = (1e-5, 1e2)
NOISE_LEVEL_BOUNDS = (1e-16, 1e2)
N_RESTARTS_OPTIMIZER = 100

# config.py
"""Configuration for Compressible Euler equations experiments.

This experiment reduces the lifted variables (u, q, 1/rho) jointly and 
learns a ROM with the quadratic structure:
    dq/dt = c + Aq + H[q x q]
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
import types
import numpy as np
import jax.numpy as jnp
import diffrax

import opinf
from opinf.operators import (
    ConstantOperator,
    LinearOperator,
    QuadraticOperator,
    CubicOperator,
    InputOperator,
    StateInputOperator,
)

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core import pde_models as pdes


# =============================================================================
# Simulation specifications
# =============================================================================
spatial_domain = np.linspace(0, 2, 201)[:-1]  # Spatial domain x
time_domain = np.linspace(0, 0.15, 401)       # Temporal domain t
init_params = [22, 20, 24, 95, 105, 100]
initial_conditions = pdes.Euler(spatial_domain).initial_conditions(
    init_params=init_params,
    plot=False,
)


# =============================================================================
# Full-Order Model
# =============================================================================
class FullOrderModel(pdes.Euler):
    """Full-order model for compressible Euler equations."""

    def __init__(self):
        super().__init__(spatial_domain)


# =============================================================================
# POD Basis with Nondimensionalization
# =============================================================================
class Basis(opinf.basis.PODBasis):
    """Basis for this problem: POD, treating all three variables jointly,
    with some nondimensionalization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _v, _rho = 100, 10
        self.__scalers = np.array([_v, _rho * _v**2, 1 / _rho])

    @property
    def scalers(self):
        return self.__scalers

    def nondimensionalize(self, states):
        return np.concatenate(
            [var / s for var, s in zip(np.split(states, 3), self.scalers)]
        )

    def redimensionalize(self, states):
        return np.concatenate(
            [var * s for var, s in zip(np.split(states, 3), self.scalers)]
        )

    def fit(self, states):
        """Fit once; subsequent calls are a no-op so that
        opinf.ROM.fit() does not silently overwrite the basis.
        Call refit() to force re-fitting.
        """
        if self.entries is not None:
            return self
        states, self.shift_ = opinf.pre.shift(states)
        return super().fit(self.nondimensionalize(states))

    def refit(self, states):
        """Force re-fit (clears existing basis first)."""
        self._LinearBasis__entries = None
        return self.fit(states)

    def compress(self, states):
        is_1d = states.ndim == 1
        if is_1d:
            states = states.reshape(-1, 1)
        states = opinf.pre.shift(states, shift_by=self.shift_)
        states = self.nondimensionalize(states)
        compressed = super().compress(states)
        if is_1d:
            return compressed.flatten()
        return compressed

    def decompress(self, states, locs=None):
        decompressed = super().decompress(states, locs=locs)
        decompressed = self.redimensionalize(decompressed)
        decompressed = opinf.pre.shift(decompressed, shift_by=-self.shift_)
        return decompressed


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


def cubic_apply(state, entries, _mask):
    if entries.shape[0] == 1:
        return entries[0, 0] * state**3
    return entries @ np.prod(state[_mask], axis=1)


def input_apply(input_, entries):
    if entries.shape[1] == 1 and (dim := jnp.ndim(input_)) != 2:
        if entries.shape[0] == 1:
            return entries[0, 0] * input_
        if dim == 1 and input_.size > 1:
            return jnp.outer(entries[:, 0], input_)
        return entries[:, 0] * input_
    return entries @ input_


def state_input_apply(state, input_, entries, shape):
    multi = (sdim := np.ndim(state)) > 1
    multi |= (idim := np.ndim(input_)) > 1
    multi |= shape[0] == 1 and sdim == 1 and state.shape[0] > 1
    multi |= shape[1] == 1 and idim == 1 and input_.shape[0] > 1
    single = not multi

    if shape[1] == 1:
        return entries[0, 0] * input_ * state
    if single:
        return entries @ jnp.kron(input_, state)
    Q_ = np.atleast_2d(state)
    U = np.atleast_2d(input_)
    return entries @ khatri_rao(U, Q_)


# =============================================================================
# Reduced-Order Model
# =============================================================================
class ReducedOrderModelOriginal(opinf.models.ContinuousModel):
    """Original reduced-order model for Euler equations."""
    
    ivp_method = "RK45"
    input_dimension = 0

    def __init__(self):
        super().__init__("cAH")

    input_func = None


class ReducedOrderModel(opinf.models.ContinuousModel):
    """Reduced-order model for Euler equations with JAX support."""

    ivp_method = "RK45"
    input_dimension = 0
    input_func = None

    def rhs(self, state, input_):
        out = jnp.zeros_like(state)
        for op in self.operators:
            if isinstance(op, ConstantOperator):
                out += constant_apply(state, input_, op.entries)
            elif isinstance(op, LinearOperator):
                out += linear_apply(state, op.entries)
            elif isinstance(op, QuadraticOperator):
                out += quadratic_apply(state, op.entries, op._mask)
            elif isinstance(op, CubicOperator):
                out += cubic_apply(state, op.entries, op._mask)
            elif isinstance(op, InputOperator):
                out += input_apply(input_, op.entries)
            elif isinstance(op, StateInputOperator):
                out += state_input_apply(state, input_, op.entries, op.shape)
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
            saveat=saveat
        )
        return sol.ys.T


# =============================================================================
# GP Kernel Hyperparameters
# =============================================================================
CONSTANT_VALUE_BOUNDS = (1e-5, 1e5)
LENGTH_SCALE_BOUNDS = (1e-5, 1e2)
NOISE_LEVEL_BOUNDS = (1e-16, 1e2)
N_RESTARTS_OPTIMIZER = 100

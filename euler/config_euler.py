# config_heatcubiclifted.py
"""Configuration for Euler equations in specific volume variables.

This experiment reduces the lifted variables (u, q, 1/rho) jointly and learns
a ROM with the quadratic structure dq/dt = H[q x q].
"""

__all__ = [
    # Simulation specifics
    "spatial_domain",
    "time_domain",
    "init_params",
    "initial_conditions",
    # Simulation classes
    "FullOrderModel",
    "Basis",
    "ReducedOrderModel",
    # GP kernel fitting hyperparameters
    "CONSTANT_VALUE_BOUNDS",
    "LENGTH_SCALE_BOUNDS",
    "NOISE_LEVEL_BOUNDS",
    "N_RESTARTS_OPTIMIZER",
]

import numpy as np
import jax.numpy as jnp

import opinf
import jax.numpy as jnp
import types
from opinf.operators import (
    ConstantOperator,
    LinearOperator,
    QuadraticOperator,
    CubicOperator,
    InputOperator,
    StateInputOperator,
)
from opinf.models import ContinuousModel

# 1) Save originals so your overrides can invoke them:
_orig_constant_set_entries      = ConstantOperator.set_entries
_orig_linear_set_entries        = LinearOperator.set_entries
_orig_quadratic_set_entries     = QuadraticOperator.set_entries
_orig_cubic_set_entries         = CubicOperator.set_entries
_orig_input_set_entries         = InputOperator.set_entries
_orig_state_input_set_entries   = StateInputOperator.set_entries
import diffrax

import pde_models as pdes


# Simulation specifications  --------------------------------------------------
spatial_domain = np.linspace(0, 2, 201)[:-1]  # Spatial domain x.
time_domain = np.linspace(0, 0.15, 401)  # Temporal domain t.
init_params = [22, 20, 24, 95, 105, 100]
initial_conditions = pdes.Euler(spatial_domain).initial_conditions(
    init_params=init_params,
    plot=False,
)  # Initial conditions q(x, 0).


# Simulation classes ----------------------------------------------------------
class FullOrderModel(pdes.Euler):
    """Full-order model for this problem."""

    def __init__(self):
        super().__init__(spatial_domain)


class Basis(opinf.basis.PODBasis):
    """Basis for this problem: POD, treating all three variables jointly,
    with some nondimensionaliziation.
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
        states, self.shift_ = opinf.pre.shift(states)
        print(states.shape, self.nondimensionalize(states).shape)
        return super().fit(self.nondimensionalize(states))

    def compress(self, states):
        # Handle both 1D (initial condition) and 2D (snapshot matrix) inputs
        is_1d = states.ndim == 1
    
        if is_1d:
            # Reshape 1D to 2D for processing
            states = states.reshape(-1, 1)
        # Apply shift and nondimensionalization
        states = opinf.pre.shift(states, shift_by=self.shift_)
        states = self.nondimensionalize(states)
        
        # Apply parent compress method
        compressed = super().compress(states)
        
        if is_1d:
            # Return 1D result for 1D input
            return compressed.flatten()
        
        return compressed

    def decompress(self, states, locs=None):
        # First apply parent decompress method (which handles the locs parameter)
        decompressed = super().decompress(states, locs=locs)
        
        # Then apply your custom post-processing in reverse order:
        # parent.decompress -> redimensionalize -> unshift
        decompressed = self.redimensionalize(decompressed)
        decompressed = opinf.pre.shift(decompressed, shift_by=-self.shift_)
        
        return decompressed


class ReducedOrderModel(opinf.models.ContinuousModel):
    """Reduced-order model for this problem."""

    ivp_method = "RK45"
    input_dimension = 0

    # def __init__(self, *args, **kwargs):
    #     self.input_func = None
    #     super().__init__("cAH", *args, **kwargs)

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
                raise RuntimeError
            # out += op.apply(state, input_)
        return out

    def predict(self, state0, t):
        # 1) Wrap your vector field as f(t, y, args)
        def f(t, y, args):
            input_func, = args        # unpack your args‐tuple
            return self.rhs(t, y, self.input_func)

        # 2) Choose solver and tolerances
        solver = diffrax.Tsit5()      # classical Runge–Kutta–Fehlberg 4(5)
        rtol, atol = 1e-6, 1e-9

        # 3) Set up the SaveAt to sample exactly at your t grid
        saveat = diffrax.SaveAt(ts=t)

        # 4) Call diffeqsolve
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f),
            solver=solver,
            t0=float(t[0]),
            t1=float(t[-1]),
            dt0= (t[1] - t[0]),       # initial guess for step size
            y0=state0,
            args=(self.input_func,),
            rtol=rtol,
            atol=atol,
            saveat=saveat
        )
    
    def _extract_operators(self, Ohat):
        index = 0
        for i in self._indices_of_operators_to_infer:
            op  = self.operators[i]
            dim = op.operator_dimension(self.state_dimension,
                                        self.input_dimension)
            end = index + dim
            blk = Ohat[:, index:end]

            # bind & call the right override on this instance:
            if isinstance(op, ConstantOperator):
                op.set_entries = types.MethodType(constant_set_entries, op)
            elif isinstance(op, LinearOperator):
                op.set_entries = types.MethodType(linear_set_entries, op)
            elif isinstance(op, QuadraticOperator):
                op.set_entries = types.MethodType(quadratic_set_entries, op)
            elif isinstance(op, CubicOperator):
                op.set_entries = types.MethodType(cubic_set_entries, op)
            elif isinstance(op, InputOperator):
                op.set_entries = types.MethodType(input_set_entries, op)
            elif isinstance(op, StateInputOperator):
                op.set_entries = types.MethodType(state_input_set_entries, op)
            else:
                raise RuntimeError(f"Unknown operator: {type(op)}")

            op.set_entries(blk)
            index = end

def khatri_rao(a,b):
    c = jnp.vstack([jnp.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T 
    return c

def constant_apply(state, input_, entries):
    if entries.shape[0] == 1:
            if state is None or np.isscalar(state):  # r = k = 1.
                return entries[0]
            return np.full_like(state, entries[0])  # r = 1, k > 1.
        # if state is None or np.ndim(state) == 1:
        #     return self.entries
    if np.ndim(state) == 2:  # r, k > 1.
        return np.outer(entries, np.ones(state.shape[-1]))
    return entries  # r > 1, k = 1.

def linear_apply(state, entries,):
    if entries.shape[0] == 1:
        return entries[0, 0] * state  # r = 1.
    return entries @ state  # r > 1.

def quadratic_apply(state, entries, _mask):
    if entries.shape[0] == 1:
        return entries[0, 0] * state**2  # r = 1
    return entries @ jnp.prod(state[_mask], axis=1)

def cubic_apply(state, entries, _mask):
    if entries.shape[0] == 1:
        return entries[0, 0] * state**3  # r = 1.
    return entries @ np.prod(state[_mask], axis=1)

def input_apply(input_, entries):
    if entries.shape[1] == 1 and (dim := jnp.ndim(input_)) != 2:
        if entries.shape[0] == 1:
            return entries[0, 0] * input_  # r = m = 1.
        if dim == 1 and input_.size > 1:  # r, k > 1, m = 1.
            return jnp.outer(entries[:, 0], input_)
        return entries[:, 0] * input_  # r > 1, m = k = 1.
    return entries @ input_  # m > 1.

def state_input_apply(state, input_, entries, shape):
    # Determine if arguments represent one snapshot or several.
    multi = (sdim := np.ndim(state)) > 1
    multi |= (idim := np.ndim(input_)) > 1
    multi |= shape[0] == 1 and sdim == 1 and state.shape[0] > 1
    multi |= shape[1] == 1 and idim == 1 and input_.shape[0] > 1
    single = not multi

    if shape[1] == 1:
        return entries[0, 0] * input_ * state  # r = m = 1.
    if single:
        return entries @ jnp.kron(input_, state)  # k = 1, rm > 1.
    Q_ = np.atleast_2d(state)
    U = np.atleast_2d(input_)
    return entries @ khatri_rao(U, Q_)  # k > 1, rm > 1.

def constant_set_entries(self, entries):
    # always 1D
    arr = jnp.array(entries)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
    elif arr.ndim != 1:
        raise ValueError("ConstantOperator entries must be one-dimensional")
    # directly assign
    self.entries = arr

def linear_set_entries(self, entries):
    arr = jnp.array(entries)
    # allow scalars or 1×1
    if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
        arr = arr.reshape((1,1))
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("LinearOperator entries must be square (r×r)")
    self.entries = arr

def quadratic_set_entries(self, entries):
    arr = jnp.array(entries)
    r = arr.shape[0]
    # handle full tensor form
    if arr.ndim == 3 and arr.shape == (r,r,r):
        arr = arr.reshape((r, r*r))
    if arr.ndim != 2:
        raise ValueError("QuadraticOperator entries must be 2D or 3D symmetric")
    # compress full-r^2 -> r(r+1)/2 if needed
    if arr.shape[1] == r*r:
        # use numpy helper then wrap
        from opinf.operators import QuadraticOperator as Qop
        arr = jnp.array(Qop.compress_entries(arr))
    # precompute mask for apply/jacobian
    self._mask = QuadraticOperator.ckron_indices(r)
    self._prejac = None
    self.entries = arr

def cubic_set_entries(self, entries):
    arr = jnp.array(entries)
    r = arr.shape[0]
    # full tensor
    if arr.ndim == 4 and arr.shape == (r,r,r,r):
        arr = arr.reshape((r, r**3))
    if arr.ndim != 2:
        raise ValueError("CubicOperator entries must be 2D or 4D symmetric")
    # compress if full r^3
    if arr.shape[1] == r**3:
        from opinf.operators import CubicOperator as Cop
        arr = jnp.array(Cop.compress_entries(arr))
    self._mask = CubicOperator.ckron_indices(r)
    self._prejac = None
    self.entries = arr

def input_set_entries(self, entries):
    arr = jnp.array(entries)
    # scalars or 1D -> (r,1)
    if arr.ndim == 0 or (arr.ndim == 1 and arr.size > 1):
        arr = arr.reshape((-1, 1))
    if arr.ndim != 2:
        raise ValueError("InputOperator entries must be two-dimensional")
    self.entries = arr

def state_input_set_entries(self, entries):
    arr = jnp.array(entries)
    if arr.ndim != 2:
        raise ValueError("StateInputOperator entries must be two-dimensional")
    r, rm = arr.shape
    m, remainder = divmod(rm, r)
    if remainder != 0:
        raise ValueError("invalid StateInputOperator entries dimensions")
    self.entries = arr

# Gaussian process kernel fitting hyperparameters -----------------------------
CONSTANT_VALUE_BOUNDS = (1e-5, 1e5)
LENGTH_SCALE_BOUNDS = (1e-5, 1e2)
NOISE_LEVEL_BOUNDS = (1e-16, 1e2)
N_RESTARTS_OPTIMIZER = 100
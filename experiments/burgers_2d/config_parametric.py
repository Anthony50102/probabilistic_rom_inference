"""Parametric-IC configuration for the 2D diffusion-reaction ("burgers") experiment.

Heat-style parametric variation: operators are shared across trajectories and
the parameter enters only through the initial condition. The PDE is unchanged.

Parameterization (2D, μ = (a, b)):

    u0(x, y; a, b) = sin(πX)·sin(πY)
                   + a·sin(2πX)·sin(πY)
                   + b·sin(πX)·sin(2πY)
                   + 0.2·sin(2πX)·sin(2πY)

The base mode and a fixed high-frequency tail ensure non-trivial reaction
dynamics even at μ = 0; the two middle-mode amplitudes are the parameters.

The ROM structure stays `cAH` (no B, no N) — μ never enters the operators,
only the IC.
"""

import numpy as np

from config import (
    FullOrderModel,
    Basis,
    ReducedOrderModel,
    CONSTANT_VALUE_BOUNDS,
    LENGTH_SCALE_BOUNDS,
    NOISE_LEVEL_BOUNDS,
    N_RESTARTS_OPTIMIZER,
    spatial_domain,
    time_domain,
)

__all__ = [
    "spatial_domain",
    "time_domain",
    "FullOrderModel",
    "Basis",
    "ReducedOrderModel",
    "initial_conditions",
    "TRAINING_MUS",
    "TEST_MU",
    "CONSTANT_VALUE_BOUNDS",
    "LENGTH_SCALE_BOUNDS",
    "NOISE_LEVEL_BOUNDS",
    "N_RESTARTS_OPTIMIZER",
]


def initial_conditions(a: float, b: float) -> np.ndarray:
    """Parametric IC on the interior grid of the unit square.

    Returns a flat (nx_int * ny_int,) array matching the state layout used
    by `DiffusionReaction2D`.
    """
    fom = FullOrderModel()
    x_int = fom.x[1:-1]
    y_int = fom.y[1:-1]
    X, Y = np.meshgrid(x_int, y_int, indexing="xy")
    u0 = (
        np.sin(np.pi * X) * np.sin(np.pi * Y)
        + a * np.sin(2 * np.pi * X) * np.sin(np.pi * Y)
        + b * np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
        + 0.2 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    )
    return u0.ravel()


# Training parameters: 5 points in [0.2, 0.8]^2, diverse mix of (a, b).
TRAINING_MUS = (
    (0.3, 0.2),
    (0.5, 0.4),
    (0.7, 0.3),
    (0.4, 0.6),
    (0.6, 0.5),
)

# Held-out test parameter (interior to the training box).
TEST_MU = (0.5, 0.3)

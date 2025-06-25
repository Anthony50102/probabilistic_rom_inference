# config_fhnlifted.py
"""Configuration for FitzHugh-Nagumo experiments with quadratic lifting.

This experiment reduces the lifted variables (q1, q2, q3=q1^2) jointly and
learns a ROM with the quadratic structure
dq/dt = c + Aq + H[q x q] + B[u] + N[u x q].
"""

__all__ = [
    # Simulation specifics
    "spatial_domain",
    "time_domain",
    # Simulation classes
    "monolithic",
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

import opinf

import pde_models as pdes


# Simulation specifications  --------------------------------------------------
spatial_domain = np.linspace(0, 1, 512)  # Spatial domain x.
time_domain = np.linspace(0, 4, 401)  # Temporal domain t.
initial_conditions = None
a = 50000.0  # first parameter for Neumann BC.
b = 15.0  # second parameter for Neumann BC.


# Simulation classes ----------------------------------------------------------
class FullOrderModel(pdes.FitzHughNagumo):
    """Full-order model for this problem."""

    def __init__(self):
        """Initialized solver with default parameters."""
        super().__init__(spatial_domain, a=a, b=b)


class Basis(opinf.basis.PODBasis):
    """Basis for states of the form (q1, q2, q1^2).
    A separate POD basis is used for each state variable.
    """

    # def fit(self, states, r):
    #     """Construct the bases."""
    #     q1, q2 = np.split(states, 2, axis=0)

    #     return super().fit(
    #         np.concatenate((q1, q2, q1**2)),
    #         r,
    #     )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_vectors = kwargs['num_vectors']

    def fit(self, states):
        """Construct the bases."""
        q1, q2 = np.split(states, 2, axis=0)

        print(q1.shape, q2.shape)
        return super().fit(
            np.concatenate((q1, q2, q1**2)),
            )


    def compress(self, states):
        """Map high-dimensional states to low-dimensional coordinates."""
        q1, q2 = np.split(states, 2, axis=0)
        return super().compress(
            np.concatenate((q1, q2, q1**2)),
        )

    def decompress(self, states_compressed, **kwargs):
        """Map low-dimensional coordinates to high-dimensional states."""
        q = super().decompress(states_compressed)
        q1, q2, _ = np.split(q, 3, axis=0)
        return np.concatenate((q1, q2))


class ReducedOrderModel(opinf.models.ContinuousModel):
    """Reduced-order model for this problem."""

    ivp_method = "Radau"
    input_dimension = 1

    def __init__(self, *args, **kwargs):
        # ensure that the base class sees your default operator string
        kwargs.setdefault('operators', "cAHBN")
        super().__init__(*args, **kwargs)

    @staticmethod
    def input_func(t):
        return FullOrderModel.left_neumann_condition(t, a, b)
    
    @staticmethod
    def input_func_jax(t):
       return FullOrderModel.left_neumann_condition_jax(t, a, b) 

    @staticmethod
    def full_rhs(t):
        pass


monolithic = True


# Gaussian process kernel fitting hyperparameters -----------------------------
CONSTANT_VALUE_BOUNDS = (1e-5, 1e5)
LENGTH_SCALE_BOUNDS = (1e-5, 1e2)
NOISE_LEVEL_BOUNDS = (1e-16, 1e2)
N_RESTARTS_OPTIMIZER = 100

# config.py
"""General configuration file for logger, figures folders, etc."""
import os
import sys
import time
import logging
import numpy as np

# Paths -----------------------------------------------------------------------
FIGURES_FOLDER = os.path.join(
    "figures",
    time.strftime("%b%d").lower(),
    time.strftime("%H-%M-%S"),
)
LOG_FILE = "log.log"


def TRNFMT(k: int) -> str:
    """String format for training sizes."""
    return f"trainsize{k:0>3d}"


def SPRSFMT(sparsity: float) -> str:
    """String format for sparsity percentages."""
    return f"sparsity{int(sparsity*100):0>3d}"


def NOISEFMT(level: float) -> str:
    """Label for datasets with noise percentage ``level``."""
    return "noise000" if not level else f"noise{int(level*100):0>3d}"


def DIMFMT(stateindex: int) -> str:
    """String format for state variable index."""
    return f"r_{int(stateindex)+1:0>2d}"


def _makefolder(*args) -> str:
    """Join arguments into a path to a folder. If the folder doesn't exist,
    make the folder as well. Return the resulting path.
    """
    folder = os.path.join(*args)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def figures_path() -> str:
    """Return the path to the folder containing all results figures."""
    # return _makefolder(BASE_FOLDER, FIGURES_FOLDER)   # Figures live by data.
    return _makefolder(os.getcwd(), FIGURES_FOLDER)  # Figures live by code.


# Initialize logger -----------------------------------------------------------
_handler = logging.FileHandler(LOG_FILE, "a")
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
_handler.setLevel(logging.INFO)
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)

# Log the session header.
if hasattr(sys.modules["__main__"], "__file__"):
    _front = f"({os.path.basename(sys.modules['__main__'].__file__)})"
    _end = time.strftime("%Y-%m-%d %H:%M:%S")
    _mid = "-" * (79 - len(_front) - len(_end) - 20)
    _header = f"NEW SESSION {_front} {_mid} {_end}"
else:
    _header = f"NEW SESSION {time.strftime(' %Y-%m-%d %H:%M:%S'):->61}"
logging.info(_header)
print(f"Logging to {LOG_FILE}")


# Random seed -----------------------------------------------------------------
np.random.seed(27092023)
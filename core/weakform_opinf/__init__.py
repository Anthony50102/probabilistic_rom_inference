"""Centralised marginalised-O × weak-form Bayesian Operator Inference.

One canonical implementation of the method shared by every PDE case study
(euler, burgers, heat, tumor, tumor+chemo). Per-experiment behaviour is
expressed through :class:`WeakFormConfig` fields and an
:class:`ExperimentSpec` adapter; the algorithm itself lives here so all case
studies provably run the identical method.

No environment-variable toggles and no MLE: GP-hyperparameter priors are
spectrum-anchored and explored by SVI/NUTS.
"""

from .config import WeakFormConfig
from .experiment import ExperimentSpec, PreparedRun, EvalTarget
from .model import build_model
from .pipeline import run_experiment

__all__ = [
    "WeakFormConfig",
    "ExperimentSpec",
    "PreparedRun",
    "EvalTarget",
    "build_model",
    "run_experiment",
]

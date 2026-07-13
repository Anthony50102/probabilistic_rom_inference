"""Experiment adapter interface for the weak-form OpInf pipeline.

Each experiment provides an :class:`ExperimentSpec` whose ``prepare`` builds the
genuinely experiment-specific objects — data source, POD basis, ROM, input
function — and returns a :class:`PreparedRun`. Everything downstream (model
build, SVI/NUTS, operator sampling, prediction, metrics, saving) is shared by
:mod:`core.weakform_opinf.pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np


@dataclass
class EvalTarget:
    """A trajectory to predict + score after inference (one operator, one IC)."""
    t_pred: np.ndarray
    true_comp: np.ndarray          # (num_modes, len(t_pred)) projected truth
    true_states: np.ndarray        # full-order truth for full-order error
    state0_comp: np.ndarray        # (num_modes,) initial reduced state
    t_full: np.ndarray
    input_func: Optional[Callable] = None
    label: str = ""


@dataclass
class PreparedRun:
    """Everything the shared pipeline needs after experiment-specific prep."""
    rom: Any
    trajectories: list                 # list of {t_sampled, snapshots_comp, inputs_eval}
    basis: Any
    eval_targets: list                 # list[EvalTarget]
    training_span: tuple
    # data used for IC-uncertainty GP and diagnostics (first trajectory)
    snapshots_comp: np.ndarray
    t_sampled: np.ndarray
    extra: dict = field(default_factory=dict)


class ExperimentSpec(Protocol):
    """Protocol every experiment adapter implements."""

    name: str

    def prepare(self, cfg, schema) -> PreparedRun:
        """Build rom, trajectories, basis, and eval targets for one data regime."""
        ...

    def plot(self, result: dict, save_dir: str) -> None:
        """Optional experiment-specific plotting. May be a no-op."""
        ...

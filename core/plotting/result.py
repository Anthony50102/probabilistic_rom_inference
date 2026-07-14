"""Method-agnostic result contract for plotting.

Every method (Bayesian OpInf, Neural ODE) and every PDE case study emits a
:class:`RunResult`. A single set of plotters then serves per-run figures,
cross-method comparisons, and paper figures. This module has **no dependency**
on any particular method or on ``core.weakform_opinf`` — it is the neutral
contract both sides agree on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class TargetResult:
    """Predictions + metrics for one evaluated trajectory (one IC)."""
    label: str
    t_pred: np.ndarray
    rom_solves: np.ndarray          # (S, r, T) stable posterior/ensemble solves
    true_comp: np.ndarray           # (r, len(t_full)) projected truth
    true_states: np.ndarray         # full-order truth (for full-order error)
    t_full: np.ndarray
    state0_comp: Optional[np.ndarray] = None
    train_error: float = float("nan")
    pred_error: float = float("nan")
    stability_pct: float = float("nan")
    ci_coverage: float = float("nan")
    ci_width: float = float("nan")

    @property
    def n_stable(self) -> int:
        return 0 if self.rom_solves is None else len(self.rom_solves)


@dataclass
class RunResult:
    """One method × one data regime. The unit every plotter consumes."""
    method_name: str                # e.g. "04_unified", "05_neural_ode"
    schema: dict
    basis: Any
    training_span: tuple
    num_modes: int
    targets: list                   # list[TargetResult] (len 1 for single-IC)
    losses: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    t_samp: Optional[np.ndarray] = None
    snapshots_comp: Optional[np.ndarray] = None  # noisy obs (for scatter)
    O_samples: Optional[np.ndarray] = None
    runtime: float = float("nan")
    method_label: Optional[str] = None
    color: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        from .style import method_color, method_label
        if self.method_label is None:
            self.method_label = method_label(self.method_name)
        if self.color is None:
            self.color = method_color(self.method_name)

    @property
    def primary(self) -> TargetResult:
        return self.targets[0]

    def aggregate(self) -> dict:
        """Mean metrics over targets that produced finite errors."""
        fin = [t for t in self.targets if np.isfinite(t.train_error)]
        m = lambda k: float(np.mean([getattr(t, k) for t in fin])) if fin else float("nan")
        return dict(
            stability_pct=float(np.mean([t.stability_pct for t in self.targets])),
            train_error=m("train_error"), pred_error=m("pred_error"),
            ci_coverage=m("ci_coverage"), ci_width=m("ci_width"),
        )

    def save_npz(self, out_dir, suffix=""):
        """Write the standardised comparison-schema npz for this run.

        The primary target's solves + basis are stored so the comparison layer
        can recompute full-order errors; aggregate metrics are stored as scalars.
        """
        os.makedirs(out_dir, exist_ok=True)
        agg = self.aggregate()
        tgt = self.primary
        rom_arr = (np.asarray(tgt.rom_solves) if tgt.n_stable > 0
                   else np.empty((0, self.num_modes, len(tgt.t_pred))))
        path = os.path.join(out_dir, f"{self.method_name}{suffix}.npz")
        np.savez(
            path,
            rom_solves=rom_arr, t_pred=tgt.t_pred,
            train_error=agg["train_error"], pred_error=agg["pred_error"],
            stability_pct=agg["stability_pct"], ci_coverage=agg["ci_coverage"],
            ci_width=agg["ci_width"], runtime=self.runtime,
            num_modes=self.num_modes,
            training_span=np.array(self.training_span),
            losses=np.asarray(self.losses),
            basis_entries=np.asarray(self.basis.entries),
            true_states=np.asarray(tgt.true_states),
            **{f"extra_{k}": v for k, v in self.extra.items()
               if np.ndim(v) <= 2},
        )
        return path


@dataclass
class MethodData:
    """Lightweight per-method record for the comparison layer (from npz)."""
    name: str
    label: str
    color: str
    rom_solves: np.ndarray
    t_pred: np.ndarray
    train_error: float
    pred_error: float
    stability_pct: float
    ci_coverage: float
    ci_width: float
    runtime: float
    rom_errors: Optional[np.ndarray] = None   # filled by comparison layer


def load_comparison_npz(path, name, label=None, color=None):
    """Load a comparison-schema npz into a :class:`MethodData` (or None)."""
    if not os.path.exists(path):
        return None
    from .style import method_color, method_label
    d = np.load(path, allow_pickle=True)
    return MethodData(
        name=name,
        label=label or method_label(name),
        color=color or method_color(name),
        rom_solves=d["rom_solves"],
        t_pred=d["t_pred"],
        train_error=float(d["train_error"]),
        pred_error=float(d["pred_error"]),
        stability_pct=float(d["stability_pct"]),
        ci_coverage=float(d["ci_coverage"]) if "ci_coverage" in d else float("nan"),
        ci_width=float(d["ci_width"]) if "ci_width" in d else float("nan"),
        runtime=float(d["runtime"]) if "runtime" in d else float("nan"),
    )

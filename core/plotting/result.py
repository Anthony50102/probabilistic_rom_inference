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

    @staticmethod
    def from_flat(result, method_name):
        rs = np.asarray(result["rom_solves"]) if len(result["rom_solves"]) \
            else np.empty((0, result["num_modes"], len(result["t_pred"])))
        snaps = result.get("snaps_comp")
        tgt = TargetResult(
            label="", t_pred=np.asarray(result["t_pred"]), rom_solves=rs,
            true_comp=result["true_comp"], true_states=result["true_states"],
            t_full=result["t_full"],
            state0_comp=(snaps[:, 0] if snaps is not None else None),
            train_error=float(result.get("train_error", np.nan)),
            pred_error=float(result.get("pred_error", np.nan)),
            stability_pct=float(result.get("stability_pct", np.nan)),
            ci_coverage=float(result.get("ci_coverage", np.nan)),
            ci_width=float(result.get("ci_width", np.nan)))
        return RunResult(
            method_name=method_name, schema=result["schema"],
            basis=result["basis"], training_span=result["training_span"],
            num_modes=result["num_modes"], targets=[tgt],
            losses=np.asarray(result.get("losses", [0.0])),
            t_samp=result.get("t_samp"), snapshots_comp=snaps,
            O_samples=result.get("O_samples"),
            runtime=float(result.get("runtime", np.nan)))

    @staticmethod
    def from_multi(result, method_name):
        """Build a multi-target RunResult from a multi-IC 05 result dict.

        Expects ``all_rom_solves``/``all_true_comp``/``all_snaps_comp`` lists
        (one entry per evaluated IC) plus shared t_full/t_pred/basis. Truth
        states are read from ``all_true_states`` or ``all_true_states_full``.
        """
        all_rs = result["all_rom_solves"]
        all_tc = result["all_true_comp"]
        all_ts = result.get("all_true_states",
                            result.get("all_true_states_full"))
        all_sn = result.get("all_snaps_comp")
        labels = result.get("eval_labels", [""] * len(all_rs))
        t_pred = np.asarray(result["t_pred"])
        num_modes = result["num_modes"]
        targets = []
        for k in range(len(all_rs)):
            rs = np.asarray(all_rs[k]) if len(all_rs[k]) \
                else np.empty((0, num_modes, len(t_pred)))
            sn = all_sn[k] if all_sn is not None else None
            # true_states may be shorter than the eval list (e.g. a held-out
            # test IC without cached full-order states); guard the index. Only
            # the primary target's full-order error is plotted, so a missing
            # entry on later targets is harmless.
            ts_k = (all_ts[k] if (all_ts is not None and k < len(all_ts))
                    else None)
            targets.append(TargetResult(
                label=labels[k] if k < len(labels) else "",
                t_pred=t_pred, rom_solves=rs, true_comp=all_tc[k],
                true_states=ts_k, t_full=result["t_full"],
                state0_comp=(sn[:, 0] if sn is not None else None)))
        sn0 = all_sn[0] if all_sn is not None else None
        ts0 = result.get("all_t_samp", [None])[0]
        return RunResult(
            method_name=method_name, schema=result["schema"],
            basis=result["basis"], training_span=result["training_span"],
            num_modes=num_modes, targets=targets,
            losses=np.asarray(result.get("losses",
                              result.get("all_member_losses", [0.0]))),
            t_samp=ts0, snapshots_comp=sn0,
            runtime=float(result.get("runtime", np.nan)))
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

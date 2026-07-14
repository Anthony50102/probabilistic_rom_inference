"""Standard diagnostic plots for weak-form OpInf runs.

Thin shim over :mod:`core.plotting.figures`: converts the pipeline result dict
into the shared :class:`core.plotting.RunResult` and emits the standard
four-figure set. Kept so the experiment adapters' ``plot_standard(...)`` call
site is unchanged.
"""

from __future__ import annotations

import numpy as np

from core.plotting import RunResult, TargetResult, figures


def _to_run_result(result, method_name):
    targets = []
    for et, pt in zip(result["eval_targets"], result["per_target"]):
        rs = pt["rom_solves"]
        rs = np.asarray(rs) if len(rs) else np.empty((0, result["num_modes"], len(pt["t_pred"])))
        targets.append(TargetResult(
            label=getattr(et, "label", ""),
            t_pred=pt["t_pred"], rom_solves=rs,
            true_comp=et.true_comp, true_states=et.true_states,
            t_full=et.t_full, state0_comp=et.state0_comp,
            train_error=pt["train_error"], pred_error=pt["pred_error"],
            stability_pct=pt["stability_pct"], ci_coverage=pt["ci_coverage"],
            ci_width=pt["ci_width"]))
    return RunResult(
        method_name=method_name, schema=result["schema"],
        basis=result["basis"], training_span=result["training_span"],
        num_modes=result["num_modes"], targets=targets,
        losses=np.asarray(result["losses"]), O_samples=result.get("O_samples"),
        runtime=result.get("runtime", float("nan")))


def plot_standard(result, save_dir, prefix, dose_days=None,
                  method_name="04_unified"):
    """Emit rom-trajectory, loss, full-order-error, and operator-trace figures."""
    run = _to_run_result(result, method_name)
    figures.standard(run, save_dir, prefix, dose_days=dose_days)

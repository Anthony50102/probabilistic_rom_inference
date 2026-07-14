"""Centralised plotting for the probabilistic ROM inference experiments.

One plotting library shared by every method (Bayesian OpInf, Neural ODE) and
every PDE case study. The key enabler is a method-agnostic result contract
(:class:`RunResult` / :class:`TargetResult`) that every experiment emits, so a
single set of plotters serves per-run figures, cross-method comparisons, and
paper figures.

Modules
-------
- ``result``     : RunResult / TargetResult / MethodData + npz (de)serialisation
- ``style``      : shared rc params, method colours, save_figure
- ``figures``    : per-run figures (rom trajectories, loss, full-order error, traces)
- ``comparison`` : multi-method comparison (error curves, metric bars, tables)
- ``physical``   : domain figures (spatial comparison, tumor volume, 2D contour)
- ``gp``         : GP-diagnostic plots + the legacy ``Plotter`` class

``core.plotting`` re-exports the legacy function/``Plotter`` API below so
existing ``from core.plotting import X`` imports keep working (back-compat shim).
"""

from .result import RunResult, TargetResult, MethodData, load_comparison_npz
from .style import (
    METHOD_COLORS, METHOD_LABELS, method_color, method_label,
    save_figure, save_paper_figure, apply_style,
)
from . import figures, comparison, physical, gp

# ── Back-compat re-exports (legacy `from core.plotting import X`) ─────────────
from .gp import (
    Plotter,
    plot_gp_fit,
    plot_operator_derivative_fit,
    plot_deterministic_rom_solves,
    rbf_eval,
    flatten_time,
    compute_derivatives_fourth_order,
    save_plot_data,
    load_plot_data,
)
from .figures import plot_full_order_error
from .comparison import save_metrics_table

__all__ = [
    "RunResult", "TargetResult", "MethodData", "load_comparison_npz",
    "METHOD_COLORS", "METHOD_LABELS", "method_color", "method_label",
    "save_figure", "save_paper_figure", "apply_style",
    "figures", "comparison", "physical", "gp",
    "Plotter", "plot_gp_fit", "plot_operator_derivative_fit",
    "plot_deterministic_rom_solves", "plot_full_order_error",
    "save_metrics_table", "rbf_eval", "flatten_time",
    "compute_derivatives_fourth_order", "save_plot_data", "load_plot_data",
]

"""GP-diagnostic plots and the legacy ``Plotter`` class.

These are re-exported from the (self-contained) legacy implementation. Kept as
a distinct module so the package's public surface is organised; the underlying
code is unchanged for back-compat.
"""

from __future__ import annotations

from ._legacy import (  # noqa: F401
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

__all__ = [
    "Plotter",
    "plot_gp_fit",
    "plot_operator_derivative_fit",
    "plot_deterministic_rom_solves",
    "rbf_eval",
    "flatten_time",
    "compute_derivatives_fourth_order",
    "save_plot_data",
    "load_plot_data",
]

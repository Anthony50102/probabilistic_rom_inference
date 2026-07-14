"""Shared plotting style: method colours, rc params, figure saving."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Canonical colour + label per method, used consistently across per-run
# figures, comparison plots, and paper figures.
METHOD_COLORS = {
    "04_unified": "tab:purple",
    "04_unified_chemo": "tab:purple",
    "05_neural_ode": "tab:orange",
    "05_neural_ode_chemo": "tab:orange",
    "fom": "black",
    "truth": "black",
    "projection": "tab:green",
}

METHOD_LABELS = {
    "04_unified": "Bayesian OpInf",
    "04_unified_chemo": "Bayesian OpInf",
    "05_neural_ode": "Neural ODE",
    "05_neural_ode_chemo": "Neural ODE",
}


def method_color(name, default="tab:blue"):
    return METHOD_COLORS.get(name, default)


def method_label(name):
    return METHOD_LABELS.get(name, name)


def apply_style():
    """Apply the shared matplotlib rc params (idempotent)."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 8,
        "legend.frameon": False,
    })


def save_figure(fig, path, dpi=200, close=True):
    """Save a figure, creating parent dirs; optionally close it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def save_paper_figure(fig, name, directory, dpi=300):
    """Publication-quality save (PNG + PDF) into ``directory``."""
    os.makedirs(directory, exist_ok=True)
    png = os.path.join(directory, f"{name}.png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    try:
        fig.savefig(os.path.join(directory, f"{name}.pdf"), bbox_inches="tight")
    except Exception:
        pass
    return png

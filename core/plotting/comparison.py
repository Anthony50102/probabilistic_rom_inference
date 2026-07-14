"""Cross-method comparison figures: overlaid full-order error curves, metric
bar charts, and metric tables. Consume ``MethodData`` records (loaded from the
standardised comparison npz via :func:`core.plotting.load_comparison_npz`).
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from .style import save_figure
from ._legacy import save_metrics_table  # noqa: F401  (re-exported)


def compute_full_order_errors(rom_solves, t_pred, basis, t_full, true_states):
    """Return (per-sample rom_errors (S,T), projection_error (T,)) on ``t_pred``."""
    true_interp = interp1d(t_full, true_states, axis=1, kind="linear",
                           fill_value="extrapolate")(t_pred)
    true_comp = basis.compress(true_interp)
    true_proj = basis.decompress(true_comp)
    norm_truth = np.maximum(np.linalg.norm(true_interp, axis=0), 1e-10)
    projection_error = np.linalg.norm(true_interp - true_proj, axis=0) / norm_truth
    rom_errors = []
    for i in range(rom_solves.shape[0]):
        rom_full = basis.decompress(rom_solves[i])
        rom_errors.append(np.linalg.norm(true_interp - rom_full, axis=0) / norm_truth)
    return np.array(rom_errors), projection_error


def fill_errors(methods, basis, t_full, true_states):
    """Populate ``m.rom_errors`` for each method; return shared projection_error."""
    projection_error = None
    for m in methods:
        rom_errors, proj = compute_full_order_errors(
            m.rom_solves, m.t_pred, basis, t_full, true_states)
        m.rom_errors = rom_errors
        if projection_error is None:
            projection_error = proj
    return projection_error


def error_comparison(methods, projection_error, t_pred, training_span,
                     title, save_path):
    """3-panel: ROM error, projection error, excess error (all methods overlaid)."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax in axes:
        ax.axvspan(training_span[0], training_span[1], color="gray", alpha=0.10)

    for m in methods:
        med = np.median(m.rom_errors, axis=0)
        p5 = np.percentile(m.rom_errors, 5, axis=0)
        p95 = np.percentile(m.rom_errors, 95, axis=0)
        axes[0].plot(t_pred, med, color=m.color, lw=2, label=f"{m.label} (median)")
        axes[0].fill_between(t_pred, p5, p95, color=m.color, alpha=0.10)
    axes[0].set(ylabel="Relative Error", title=f"ROM Prediction Error — {title}")
    axes[0].set_yscale("log"); axes[0].legend(loc="upper left", fontsize=9)

    axes[1].plot(t_pred, projection_error, "k--", lw=2, label="Projection (basis limit)")
    axes[1].set(ylabel="Relative Error", title="Projection Error (Basis Limit)")
    axes[1].set_yscale("log"); axes[1].legend(loc="upper left", fontsize=9)

    for m in methods:
        med = np.median(m.rom_errors, axis=0)
        excess = np.maximum(med - projection_error, 1e-16)
        axes[2].plot(t_pred, excess, color=m.color, lw=2, label=m.label)
    axes[2].set(xlabel="Time", ylabel="Relative Error",
                title="Excess ROM Error (Above Basis Limit)")
    axes[2].set_yscale("log"); axes[2].legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    return save_figure(fig, save_path, dpi=150)


def metrics_bars(methods, title, save_path):
    """Bar chart: train error, prediction error, CI coverage."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    labels = [m.label for m in methods]
    colors = [m.color for m in methods]
    x = np.arange(len(labels))

    def _bars(ax, vals, fmt, ylabel, title_):
        b = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        for bar, v in zip(b, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt(v), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set(ylabel=ylabel, title=title_)

    _bars(axes[0], [m.train_error for m in methods], lambda v: f"{v:.3f}",
          "Relative L2 Error", "Training-region Error")
    _bars(axes[1], [m.pred_error for m in methods], lambda v: f"{v:.3f}",
          "Relative L2 Error", "Prediction-region Error")
    covs = [(m.ci_coverage * 100) if not np.isnan(m.ci_coverage) else 0.0
            for m in methods]
    _bars(axes[2], covs, lambda v: f"{v:.0f}%", "CI Coverage (%)", "90% CI Coverage")
    axes[2].axhline(90.0, color="k", ls="--", lw=1, alpha=0.6, label="Target 90%")
    axes[2].set_ylim(0, 105); axes[2].legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Method Comparison — {title}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save_figure(fig, save_path, dpi=150)


def print_summary_table(rows, header=None):
    """Print a plain-text metrics summary. ``rows`` = list of (regime, MethodData)."""
    print(f"\n{'Regime':<24}{'Method':<22}{'Train':>9}{'Pred':>9}"
          f"{'Stab':>8}{'CIcov':>8}{'Time':>9}")
    print("-" * 89)
    for regime, m in rows:
        ci = f"{m.ci_coverage*100:.1f}" if not np.isnan(m.ci_coverage) else " n/a"
        print(f"{regime:<24}{m.label:<22}{m.train_error*100:>8.2f}%"
              f"{m.pred_error*100:>8.2f}%{m.stability_pct:>7.1f}%{ci:>8}"
              f"{m.runtime:>8.1f}s")

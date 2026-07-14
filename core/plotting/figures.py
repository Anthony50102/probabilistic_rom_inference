"""Per-run figures: reduced-coordinate trajectories, loss, full-order error,
operator traces. All consume the method-agnostic :class:`RunResult`.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from .style import save_figure
# full-order error implementation still lives in the legacy module.
from ._legacy import plot_full_order_error  # noqa: F401  (re-exported)


def rom_trajectories(run, save_path, target=None, dose_days=None):
    """Reduced-coordinate ROM trajectories with 5–95% band vs truth."""
    tgt = target or run.primary
    if tgt.n_stable == 0:
        return None
    rom = np.asarray(tgt.rom_solves)
    med = np.median(rom, axis=0)
    q05 = np.percentile(rom, 5, axis=0)
    q95 = np.percentile(rom, 95, axis=0)
    truth = interp1d(tgt.t_full, tgt.true_comp, kind="cubic",
                     fill_value="extrapolate")(tgt.t_pred)
    r = run.num_modes
    ncol = min(r, 4)
    nrow = int(np.ceil(r / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow),
                             squeeze=False)
    for i in range(r):
        ax = axes[i // ncol][i % ncol]
        if dose_days is not None:
            for dd in np.asarray(dose_days):
                ax.axvline(dd, color="tab:red", lw=0.5, alpha=0.15, zorder=0)
        if run.t_samp is not None and run.snapshots_comp is not None:
            ax.scatter(run.t_samp, run.snapshots_comp[i], s=9,
                       color="tab:blue", alpha=0.35, zorder=4, label="Noisy obs")
        ax.plot(tgt.t_pred, truth[i], "k-", lw=1.5, label="Truth", zorder=3)
        ax.plot(tgt.t_pred, med[i], color=run.color, lw=1.8,
                label=f"{run.method_label} median", zorder=5)
        ax.fill_between(tgt.t_pred, q05[i], q95[i], color=run.color,
                        alpha=0.18, label="5–95%", zorder=2)
        ax.axvline(run.training_span[1], color="gray", ls=":", lw=0.9, alpha=0.6)
        ax.set(title=f"Mode {i}", xlabel="time")
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(r, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{run.method_label} — {run.schema.get('label', '')}"
                 + (f"  [{tgt.label}]" if tgt.label else ""))
    fig.tight_layout()
    return save_figure(fig, save_path)


def loss(run, save_path):
    """SVI/training loss convergence (full + last-50%)."""
    losses = np.asarray(run.losses)
    if losses.ndim > 1:            # ensemble (per-member) → mean
        losses = np.mean(losses, axis=0)
    if losses.size <= 1:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(losses, lw=0.8, color=run.color)
    ax[0].set(xlabel="step", ylabel="loss", title="Loss convergence")
    half = len(losses) // 2
    ax[1].plot(range(half, len(losses)), losses[half:], lw=0.8, color=run.color)
    ax[1].set(xlabel="step", ylabel="loss", title="Loss (last 50%)")
    fig.tight_layout()
    return save_figure(fig, save_path)


def full_order_error(run, save_path, target=None):
    """Full-order error decomposition (ROM error, projection error, excess)."""
    tgt = target or run.primary
    if tgt.n_stable == 0:
        return None
    fig, _ = plot_full_order_error(
        rom_solves=np.asarray(tgt.rom_solves), basis=run.basis,
        true_states=tgt.true_states, time_domain_full=tgt.t_full,
        time_domain_eval=tgt.t_pred, training_span=tuple(run.training_span),
        suptitle=f"{run.method_label} — {run.schema.get('label', '')}")
    return save_figure(fig, save_path)


def operator_traces(run, save_path, n_random=6):
    """Posterior trace plots for operator entries (Bayesian methods only)."""
    if run.O_samples is None:
        return None
    from core.diagnostics import plot_trace
    fig, _ = plot_trace({"O": np.asarray(run.O_samples)}, param_name="O",
                        n_random=n_random)
    return save_figure(fig, save_path)


def standard(run, save_dir, prefix, dose_days=None):
    """Emit the standard four-figure diagnostic set for one run."""
    os.makedirs(save_dir, exist_ok=True)
    rom_trajectories(run, os.path.join(save_dir, f"{prefix}_rom_trajectories.png"),
                     dose_days=dose_days)
    loss(run, os.path.join(save_dir, f"{prefix}_loss.png"))
    try:
        full_order_error(run, os.path.join(save_dir, f"{prefix}_full_order_error.png"))
    except Exception as e:
        print(f"  [plot] full_order_error skipped: {e}")
    try:
        operator_traces(run, os.path.join(save_dir, f"{prefix}_operator_traces.png"))
    except Exception as e:
        print(f"  [plot] operator_traces skipped: {e}")

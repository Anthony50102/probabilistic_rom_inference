"""Domain-specific ("physical space") figures shared across experiments.

- ``spatial_comparison`` / ``tumor_volume``: tumor (TumorTwin) case studies.
- ``contour_2d``: 2D-field cases (e.g. Burgers 2D).

All consume a list of :class:`core.plotting.result.MethodData` plus a shared
context (full-order model, basis, truth) so both the per-method scripts and the
paper generator use one implementation.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import save_figure


def tumor_volume(methods, fom, basis, true_states, t_full, training_span,
                 save_path):
    """Total tumor burden over time: FOM truth vs each method's median + 90% CI."""
    voxel_vol = float(np.prod(fom.spacing))
    fom_vol = true_states.sum(axis=0) * voxel_vol
    V = basis.entries
    vol_proj = V.T @ np.ones(V.shape[0])
    shift_vol = np.ones(V.shape[0]) @ basis.shift_

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(t_full, fom_vol, "k-", lw=2.5, label="FOM Truth", zorder=10)
    for m in methods:
        rom = m.rom_solves
        vols = np.array([vol_proj @ rom[s] + shift_vol
                         for s in range(rom.shape[0])]) * voxel_vol
        med = np.median(vols, axis=0)
        lo = np.percentile(vols, 5, axis=0)
        hi = np.percentile(vols, 95, axis=0)
        ax.plot(m.t_pred, med, color=m.color, lw=2, label=f"{m.label} median")
        ax.fill_between(m.t_pred, lo, hi, color=m.color, alpha=0.15,
                        label=f"{m.label} 90% CI")
    ax.axvline(training_span[1], color="gray", ls="--", alpha=0.5,
               label="Train/Predict")
    ax.set(xlabel="Time (days)", ylabel="Total Tumor Burden (mm³)",
           title="Tumor Volume Over Time")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save_figure(fig, save_path)


def spatial_comparison(methods, fom, basis, true_states, t_full, save_path,
                       timepoints=(5, 15, 30, 45, 60, 90)):
    """Axial-slice cellularity: FOM truth + per-method prediction + |error|."""
    n_times = len(timepoints)
    n_methods = len(methods)
    n_rows = 1 + 2 * n_methods
    fig, axes = plt.subplots(n_rows, n_times,
                             figsize=(3.5 * n_times, 3.0 * n_rows), squeeze=False)
    im_fom = im_pred = im_err = None
    for col, t_target in enumerate(timepoints):
        idx = int(np.argmin(np.abs(t_full - t_target)))
        fom_state = true_states[:, idx]
        im_fom = axes[0][col].imshow(fom.get_center_slices(fom_state)["axial"].T,
                                     origin="lower", cmap="hot_r", vmin=0, vmax=1,
                                     aspect="equal")
        axes[0][col].set_title(f"Day {t_full[idx]:.0f}", fontsize=11)
        axes[0][col].set_xticks([]); axes[0][col].set_yticks([])
        for mi, m in enumerate(methods):
            med = np.median(m.rom_solves, axis=0)
            ip = int(np.argmin(np.abs(m.t_pred - t_target)))
            rom_full = basis.decompress(med[:, ip])
            rp, re = 1 + 2 * mi, 2 + 2 * mi
            im_pred = axes[rp][col].imshow(
                fom.get_center_slices(rom_full)["axial"].T, origin="lower",
                cmap="hot_r", vmin=0, vmax=1, aspect="equal")
            im_err = axes[re][col].imshow(
                fom.get_center_slices(np.abs(fom_state - rom_full))["axial"].T,
                origin="lower", cmap="Reds", vmin=0, aspect="equal")
            for r in (rp, re):
                axes[r][col].set_xticks([]); axes[r][col].set_yticks([])
    axes[0][0].set_ylabel("FOM Truth", fontsize=11, fontweight="bold")
    for mi, m in enumerate(methods):
        axes[1 + 2 * mi][0].set_ylabel(m.label, fontsize=11, fontweight="bold")
        axes[2 + 2 * mi][0].set_ylabel("|Error|", fontsize=10)
    fig.suptitle("FOM vs ROM (axial slice)", fontsize=14, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.92, 0.98])
    return save_figure(fig, save_path)


def contour_2d(methods, fom, basis, true_states, t_full, save_path,
               timepoints=None, field_shape=None):
    """2D field snapshots: FOM truth vs each method's median at several times."""
    if timepoints is None:
        timepoints = np.linspace(t_full[0], t_full[-1], 5)
    n_times = len(timepoints)
    n_rows = 1 + len(methods)
    fig, axes = plt.subplots(n_rows, n_times,
                             figsize=(3.0 * n_times, 3.0 * n_rows), squeeze=False)

    def _reshape(v):
        return v.reshape(field_shape) if field_shape is not None else v

    for col, t_target in enumerate(timepoints):
        idx = int(np.argmin(np.abs(t_full - t_target)))
        axes[0][col].imshow(_reshape(true_states[:, idx]), origin="lower",
                            aspect="auto")
        axes[0][col].set_title(f"t={t_full[idx]:.2f}", fontsize=10)
        axes[0][col].set_xticks([]); axes[0][col].set_yticks([])
        for mi, m in enumerate(methods):
            med = np.median(m.rom_solves, axis=0)
            ip = int(np.argmin(np.abs(m.t_pred - t_target)))
            axes[1 + mi][col].imshow(_reshape(basis.decompress(med[:, ip])),
                                     origin="lower", aspect="auto")
            axes[1 + mi][col].set_xticks([]); axes[1 + mi][col].set_yticks([])
    axes[0][0].set_ylabel("FOM Truth", fontsize=11, fontweight="bold")
    for mi, m in enumerate(methods):
        axes[1 + mi][0].set_ylabel(m.label, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, save_path)

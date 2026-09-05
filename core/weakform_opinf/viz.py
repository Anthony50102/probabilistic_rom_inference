"""Shared standard diagnostic plots for weak-form OpInf experiments.

Every adapter can call :func:`plot_standard` to emit the same four figures —
reduced-coordinate ROM trajectories with 5–95% bands, the SVI/NUTS loss, the
full-order error decomposition, and operator posterior traces — keeping plotting
out of the per-experiment adapters.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from core.plotting import plot_full_order_error
from core.diagnostics import plot_trace


def plot_standard(result, save_dir, prefix, dose_days=None):
    """Write the standard diagnostic figure set for one regime result.

    Parameters
    ----------
    result : dict returned by pipeline.run_experiment
    save_dir : str
    prefix : str   figure filename prefix
    dose_days : array-like or None   optional vertical markers (chemo doses)
    """
    os.makedirs(save_dir, exist_ok=True)
    tgt = result["eval_targets"][0]
    sc = result["per_target"][0]
    rom_solves = sc["rom_solves"]
    t_pred = sc["t_pred"]
    span = result["training_span"]
    nmodes = result["num_modes"]

    if len(rom_solves) > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        q05 = np.percentile(rom_arr, 5, axis=0)
        q95 = np.percentile(rom_arr, 95, axis=0)
        ti = interp1d(tgt.t_full, tgt.true_comp, kind="cubic",
                      fill_value="extrapolate")
        truth = ti(t_pred)
        ncol = min(nmodes, 4)
        nrow = int(np.ceil(nmodes / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow),
                                 squeeze=False)
        for i in range(nmodes):
            ax = axes[i // ncol][i % ncol]
            if dose_days is not None:
                for dd in np.asarray(dose_days):
                    ax.axvline(dd, color="tab:red", lw=0.5, alpha=0.15, zorder=0)
            ax.plot(t_pred, truth[i], "k-", lw=1.5, label="Truth", zorder=3)
            ax.plot(t_pred, rom_med[i], color="tab:purple", lw=1.8,
                    label="ROM median", zorder=5)
            ax.fill_between(t_pred, q05[i], q95[i], color="tab:purple",
                            alpha=0.18, label="ROM 5–95%", zorder=2)
            ax.axvline(span[1], color="gray", ls=":", lw=0.9, alpha=0.6)
            ax.set_title(f"Mode {i}")
            ax.set_xlabel("time")
            if i == 0:
                ax.legend(fontsize=7)
        for j in range(nmodes, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"ROM trajectories — {result['schema'].get('label', '')}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{prefix}_rom_trajectories.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    losses = np.asarray(result["losses"])
    if losses.size > 1:
        fig_l, ax_l = plt.subplots(1, 2, figsize=(12, 4))
        ax_l[0].plot(losses, lw=0.8, color="tab:purple")
        ax_l[0].set(xlabel="step", ylabel="-ELBO", title="Loss convergence")
        half = len(losses) // 2
        ax_l[1].plot(range(half, len(losses)), losses[half:], lw=0.8,
                     color="tab:purple")
        ax_l[1].set(xlabel="step", ylabel="-ELBO", title="Loss (last 50%)")
        fig_l.tight_layout()
        fig_l.savefig(os.path.join(save_dir, f"{prefix}_loss.png"),
                      dpi=200, bbox_inches="tight")
        plt.close(fig_l)

    if len(rom_solves) > 0:
        try:
            fig_e, _ = plot_full_order_error(
                rom_solves=np.array(rom_solves), basis=result["basis"],
                true_states=tgt.true_states, time_domain_full=tgt.t_full,
                time_domain_eval=t_pred, training_span=tuple(span),
                suptitle=result["schema"].get("label", ""))
            fig_e.savefig(os.path.join(save_dir, f"{prefix}_full_order_error.png"),
                          dpi=200, bbox_inches="tight")
            plt.close(fig_e)
        except Exception as e:
            print(f"  [plot] full_order_error skipped: {e}")

    try:
        fig_t, _ = plot_trace({"O": np.asarray(result["O_samples"])},
                              param_name="O", n_random=6)
        fig_t.savefig(os.path.join(save_dir, f"{prefix}_operator_traces.png"),
                      dpi=200, bbox_inches="tight")
        plt.close(fig_t)
    except Exception as e:
        print(f"  [plot] operator_traces skipped: {e}")

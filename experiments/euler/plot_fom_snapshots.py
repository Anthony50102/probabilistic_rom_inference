"""
Plot Euler FOM spatial profiles at specific time instances.

Produces a 1×3 figure (velocity, pressure, 1/rho) with FOM profiles
overlaid at a few chosen times, for the paper.

Usage:
    python plot_fom_snapshots.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..'))

import config


SNAPSHOT_TIMES = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]
FIGURE_PATH = os.path.join(SCRIPT_DIR, "figures", "euler_fom_snapshots.png")
VARIABLE_LABELS = [r"Velocity $v$", r"Pressure $p$", r"Specific volume $1/\rho$"]


def main():
    fom = config.FullOrderModel()
    t_full = config.time_domain
    x = config.spatial_domain
    nx = len(x)

    # Clean FOM trajectory (no noise).
    true_states = fom.solve(config.initial_conditions, t_full)

    # true_states has shape (3*nx, n_t) — concatenation of [v, p, 1/rho].
    v_all   = true_states[0:nx,       :]
    p_all   = true_states[nx:2*nx,    :]
    zeta_all = true_states[2*nx:3*nx, :]
    vars_all = [v_all, p_all, zeta_all]

    snapshot_idx = [int(np.argmin(np.abs(t_full - ts))) for ts in SNAPSHOT_TIMES]
    times_used = [t_full[i] for i in snapshot_idx]
    colors = cm.viridis(np.linspace(0.05, 0.9, len(snapshot_idx)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, (ax, var_block, label) in enumerate(zip(axes, vars_all, VARIABLE_LABELS)):
        for k, (idx, t) in enumerate(zip(snapshot_idx, times_used)):
            ax.plot(x, var_block[:, idx], color=colors[k], lw=1.5,
                    label=f"t = {t:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if j == 2:
            ax.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle("Euler FOM — Spatial Profiles at Selected Times", fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    print(f"📊 Saved: {FIGURE_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""
Plot Euler FOM spatial profiles at specific time instances, overlaid with
04 (Bayesian OpInf) and 05 (Neural ODE) median predictions and 5-95% bands.

Produces a 3-row × N-col panel: rows = (velocity, pressure, 1/rho),
cols = snapshot times. Each subplot shows FOM (solid black), 04 (blue
median + band), and 05 (orange median + band).

Usage:
    python plot_fom_with_predictions.py [--schema sparse_low_noise]
"""

import os
import sys
import argparse
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)

import config


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SNAPSHOT_TIMES = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]
VARIABLE_LABELS = [r"Velocity $v$", r"Pressure $p$", r"Specific volume $1/\rho$"]


def _split_state(state_full, nx):
    """state_full: (3*nx,) or (3*nx, T) → (v, p, 1/rho) tuple."""
    return state_full[0:nx], state_full[nx:2*nx], state_full[2*nx:3*nx]


def _physical_predictions(rom_arr, basis, nx):
    """rom_arr: (n_samples, num_modes, n_t) → (n_samples, 3*nx, n_t)."""
    n_s, _, n_t = rom_arr.shape
    out = np.empty((n_s, 3 * nx, n_t))
    for s in range(n_s):
        out[s] = basis.decompress(rom_arr[s])
    return out


def main(schema_name="sparse_low_noise"):
    mod04 = _load_module("euler_04", "04_conditional_integral.py")
    mod05 = _load_module("euler_05", "05_neural_ode.py")

    # Find schema in 04 (assume names match between scripts)
    schemas_04 = {s["name"]: s for s in mod04.SCHEMAS}
    schemas_05 = {s["name"]: s for s in mod05.SCHEMAS}
    if schema_name not in schemas_04 or schema_name not in schemas_05:
        raise SystemExit(f"Schema '{schema_name}' not in both 04 and 05. "
                         f"04 has {list(schemas_04)}, 05 has {list(schemas_05)}")

    print(f"\n=== Running 04 (Bayesian OpInf) on '{schema_name}' ===")
    r04 = mod04.run_experiment(schemas_04[schema_name])
    print(f"\n=== Running 05 (Neural ODE) on '{schema_name}' ===")
    r05 = mod05.run_experiment(schemas_05[schema_name])

    x = config.spatial_domain
    nx = len(x)
    t_full = config.time_domain

    true_states = r04["true_states"]  # same FOM in both
    snapshot_idx_full = [int(np.argmin(np.abs(t_full - ts))) for ts in SNAPSHOT_TIMES]
    times_used = [t_full[i] for i in snapshot_idx_full]

    fom_v, fom_p, fom_z = _split_state(true_states, nx)
    fom_blocks = [fom_v, fom_p, fom_z]

    def _get_pred_blocks(result):
        rom_arr = np.array(result["rom_solves"])
        basis = result["basis"]
        t_pred = result["t_pred"]
        if rom_arr.size == 0:
            return None, None, None, t_pred
        full_arr = _physical_predictions(rom_arr, basis, nx)  # (n_s, 3*nx, n_t)
        med = np.median(full_arr, axis=0)
        q05 = np.percentile(full_arr, 5, axis=0)
        q95 = np.percentile(full_arr, 95, axis=0)
        med_blocks = [med[0:nx], med[nx:2*nx], med[2*nx:3*nx]]
        lo_blocks = [q05[0:nx], q05[nx:2*nx], q05[2*nx:3*nx]]
        hi_blocks = [q95[0:nx], q95[nx:2*nx], q95[2*nx:3*nx]]
        return med_blocks, lo_blocks, hi_blocks, t_pred

    med04, lo04, hi04, t_pred04 = _get_pred_blocks(r04)
    med05, lo05, hi05, t_pred05 = _get_pred_blocks(r05)

    # Noisy training data overlay (Option 1): for each plotted time, find
    # the nearest training snapshot and overlay it as faded markers. Only
    # show if the nearest training time is within `tol_t` of the panel time.
    snaps_noisy = r04.get("snaps_noisy")  # (3*nx, n_samples) in physical space
    t_samp = r04.get("t_samp")
    if t_samp is not None and len(t_samp) > 1:
        # Tolerance = average training spacing → every plotted time inside
        # the training window picks up the nearest snapshot; plotted times
        # outside the training window are left blank (visually flags
        # extrapolation).
        tol_t = float(np.mean(np.diff(t_samp)))
        t_train_lo, t_train_hi = float(t_samp.min()), float(t_samp.max())
    else:
        tol_t = 0.0
        t_train_lo, t_train_hi = 0.0, -1.0
    if snaps_noisy is not None and t_samp is not None:
        noisy_blocks_per_col = []
        for t_snap in times_used:
            in_train = (t_train_lo - tol_t) <= t_snap <= (t_train_hi + tol_t)
            j = int(np.argmin(np.abs(t_samp - t_snap)))
            if in_train and abs(t_samp[j] - t_snap) <= tol_t:
                v_n, p_n, z_n = _split_state(snaps_noisy[:, j], nx)
                noisy_blocks_per_col.append(([v_n, p_n, z_n], t_samp[j]))
            else:
                noisy_blocks_per_col.append((None, None))
    else:
        noisy_blocks_per_col = [(None, None)] * len(times_used)

    n_cols = len(SNAPSHOT_TIMES)
    fig, axes = plt.subplots(3, n_cols, figsize=(2.6 * n_cols, 7.5),
                              sharex=True)

    for col, (idx_full, t_snap) in enumerate(zip(snapshot_idx_full, times_used)):
        idx04 = int(np.argmin(np.abs(t_pred04 - t_snap))) if med04 is not None else None
        idx05 = int(np.argmin(np.abs(t_pred05 - t_snap))) if med05 is not None else None
        noisy_blocks, t_noisy_used = noisy_blocks_per_col[col]

        for row in range(3):
            ax = axes[row, col]

            # Noisy training snapshot at the nearest training time (faded)
            if noisy_blocks is not None:
                ax.plot(x, noisy_blocks[row], color='0.55', lw=0,
                        marker='.', ms=2.5, alpha=0.45,
                        label=('noisy training data'
                               if (row == 0 and col == 0) else None))

            if med05 is not None:
                ax.fill_between(x, lo05[row][:, idx05], hi05[row][:, idx05],
                                color='tab:orange', alpha=0.18, lw=0,
                                label='05 5-95%' if (row == 0 and col == 0) else None)
                ax.plot(x, med05[row][:, idx05], color='tab:orange', lw=1.4,
                        ls=':', label='05 median' if (row == 0 and col == 0) else None)

            if med04 is not None:
                ax.fill_between(x, lo04[row][:, idx04], hi04[row][:, idx04],
                                color='tab:blue', alpha=0.18, lw=0,
                                label='04 5-95%' if (row == 0 and col == 0) else None)
                ax.plot(x, med04[row][:, idx04], color='tab:blue', lw=1.4,
                        ls='--', label='04 median' if (row == 0 and col == 0) else None)

            ax.plot(x, fom_blocks[row][:, idx_full], color='black', lw=1.5,
                    label='FOM' if (row == 0 and col == 0) else None)

            if row == 0:
                if t_noisy_used is not None and abs(t_noisy_used - t_snap) > 1e-9:
                    ax.set_title(f"t = {t_snap:.3f}\n(data @ {t_noisy_used:.3f})",
                                 fontsize=10)
                else:
                    ax.set_title(f"t = {t_snap:.3f}", fontsize=11)
            if col == 0:
                ax.set_ylabel(VARIABLE_LABELS[row], fontsize=11)
            if row == 2:
                ax.set_xlabel("x")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Euler FOM with 04 (Bayesian OpInf) & 05 (Neural ODE) Predictions"
                 f" — {schemas_04[schema_name]['label']}", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    out_path = os.path.join(SCRIPT_DIR, "figures",
                             f"euler_fom_with_predictions_{schema_name}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n📊 Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--schema", default="sparse_low_noise",
                   help="schema name (must exist in both 04 and 05 SCHEMAS)")
    args = p.parse_args()
    main(args.schema)

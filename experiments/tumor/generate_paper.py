"""
Generate paper-ready comparison figures for the Tumor Growth experiment.

Loads pre-computed predictions from 04 (Bayesian OpInf) and 05 (Neural ODE)
and generates side-by-side comparison plots suitable for publication.

Prerequisites:
    python 04_conditional_integral.py
    python 05_neural_ode.py

Usage:
    python generate_paper.py          # comparison figures only (uses saved NPZ)
    python generate_paper.py --run    # re-run both methods first, then compare
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis, TumorTwinFOM, load_fom_data, TRAINING_SPAN

SCHEMA = "dense_low_noise"
NUM_MODES = 4
NUM_EVAL_POINTS = 200

METHODS = [
    {"name": "04_conditional_integral", "script": "04_conditional_integral.py",
     "label": "Bayesian OpInf", "color": "tab:purple", "short": "04"},
    {"name": "05_neural_ode", "script": "05_neural_ode.py",
     "label": "Neural ODE", "color": "tab:orange", "short": "05"},
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# Data loading
# =============================================================================
def load_shared_data():
    """Load FOM data and build clean POD basis (shared between methods)."""
    t_pred = np.linspace(TRAINING_SPAN[0], config.PREDICTION_DAYS, NUM_EVAL_POINTS)
    fom, t_full, true_states, t_samp, snaps_noisy = \
        load_fom_data(t_pred, TRAINING_SPAN, num_samples=80, noise_level=0.01)

    snaps_clean = fom.get_states(t_samp)
    basis = Basis(num_vectors=NUM_MODES)
    basis.fit(snaps_clean)
    true_comp = basis.compress(true_states)

    return {
        'fom': fom, 'basis': basis,
        't_full': t_full, 't_pred': t_pred,
        'true_states': true_states, 'true_comp': true_comp,
    }


def load_method_npz(method):
    """Load saved NPZ predictions for one method."""
    path = os.path.join(
        SCRIPT_DIR, "results", "comparison", SCHEMA, f"{method['name']}.npz"
    )
    if not os.path.exists(path):
        print(f"  ⚠ Missing: {path}")
        return None
    data = np.load(path)
    if data["rom_solves"].size == 0:
        return None
    return {
        "rom_solves": data["rom_solves"],
        "t_pred": data["t_pred"],
        "train_error": float(data["train_error"]),
        "pred_error": float(data["pred_error"]),
        "stability_pct": float(data["stability_pct"]),
        "ci_coverage": float(data.get("ci_coverage", float("nan"))),
        "ci_width": float(data.get("ci_width", float("nan"))),
        "runtime": float(data["runtime"]),
    }


# =============================================================================
# Comparison figures
# =============================================================================
def plot_rom_trajectories_comparison(shared, methods_data, save_dir):
    """Side-by-side ROM trajectory plots for both methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_comp = shared['true_comp']
    t_full = shared['t_full']
    n_methods = len(methods_data)

    fig, axes = plt.subplots(NUM_MODES, n_methods, figsize=(7 * n_methods, 2.5 * NUM_MODES),
                              sharey='row')
    if NUM_MODES == 1:
        axes = axes.reshape(1, -1)

    for col, (method, mdata) in enumerate(methods_data):
        rom_arr = mdata['rom_solves']
        t_pred = mdata['t_pred']
        color = method['color']

        rom_med = np.median(rom_arr, axis=0)
        rom_q05 = np.percentile(rom_arr, 5, axis=0)
        rom_q95 = np.percentile(rom_arr, 95, axis=0)

        true_interp = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        true_at = true_interp(t_pred)

        for i in range(NUM_MODES):
            ax = axes[i, col]
            ax.axvspan(TRAINING_SPAN[0], TRAINING_SPAN[1], color='gray', alpha=0.08, zorder=0)
            ax.plot(t_pred, true_at[i], color='tab:gray', lw=2, label='Truth')
            ax.plot(t_pred, rom_med[i], color=color, ls='--', lw=2, alpha=0.9, label='Median')
            ax.fill_between(t_pred, rom_q05[i], rom_q95[i], color=color, alpha=0.15, label='90% CI')
            ax.axvline(TRAINING_SPAN[1], color='k', ls=':', lw=0.8, alpha=0.5)
            if col == 0:
                ax.set_ylabel(f'Mode {i}')
            yvals = true_at[i]
            ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
            pad = max(abs(ymax - ymin) * 0.3, 1e-6)
            ax.set_ylim(ymin - pad, ymax + pad)
            if i == 0:
                ax.legend(fontsize=7, loc='upper right')

        axes[0, col].set_title(
            f"{method['label']}  (train {mdata['train_error']:.1%}, "
            f"pred {mdata['pred_error']:.1%})", fontsize=12)
        axes[-1, col].set_xlabel('Time (days)')

    fig.suptitle('ROM Trajectories — Method Comparison', fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(save_dir, "comparison_rom_trajectories.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


def plot_spatial_comparison(shared, methods_data, save_dir):
    """Side-by-side spatial tumor density slices for both methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fom = shared['fom']
    basis = shared['basis']
    true_states = shared['true_states']
    t_full = shared['t_full']

    timepoints = [5, 15, 30, 45, 60, 90]
    n_times = len(timepoints)
    n_methods = len(methods_data)

    # Layout: row 0 = FOM truth, then 2 rows per method (prediction + error)
    n_rows = 1 + 2 * n_methods
    fig, axes = plt.subplots(n_rows, n_times, figsize=(3.5 * n_times, 3.0 * n_rows))

    for col, t_target in enumerate(timepoints):
        idx_full = np.argmin(np.abs(t_full - t_target))
        fom_state = true_states[:, idx_full]
        fom_slices = fom.get_center_slices(fom_state)

        # Row 0: FOM truth
        im_fom = axes[0, col].imshow(fom_slices['axial'].T, origin='lower',
                                      cmap='hot_r', vmin=0, vmax=1, aspect='equal')
        axes[0, col].set_title(f'Day {t_full[idx_full]:.0f}', fontsize=11)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        for m_idx, (method, mdata) in enumerate(methods_data):
            rom_arr = mdata['rom_solves']
            t_pred = mdata['t_pred']
            rom_med = np.median(rom_arr, axis=0)

            idx_pred = np.argmin(np.abs(t_pred - t_target))
            rom_full = basis.decompress(rom_med[:, idx_pred])
            rom_slices = fom.get_center_slices(rom_full)
            err_slices = fom.get_center_slices(np.abs(fom_state - rom_full))

            row_pred = 1 + 2 * m_idx
            row_err = 2 + 2 * m_idx

            im_pred = axes[row_pred, col].imshow(
                rom_slices['axial'].T, origin='lower',
                cmap='hot_r', vmin=0, vmax=1, aspect='equal')
            im_err = axes[row_err, col].imshow(
                err_slices['axial'].T, origin='lower',
                cmap='Reds', vmin=0, aspect='equal')

            for r in [row_pred, row_err]:
                axes[r, col].set_xticks([])
                axes[r, col].set_yticks([])

    # Row labels
    axes[0, 0].set_ylabel('FOM Truth', fontsize=11, fontweight='bold')
    for m_idx, (method, _) in enumerate(methods_data):
        axes[1 + 2 * m_idx, 0].set_ylabel(method['label'], fontsize=11, fontweight='bold')
        axes[2 + 2 * m_idx, 0].set_ylabel('|Error|', fontsize=10)

    fig.colorbar(im_fom, ax=axes[0, :].tolist(), shrink=0.8, label='Cellularity')
    for m_idx in range(n_methods):
        fig.colorbar(im_pred, ax=axes[1 + 2 * m_idx, :].tolist(), shrink=0.8, label='Cellularity')
        fig.colorbar(im_err, ax=axes[2 + 2 * m_idx, :].tolist(), shrink=0.8, label='|Error|')

    fig.suptitle('Tumor Growth: FOM vs ROM Methods (axial slice)', fontsize=14, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.92, 0.98])
    path = os.path.join(save_dir, "comparison_spatial.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


def plot_tumor_volume_comparison(shared, methods_data, save_dir):
    """Combined tumor volume over time for all methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fom = shared['fom']
    basis = shared['basis']
    true_states = shared['true_states']
    t_full = shared['t_full']

    voxel_vol = float(np.prod(fom.spacing))
    fom_vol = np.array([true_states[:, i].sum() * voxel_vol
                        for i in range(true_states.shape[1])])

    # Efficient volume projection
    V = basis.entries
    ones = np.ones(V.shape[0])
    vol_proj = V.T @ ones
    shift_vol = ones @ basis.shift_

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(t_full, fom_vol, 'k-', lw=2.5, label='FOM Truth', zorder=10)

    for method, mdata in methods_data:
        rom_arr = mdata['rom_solves']
        t_pred = mdata['t_pred']
        color = method['color']

        rom_vols = np.array([vol_proj @ rom_arr[s] + shift_vol
                             for s in range(rom_arr.shape[0])]) * voxel_vol

        med = np.median(rom_vols, axis=0)
        lo = np.percentile(rom_vols, 5, axis=0)
        hi = np.percentile(rom_vols, 95, axis=0)

        ax.plot(t_pred, med, color=color, lw=2, label=f'{method["label"]} Median')
        ax.fill_between(t_pred, lo, hi, color=color, alpha=0.15,
                        label=f'{method["label"]} 90% CI')

    ax.axvline(TRAINING_SPAN[1], color='gray', ls='--', alpha=0.5, label='Train/Predict')
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Total Tumor Burden (mm³)', fontsize=12)
    ax.set_title('Tumor Volume Over Time', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = os.path.join(save_dir, "comparison_tumor_volume.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


def plot_metrics_table(methods_data, save_dir):
    """Bar chart comparing key metrics across methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [m['label'] for m, _ in methods_data]
    colors = [m['color'] for m, _ in methods_data]
    metrics = {
        'Train Error (%)': [d['train_error'] * 100 for _, d in methods_data],
        'Pred Error (%)': [d['pred_error'] * 100 for _, d in methods_data],
        'CI Coverage (%)': [d['ci_coverage'] * 100 for _, d in methods_data],
        'Runtime (s)': [d['runtime'] for _, d in methods_data],
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', lw=0.5)
        ax.set_title(metric_name, fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add target line on CI coverage
    axes[2].axhline(90, color='red', ls='--', alpha=0.5, label='Target 90%')
    axes[2].legend(fontsize=8)

    fig.suptitle('Method Comparison — Key Metrics', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(save_dir, "comparison_metrics.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


# =============================================================================
# Script runners
# =============================================================================
def _run_script(method):
    """Execute a method script and return status info."""
    script_path = os.path.join(SCRIPT_DIR, method["script"])
    result = {"method": method["label"], "status": "pending",
              "elapsed_s": None, "error": None}
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, script_path, SCHEMA],
            capture_output=True, text=True, timeout=900, cwd=SCRIPT_DIR,
        )
        result["status"] = "success" if proc.returncode == 0 else "failed"
        result["error"] = proc.stderr[-500:] if proc.returncode != 0 else None
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Exceeded 900s timeout"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:500]
    finally:
        result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def _ensure_fom_data():
    """Check that TumorTwin FOM data exists; generate if not."""
    fom_path = os.path.join(SCRIPT_DIR, "data", "TNBC_demo_001_fom.npz")
    if os.path.exists(fom_path):
        return True
    print("  FOM data not found — generating with TumorTwin...")
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "generate_fom_data.py")],
            capture_output=True, text=True, timeout=300, cwd=SCRIPT_DIR,
        )
        return proc.returncode == 0
    except Exception:
        return False


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate paper comparison figures")
    parser.add_argument("--run", action="store_true",
                        help="Re-run both method scripts before generating figures")
    args = parser.parse_args()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    print(f"Tumor Growth — Paper Figure Generation  [{SCHEMA}]")
    print(f"Figures → {FIGURE_DIR}\n")

    if not _ensure_fom_data():
        print("Cannot proceed without FOM data. Run generate_fom_data.py first.")
        sys.exit(1)

    # ── Optionally re-run both scripts ───────────────────────────────────
    if args.run:
        for i, method in enumerate(METHODS, 1):
            print(f"  [{i}/{len(METHODS)}] Running {method['label']}...")
            r = _run_script(method)
            icon = "✓" if r["status"] == "success" else "✗"
            print(f"    {icon} {r['status']} ({r['elapsed_s']}s)")
            if r["status"] != "success":
                print(f"    Error: {r['error']}")

    # ── Load predictions ─────────────────────────────────────────────────
    print("\nLoading saved predictions...")
    methods_data = []
    for method in METHODS:
        mdata = load_method_npz(method)
        if mdata is not None:
            methods_data.append((method, mdata))
            print(f"  ✓ {method['label']}: train={mdata['train_error']:.2%}, "
                  f"pred={mdata['pred_error']:.2%}, runtime={mdata['runtime']:.0f}s")
        else:
            print(f"  ✗ {method['label']}: no predictions found "
                  f"(run the script first or use --run)")

    if len(methods_data) < 2:
        print("\nNeed predictions from both methods to generate comparisons.")
        print("Run: python 04_conditional_integral.py && python 05_neural_ode.py")
        sys.exit(1)

    # ── Load shared FOM data & basis ─────────────────────────────────────
    print("\nLoading shared FOM data...")
    shared = load_shared_data()
    print(f"  FOM: {shared['true_states'].shape[0]:,} DOFs × "
          f"{shared['true_states'].shape[1]} time steps")
    print(f"  Basis: {NUM_MODES} modes, energy {shared['basis'].cumulative_energy:.4%}")

    # ── Generate comparison figures ──────────────────────────────────────
    print(f"\nGenerating comparison figures...")
    plot_rom_trajectories_comparison(shared, methods_data, FIGURE_DIR)
    plot_spatial_comparison(shared, methods_data, FIGURE_DIR)
    plot_tumor_volume_comparison(shared, methods_data, FIGURE_DIR)
    plot_metrics_table(methods_data, FIGURE_DIR)

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':<20s} {'Train':>8s} {'Pred':>8s} "
          f"{'Stab':>6s} {'CI_cov':>7s} {'Time':>6s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*6}")
    for method, mdata in methods_data:
        ci = mdata["ci_coverage"]
        ci_str = f"{ci:>6.1%}" if not np.isnan(ci) else "   N/A"
        print(f"  {method['label']:<20s} "
              f"{mdata['train_error']:>7.2%} {mdata['pred_error']:>7.2%} "
              f"{mdata['stability_pct']:>5.0f}% {ci_str} "
              f"{mdata['runtime']:>5.0f}s")

    print(f"\n  All figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()

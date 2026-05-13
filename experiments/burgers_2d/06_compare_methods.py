"""
06_compare_methods.py — Aggregator script for comparing all methods.
    2D Diffusion-Reaction Equation: ∂u/∂t = κ∇²u − βu²

Loads .npz predictions from each method and creates comparison plots:
  - Overlaid full-order error curves (ROM error, projection error, excess)
  - Summary bar charts (train error, pred error, stability)
  - 2D contour comparison plots (True vs method reconstructions)

Usage:
    python 06_compare_methods.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis
from core import generate_trajectory

# ── Constants ────────────────────────────────────────────────────────
SCHEMAS = [
    {
        "name": "dense_medium_noise",
        "label": "Dense data, medium noise",
        "NUM_SAMPLES": 60,
        "NOISE_LEVEL": 0.03,
    },
]

METHODS = [
    {
        "name": "04_conditional_integral",
        "label": "Conditional Integral (2-stage)",
        "color": "tab:purple",
        "linestyle": "--",
    },
    {
        "name": "04_unified",
        "label": "Marg-O × Weak-Form",
        "color": "tab:green",
        "linestyle": "-",
    },
    {
        "name": "05_neural_ode",
        "label": "MLP Ensemble (baseline)",
        "color": "tab:orange",
        "linestyle": "-.",
    },
]

TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 3.0)
SEED = 42
NUM_MODES = 3
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── I/O helpers ──────────────────────────────────────────────────────

def load_method_data(schema_name, method):
    """Load .npz predictions for one method+regime. Returns None if missing."""
    path = os.path.join(
        SCRIPT_DIR, "results", "comparison", schema_name, f"{method['name']}.npz"
    )
    if not os.path.exists(path):
        print(f"  ⚠ Not found: {path}")
        return None
    data = np.load(path)
    rom_solves = data["rom_solves"]
    if rom_solves.size == 0:
        print(f"  ⚠ Empty rom_solves in {path}")
        return None
    return {
        "rom_solves": rom_solves,
        "t_pred": data["t_pred"],
        "train_error": float(data["train_error"]),
        "pred_error": float(data["pred_error"]),
        "stability_pct": float(data["stability_pct"]),
        "ci_coverage": float(data["ci_coverage"]) if "ci_coverage" in data else float("nan"),
        "ci_width": float(data["ci_width"]) if "ci_width" in data else float("nan"),
        "runtime": float(data["runtime"]),
        **method,  # color, label, linestyle, name
    }


def generate_shared_data(schema):
    """Re-generate deterministic data to obtain basis, true_states, and fom."""
    np.random.seed(SEED)
    num_samples = schema["NUM_SAMPLES"]
    noise_level = schema["NOISE_LEVEL"]

    fom, t_full, true_states, t_samp, snaps_samp = generate_trajectory(
        config, config.time_domain, TRAINING_SPAN, num_samples, noise_level
    )
    basis = Basis(num_vectors=NUM_MODES)
    basis.fit(snaps_samp)
    return fom, t_full, true_states, basis


def compute_errors(rom_solves, t_pred, basis, t_full, true_states):
    """Compute full-order ROM errors and projection error on the pred grid."""
    interp_truth = interp1d(t_full, true_states, axis=1, kind="linear",
                            fill_value="extrapolate")
    true_interp = interp_truth(t_pred)

    # Projection error (basis limit)
    true_comp = basis.compress(true_interp)
    true_proj = basis.decompress(true_comp)
    norm_truth = np.linalg.norm(true_interp, axis=0)
    norm_truth = np.maximum(norm_truth, 1e-10)
    projection_error = np.linalg.norm(true_interp - true_proj, axis=0) / norm_truth

    # Per-sample ROM errors
    n_stable = rom_solves.shape[0]
    rom_errors = []
    for i in range(n_stable):
        rom_full = basis.decompress(rom_solves[i])
        error = np.linalg.norm(true_interp - rom_full, axis=0) / norm_truth
        rom_errors.append(error)
    rom_errors = np.array(rom_errors)

    return rom_errors, projection_error


# ── Plotting ─────────────────────────────────────────────────────────

def plot_error_comparison(methods_data, projection_error, t_pred, training_span,
                          title_suffix, save_path):
    """3-panel plot: ROM error, projection error, excess error."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1]})

    # Panel 1: ROM prediction error — all methods overlaid
    ax_rom = axes[0]
    ax_rom.axvspan(training_span[0], training_span[1], color="gray", alpha=0.10)
    for m in methods_data:
        rom_errors = m["rom_errors"]
        median = np.median(rom_errors, axis=0)
        p5 = np.percentile(rom_errors, 5, axis=0)
        p95 = np.percentile(rom_errors, 95, axis=0)
        ax_rom.plot(t_pred, median, color=m["color"],
                    linestyle=m["linestyle"], lw=2,
                    label=f"{m['label']} (median)")
        ax_rom.fill_between(t_pred, p5, p95, color=m["color"], alpha=0.10)
    ax_rom.set_ylabel("Relative Error")
    ax_rom.set_title(f"ROM Prediction Error — {title_suffix}")
    ax_rom.legend(loc="upper left", fontsize=9)
    ax_rom.set_yscale("log")

    # Panel 2: Projection error (same for all methods)
    ax_proj = axes[1]
    ax_proj.axvspan(training_span[0], training_span[1], color="gray", alpha=0.10)
    ax_proj.plot(t_pred, projection_error, "k--", lw=2,
                 label="Projection error (basis limit)")
    ax_proj.set_ylabel("Relative Error")
    ax_proj.set_title("Projection Error (Basis Limit)")
    ax_proj.legend(loc="upper left", fontsize=9)
    ax_proj.set_yscale("log")

    # Panel 3: Excess error (ROM − projection)
    ax_diff = axes[2]
    ax_diff.axvspan(training_span[0], training_span[1], color="gray", alpha=0.10)
    for m in methods_data:
        median = np.median(m["rom_errors"], axis=0)
        excess = np.maximum(median - projection_error, 1e-16)
        ax_diff.plot(t_pred, excess, color=m["color"],
                     linestyle=m["linestyle"], lw=2, label=m["label"])
    ax_diff.set_xlabel("Time")
    ax_diff.set_ylabel("Relative Error")
    ax_diff.set_title("Excess ROM Error (Above Basis Limit)")
    ax_diff.legend(loc="upper left", fontsize=9)
    ax_diff.set_yscale("log")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {save_path}")


def plot_metrics_comparison(methods_data, title_suffix, save_path):
    """Bar chart comparing train error, pred error, stability across methods."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    labels = [m["label"] for m in methods_data]
    colors = [m["color"] for m in methods_data]
    x = np.arange(len(labels))

    # Training error
    train_errors = [m["train_error"] for m in methods_data]
    axes[0].bar(x, train_errors, color=colors, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Relative Error")
    axes[0].set_title("Training Error")

    # Prediction error
    pred_errors = [m["pred_error"] for m in methods_data]
    axes[1].bar(x, pred_errors, color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Relative Error")
    axes[1].set_title("Prediction Error")

    # Stability
    stabilities = [m["stability_pct"] for m in methods_data]
    axes[2].bar(x, stabilities, color=colors, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[2].set_ylabel("Stability %")
    axes[2].set_title("Stability")
    axes[2].set_ylim(0, 105)

    fig.suptitle(f"Method Comparison — {title_suffix}", fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {save_path}")


def plot_2d_contour_comparison(methods_data, fom, basis, t_full, true_states,
                                title_suffix, save_path):
    """2D contour comparison: True vs each method at selected time snapshots."""
    snapshot_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    n_methods = len(methods_data)
    n_cols = len(snapshot_times)
    n_rows = 1 + n_methods  # True + each method

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.0 * n_rows))
    x, y = fom.spatial_domain

    for col, t_snap in enumerate(snapshot_times):
        # True solution
        t_idx = np.argmin(np.abs(t_full - t_snap))
        u_true = fom.reconstruct_2d(true_states[:, t_idx])

        # Compute shared colorbar range from true solution
        vmin_true, vmax_true = u_true.min(), u_true.max()
        levels = np.linspace(vmin_true, vmax_true, 30)

        im0 = axes[0, col].contourf(x, y, u_true, levels=levels,
                                     cmap='RdBu_r', extend='both')
        axes[0, col].set_aspect('equal')
        axes[0, col].set_title(f't = {t_snap:.1f}', fontsize=11)
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

        # Each method
        for row_idx, md in enumerate(methods_data, start=1):
            t_pred = md["t_pred"]
            rom_solves = md["rom_solves"]
            rom_med = np.median(rom_solves, axis=0)

            t_idx_pred = np.argmin(np.abs(t_pred - t_snap))
            u_rom_full = basis.decompress(rom_med[:, t_idx_pred])
            u_rom = fom.reconstruct_2d(u_rom_full)

            im = axes[row_idx, col].contourf(x, y, u_rom, levels=levels,
                                              cmap='RdBu_r', extend='both')
            axes[row_idx, col].set_aspect('equal')
            plt.colorbar(im, ax=axes[row_idx, col], fraction=0.046, pad=0.04)

        # Clean up axis labels
        for row in range(n_rows):
            if col > 0:
                axes[row, col].set_yticklabels([])

    # Row labels
    axes[0, 0].set_ylabel('True', fontsize=11)
    for row_idx, md in enumerate(methods_data, start=1):
        axes[row_idx, 0].set_ylabel(md["label"], fontsize=10)

    fig.suptitle(f'2D Field Comparison — {title_suffix}', fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {save_path}")


# ── Main logic ───────────────────────────────────────────────────────

def compare_regime(schema):
    """Generate comparison plots for one data regime."""
    name = schema["name"]
    label = schema["label"]
    print(f"\n{'='*60}")
    print(f"Regime: {label}  ({name})")
    print(f"{'='*60}")

    # Load method results
    methods_data = []
    for method in METHODS:
        md = load_method_data(name, method)
        if md is not None:
            methods_data.append(md)

    if not methods_data:
        print("  ⚠ No method results found — skipping this regime.")
        return methods_data

    # Re-generate shared data (deterministic)
    print("  Regenerating shared data (basis + true states)…")
    fom, t_full, true_states, basis = generate_shared_data(schema)

    # Compute full-order errors for each method
    t_pred = methods_data[0]["t_pred"]
    projection_error = None
    for md in methods_data:
        rom_errors, proj_err = compute_errors(
            md["rom_solves"], md["t_pred"], basis, t_full, true_states
        )
        md["rom_errors"] = rom_errors
        if projection_error is None:
            projection_error = proj_err

    # Save directory
    out_dir = os.path.join(SCRIPT_DIR, "results", "comparison", name)

    # Plot 1: overlaid error curves
    plot_error_comparison(
        methods_data, projection_error, t_pred, TRAINING_SPAN,
        title_suffix=label,
        save_path=os.path.join(out_dir, "full_order_error_comparison.png"),
    )

    # Plot 2: summary bar charts
    plot_metrics_comparison(
        methods_data,
        title_suffix=label,
        save_path=os.path.join(out_dir, "metrics_comparison.png"),
    )

    # Plot 3: 2D contour comparison
    plot_2d_contour_comparison(
        methods_data, fom, basis, t_full, true_states,
        title_suffix=label,
        save_path=os.path.join(out_dir, "2d_contour_comparison.png"),
    )

    return methods_data


def print_summary_table(all_results):
    """Print a summary table across all regimes and methods."""
    print(f"\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    header = f"{'Regime':<22} {'Method':<24} {'Train %':>8} {'Pred %':>8} {'Stab %':>7} {'CI Cov':>7} {'Time(s)':>8}"
    print(header)
    print("-" * 90)
    for regime_name, methods_data in all_results:
        if not methods_data:
            print(f"{regime_name:<22}  (no results)")
            continue
        for md in methods_data:
            ci_str = f"{md['ci_coverage']:.1f}" if not np.isnan(md["ci_coverage"]) else "  n/a"
            print(
                f"{regime_name:<22} {md['label']:<24} "
                f"{md['train_error']*100:>7.2f}% "
                f"{md['pred_error']*100:>7.2f}% "
                f"{md['stability_pct']:>6.1f}% "
                f"{ci_str:>7} "
                f"{md['runtime']:>7.1f}s"
            )
    print(f"{'='*90}")


def main(schema_names=None):
    schemas = SCHEMAS
    if schema_names:
        schemas = [s for s in SCHEMAS if s["name"] in schema_names]
    if not schemas:
        print(f"No matching schemas for: {schema_names}")
        print(f"Available: {[s['name'] for s in SCHEMAS]}")
        sys.exit(1)

    all_results = []
    for schema in schemas:
        methods_data = compare_regime(schema)
        all_results.append((schema["name"], methods_data))

    print_summary_table(all_results)


if __name__ == "__main__":
    schema_names = sys.argv[1:] if len(sys.argv) > 1 else None
    main(schema_names)

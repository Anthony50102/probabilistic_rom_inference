"""
06_compare_methods.py — Aggregator script for comparing all methods (Tumor).

Loads .npz predictions from each method and creates comparison plots:
  - Overlaid full-order error curves (ROM error, projection error, excess)
  - Summary bar charts (train error, pred error, stability)

Add new methods to the METHODS list — each is expected to write
results/comparison/<schema>/<method>.npz with keys
{rom_solves, t_pred, train_error, pred_error, stability_pct,
 ci_coverage, ci_width, runtime}.

Tumor uses adaptive POD: different methods may use different effective
mode counts.  We fit a basis with NUM_MODES_MAX modes once per schema
and use the first rom_solves.shape[1] columns to decompress.

Usage:
    python 06_compare_methods.py                    # all regimes
    python 06_compare_methods.py dense_medium_noise # single regime
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
from config import Basis, load_fom_data

# ── Constants ────────────────────────────────────────────────────────
SCHEMAS = [
    {"name": "dense_low_noise",    "label": "Dense data, low noise",
     "NUM_SAMPLES": 80, "NOISE_LEVEL": 0.01, "NUM_EVAL_POINTS": 200},
    {"name": "dense_medium_noise", "label": "Dense data, medium noise",
     "NUM_SAMPLES": 80, "NOISE_LEVEL": 0.03, "NUM_EVAL_POINTS": 200},
    {"name": "dense_high_noise",   "label": "Dense data, high noise",
     "NUM_SAMPLES": 80, "NOISE_LEVEL": 0.05, "NUM_EVAL_POINTS": 200},
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
        "label": "Neural ODE (baseline)",
        "color": "tab:orange",
        "linestyle": "-.",
    },
]

TRAINING_SPAN = config.TRAINING_SPAN
SEED = 42
NUM_MODES_MAX = 4
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
        **method,
    }


def generate_shared_data(schema):
    """Re-load FOM data to obtain basis and true_states for this regime."""
    np.random.seed(SEED)
    t_pred = np.linspace(
        TRAINING_SPAN[0], config.PREDICTION_DAYS, schema["NUM_EVAL_POINTS"])
    fom, t_full, true_states, t_samp, snaps_noisy = load_fom_data(
        t_pred, TRAINING_SPAN, schema["NUM_SAMPLES"], schema["NOISE_LEVEL"])
    basis = Basis(num_vectors=NUM_MODES_MAX)
    basis.fit(snaps_noisy)
    return fom, t_full, true_states, basis


def compute_errors(rom_solves, t_pred, basis, t_full, true_states):
    """Full-order ROM error and projection error.  rom_solves may have
    fewer modes than basis (adaptive POD) — we pad with zeros to basis.r
    before decompressing (basis.decompress handles the mean shift)."""
    interp_truth = interp1d(t_full, true_states, axis=1, kind="linear",
                            fill_value="extrapolate")
    true_interp = interp_truth(t_pred)

    n_modes_method = rom_solves.shape[1]
    full_r = basis.entries.shape[1]

    def _pad(coeffs):
        if coeffs.shape[0] == full_r:
            return coeffs
        out = np.zeros((full_r, coeffs.shape[1]))
        out[:n_modes_method] = coeffs
        return out

    true_comp = basis.compress(true_interp)
    true_comp_truncated = np.zeros_like(true_comp)
    true_comp_truncated[:n_modes_method] = true_comp[:n_modes_method]
    true_proj = basis.decompress(true_comp_truncated)
    norm_truth = np.maximum(np.linalg.norm(true_interp, axis=0), 1e-10)
    projection_error = np.linalg.norm(true_interp - true_proj, axis=0) / norm_truth

    rom_errors = []
    for i in range(rom_solves.shape[0]):
        rom_full = basis.decompress(_pad(rom_solves[i]))
        err = np.linalg.norm(true_interp - rom_full, axis=0) / norm_truth
        rom_errors.append(err)
    return np.array(rom_errors), projection_error


# ── Plotting ─────────────────────────────────────────────────────────

def plot_error_comparison(methods_data, t_pred, training_span,
                          title_suffix, save_path):
    """3-panel plot: ROM error, projection error, excess error.
    Each method has its own projection-error curve (because mode counts differ)."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1]})

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

    ax_proj = axes[1]
    ax_proj.axvspan(training_span[0], training_span[1], color="gray", alpha=0.10)
    for m in methods_data:
        ax_proj.plot(t_pred, m["projection_error"], color=m["color"],
                     linestyle=":", lw=1.5,
                     label=f"{m['label']} ({m['rom_solves'].shape[1]} modes)")
    ax_proj.set_ylabel("Relative Error")
    ax_proj.set_title("Projection Error (Basis Limit)")
    ax_proj.legend(loc="upper left", fontsize=9)
    ax_proj.set_yscale("log")

    ax_diff = axes[2]
    ax_diff.axvspan(training_span[0], training_span[1], color="gray", alpha=0.10)
    for m in methods_data:
        median = np.median(m["rom_errors"], axis=0)
        excess = np.maximum(median - m["projection_error"], 1e-16)
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
    """Bar chart: train error, pred error, stability."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    labels = [m["label"] for m in methods_data]
    colors = [m["color"] for m in methods_data]
    x = np.arange(len(labels))

    axes[0].bar(x, [m["train_error"] for m in methods_data],
                color=colors, alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20,
                                                     ha="right", fontsize=9)
    axes[0].set_ylabel("Relative Error"); axes[0].set_title("Training Error")

    axes[1].bar(x, [m["pred_error"] for m in methods_data],
                color=colors, alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20,
                                                     ha="right", fontsize=9)
    axes[1].set_ylabel("Relative Error"); axes[1].set_title("Prediction Error")

    axes[2].bar(x, [m["stability_pct"] for m in methods_data],
                color=colors, alpha=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=20,
                                                     ha="right", fontsize=9)
    axes[2].set_ylabel("Stability %"); axes[2].set_title("Stability")
    axes[2].set_ylim(0, 105)

    fig.suptitle(f"Method Comparison — {title_suffix}", fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────

def compare_regime(schema):
    name = schema["name"]; label = schema["label"]
    print(f"\n{'='*60}\nRegime: {label}  ({name})\n{'='*60}")

    methods_data = []
    for method in METHODS:
        md = load_method_data(name, method)
        if md is not None:
            methods_data.append(md)

    if not methods_data:
        print("  ⚠ No method results found — skipping.")
        return methods_data

    print("  Regenerating shared data (basis + true states)…")
    fom, t_full, true_states, basis = generate_shared_data(schema)

    t_pred = methods_data[0]["t_pred"]
    for md in methods_data:
        rom_errs, proj_err = compute_errors(
            md["rom_solves"], md["t_pred"], basis, t_full, true_states)
        md["rom_errors"] = rom_errs
        md["projection_error"] = proj_err

    out_dir = os.path.join(SCRIPT_DIR, "results", "comparison", name)
    plot_error_comparison(methods_data, t_pred, TRAINING_SPAN, label,
                          os.path.join(out_dir, "full_order_error_comparison.png"))
    plot_metrics_comparison(methods_data, label,
                            os.path.join(out_dir, "metrics_comparison.png"))
    return methods_data


def print_summary_table(all_results):
    print(f"\n{'='*90}\nSUMMARY TABLE\n{'='*90}")
    print(f"{'Regime':<22} {'Method':<32} {'Train %':>8} {'Pred %':>8} "
          f"{'Stab %':>7} {'CI Cov':>7} {'Time(s)':>8}")
    print("-" * 90)
    for regime_name, methods_data in all_results:
        if not methods_data:
            print(f"{regime_name:<22}  (no results)")
            continue
        for md in methods_data:
            ci_str = (f"{md['ci_coverage']*100:.1f}%"
                      if not np.isnan(md['ci_coverage']) else "  n/a")
            print(f"{regime_name:<22} {md['label']:<32} "
                  f"{md['train_error']*100:>7.2f}% "
                  f"{md['pred_error']*100:>7.2f}% "
                  f"{md['stability_pct']:>6.1f}% "
                  f"{ci_str:>7} "
                  f"{md['runtime']:>7.1f}s")
    print(f"{'='*90}")


def main(schema_names=None):
    schemas = SCHEMAS
    if schema_names:
        schemas = [s for s in SCHEMAS if s["name"] in schema_names]
    if not schemas:
        print(f"No matching schemas for: {schema_names}")
        print(f"Available: {[s['name'] for s in SCHEMAS]}")
        sys.exit(1)

    all_results = [(s["name"], compare_regime(s)) for s in schemas]
    print_summary_table(all_results)


if __name__ == "__main__":
    schema_names = sys.argv[1:] if len(sys.argv) > 1 else None
    main(schema_names)

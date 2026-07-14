"""
06_compare_methods.py — Aggregator script for comparing all methods.

Loads .npz predictions from each method and creates comparison plots:
  - Overlaid full-order error curves (ROM error, projection error, excess)
  - Summary bar charts (train error, pred error, stability)

Usage:
    python 06_compare_methods.py                    # all regimes
    python 06_compare_methods.py dense_low_noise    # single regime
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
from core.plotting import comparison
from config import Basis
from core.plotting import save_metrics_table

from core import generate_trajectory

# ── Constants ────────────────────────────────────────────────────────
SCHEMAS = [
    {
        "name": "dense_low_noise",
        "label": "Dense data, low noise",
        "NUM_SAMPLES": 250,
        "NOISE_LEVEL": 0.01,
    },
    {
        "name": "sparse_low_noise",
        "label": "Sparse data, low noise",
        "NUM_SAMPLES": 55,
        "NOISE_LEVEL": 0.03,
    },
    {
        "name": "dense_high_noise",
        "label": "Dense data, high noise",
        "NUM_SAMPLES": 250,
        "NOISE_LEVEL": 0.10,
    },
]

METHODS = [
    {
        "name": "04_unified",
        "label": "Bayesian OpInf",
        "color": "tab:purple",
        "linestyle": "-",
    },
    {
        "name": "05_neural_ode",
        "label": "Neural ODE",
        "color": "tab:orange",
        "linestyle": "-.",
    },
]

TRAINING_SPAN = (0, 0.08)
PREDICTION_SPAN = (0, 0.15)
SEED = 42
NUM_MODES = 6
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
    """Re-generate deterministic data to obtain basis and true_states."""
    np.random.seed(SEED)
    num_samples = schema["NUM_SAMPLES"]
    noise_level = schema["NOISE_LEVEL"]

    fom, t_full, true_states, t_samp, snaps_samp = generate_trajectory(
        config, config.time_domain, TRAINING_SPAN, num_samples, noise_level
    )
    basis = Basis(num_vectors=NUM_MODES)
    basis.fit(snaps_samp)
    return t_full, true_states, basis


def compute_errors(rom_solves, t_pred, basis, t_full, true_states):
    return comparison.compute_full_order_errors(rom_solves, t_pred, basis, t_full, true_states)

def plot_error_comparison(methods_data, projection_error, t_pred, training_span,
                          title_suffix, save_path):
    comparison.error_comparison(methods_data, projection_error, t_pred, training_span, title_suffix, save_path)
    print(f"  Saved: {save_path}")

def plot_metrics_comparison(methods_data, title_suffix, save_path):
    comparison.metrics_bars(methods_data, title_suffix, save_path)
    print(f"  Saved: {save_path}")

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
    t_full, true_states, basis = generate_shared_data(schema)

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

    # Plot 3: ML-style metrics table
    save_metrics_table(
        methods_data,
        title=f"Method Comparison — {label}",
        png_path=os.path.join(out_dir, "metrics_table.png"),
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

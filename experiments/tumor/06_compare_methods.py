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
from core.plotting import comparison
from config import Basis, load_fom_data
from core.plotting import save_metrics_table

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
    if "rom_solves" in data.files:
        rom_solves = data["rom_solves"]
    elif "rom_solves_0" in data.files:
        rom_solves = data["rom_solves_0"]
    else:
        print(f"  ⚠ No rom_solves[_0] in {path}")
        return None
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
    return comparison.compute_full_order_errors(rom_solves, t_pred, basis, t_full, true_states)

def plot_error_comparison(methods_data, t_pred, training_span,
                          title_suffix, save_path):
    comparison.error_comparison(methods_data, None, t_pred, training_span, title_suffix, save_path)
    print(f"  Saved: {save_path}")

def plot_metrics_comparison(methods_data, title_suffix, save_path):
    comparison.metrics_bars(methods_data, title_suffix, save_path)
    print(f"  Saved: {save_path}")

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
    save_metrics_table(
        methods_data,
        title=f"Method Comparison — {label}",
        png_path=os.path.join(out_dir, "metrics_table.png"),
    )
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

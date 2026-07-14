"""
06_compare_methods.py — Aggregator for heat experiment method comparison.

Loads predictions from all 3 methods (02_two_stage_svi, 04_unified,
05_neural_ode) and creates overlaid comparison plots for each data regime.

Usage:
    python 06_compare_methods.py                          # all regimes
    python 06_compare_methods.py dense_low_noise          # single regime
    python 06_compare_methods.py dense_low_noise dense_high_noise
"""

import sys
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))

import config
from core.plotting import comparison
from config import Basis, input_func_factory, input_parameters, test_parameters
from core.plotting import save_metrics_table
from step1_generate_data import TrajectorySampler

# ── Data regimes (identical to 04/05) ────────────────────────────────────────
SCHEMAS = [
    {
        "name": "sparse_low_noise",
        "label": "Sparse data, low noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.01,
        "NUM_EVAL_POINTS": 100,
    },
    {
        "name": "sparse_medium_noise",
        "label": "Sparse data, medium noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.03,
        "NUM_EVAL_POINTS": 100,
    },
    {
        "name": "sparse_high_noise",
        "label": "Sparse data, high noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.05,
        "NUM_EVAL_POINTS": 100,
    },
]

TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 2.0)
SEED = 42
NUM_MODES = 5
NUM_ICS = 5

# ── Method styling ───────────────────────────────────────────────────────────
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


# ── Data loading ─────────────────────────────────────────────────────────────
def load_method_data(schema_name, method):
    """Load .npz predictions for one method+regime (multi-IC format)."""
    path = os.path.join(
        SCRIPT_DIR, "results", "comparison", schema_name, f"{method['name']}.npz"
    )
    if not os.path.exists(path):
        print(f"  ⚠ Not found: {path}")
        return None
    data = np.load(path, allow_pickle=True)
    n_ics = int(data["n_ics"])

    all_rom_solves = []
    for ic in range(n_ics):
        key = f"rom_solves_{ic}"
        if key in data:
            all_rom_solves.append(data[key])
        else:
            all_rom_solves.append(np.empty((0, NUM_MODES, len(data["t_pred"]))))

    return {
        "all_rom_solves": all_rom_solves,
        "t_pred": data["t_pred"],
        "train_error": float(data["train_error"]),
        "pred_error": float(data["pred_error"]),
        "stability_pct": float(data["stability_pct"]),
        "ci_coverage": float(data.get("ci_coverage", float("nan"))),
        "ci_width": float(data.get("ci_width", float("nan"))),
        "runtime": float(data["runtime"]),
        "n_ics": n_ics,
        **method,
    }


def regenerate_data(schema):
    """Re-generate the shared data (deterministic, same seed) for one regime."""
    num_samples = schema["NUM_SAMPLES"]
    noise_level = schema["NOISE_LEVEL"]
    num_eval_points = schema["NUM_EVAL_POINTS"]

    train_params = list(input_parameters[:NUM_ICS])

    np.random.seed(SEED)
    sampler = TrajectorySampler(
        training_span=TRAINING_SPAN,
        num_samples=num_samples,
        noiselevel=noise_level,
        num_regression_points=num_eval_points,
        synced=False,
    )
    all_true_states, all_time_sampled, all_snapshots, all_training_inputs = (
        sampler.multisample(train_params)
    )

    # Fit shared basis
    snapshots_train = np.hstack(all_snapshots)
    basis = Basis(num_vectors=NUM_MODES)
    basis.fit(snapshots_train)

    # Generate test IC
    test_sampler = TrajectorySampler(
        training_span=TRAINING_SPAN,
        num_samples=num_samples,
        noiselevel=noise_level,
        num_regression_points=num_eval_points,
        synced=False,
    )
    test_true_list, test_t_list, test_snap_list, test_inp_list = (
        test_sampler.multisample([test_parameters])
    )

    # Combine: train ICs + test IC
    all_true = list(all_true_states) + list(test_true_list)

    return basis, all_true


# ── Error computation ────────────────────────────────────────────────────────
def compute_rom_errors(basis, true_states_full_ic, rom_solves_ic, t_pred):
    """Compute per-sample relative error curves for one IC."""
    true_interp = interp1d(
        config.time_domain,
        true_states_full_ic,
        axis=1,
        kind="linear",
        fill_value="extrapolate",
    )
    true_at_pred = true_interp(t_pred)

    norm_truth = np.linalg.norm(true_at_pred, axis=0)
    norm_truth = np.maximum(norm_truth, 1e-10)

    rom_errors = []
    for i in range(len(rom_solves_ic)):
        rom_full = basis.decompress(rom_solves_ic[i])
        err = np.linalg.norm(true_at_pred - rom_full, axis=0) / norm_truth
        rom_errors.append(err)
    return np.array(rom_errors)


def compute_projection_error(basis, true_states_full_ic, t_pred):
    """Projection error = basis truncation limit."""
    true_interp = interp1d(
        config.time_domain,
        true_states_full_ic,
        axis=1,
        kind="linear",
        fill_value="extrapolate",
    )
    true_at_pred = true_interp(t_pred)

    true_comp = basis.compress(true_at_pred)
    true_proj = basis.decompress(true_comp)

    norm_truth = np.linalg.norm(true_at_pred, axis=0)
    norm_truth = np.maximum(norm_truth, 1e-10)
    return np.linalg.norm(true_at_pred - true_proj, axis=0) / norm_truth


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_error_comparison(methods_with_data, projection_error, t_pred, schema,
                          ic_label, save_path):
    comparison.error_comparison(methods_with_data, projection_error, t_pred, TRAINING_SPAN, schema['label'], save_path)
    print(f"  Saved: {save_path}")

def plot_metrics_comparison(methods_with_data, schema, save_path,
                            train_test_errors=None):
    comparison.metrics_bars(methods_with_data, schema['label'], save_path)
    print(f"  Saved: {save_path}")

def compare_regime(schema):
    """Load all methods, regenerate shared data, and create comparison plots."""
    schema_name = schema["name"]
    print(f"\n{'='*60}")
    print(f"  Regime: {schema['label']} ({schema_name})")
    print(f"{'='*60}")

    # Load method predictions
    methods_data = []
    for method in METHODS:
        md = load_method_data(schema_name, method)
        if md is not None:
            methods_data.append(md)

    if not methods_data:
        print("  ⚠ No method data found for this regime — skipping.")
        return None

    # Re-generate shared data for basis and true_states
    print(f"  Regenerating shared data (seed={SEED})...")
    basis, all_true = regenerate_data(schema)

    # Use t_pred from the first available method
    t_pred = methods_data[0]["t_pred"]

    # Output directory
    out_dir = os.path.join(SCRIPT_DIR, "results", "comparison", schema_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Plot 1: Training IC 0 error comparison ──
    print("  Computing full-order errors (IC 0)...")
    true_ic0 = all_true[0]
    projection_error = compute_projection_error(basis, true_ic0, t_pred)

    for md in methods_data:
        rom_solves_ic = md["all_rom_solves"][0]
        if len(rom_solves_ic) == 0:
            md["rom_errors"] = np.empty((0, len(t_pred)))
            continue
        md["rom_errors"] = compute_rom_errors(basis, true_ic0, rom_solves_ic, t_pred)

    methods_with_data = [md for md in methods_data
                         if "rom_errors" in md and len(md["rom_errors"]) > 0]

    if methods_with_data:
        plot_error_comparison(
            methods_with_data, projection_error, t_pred, schema,
            ic_label="IC 0",
            save_path=os.path.join(out_dir, "full_order_error_comparison.png"),
        )

    # ── Plot 2: Metrics bar chart ──
    # Compute per-IC train/pred relative L2 errors split by Train ICs vs Test IC
    print("  Computing per-IC train/test relative L2 errors...")
    tt_errors = {}
    train_mask = t_pred <= TRAINING_SPAN[1]
    pred_mask = t_pred > TRAINING_SPAN[1]
    for md in methods_data:
        n_ics = md["n_ics"]
        train_ic_indices = list(range(n_ics - 1))  # all but last
        test_ic_idx = n_ics - 1

        def _rel_l2(rom_solves_ic, true_ic):
            """Median-prediction relative L2 (train and pred region)."""
            if len(rom_solves_ic) == 0:
                return float("nan"), float("nan")
            interp = interp1d(config.time_domain, true_ic, axis=1,
                              kind="linear", fill_value="extrapolate")
            true_at = interp(t_pred)
            rom_full = np.array([basis.decompress(rs) for rs in rom_solves_ic])
            rom_med = np.median(rom_full, axis=0)
            tr = (np.linalg.norm(rom_med[:, train_mask] - true_at[:, train_mask])
                  / max(np.linalg.norm(true_at[:, train_mask]), 1e-12))
            pr = (np.linalg.norm(rom_med[:, pred_mask] - true_at[:, pred_mask])
                  / max(np.linalg.norm(true_at[:, pred_mask]), 1e-12))
            return float(tr), float(pr)

        train_l2_vals, pred_l2_vals = [], []
        for ic in train_ic_indices:
            tr, pr = _rel_l2(md["all_rom_solves"][ic], all_true[ic])
            if not np.isnan(tr):
                train_l2_vals.append(tr); pred_l2_vals.append(pr)
        test_tr, test_pr = (float("nan"), float("nan"))
        if 0 <= test_ic_idx < len(md["all_rom_solves"]):
            test_tr, test_pr = _rel_l2(
                md["all_rom_solves"][test_ic_idx], all_true[test_ic_idx])
        tt_errors[md["label"]] = {
            "train_l2": float(np.mean(train_l2_vals)) if train_l2_vals else float("nan"),
            "pred_l2":  float(np.mean(pred_l2_vals))  if pred_l2_vals  else float("nan"),
            "test_train_l2": test_tr,
            "test_pred_l2":  test_pr,
        }

    plot_metrics_comparison(
        methods_data, schema,
        save_path=os.path.join(out_dir, "metrics_comparison.png"),
        train_test_errors=tt_errors,
    )

    # ── ML-style metrics table ──
    save_metrics_table(
        methods_data,
        title=f"Method Comparison — {schema['label']}",
        png_path=os.path.join(out_dir, "metrics_table.png"),
    )

    # ── Plot 3: Test IC error comparison ──
    print("  Computing full-order errors (Test IC)...")
    for md in methods_data:
        test_ic_idx = md["n_ics"] - 1
        if test_ic_idx < 0 or test_ic_idx >= len(md["all_rom_solves"]):
            md["rom_errors_test"] = np.empty((0, len(t_pred)))
            continue
        rom_solves_test = md["all_rom_solves"][test_ic_idx]
        if len(rom_solves_test) == 0:
            md["rom_errors_test"] = np.empty((0, len(t_pred)))
            continue
        md["rom_errors_test"] = compute_rom_errors(
            basis, all_true[test_ic_idx], rom_solves_test, t_pred
        )

    # Projection error for test IC
    test_ic_idx = methods_data[0]["n_ics"] - 1
    if test_ic_idx < len(all_true):
        proj_err_test = compute_projection_error(basis, all_true[test_ic_idx], t_pred)

        # Swap rom_errors for test plotting
        test_methods = []
        for md in methods_data:
            md_copy = dict(md)
            md_copy["rom_errors"] = md.get("rom_errors_test", np.empty((0, len(t_pred))))
            test_methods.append(md_copy)
        test_with_data = [md for md in test_methods
                          if len(md["rom_errors"]) > 0]

        if test_with_data:
            plot_error_comparison(
                test_with_data, proj_err_test, t_pred, schema,
                ic_label="Test IC",
                save_path=os.path.join(out_dir, "full_order_error_comparison_test.png"),
            )

    return methods_data


# ── Summary table ────────────────────────────────────────────────────────────
def print_summary(all_results):
    """Print a summary table of all regimes and methods."""
    print(f"\n{'='*90}")
    print("  SUMMARY TABLE")
    print(f"{'='*90}")
    header = f"{'Regime':<25s} {'Method':<25s} {'Train%':>8s} {'Pred%':>8s} {'Stab%':>7s} {'CI Cov':>7s} {'Time':>7s}"
    print(header)
    print("-" * 90)

    for schema_name, methods_data in all_results.items():
        if methods_data is None:
            print(f"{schema_name:<25s}  (no data)")
            continue
        for md in methods_data:
            ci_str = f"{md['ci_coverage']*100:.0f}%" if not np.isnan(md["ci_coverage"]) else "N/A"
            print(
                f"{schema_name:<25s} {md['label']:<25s} "
                f"{md['train_error']*100:>7.1f}% {md['pred_error']*100:>7.1f}% "
                f"{md['stability_pct']:>6.0f}% {ci_str:>7s} "
                f"{md['runtime']:>6.1f}s"
            )


# ── Main ─────────────────────────────────────────────────────────────────────
def main(schema_names=None):
    """Run comparison for selected (or all) regimes."""
    if schema_names:
        schemas = [s for s in SCHEMAS if s["name"] in schema_names]
        missing = set(schema_names) - {s["name"] for s in schemas}
        if missing:
            print(f"Warning: unknown schema(s): {missing}")
    else:
        schemas = SCHEMAS

    all_results = {}
    for schema in schemas:
        result = compare_regime(schema)
        all_results[schema["name"]] = result

    print_summary(all_results)
    print("\nDone.")


if __name__ == "__main__":
    schema_names = sys.argv[1:] if len(sys.argv) > 1 else None
    main(schema_names)

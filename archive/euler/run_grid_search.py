"""
Hyperparameter grid search over GAMMA and GAMMA2 using papermill.

Runs 02_full_bayesian.ipynb for each (GAMMA, GAMMA2) combination,
saving executed notebooks and a summary CSV to results/grid_search/.

Usage:
    python run_grid_search.py
"""

import os
import itertools
import json
import time
from datetime import datetime

import numpy as np
import papermill as pm

# ── Grid definition ──────────────────────────────────────────────────────────
GAMMA_VALUES = [1e-1, 1e0, 1e1, 1e2]
GAMMA2_VALUES = [1e-1, 1e0, 1e1, 1e2]

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(SCRIPT_DIR, "02_full_bayesian.ipynb")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "grid_search",
                          datetime.now().strftime("%Y%m%d_%H%M%S"))

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single(gamma: float, gamma2: float, output_dir: str) -> dict:
    """Execute the notebook with a single (GAMMA, GAMMA2) pair."""
    tag = f"g1_{gamma:.0e}_g2_{gamma2:.0e}".replace("+", "")
    output_path = os.path.join(output_dir, f"{tag}.ipynb")

    params = {
        "GAMMA": gamma,
        "GAMMA2": gamma2,
        "VERBOSE": False,
    }

    print(f"\n{'='*60}")
    print(f"  GAMMA={gamma:.1e}  GAMMA2={gamma2:.1e}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    result = {
        "gamma": gamma,
        "gamma2": gamma2,
        "output_notebook": output_path,
        "status": "pending",
        "elapsed_s": None,
        "error": None,
    }

    t0 = time.time()
    try:
        pm.execute_notebook(
            NOTEBOOK,
            output_path,
            parameters=params,
            cwd=SCRIPT_DIR,
            kernel_name="python3",
        )
        result["status"] = "success"
    except pm.PapermillExecutionError as exc:
        result["status"] = "failed"
        result["error"] = str(exc)[:500]
        print(f"  !! Execution failed: {exc}")
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:500]
        print(f"  !! Unexpected error: {exc}")
    finally:
        result["elapsed_s"] = round(time.time() - t0, 1)

    return result


def main():
    combos = list(itertools.product(GAMMA_VALUES, GAMMA2_VALUES))
    print(f"Running grid search: {len(combos)} combinations")
    print(f"  GAMMA  values: {GAMMA_VALUES}")
    print(f"  GAMMA2 values: {GAMMA2_VALUES}")
    print(f"  Output dir:    {OUTPUT_DIR}\n")

    results = []
    for i, (g1, g2) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}]", end="")
        res = run_single(g1, g2, OUTPUT_DIR)
        results.append(res)
        print(f"  -> {res['status']} ({res['elapsed_s']}s)")

    # ── Save summary ─────────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also write a quick CSV for easy inspection
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("gamma,gamma2,status,elapsed_s,error\n")
        for r in results:
            err = (r["error"] or "").replace(",", ";").replace("\n", " ")
            f.write(f"{r['gamma']},{r['gamma2']},{r['status']},{r['elapsed_s']},{err}\n")

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    n_ok = sum(1 for r in results if r["status"] == "success")
    print(f"  Success: {n_ok}/{len(results)}")
    print(f"  Summary: {summary_path}")
    print(f"  CSV:     {csv_path}")
    print()
    print(f"  {'GAMMA':>10s}  {'GAMMA2':>10s}  {'status':>8s}  {'time':>8s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for r in results:
        print(f"  {r['gamma']:>10.1e}  {r['gamma2']:>10.1e}  {r['status']:>8s}  {r['elapsed_s']:>7.1f}s")


if __name__ == "__main__":
    main()

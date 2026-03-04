"""
Generate paper figures for the FitzHugh-Nagumo equation.

Runs both 01_gpbayes_opinf.ipynb (baseline) and 02_full_bayesian.ipynb
for each data regime:
  1. Dense data, low noise (600 samples, 1% noise)

Each configuration is executed via papermill with SVI (AutoDelta),
saving executed notebooks and a summary to results/paper_runs/.

Usage:
    python generate_paper.py
"""

import os
import json
import time
from datetime import datetime

import papermill as pm

# ── Data regime definitions ──────────────────────────────────────────────────
# Configuration copied from 02_full_bayesian.ipynb defaults.
SCHEMAS = [
    {
        "name": "dense_low_noise",
        "label": "Dense data, low noise",
        "NUM_SAMPLES": 600,
        "NOISE_LEVEL": 0.01,
        "NUM_MODES": 3,
        "GAMMA": 1e0,
        "GAMMA2": 1e0,
        "GAMMA3": 1e-1,
        "GAMMA4": 1e-1,
    },
]

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_BAYESIAN = os.path.join(SCRIPT_DIR, "02_full_bayesian.ipynb")
NOTEBOOK_GPBAYES = os.path.join(SCRIPT_DIR, "01_gpbayes_opinf.ipynb")
OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "results", "paper_runs", datetime.now().strftime("%Y%m%d_%H%M%S")
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _execute_notebook(notebook_path, output_path, params):
    """Execute a single notebook via papermill and return status info."""
    result = {
        "output_notebook": output_path,
        "status": "pending",
        "elapsed_s": None,
        "error": None,
    }
    t0 = time.time()
    try:
        pm.execute_notebook(
            notebook_path,
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


def run_schema(schema, output_dir):
    """Execute both GP-Bayes and Full Bayesian notebooks for a data regime."""
    tag = schema["name"]

    print(f"\n{'=' * 60}")
    print(f"  Schema: {schema['label']}")
    print(f"  NUM_SAMPLES={schema['NUM_SAMPLES']}, "
          f"NOISE_LEVEL={schema['NOISE_LEVEL']:.3f}")
    print(f"  GAMMA={schema['GAMMA']:.1e}, GAMMA2={schema['GAMMA2']:.1e}, "
          f"GAMMA3={schema['GAMMA3']:.1e}, GAMMA4={schema['GAMMA4']:.1e}")
    print(f"{'=' * 60}")

    base_info = {
        "schema": tag,
        "label": schema["label"],
        "num_samples": schema["NUM_SAMPLES"],
        "noise_level": schema["NOISE_LEVEL"],
        "gamma": schema["GAMMA"],
        "gamma2": schema["GAMMA2"],
        "gamma3": schema["GAMMA3"],
        "gamma4": schema["GAMMA4"],
    }

    results = []

    # --- GP-Bayes OpInf (baseline) ---
    gpbayes_path = os.path.join(output_dir, f"{tag}_gpbayes.ipynb")
    gpbayes_params = {
        "num_samples": schema["NUM_SAMPLES"],
        "noiselevel": schema["NOISE_LEVEL"],
        "numPODmodes": schema["NUM_MODES"],
        "VERBOSE": False,
    }
    print(f"  [GP-Bayes] Output: {gpbayes_path}")
    gpbayes_result = {**base_info, "method": "gpbayes", **_execute_notebook(
        NOTEBOOK_GPBAYES, gpbayes_path, gpbayes_params
    )}
    results.append(gpbayes_result)
    print(f"  [GP-Bayes] -> {gpbayes_result['status']} ({gpbayes_result['elapsed_s']}s)")

    # --- Full Bayesian ---
    bayesian_path = os.path.join(output_dir, f"{tag}_bayesian.ipynb")
    bayesian_params = {
        "NUM_SAMPLES": schema["NUM_SAMPLES"],
        "NOISE_LEVEL": schema["NOISE_LEVEL"],
        "GAMMA": schema["GAMMA"],
        "GAMMA2": schema["GAMMA2"],
        "GAMMA3": schema["GAMMA3"],
        "GAMMA4": schema["GAMMA4"],
        "RUN_SVI": True,
        "RUN_MCMC": True,
        "VERBOSE": False,
    }
    print(f"  [Bayesian] Output: {bayesian_path}")
    bayesian_result = {**base_info, "method": "bayesian", **_execute_notebook(
        NOTEBOOK_BAYESIAN, bayesian_path, bayesian_params
    )}
    results.append(bayesian_result)
    print(f"  [Bayesian] -> {bayesian_result['status']} ({bayesian_result['elapsed_s']}s)")

    return results


def main():
    print(f"Generating paper results: {len(SCHEMAS)} data regimes")
    print(f"Output dir: {OUTPUT_DIR}\n")

    for s in SCHEMAS:
        print(f"  * {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  "
              f"noise={s['NOISE_LEVEL']:.3f}  "
              f"gamma={s['GAMMA']:.0e}  gamma2={s['GAMMA2']:.0e}")

    results = []
    for i, schema in enumerate(SCHEMAS, 1):
        print(f"\n[{i}/{len(SCHEMAS)}]")
        schema_results = run_schema(schema, OUTPUT_DIR)
        results.extend(schema_results)

    # ── Save summary ─────────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("schema,method,num_samples,noise_level,gamma,gamma2,gamma3,gamma4,status,elapsed_s,error\n")
        for r in results:
            err = (r.get("error") or "").replace(",", ";").replace("\n", " ")
            f.write(
                f"{r['schema']},{r.get('method','')},{r['num_samples']},"
                f"{r['noise_level']},{r['gamma']},{r['gamma2']},"
                f"{r.get('gamma3','')},{r.get('gamma4','')},"
                f"{r['status']},{r['elapsed_s']},{err}\n"
            )

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PAPER GENERATION COMPLETE")
    print(f"{'=' * 60}")
    n_ok = sum(1 for r in results if r["status"] == "success")
    print(f"  Success: {n_ok}/{len(results)}")
    print(f"  Summary: {summary_path}")
    print(f"  CSV:     {csv_path}")
    print()
    print(f"  {'Schema':>25s}  {'Method':>10s}  {'Samples':>7s}  {'Noise':>7s}  "
          f"{'Status':>8s}  {'Time':>8s}")
    print(f"  {'-' * 25}  {'-' * 10}  {'-' * 7}  {'-' * 7}  {'-' * 8}  {'-' * 8}")
    for r in results:
        print(f"  {r['label']:>25s}  {r.get('method',''):>10s}  "
              f"{r['num_samples']:>7d}  "
              f"{r['noise_level']:>7.3f}  {r['status']:>8s}  "
              f"{r['elapsed_s']:>7.1f}s")


if __name__ == "__main__":
    main()

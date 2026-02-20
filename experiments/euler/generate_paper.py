"""
Generate paper figures for the Compressible Euler equation.

Runs 02_full_bayesian.ipynb for three data regimes:
  1. Dense data, low noise    (250 samples, 3% noise)
  2. Sparse data, medium noise (55 samples, 5% noise)
  3. Dense data, high noise   (250 samples, 15% noise)

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
SCHEMAS = [
    {
        "name": "dense_low_noise",
        "label": "Dense data, low noise",
        "NUM_SAMPLES": 250,
        "NOISE_LEVEL": 0.03,
        "NUM_EVAL_POINTS": 400,
        "NUM_MODES": 6,
        "GAMMA": 1e1,
        "GAMMA2": 1e1,
    },
    {
        "name": "sparse_medium_noise",
        "label": "Sparse data, medium noise",
        "NUM_SAMPLES": 55,
        "NOISE_LEVEL": 0.05,
        "NUM_EVAL_POINTS": 150,
        "NUM_MODES": 6,
        "GAMMA": 1e1,
        "GAMMA2": 1e1,
    },
    {
        "name": "dense_high_noise",
        "label": "Dense data, high noise",
        "NUM_SAMPLES": 250,
        "NOISE_LEVEL": 0.15,
        "NUM_EVAL_POINTS": 400,
        "NUM_MODES": 6,
        "GAMMA": 1e1,
        "GAMMA2": 1e1,
    },
]

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(SCRIPT_DIR, "02_full_bayesian.ipynb")
OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "results", "paper_runs", datetime.now().strftime("%Y%m%d_%H%M%S")
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_schema(schema, output_dir):
    """Execute the notebook with a single data-regime configuration."""
    tag = schema["name"]
    output_path = os.path.join(output_dir, f"{tag}.ipynb")

    # Parameters injected into the notebook's config cell via papermill
    params = {
        "NUM_SAMPLES": schema["NUM_SAMPLES"],
        "NOISE_LEVEL": schema["NOISE_LEVEL"],
        "NUM_EVAL_POINTS": schema["NUM_EVAL_POINTS"],
        "NUM_MODES": schema["NUM_MODES"],
        "GAMMA": schema["GAMMA"],
        "GAMMA2": schema["GAMMA2"],
        "RUN_SVI": True,
        "RUN_MCMC": False,
    }

    print(f"\n{'=' * 60}")
    print(f"  Schema: {schema['label']}")
    print(f"  NUM_SAMPLES={schema['NUM_SAMPLES']}, "
          f"NOISE_LEVEL={schema['NOISE_LEVEL']:.3f}, "
          f"NUM_EVAL_POINTS={schema['NUM_EVAL_POINTS']}")
    print(f"  GAMMA={schema['GAMMA']:.1e}, GAMMA2={schema['GAMMA2']:.1e}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}")

    result = {
        "schema": tag,
        "label": schema["label"],
        "num_samples": schema["NUM_SAMPLES"],
        "noise_level": schema["NOISE_LEVEL"],
        "gamma": schema["GAMMA"],
        "gamma2": schema["GAMMA2"],
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
    print(f"Generating paper results: {len(SCHEMAS)} data regimes")
    print(f"Output dir: {OUTPUT_DIR}\n")

    for s in SCHEMAS:
        print(f"  * {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  "
              f"noise={s['NOISE_LEVEL']:.3f}  "
              f"gamma={s['GAMMA']:.0e}  gamma2={s['GAMMA2']:.0e}")

    results = []
    for i, schema in enumerate(SCHEMAS, 1):
        print(f"\n[{i}/{len(SCHEMAS)}]", end="")
        res = run_schema(schema, OUTPUT_DIR)
        results.append(res)
        print(f"  -> {res['status']} ({res['elapsed_s']}s)")

    # ── Save summary ─────────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("schema,num_samples,noise_level,gamma,gamma2,status,elapsed_s,error\n")
        for r in results:
            err = (r["error"] or "").replace(",", ";").replace("\n", " ")
            f.write(
                f"{r['schema']},{r['num_samples']},{r['noise_level']},"
                f"{r['gamma']},{r['gamma2']},{r['status']},"
                f"{r['elapsed_s']},{err}\n"
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
    print(f"  {'Schema':>25s}  {'Samples':>7s}  {'Noise':>7s}  "
          f"{'Status':>8s}  {'Time':>8s}")
    print(f"  {'-' * 25}  {'-' * 7}  {'-' * 7}  {'-' * 8}  {'-' * 8}")
    for r in results:
        print(f"  {r['label']:>25s}  {r['num_samples']:>7d}  "
              f"{r['noise_level']:>7.3f}  {r['status']:>8s}  "
              f"{r['elapsed_s']:>7.1f}s")


if __name__ == "__main__":
    main()

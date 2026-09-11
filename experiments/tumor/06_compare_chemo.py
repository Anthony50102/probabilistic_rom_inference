"""Run matched nominal-dose fits and fixed-model dose generalization.

python 06_compare_chemo.py --method bayes
python 06_compare_chemo.py --method neural
python 06_compare_chemo.py --method report

Fits/checkpoints and results are isolated under chemo_protocol.OUTPUT_ROOT.
The default runs both methods at all three noise levels and then reports them.
"""

import argparse
import csv
import gc
import importlib.util
import json
from pathlib import Path
import sys

import jax
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from chemo_protocol import (
    OUTPUT_ROOT, INPUT_AWARE_OUTPUT_ROOT, SCHEMAS, prepare_data, save_protocol,
)
from chemo_evaluation import plot_comparison, write_json


def load_method(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_schema(schema, method, output_root=OUTPUT_ROOT, bayes_profile="historical"):
    directory = Path(output_root) / schema["name"]
    data = prepare_data(schema)
    save_protocol(data, directory)
    print(f"\nMatched observations: {data['fingerprint']}", flush=True)
    if method in ("bayes", "both"):
        module = load_method("matched_bayes_chemo", "04_unified_chemo.py")
        result = module.run_matched(
            schema, data=data, out_dir=directory, profile=bayes_profile)
        module.evaluate_dose_variation(result, out_dir=directory)
        del result
        jax.clear_caches()
        gc.collect()
    if method in ("neural", "both"):
        module = load_method("matched_neural_chemo", "05_neural_ode_chemo.py")
        result = module.run_experiment(
            schema, data=data, checkpoint_dir=str(directory / "neural_checkpoints"))
        nominal_path = directory / "neural_fit.npz"
        np.savez_compressed(
            nominal_path, rom_solves=result["rom_solves"], losses=result["losses"],
            t_pred=data["t_pred"], t_samp=data["t_samp"], snaps_comp=data["snaps_comp"],
            basis_entries=data["basis"].entries, basis_shift=data["basis"].shift_,
            model_id=result["model_id"], data_fingerprint=data["fingerprint"],
            n_kept=result["n_kept"], n_dropped=result["n_dropped"],
            runtime=result["runtime"])
        write_json(directory / "neural_fit.json", dict(
            data_fingerprint=data["fingerprint"], model_id=result["model_id"],
            schema=schema, config=module.MODEL_PARAMS,
            n_kept=result["n_kept"], n_dropped=result["n_dropped"]))
        module.evaluate_dose_variation(result, save_dir=str(directory))
        del result
        jax.clear_caches()
        gc.collect()
    del data
    gc.collect()


def report(schemas, output_root=OUTPUT_ROOT):
    all_rows = []
    for schema in schemas:
        directory = Path(output_root) / schema["name"]
        rows = plot_comparison(directory)
        fingerprints = {row["data_fingerprint"] for row in rows}
        if len(fingerprints) != 1:
            raise ValueError(f"Methods used different data: {directory}")
        for method in ("04_unified_chemo", "05_neural_ode_chemo"):
            if len({r["model_id"] for r in rows if r["method"] == method}) != 1:
                raise ValueError(f"Model changed between doses: {method}, {directory}")
        all_rows.extend(rows)
    root = Path(output_root)
    write_json(root / "comparison_all.json", all_rows)
    fields = sorted({key for row in all_rows for key in row})
    with (root / "comparison_all.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nNoise Method               Dose Stable  Field-fit Field-forecast ROM-CI-forecast")
    for row in all_rows:
        def percent(key):
            value = row.get(key)
            return "n/a" if value is None else f"{value:.2%}"
        label = "Bayesian OpInf" if row["method"].startswith("04") else "Neural ODE"
        print(f"{row['noise']:>4.0%}  {label:<20} {row['dose_scale']:>3.1f} "
              f"{row['n_stable']:>3}/{row['n_total']:<3} "
              f"{percent('field_error_fit'):>9} {percent('field_error_forecast'):>14} "
              f"{percent('reduced_coverage_forecast'):>15}")
    print(f"\nResults: {root / 'comparison_all.csv'}")
    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schemas", nargs="*")
    parser.add_argument("--method", choices=("bayes", "neural", "both", "report"),
                        default="both")
    parser.add_argument("--bayes-profile", choices=("historical", "input-aware"),
                        default="historical")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    names = args.schemas or [s["name"] for s in SCHEMAS]
    unknown = set(names) - {s["name"] for s in SCHEMAS}
    if unknown:
        parser.error(f"Unknown schemas: {sorted(unknown)}")
    schemas = [s for s in SCHEMAS if s["name"] in names]
    output_root = args.output_root or Path(
        OUTPUT_ROOT if args.bayes_profile == "historical" else INPUT_AWARE_OUTPUT_ROOT)
    if args.method != "report":
        for schema in schemas:
            run_schema(schema, args.method, output_root, args.bayes_profile)
    if args.method in ("both", "report"):
        report(schemas, output_root)


if __name__ == "__main__":
    main()

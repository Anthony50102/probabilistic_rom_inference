"""
compare_weak_form.py — head-to-head comparison of:
  (A) 04  — indicator-window integral constraint     (current dual constraint)
  (B) 04b — smooth-bump weak-form constraint         (WSINDy-style ψ_k)

Runs the same data regime through both models and prints a side-by-side
summary of stability, train/prediction error, CI coverage, and runtime.

Usage:
    python compare_weak_form.py                       # runs all three regimes
    python compare_weak_form.py dense_low_noise       # one regime
"""

import sys, os, importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util
def _load(modname, filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_A = _load("ci_indicator", "04_conditional_integral.py")
mod_B = _load("ci_weakform",  "04b_weak_form.py")


def _row(label, r):
    return (f"{label:<20s}  "
            f"stab={r['stability_pct']:5.1f}%  "
            f"train={r['train_error']:7.2%}  "
            f"pred={r['pred_error']:7.2%}  "
            f"CI_cov={r['ci_coverage']:6.1%}  "
            f"CI_w={r['ci_width']:7.4f}  "
            f"runtime={r['runtime']:5.0f}s")


def main(schema_names=None):
    schemas = mod_A.SCHEMAS
    if schema_names:
        schemas = [s for s in mod_A.SCHEMAS if s['name'] in schema_names]

    out = []
    for schema in schemas:
        print(f"\n\n############  {schema['label']}  ############")
        print(f"\n>>> Method A (04) — indicator-window integral")
        rA = mod_A.run_experiment(schema)
        mod_A.save_predictions(rA)

        print(f"\n>>> Method B (04b) — smooth-bump weak form  (p={mod_B.MODEL_PARAMS['BUMP_P']})")
        rB = mod_B.run_experiment(schema)
        mod_B.save_predictions(rB)

        out.append((schema, rA, rB))

    print("\n\n" + "=" * 100)
    print("HEAD-TO-HEAD SUMMARY")
    print("=" * 100)
    for schema, rA, rB in out:
        print(f"\n[{schema['label']}]   samples={schema['NUM_SAMPLES']}, noise={schema['NOISE_LEVEL']:.0%}")
        print(_row("  A) indicator", rA))
        print(_row("  B) weak-form", rB))


if __name__ == "__main__":
    schemas = sys.argv[1:] if len(sys.argv) > 1 else None
    main(schemas)

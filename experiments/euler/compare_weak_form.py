"""
compare_weak_form.py — head-to-head comparison of:
  (A) 04  — indicator-window integral constraint  (current dual constraint)
  (B) 04b — smooth-bump weak-form constraint      (WSINDy-style ψ_k)

Each method runs in its own Python subprocess so JIT caches, NumPyro state,
and JAX device buffers don't interfere across methods.  After both finish,
results are loaded from the .npz files saved by save_predictions() and a
side-by-side table is printed.

Usage:
    python compare_weak_form.py                       # runs all three regimes
    python compare_weak_form.py dense_low_noise       # one regime
    python compare_weak_form.py dense_high_noise dense_low_noise
"""

import os
import sys
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "comparison")

SCHEMAS = ["dense_low_noise", "sparse_low_noise", "dense_high_noise"]
METHODS = [
    ("A) indicator", "04_conditional_integral.py", "04_conditional_integral.npz"),
    ("B) weak-form", "04b_weak_form.py",            "04b_weak_form.npz"),
]


def _run(script, schema):
    cmd = [sys.executable, script, schema]
    print(f"\n>>> {' '.join(cmd)}")
    t0 = __import__("time").time()
    rc = subprocess.run(cmd, cwd=HERE).returncode
    if rc != 0:
        raise RuntimeError(f"{script} on {schema} failed (rc={rc})")
    print(f"<<< wallclock {__import__('time').time()-t0:.0f}s")


def _load(schema, npz_name):
    p = os.path.join(RESULTS, schema, npz_name)
    if not os.path.exists(p):
        return None
    return dict(np.load(p, allow_pickle=True))


def _fmt(r):
    if r is None:
        return "  (no result)"
    return (f"stab={float(r['stability_pct']):5.1f}%  "
            f"train={float(r['train_error']):7.2%}  "
            f"pred={float(r['pred_error']):7.2%}  "
            f"CI_cov={float(r['ci_coverage']):6.1%}  "
            f"CI_w={float(r['ci_width']):7.4f}  "
            f"runtime={float(r['runtime']):5.0f}s")


def main(schema_names=None):
    schemas = schema_names if schema_names else SCHEMAS
    bad = [s for s in schemas if s not in SCHEMAS]
    if bad:
        print(f"Unknown schema(s): {bad}\nAvailable: {SCHEMAS}")
        sys.exit(1)

    for schema in schemas:
        print(f"\n############  {schema}  ############")
        for label, script, _ in METHODS:
            _run(script, schema)

    print("\n\n" + "=" * 110)
    print("HEAD-TO-HEAD SUMMARY  (each method in its own subprocess)")
    print("=" * 110)
    for schema in schemas:
        print(f"\n[{schema}]")
        for label, _, npz in METHODS:
            r = _load(schema, npz)
            print(f"  {label:<14s}  {_fmt(r)}")


if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    main(args)

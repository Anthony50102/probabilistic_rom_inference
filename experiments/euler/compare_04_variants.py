"""
compare_04_variants.py — head-to-head comparison of the four 04-family methods:

  (A) 04   — indicator-window integral constraint           (baseline)
  (B) 04b  — smooth-bump weak-form constraint               (WSINDy-style ψ_k)
  (C) 04g  — marginalised-O Bayesian, indicator integral    (closed-form O)
  (D) 04u  — marginalised-O × weak-form                     (this work)

Each method runs in its own Python subprocess so JIT caches, NumPyro state, and
JAX device buffers don't interfere across methods.  After all finish, results
are loaded from the per-method .npz files and a side-by-side table is printed.

Usage:
    python compare_04_variants.py                       # all 3 regimes
    python compare_04_variants.py dense_low_noise       # one regime
    python compare_04_variants.py dense_high_noise dense_low_noise
"""

import os
import sys
import subprocess
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "comparison")

SCHEMAS = ["dense_low_noise", "sparse_low_noise", "dense_high_noise"]
METHODS = [
    ("A) indicator   ", "04_conditional_integral.py", "04_conditional_integral.npz"),
    ("B) weak-form   ", "04b_weak_form.py",            "04b_weak_form.npz"),
    ("C) marg-O      ", "04g_marginal_O.py",           "04g_marginal_O.npz"),
    ("D) marg-O+weak ", "04_unified.py",               "04_unified.npz"),
]


def _run(script, schema):
    cmd = [sys.executable, script, schema]
    print(f"\n>>> {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=HERE).returncode
    if rc != 0:
        raise RuntimeError(f"{script} on {schema} failed (rc={rc})")
    print(f"<<< wallclock {time.time()-t0:.0f}s")


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
        for _, script, _ in METHODS:
            _run(script, schema)

    print("\n\n" + "=" * 120)
    print("HEAD-TO-HEAD SUMMARY  (each method in its own subprocess)")
    print("=" * 120)
    for schema in schemas:
        print(f"\n[{schema}]")
        for label, _, npz in METHODS:
            r = _load(schema, npz)
            print(f"  {label}  {_fmt(r)}")


if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    main(args)

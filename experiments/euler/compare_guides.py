"""
compare_guides.py — sweep over SVI guide families on the same Euler model:
  - normal       : AutoNormal (mean-field, baseline)
  - lowrank      : AutoLowRankMultivariateNormal (paper's recommendation)
  - multivariate : AutoMultivariateNormal (full covariance)

Goal: see if a richer guide closes the CI-coverage gap (target 90%; AutoNormal
hits ~62% on dense_high_noise).

Each (guide, schema) pair runs in its own Python subprocess so JIT caches don't
interfere across runs.  Results are aggregated from .npz files saved by
04c_richer_guide.save_predictions().

Usage:
    python compare_guides.py                        # all guides × all regimes
    python compare_guides.py dense_high_noise       # all guides on one regime
    python compare_guides.py lowrank multivariate   # subset of guides, all regimes
"""

import os
import sys
import time
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "comparison")

GUIDES = ['normal', 'lowrank', 'multivariate']
SCHEMAS = ['dense_low_noise', 'sparse_low_noise', 'dense_high_noise']
SCRIPT = '04c_richer_guide.py'


def _run(guide, schema):
    cmd = [sys.executable, SCRIPT, guide, schema]
    print(f"\n>>> {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=HERE).returncode
    if rc != 0:
        raise RuntimeError(f"{SCRIPT} {guide} {schema} failed (rc={rc})")
    print(f"<<< wallclock {time.time()-t0:.0f}s")


def _load(schema, guide):
    p = os.path.join(RESULTS, schema, f"04c_richer_guide_{guide}.npz")
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


def main(args):
    guides = [a for a in args if a in GUIDES] or GUIDES
    schemas = [a for a in args if a in SCHEMAS] or SCHEMAS
    bad = [a for a in args if a not in GUIDES and a not in SCHEMAS]
    if bad:
        print(f"Unknown arg(s): {bad}\nGuides: {GUIDES}\nSchemas: {SCHEMAS}")
        sys.exit(1)

    for schema in schemas:
        print(f"\n############  {schema}  ############")
        for guide in guides:
            _run(guide, schema)

    print("\n\n" + "=" * 110)
    print("GUIDE COMPARISON SUMMARY  (each run in its own subprocess)")
    print("=" * 110)
    for schema in schemas:
        print(f"\n[{schema}]")
        for guide in guides:
            r = _load(schema, guide)
            print(f"  {guide:<13s}  {_fmt(r)}")


if __name__ == "__main__":
    main(sys.argv[1:])

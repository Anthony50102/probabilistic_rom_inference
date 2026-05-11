"""
sweep_init_prior.py — 2x2 sweep over (PRIOR, INIT) on a chosen regime.

Tests whether VI alone can recover good operator estimates without the MLE
warm start and without MLE-centred priors.

Cells:
  (informative, mle)     — current 04 baseline
  (informative, prior)   — same priors, but init at prior median (tests init only)
  (broad,       mle)     — paper-style broad priors, but warm-started
  (broad,       prior)   — "just run VI": broad priors AND prior-median init

Usage:
    python sweep_init_prior.py                       # all 4 cells, all 3 regimes
    python sweep_init_prior.py dense_low_noise       # all 4 cells, one regime
"""
import os, sys, time, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "comparison")
SCRIPT = "04d_no_warmstart.py"

CELLS = [
    ("informative", "mle"),
    ("informative", "prior"),
    ("broad",       "mle"),
    ("broad",       "prior"),
]
SCHEMAS = ["dense_low_noise", "sparse_low_noise", "dense_high_noise"]


def _run(prior, init, schema):
    cmd = [sys.executable, SCRIPT, prior, init, schema]
    print(f"\n>>> {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=HERE).returncode
    print(f"<<< wallclock {time.time()-t0:.0f}s  rc={rc}")
    return rc


def _load(schema, prior, init):
    p = os.path.join(RESULTS, schema, f"04d_{prior}_{init}.npz")
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def _fmt(r):
    if r is None:
        return "  (no result; failed)"
    return (f"stab={float(r['stability_pct']):5.1f}%  "
            f"train={float(r['train_error']):7.2%}  "
            f"pred={float(r['pred_error']):7.2%}  "
            f"CI_cov={float(r['ci_coverage']):6.1%}  "
            f"CI_w={float(r['ci_width']):7.4f}  "
            f"runtime={float(r['runtime']):5.0f}s")


def main(args):
    schemas = [a for a in args if a in SCHEMAS] or SCHEMAS
    for schema in schemas:
        print(f"\n############  {schema}  ############")
        for prior, init in CELLS:
            _run(prior, init, schema)

    print("\n\n" + "=" * 110)
    print("PRIOR × INIT SWEEP SUMMARY  (each cell in its own subprocess)")
    print("=" * 110)
    for schema in schemas:
        print(f"\n[{schema}]")
        for prior, init in CELLS:
            r = _load(schema, prior, init)
            print(f"  {prior:<12s} × {init:<5s}  {_fmt(r)}")


if __name__ == "__main__":
    main(sys.argv[1:])

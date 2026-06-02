#!/usr/bin/env python3
"""Re-render all plots from previously snapshotted plot_data/*.pkl files.

Walks experiments/<pde>/<figures_dir>/plot_data/*.pkl and invokes the matching
script with --replot. This lets you iterate on plot styling without re-running
inference.

Usage:
    python replot_all.py                     # all PDEs, default figures dirs
    python replot_all.py --tag paper_v2      # use figures_rerun_paper_v2/
    python replot_all.py --pde heat tumor    # subset of PDEs
"""
from __future__ import annotations
import argparse
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

PDE_SCRIPTS = {
    "04": "04_conditional_integral.py",
    "05": "05_neural_ode.py",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None,
                    help="Suffix tag: looks in figures_rerun_<tag>/ instead of figures/")
    ap.add_argument("--pde", nargs="+",
                    default=["heat", "tumor", "euler", "burgers_2d"])
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    figures_dirname = f"figures_rerun_{args.tag}" if args.tag else "figures"

    failures = []
    for pde in args.pde:
        fig_dir = os.path.join(ROOT, "experiments", pde, figures_dirname)
        pd_dir = os.path.join(fig_dir, "plot_data")
        if not os.path.isdir(pd_dir):
            print(f"[skip] {pde}: no plot_data dir at {pd_dir}")
            continue
        for pkl in sorted(glob.glob(os.path.join(pd_dir, "*.pkl"))):
            base = os.path.basename(pkl)
            prefix = base.split("_", 1)[0]   # "04" or "05"
            script = PDE_SCRIPTS.get(prefix)
            if not script:
                print(f"[skip] unknown prefix {prefix} for {pkl}")
                continue
            script_path = os.path.join(ROOT, "experiments", pde, script)
            print(f"[replot] {pde}/{base} -> {fig_dir}")
            r = subprocess.run(
                [args.python, script_path, "--replot", pkl, fig_dir],
                cwd=ROOT,
            )
            if r.returncode != 0:
                failures.append((pde, base, r.returncode))

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" ", f)
        sys.exit(1)
    print("\n✓ All replots complete")


if __name__ == "__main__":
    main()

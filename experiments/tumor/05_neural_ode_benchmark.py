"""
05_neural_ode_benchmark.py — Neural-ODE baseline on the three tumor benchmarks.

Trains the repository's Neural-ODE baseline (the 05_neural_ode*.py
architectures: a 20-member tanh-MLP ensemble with the historical loss
filter) on exactly the reduced observations used by 04_unified_benchmark.py,
then scores the ensemble centers and every member on each dose arm.

Chemo members train one at a time (about 6-8 min each, so roughly 2-3 h per
seed); members are independent, so a seed can be split across processes
with --only-members once its data exist. The untreated-growth ensemble
trains all 20 members together (about 10 min per seed). Every fit resumes
from its latest 1000-update checkpoint.

Usage:
    python 05_neural_ode_benchmark.py untreated-growth
    python 05_neural_ode_benchmark.py multi-dose-chemo --seeds 48
    python 05_neural_ode_benchmark.py multi-dose-chemo --seeds 48 --only-members 0 1 2 3 4
    python 05_neural_ode_benchmark.py single-dose-chemo --seeds 46 --members 4 --steps 200 \\
        --output-root /tmp/tumor-smoke                      # quick smoke run
"""
import argparse
import time

import benchmark_environment

benchmark_environment.apply()

import benchmark_data as bd  # noqa: E402
import benchmark_evaluation as be  # noqa: E402
import benchmark_models as bm  # noqa: E402


def parse(argv=None):
    parser = argparse.ArgumentParser(
        description="Neural-ODE baseline on the tumor benchmarks.", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    bd.add_common_arguments(parser, case_optional=True)
    parser.add_argument("--members", type=int, help=f"Ensemble size (default {bm.NODE_MEMBERS}).")
    parser.add_argument("--steps", type=int,
                        help=f"Optimizer updates (default {bm.CHEMO_NODE_STEPS} chemo, "
                             f"{bm.GROWTH_NODE_STEPS} growth; fewer only for smoke tests).")
    parser.add_argument("--only-members", type=int, nargs="+",
                        help="Chemo only: train just these member indices in this process.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Train only.")
    args = parser.parse_args(argv)
    if args.case is None and any(value is not None for value in (
            args.seeds, args.pod_rank, args.pod_source, args.pod_centering, args.only_members)):
        parser.error("--seeds, --only-members and --pod-* options need a single benchmark case.")
    return args


def _percent(score):
    value = score["physical_percent"]
    return "incomplete" if value is None else f"{value:.2f}"


def report(case, pod, seeds, root):
    grid = bd.grid(case)
    center = be.headline_node_center(case)
    rows = {}
    for seed in seeds:
        path = bd.seed_directory(case, seed, pod, root) / "evaluation" / "neural_ode.json"
        if path.exists():
            rows[seed] = bd.read_json(path)
    if not rows:
        return
    print(f"\n{case.name} [{bd.pod_tag(pod)}]: Neural-ODE {center.replace('_', ' ')} "
          f"{grid.headline} full-field error (%)")
    print(f"  {'arm':<10}" + "".join(f"{'seed ' + str(seed):>12}" for seed in rows) + f"{'kept':>12}")
    kept = "/".join(str(len(row["filter"]["kept_indices"] or [])) for row in rows.values())
    for arm, strength in bd.arms(case).items():
        cells = "".join(f"{_percent(row['arms'][arm]['centers'][center][grid.headline]):>12}"
                        for row in rows.values())
        print(f"  {bd.arm_label(strength):<10}{cells}{kept:>12}")


def main(argv=None):
    args = parse(argv)
    if not benchmark_environment.matches_reported():
        print(f"Note: thread settings differ from the reported environment ({benchmark_environment.current()}); "
              "results may differ in the last float digits.")
    for name in [args.case] if args.case else list(bd.CASES):
        case, pod, seeds = bd.resolve(args, name)
        print(f"\n=== {case.name} [{bd.pod_tag(pod)}], seeds {list(seeds)}")
        for seed in seeds:
            started = time.monotonic()
            bd.prepare(case, seed, pod, args.output_root)
            print(f"Neural-ODE ensemble, {case.name} seed {seed}")
            summary = bm.train_node(case, seed, pod, args.output_root, members=args.members, steps=args.steps,
                                    only_members=args.only_members)
            if summary is None:
                print("  other members are still missing; evaluation waits for the full ensemble")
                continue
            print(f"  {summary['completed_members']}/{summary['members']} members complete, "
                  f"kept by the loss filter: {summary['kept_indices']}")
            if not args.skip_evaluation:
                be.evaluate_node(case, seed, pod, args.output_root)
            print(f"  seed {seed} finished in {time.monotonic() - started:.0f} s")
        if not args.skip_evaluation:
            report(case, pod, seeds, args.output_root)
    print("\nDone.")


if __name__ == "__main__":
    main()

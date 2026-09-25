"""
04_unified_benchmark.py — production Bayesian OpInf on the three tumor benchmarks.

For every acquisition seed this prepares the shared data (the reduced
observations handed to both methods, the POD decoder and evaluation-only
reference fields for each dose arm), fits the unchanged production
weak-form Bayesian OpInf model, and scores its point forecast and 64
paired posterior draws on every dose arm.

Usage:
    python 04_unified_benchmark.py                          # all three benchmarks, reported seeds
    python 04_unified_benchmark.py multi-dose-chemo         # seeds 48 49 50
    python 04_unified_benchmark.py single-dose-chemo --seeds 46
    python 04_unified_benchmark.py untreated-growth --no-draws
    python 04_unified_benchmark.py multi-dose-chemo --seeds 48 --steps 500 \\
        --output-root /tmp/tumor-smoke                      # quick smoke run

Results go to results/benchmarks/<case>/<pod>/seed<N>/{data,production,evaluation};
then run 05_neural_ode_benchmark.py and 06_compare_benchmark.py.
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
        description="Production Bayesian OpInf on the tumor benchmarks.", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    bd.add_common_arguments(parser, case_optional=True)
    parser.add_argument("--steps", type=int,
                        help="SVI updates (default: the production 12000; fewer only for smoke tests).")
    parser.add_argument("--no-draws", action="store_true",
                        help="Score the point forecast only, not the 64 posterior draws.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Prepare and fit only.")
    args = parser.parse_args(argv)
    if args.case is None and any(value is not None for value in (
            args.seeds, args.pod_rank, args.pod_source, args.pod_centering)):
        parser.error("--seeds and --pod-* options need a single benchmark case.")
    return args


def _percent(score):
    value = score["physical_percent"]
    return "incomplete" if value is None else f"{value:.2f}"


def report(case, pod, seeds, root):
    grid = bd.grid(case)
    print(f"\n{case.name} [{bd.pod_tag(pod)}]: production {grid.headline} full-field error (%)")
    print(f"  {'arm':<10}" + "".join(f"{'seed ' + str(seed):>12}" for seed in seeds) + f"{'draws<=25%':>14}")
    evaluations = [bd.read_json(bd.seed_directory(case, seed, pod, root) / "evaluation" / "production.json")
                   for seed in seeds]
    for arm, strength in bd.arms(case).items():
        cells = "".join(f"{_percent(row['arms'][arm]['point']['scores'][grid.headline]):>12}"
                        for row in evaluations)
        draws = [row["arms"][arm].get("draws") for row in evaluations]
        within = ("/".join(f"{d[f'complete_field_le{be.FIELD_TOLERANCE:g}']}" for d in draws)
                  if all(draws) else "n/a")
        print(f"  {bd.arm_label(strength):<10}{cells}{within:>14}")


def main(argv=None):
    args = parse(argv)
    environment = benchmark_environment.current()
    if not benchmark_environment.matches_reported():
        print(f"Note: thread settings differ from the reported environment ({environment}); "
              "results may differ in the last float digits.")
    for name in [args.case] if args.case else list(bd.CASES):
        case, pod, seeds = bd.resolve(args, name)
        print(f"\n=== {case.name} [{bd.pod_tag(pod)}], seeds {list(seeds)}")
        for seed in seeds:
            started = time.monotonic()
            bd.prepare(case, seed, pod, args.output_root)
            print(f"Production fit, {case.name} seed {seed}")
            fit = bm.fit_production(case, seed, pod, args.output_root, steps=args.steps)
            print(f"  {fit['status']} after {fit['steps']} updates")
            if not args.skip_evaluation:
                be.evaluate_production(case, seed, pod, args.output_root, draws=not args.no_draws)
            print(f"  seed {seed} finished in {time.monotonic() - started:.0f} s")
        if not args.skip_evaluation:
            report(case, pod, seeds, args.output_root)
    print("\nDone.")


if __name__ == "__main__":
    main()

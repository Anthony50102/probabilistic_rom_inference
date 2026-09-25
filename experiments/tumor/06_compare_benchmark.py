"""
06_compare_benchmark.py — production Bayesian OpInf versus the Neural-ODE baseline on the tumor benchmarks.

Reads the evaluations written by 04_unified_benchmark.py and
05_neural_ode_benchmark.py and writes

    results/benchmarks/comparison_segmented.json, .csv   per-seed values and medians over seeds
    figures/benchmarks/segmented/<case>/burden_seed<N>.png   tumor-burden forecasts of both methods
    figures/benchmarks/segmented/multi-dose-chemo/dose_transfer_seed<N>.png
    figures/benchmarks/segmented/summary_errors.png          headline errors of every benchmark

for the reported segmented scans; --observation oracle_masked keeps the
earlier design's untagged names (comparison.json, figures/benchmarks/<case>/).
With --output-root, tables go to that root and figures to <root>/figures.

The headline score is the relative full-field (decoded plus POD-residual)
error over the main forecast window: days 60-90 for untreated growth and
days 70-110 for both chemotherapy benchmarks. Each baseline keeps its own
ensemble recipe: the chemo Neural ODE reports the loss-filtered median and
the untreated-growth baseline the strict all-member median (the other
center is reported alongside). Posterior bands are empirical draw bands, not
calibrated uncertainty.

Usage:
    python 06_compare_benchmark.py
    python 06_compare_benchmark.py multi-dose-chemo --figure-seed 49 \\
        --paper-figure ../../../GP-Bayes-Refactor/manuscript_v2/figures/selected/tumor_multidose_transfer.png
"""
import argparse
import csv
from pathlib import Path

import benchmark_environment

benchmark_environment.apply()

import numpy as np  # noqa: E402

import benchmark_data as bd  # noqa: E402
import benchmark_evaluation as be  # noqa: E402

CENTERS = ("filtered_median", "all20_median")
FIGURE_SEEDS = {"untreated-growth": 43, "single-dose-chemo": 46, "multi-dose-chemo": 49}
EXTRA_WINDOWS = {"untreated-growth": ("forecast_60_120",)}


def parse(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare the tumor benchmark methods.", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    bd.add_common_arguments(parser, case_optional=True)
    parser.add_argument("--figure-seed", type=int, help="Seed drawn in the per-case figures (default: middle seed).")
    parser.add_argument("--paper-figure", type=Path,
                        help="Also save the multi-dose transfer figure here (e.g. a manuscript figure path).")
    parser.add_argument("--no-figures", action="store_true", help="Write the tables only.")
    args = parser.parse_args(argv)
    if args.case is None and any(value is not None for value in (
            args.seeds, args.pod_rank, args.pod_source, args.pod_centering, args.figure_seed)):
        parser.error("--seeds, --figure-seed and --pod-* options need a single benchmark case.")
    if args.paper_figure and args.case not in (None, "multi-dose-chemo"):
        parser.error("--paper-figure draws the multi-dose-chemo transfer figure.")
    return args


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------
def _median(values):
    return float(np.median(values)) if values and all(v is not None for v in values) else None


def _per_seed(seeds, values):
    return {"per_seed": dict(zip(map(str, seeds), values)), "median": _median(list(values))}


def _load(case, pod, seeds, root):
    evaluations = {}
    for seed in seeds:
        folder = bd.seed_directory(case, seed, pod, root) / "evaluation"
        missing = [name for name in ("production", "neural_ode") if not (folder / f"{name}.json").exists()]
        if missing:
            raise FileNotFoundError(
                f"{case.name} seed {seed} has no {' or '.join(missing)} evaluation in {folder}; run "
                + " and ".join({"production": "04_unified_benchmark.py",
                                "neural_ode": "05_neural_ode_benchmark.py"}[name] for name in missing) + " first.")
        evaluations[seed] = {name: bd.read_json(folder / f"{name}.json") for name in ("production", "neural_ode")}
    return evaluations


def summarize(case, pod, seeds, root):
    grid, arms = bd.grid(case), bd.arms(case)
    evaluations = _load(case, pod, seeds, root)
    center = be.headline_node_center(case)
    table = {"case": case.name, "observation": case.observation, "pod_tag": bd.pod_tag(pod),
             "tag": bd.acquisition_tag(case, pod), "seeds": list(seeds), "headline_window": grid.headline,
             "node_headline_center": center, "arms": {}}
    rows = []
    for arm, strength in arms.items():
        entry = {"label": bd.arm_label(strength), "strength": strength, "windows": {}}
        for window in grid.windows:
            production = [evaluations[s]["production"]["arms"][arm]["point"]["scores"][window] for s in seeds]
            node = {name: [evaluations[s]["neural_ode"]["arms"][arm]["centers"][name][window] for s in seeds]
                    for name in CENTERS}
            entry["windows"][window] = {
                "production": _per_seed(seeds, [row["physical_percent"] for row in production]),
                "neural_ode": {name: _per_seed(seeds, [row["physical_percent"] for row in node[name]])
                               for name in CENTERS},
                "projection_floor": _per_seed(seeds, [row["projection_floor_percent"] for row in production]),
            }
            for method, name, scores in [("production", "point", production)] + [
                    ("neural_ode", name, node[name]) for name in CENTERS]:
                rows += [{"case": case.name, "observation": case.observation, "pod_tag": bd.pod_tag(pod),
                          "seed": seed, "arm": entry["label"],
                          "quantity": "field", "window": window, "method": method, "center": name,
                          "physical_percent": score["physical_percent"], "burden_percent": score["burden_percent"],
                          "reduced_percent": score["reduced_percent"],
                          "projection_floor_percent": score["projection_floor_percent"], "final_gain": None}
                         for seed, score in zip(seeds, scores)]
        headline = entry["windows"][grid.headline]
        entry["production_wins_over_headline_node"] = sum(
            p is not None and (n is None or p < n) for p, n in zip(
                headline["production"]["per_seed"].values(), headline["neural_ode"][center]["per_seed"].values()))
        draws = [evaluations[s]["production"]["arms"][arm].get("draws") for s in seeds]
        if all(draws):
            entry["production_draws"] = {str(s): {
                "complete": d["complete"], "members": d["members"],
                f"field_le{be.FIELD_TOLERANCE:g}": d[f"complete_field_le{be.FIELD_TOLERANCE:g}"],
                "burden_truth_in_band": d["burden_truth_in_complete_band"], "queries": d["queries"]}
                for s, d in zip(seeds, draws)}
        entry["neural_ode_members"] = {str(s): {
            "complete": evaluations[s]["neural_ode"]["arms"][arm]["members"]["complete"],
            "members": evaluations[s]["neural_ode"]["arms"][arm]["members"]["members"],
            "kept_by_loss_filter": len(evaluations[s]["neural_ode"]["filter"]["kept_indices"] or []),
            f"field_le{be.FIELD_TOLERANCE:g}":
                evaluations[s]["neural_ode"]["arms"][arm]["members"][f"complete_field_le{be.FIELD_TOLERANCE:g}"]}
            for s in seeds}
        effect = evaluations[seeds[0]]["production"]["arms"][arm].get("effect")
        if effect:
            production = [evaluations[s]["production"]["arms"][arm]["effect"]["point"] for s in seeds]
            node = {name: [evaluations[s]["neural_ode"]["arms"][arm]["effect"][name] for s in seeds]
                    for name in CENTERS}
            entry["effect"] = {
                "window": f"after day {bd.first_future_dose(case):g}",
                "production": {"field": _per_seed(seeds, [row["physical_percent"] for row in production]),
                               "final_gain": _per_seed(seeds, [row["final_burden_gain_predicted_over_true"]
                                                               for row in production])},
                "neural_ode": {name: {"field": _per_seed(seeds, [row["physical_percent"] for row in node[name]]),
                                      "final_gain": _per_seed(seeds, [row["final_burden_gain_predicted_over_true"]
                                                                      for row in node[name]])}
                               for name in CENTERS},
                "true_final_change_percent_of_control": _per_seed(
                    seeds, [row["true_final_change_percent_of_control"] for row in production]),
            }
            draws = [evaluations[s]["production"]["arms"][arm]["effect"].get("draws") for s in seeds]
            if all(draws):
                entry["effect"]["production_draws"] = {str(s): {
                    "complete": d["complete"], "members": d["members"],
                    "truth_in_band": d["truth_in_complete_band_queries"], "queries": d["effect_queries"]}
                    for s, d in zip(seeds, draws)}
            for method, name, scores in [("production", "point", production)] + [
                    ("neural_ode", name, node[name]) for name in CENTERS]:
                rows += [{"case": case.name, "observation": case.observation, "pod_tag": bd.pod_tag(pod),
                          "seed": seed, "arm": entry["label"],
                          "quantity": "effect", "window": entry["effect"]["window"], "method": method,
                          "center": name, "physical_percent": score["physical_percent"],
                          "burden_percent": score["burden_percent"], "reduced_percent": score["reduced_percent"],
                          "projection_floor_percent": score["projection_floor_percent"],
                          "final_gain": score["final_burden_gain_predicted_over_true"]}
                         for seed, score in zip(seeds, scores)]
        table["arms"][arm] = entry
    return table, rows


def _cell(block):
    def number(value):
        return "  n/a" if value is None else f"{value:5.2f}"
    return "/".join(number(v) for v in block["per_seed"].values()) + f" ({number(block['median']).strip()})"


def print_table(table):
    center = table["node_headline_center"]
    other = next(name for name in CENTERS if name != center)
    windows = (table["headline_window"],) + EXTRA_WINDOWS.get(table["case"], ())
    print(f"\n{table['case']} [{table['tag']}], seeds {table['seeds']}: full-field error (%), "
          "per seed (median)")
    for window in windows:
        print(f"  {window}")
        print(f"    {'arm':<10}{'production':<30}{'NODE ' + center.replace('_', ' '):<30}"
              f"{'NODE ' + other.replace('_', ' '):<30}wins")
        for entry in table["arms"].values():
            block = entry["windows"][window]
            wins = entry["production_wins_over_headline_node"] if window == table["headline_window"] else ""
            print(f"    {entry['label']:<10}{_cell(block['production']):<30}"
                  f"{_cell(block['neural_ode'][center]):<30}{_cell(block['neural_ode'][other]):<30}"
                  f"{wins}{'/' + str(len(table['seeds'])) if wins != '' else ''}")
    effects = [entry for entry in table["arms"].values() if "effect" in entry]
    if effects:
        print(f"  treatment effect versus the unchanged regimen, {effects[0]['effect']['window']}: "
              "field error (%) and day-110 burden gain (predicted/true)")
        for entry in effects:
            block = entry["effect"]
            print(f"    {entry['label']:<10}{_cell(block['production']['field']):<30}"
                  f"{_cell(block['neural_ode'][center]['field']):<30}"
                  f"gain {_cell(block['production']['final_gain'])} vs "
                  f"{_cell(block['neural_ode'][center]['final_gain'])}")


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def _plotting():
    from core.plotting.style import apply_style, method_color, method_label, save_figure
    import matplotlib.pyplot as plt
    apply_style()
    return plt, save_figure, {
        "production": (method_color("04_unified"), method_label("04_unified")),
        "neural_ode": (method_color("05_neural_ode"), method_label("05_neural_ode")), "truth": ("black", "Truth")}


def _seed_arrays(case, pod, seed, root):
    folder = bd.seed_directory(case, seed, pod, root)
    data = bd.load(case, seed, pod, root)
    return data, bd.read_npz(folder / "evaluation" / "production.npz"), \
        bd.read_npz(folder / "evaluation" / "neural_ode.npz")


def _number(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _burden(q, observations):
    return observations["burden_weights"] @ q + float(observations["burden_offset"])


def _decorate(ax, case, *, training=True):
    start, end = case.training_span
    if training:
        ax.axvspan(start, end, color="0.5", alpha=.08, lw=0)
        ax.axvline(end, color="0.5", ls="--", lw=.8)
    for day in case.dose_days:
        ax.axvline(day, color="0.6", ls=":", lw=.7)


def _burden_panel(ax, case, arm, data, production, node, colors, center, *, observed=True):
    times = production["times"]
    observations, reference = data["observations"], data["references"][arm]
    ax.plot(times, reference["truth_burden"], color=colors["truth"][0], lw=1.6, label=colors["truth"][1])
    if f"{arm}_burden_band" in production and f"{arm}_draws" in production:
        band = production[f"{arm}_burden_band"]
        ax.fill_between(times, band[0], band[2], color=colors["production"][0], alpha=.18, lw=0,
                        label=f"{colors['production'][1]} 5–95% draw band")
    ax.plot(times, _burden(production[f"{arm}_point"], observations), color=colors["production"][0], lw=1.4,
            label=colors["production"][1])
    ax.plot(times, _burden(node[f"{arm}_{center}"], observations), color=colors["neural_ode"][0], lw=1.4, ls="--",
            label=f"{colors['neural_ode'][1]} ({center.replace('_', ' ')})")
    if observed:
        ax.plot(observations["t_sampled"], _burden(observations["y"], observations), ".", color="0.35", ms=2.2,
                label="Observations")
    _decorate(ax, case)
    ax.set_xlim(times[0], times[-1])


def figure_burden(case, pod, seed, root, table, folder):
    plt, save_figure, colors = _plotting()
    data, production, node = _seed_arrays(case, pod, seed, root)
    center = table["node_headline_center"]
    arms = sorted(bd.arms(case).items(), key=lambda item: -1 if item[1] is None else item[1])
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 3.4), sharey=True, squeeze=False)
    window = table["headline_window"]
    for ax, (arm, strength) in zip(axes[0], arms):
        _burden_panel(ax, case, arm, data, production, node, colors, center)
        block = table["arms"][arm]["windows"][window]
        ax.set_title(f"{bd.arm_label(strength)}: {window.replace('forecast_', 'days ').replace('_', '-')} error "
                     f"{_number(block['production']['per_seed'][str(seed)])}% vs "
                     f"{_number(block['neural_ode'][center]['per_seed'][str(seed)])}%", fontsize=9)
        ax.set_xlabel("Day")
    axes[0, 0].set_ylabel("Tumor burden (integrated cellularity)")
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"{case.name}, seed {seed} ({bd.acquisition_tag(case, pod)})", fontsize=10)
    path = save_figure(fig, str(folder / case.name / f"burden_seed{seed}.png"))
    print(f"  figure: {path}")


def figure_transfer(case, pod, seed, root, table, folder, extra=None):
    """Multi-dose transfer: burden at each future strength, effects versus 0.5x, and errors over seeds."""
    plt, save_figure, colors = _plotting()
    data, production, node = _seed_arrays(case, pod, seed, root)
    observations = data["observations"]
    center = table["node_headline_center"]
    arms = sorted(bd.arms(case).items(), key=lambda item: item[1])
    control = next(iter(bd.arms(case)))
    changed = [(arm, strength) for arm, strength in arms if arm != control]
    times = production["times"]
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.4))
    for ax, (arm, strength) in zip(axes[0], arms):
        _burden_panel(ax, case, arm, data, production, node, colors, center, observed=False)
        suffix = " (unchanged)" if arm == control else ""
        ax.set_title(f"Future pulses {strength:g}×{suffix}", fontsize=9)
        ax.set_xlabel("Day")
    axes[0, 0].set_ylabel("Tumour burden")
    lower = min(ax.get_ylim()[0] for ax in axes[0])
    upper = max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(lower, upper)
    switch = bd.first_future_dose(case)
    mask = times >= case.training_span[1]
    for ax, (arm, strength) in zip(axes[1], changed):
        reference = data["references"][arm]
        ax.axhline(0., color="0.5", lw=.7)
        ax.plot(times[mask], reference["effect_burden"][mask], color=colors["truth"][0], lw=1.6,
                label=colors["truth"][1])
        if f"{arm}_effect_band" in production:
            band = production[f"{arm}_effect_band"]
            ax.fill_between(times[mask], band[0][mask], band[2][mask], color=colors["production"][0], alpha=.18,
                            lw=0, label=f"{colors['production'][1]} 5–95% paired draws")
        ax.plot(times[mask], observations["burden_weights"] @ (production[f"{arm}_point"]
                                                                - production[f"{control}_point"])[:, mask],
                color=colors["production"][0], lw=1.4, label=colors["production"][1])
        ax.plot(times[mask], observations["burden_weights"] @ (node[f"{arm}_{center}"]
                                                                - node[f"{control}_{center}"])[:, mask],
                color=colors["neural_ode"][0], lw=1.4, ls="--", label=colors["neural_ode"][1])
        for day in case.dose_days:
            if day >= switch:
                ax.axvline(day, color="0.6", ls=":", lw=.7)
        ax.set_xlim(times[mask][0], times[-1])
        ax.set_title(f"Effect of {strength:g}× versus {case.training_strength:g}×", fontsize=9)
        ax.set_xlabel("Day")
    axes[1, 0].set_ylabel("Change in tumour burden")
    ax = axes[1, 3]
    window = table["headline_window"]
    positions = np.arange(len(arms))
    for offset, (method, name) in zip((-.18, .18), (("production", None), ("neural_ode", center))):
        blocks = [table["arms"][arm]["windows"][window][method] for arm, _ in arms]
        blocks = [block[name] for block in blocks] if name else blocks
        ax.bar(positions + offset, [block["median"] for block in blocks], width=.34, color=colors[method][0],
               alpha=.85)
        for position, block in zip(positions, blocks):
            values = list(block["per_seed"].values())
            ax.plot(np.full(len(values), position + offset), values, "o", ms=3, mfc="white", mec="black", mew=.7)
    ax.set_xticks(positions, [f"{strength:g}×" for _, strength in arms])
    ax.set_xlabel("Future pulse strength")
    ax.set_ylabel(f"Field error, days {window.split('_')[1]}–{window.split('_')[2]} (%)")
    ax.set_title("Median (bars), acquisitions (circles)", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9, frameon=False,
               bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .95))
    paths = [folder / case.name / f"dose_transfer_seed{seed}.png"] + ([extra] if extra else [])
    for index, path in enumerate(paths):
        save_figure(fig, str(path), dpi=300, close=index == len(paths) - 1)
        print(f"  figure: {path}")


def figure_summary(tables, folder):
    plt, save_figure, colors = _plotting()
    groups = [(table, arm) for table in tables for arm in sorted(
        table["arms"],
        key=lambda key: -1 if table["arms"][key]["strength"] is None else table["arms"][key]["strength"])]
    fig, ax = plt.subplots(figsize=(1.0 + .75 * len(groups), 3.4))
    positions = np.arange(len(groups))
    for offset, method in zip((-.18, .18), ("production", "neural_ode")):
        medians, seeds = [], []
        for table, arm in groups:
            block = table["arms"][arm]["windows"][table["headline_window"]][method]
            block = block[table["node_headline_center"]] if method == "neural_ode" else block
            medians.append(block["median"])
            seeds.append(list(block["per_seed"].values()))
        ax.bar(positions + offset, medians, width=.34, color=colors[method][0], alpha=.85, label=colors[method][1])
        for position, values in zip(positions, seeds):
            ax.plot(np.full(len(values), position + offset), values, "o", ms=3, mfc="white", mec="black", mew=.7)
    names = {"untreated-growth": "growth", "single-dose-chemo": "single", "multi-dose-chemo": "multi"}
    ax.set_xticks(positions, [f"{names.get(table['case'], table['case'])}\n{table['arms'][arm]['label']}"
                              for table, arm in groups])
    ax.set_ylabel("Headline forecast error (%)")
    ax.set_title("Median over seeds (bars) and individual seeds (circles)", fontsize=9)
    ax.legend()
    path = save_figure(fig, str(folder / "summary_errors.png"))
    print(f"  figure: {path}")


# -----------------------------------------------------------------------------
def main(argv=None):
    args = parse(argv)
    root = Path(args.output_root or bd.RESULTS)
    figures = bd.FIGURES if args.output_root is None else root / "figures"
    observation = args.observation or bd.get_case(args.case or next(iter(bd.CASES))).observation
    tagged = observation != bd.LEGACY_OBSERVATION
    figures = figures / observation if tagged else figures
    tables, rows = [], []
    for name in [args.case] if args.case else list(bd.CASES):
        case, pod, seeds = bd.resolve(args, name)
        table, case_rows = summarize(case, pod, seeds, args.output_root)
        tables.append(table)
        rows += case_rows
        print_table(table)
        if not args.no_figures:
            if args.figure_seed is not None and args.figure_seed not in seeds:
                raise SystemExit(f"--figure-seed {args.figure_seed} is not among the seeds {list(seeds)}.")
            preferred = args.figure_seed if args.figure_seed is not None else FIGURE_SEEDS.get(case.name)
            seed = preferred if preferred in seeds else seeds[len(seeds) // 2]
            figure_burden(case, pod, seed, args.output_root, table, figures)
            if case.name == "multi-dose-chemo":
                figure_transfer(case, pod, seed, args.output_root, table, figures, args.paper_figure)
    suffix = ("" if args.case is None else f"_{args.case}") + (f"_{observation}" if tagged else "")
    bd.write_json(root / f"comparison{suffix}.json", {
        "tables": tables, "bands_are_not_calibrated_uncertainty": True,
        "headline_score": "Relative full-field error (decoded state plus POD residual) over the main forecast "
                          "window; Neural-ODE centers are coordinate-wise medians of the ensemble."})
    with (root / f"comparison{suffix}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nTables: {root / f'comparison{suffix}.json'} and .csv")
    if not args.no_figures and args.case is None:
        figure_summary(tables, figures)
    print("Done.")


if __name__ == "__main__":
    main()

"""Summarize every declared fit, preserving held-out results and failures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

VARIANTS = ("hierarchical", "fixed", "contract1", "contract5", "contract20")
CASES = (("heat", "sparse_low_noise"), ("heat", "sparse_medium_noise"),
         ("heat", "sparse_high_noise"), ("burgers_2d", "dense_medium_noise"))
COLORS = ("black", "#777777", "#2878b5", "#e39c24", "#c43a3a")


def mean_complete(values):
    return None if any(v is None for v in values) else float(np.mean(values))


def collect(root):
    records, hashes, sources = [], {}, set()
    for system, schema in CASES:
        for seed in (42, 43):
            for variant in VARIANTS:
                folder = root / f"{system}--{schema}--s{seed}--{variant}"
                request = json.loads((folder / "request.json").read_text())
                summary = json.loads((folder / "summary.json").read_text())
                cfg = request["config"]
                if (cfg["seed"] != seed or request["variant"] != variant
                        or request["system"] != system or request["schema"]["name"] != schema
                        or cfg["num_steps"] != 12000 or cfg["num_posterior_samples"] != 500
                        or cfg["precision"] != "float64" or request["long_draws"] != 50):
                    raise ValueError(f"Unexpected production request: {folder}")
                sources.add(request["source_commit"])
                key = (system, schema, seed)
                previous = hashes.setdefault(key, request["training_hash"])
                if previous != request["training_hash"]:
                    raise ValueError(f"Unmatched training data/basis: {folder}")
                if len(summary["targets"]) != (6 if system == "heat" else 1):
                    raise ValueError(f"Incomplete target set: {folder}")
                for held_out in ((False, True) if system == "heat" else (False,)):
                    targets = [row for row in summary["targets"] if row["held_out"] == held_out]
                    if any(len(row["short"]) != 200 or len(row["long"]) != 50 for row in targets):
                        raise ValueError(f"Incomplete draw evaluation: {folder}")
                    mean_errors = [row["mean"]["errors"] for row in targets]
                    median_errors = [row["median_errors"] for row in targets]
                    failures = {}
                    for row in targets:
                        for window in ("short", "long"):
                            for draw in row[window]:
                                if not draw["complete"]:
                                    reason = f"{window}: {draw['reason']}"
                                    failures[reason] = failures.get(reason, 0) + 1
                    records.append(dict(
                        system=system, schema=schema, seed=seed, variant=variant, held_out=held_out,
                        mean_error=mean_complete([None if e is None else e["full_relative"] for e in mean_errors]),
                        mean_fluct_error=mean_complete([None if e is None else e["fluct_relative"] for e in mean_errors]),
                        median_error=mean_complete([None if e is None else e["full_relative"] for e in median_errors]),
                        floor=float(np.mean([row["physical_basis_floor"]["full_relative"] for row in targets])),
                        survival=float(np.mean([row["survival"] for row in targets])),
                        long_survival=float(np.mean([row["long_survival"] for row in targets])),
                        coverage=mean_complete([row["coverage"] for row in targets]),
                        band_width=mean_complete([row["relative_band_width"] for row in targets]),
                        long_mean_complete=all(row["long_mean"]["complete"] for row in targets),
                        fit_seconds=summary["fit_seconds"], final_loss_sd=summary["final_loss_sd"],
                        loss_block_change=summary["loss_block_change"], failures=failures))
    if len(sources) != 1:
        raise ValueError(f"Mixed production source revisions: {sources}")
    return records, sources.pop()


def summarize(records):
    groups = []
    for system, schema in CASES:
        for held_out in ((False, True) if system == "heat" else (False,)):
            label = (f"Heat {dict(sparse_low_noise=1, sparse_medium_noise=3, sparse_high_noise=5)[schema]}% "
                     + ("held-out" if held_out else "training cases")) if system == "heat" else "Diffusion-reaction 3%"
            rows = []
            for variant in VARIANTS:
                items = [r for r in records if (r["system"], r["schema"], r["held_out"], r["variant"])
                         == (system, schema, held_out, variant)]
                if len(items) != 2:
                    raise ValueError("Both seeds are required")
                metrics = {key: mean_complete([r[key] for r in items]) for key in
                           ("mean_error", "mean_fluct_error", "median_error", "floor", "survival",
                            "long_survival", "coverage", "band_width")}
                rows.append(dict(variant=variant, **metrics))
            groups.append(dict(system=system, schema=schema, held_out=held_out, label=label, variants=rows))
    return groups


def overview(groups, output):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    centers = np.arange(len(groups))
    width = .15
    for k, variant in enumerate(VARIANTS):
        y = centers + (k-2)*width
        rows = [g["variants"][k] for g in groups]
        for ax, metric in zip(axes, ("mean_error", "survival", "long_survival")):
            values = [np.nan if r[metric] is None else 100*r[metric] for r in rows]
            ax.barh(y, values, height=width, color=COLORS[k], label=variant)
    axes[0].set_yticks(centers, [g["label"] for g in groups])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Full-state-relative mean-operator error (%)")
    axes[0].set_xscale("symlog", linthresh=5)
    axes[1].set_xlabel("Short-window completed draws (%)")
    axes[2].set_xlabel("Long-window completed draws (%)")
    for ax in axes[1:]:
        ax.set_xlim(0, 105)
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Stability-prior transfer: all declared variants, means across two seeds", fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(output / "01_accuracy_and_survival.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    for k, variant in enumerate(VARIANTS):
        for ax, metric, scale in zip(axes, ("coverage", "band_width"), (100, 1)):
            values = [np.nan if g["variants"][k][metric] is None else scale*g["variants"][k][metric] for g in groups]
            ax.barh(centers+(k-2)*width, values, height=width, color=COLORS[k], label=variant)
    axes[0].set_yticks(centers, [g["label"] for g in groups])
    axes[0].invert_yaxis()
    axes[0].axvline(90, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Survivor-only 90% band coverage, reduced states (%)")
    axes[1].set_xlabel("Band width / reduced-truth norm")
    axes[1].set_xscale("symlog", linthresh=1)
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Uncertainty: coverage and width must be read with draw survival", fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(output / "02_uncertainty.png", dpi=160)
    plt.close(fig)


def trajectories(root, groups, output):
    for number, group in enumerate(groups, 3):
        target = 5 if group["held_out"] else 0
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for k, variant in enumerate(VARIANTS):
            folder = root / f"{group['system']}--{group['schema']}--s42--{variant}"
            with np.load(folder / "plotdata.npz") as data:
                t = data[f"t_{target}"]
                for mode, ax in enumerate(axes):
                    if k == 0:
                        ax.plot(t, data[f"truth_{target}"][mode], color="#16843d", linewidth=2, label="truth")
                        ts, obs = data[f"t_sampled_{target}"], data[f"observations_{target}"][mode]
                        if group["held_out"]:
                            ts, obs = ts[:1], obs[:1]
                        ax.plot(ts, obs, "k*", markersize=4, label="observations used")
                        ax.axvline(1, color="gray", linestyle=":", linewidth=1)
                    if f"mean_{target}" in data:
                        ax.plot(t, data[f"mean_{target}"][mode], color=COLORS[k],
                                linestyle="--" if k == 0 else "-", label=variant)
                    if variant in ("hierarchical", "contract20") and f"lo_{target}" in data:
                        ax.fill_between(t, data[f"lo_{target}"][mode], data[f"hi_{target}"][mode],
                                        color=COLORS[k], alpha=.10)
                    ax.set(xlabel="time", ylabel=f"POD coefficient {mode+1}")
        fig.suptitle(group["label"] + ": seed 42, mean operators; bands only for hierarchical / contract20")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8)
        fig.tight_layout(rect=(0, .13, 1, .92))
        fig.savefig(output / f"{number:02d}_{group['system']}_{group['schema']}"
                    f"_{'held_out' if group['held_out'] else 'train'}.png", dpi=160)
        plt.close(fig)


def main(args):
    records, source = collect(args.results)
    groups = summarize(records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(
        dict(source_commit=source, records=records, groups=groups), indent=2, allow_nan=False) + "\n")
    with (args.output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows([{**r, "failures": json.dumps(r["failures"])} for r in records])
    lines = ["All 40 fits; values averaged over seeds 42/43. Heat training rows also average over five targets.",
             "Errors use full-state normalization. Bands are survivor-only; no failed targets are dropped.",
             "variant          mean%  median%   short%    long%  cover%    width"]
    for group in groups:
        lines.append("\n" + group["label"])
        for row in group["variants"]:
            values = [(None if row[key] is None else factor * row[key]) for key, factor in
                      (("mean_error", 100), ("median_error", 100), ("survival", 100),
                       ("long_survival", 100), ("coverage", 100), ("band_width", 1))]
            lines.append(f"{row['variant']:15s} " + " ".join(
                f"{value:8.3f}" if value is not None else "  FAILED" for value in values))
    text = "\n".join(lines) + "\n"
    (args.output / "report.txt").write_text(text)
    print(text)
    overview(groups, args.output)
    trajectories(args.results, groups, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())

"""Report every shared-prior fit, including failed fits and draw censoring."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def mean_if_complete(values):
    return None if not values or any(v is None for v in values) else float(np.mean(values))


def collect(results):
    records, failures = [], []
    for path in sorted(results.rglob("job.json")):
        job = json.loads(path.read_text())
        if job.get("returncode") != 0:
            failures.append(dict(path=str(path.parent), job=job))
            continue
        request = json.loads((path.parent / "request.json").read_text())
        summary = json.loads((path.parent / "summary.json").read_text())
        case = (request["system"], request["schema"]["name"], request["config"]["num_modes"],
                request["config"]["seed"], tuple(request["schema"].get("window", [])))
        for held in (False, True):
            targets = [t for t in summary["targets"] if t["held_out"] == held]
            if not targets:
                continue
            row = dict(system=case[0], schema=case[1], rank=case[2], seed=case[3], window=list(case[4]),
                       candidate=request["candidate"], held_out=held, path=str(path.parent),
                       training_hash=request["training_hash"], targets=len(targets),
                       survival=float(np.mean([t["survival"] for t in targets])),
                       min_survival=float(min(t["survival"] for t in targets)),
                       long_survival=mean_if_complete([t.get("long_survival") for t in targets]),
                       long_mean_complete=all(t.get("long_mean", {}).get("complete", False) for t in targets),
                       coverage=mean_if_complete([t["coverage"] for t in targets]),
                       width=mean_if_complete([t["relative_band_width"] for t in targets]),
                       fit_seconds=summary["fit_seconds"],
                       loss_drift_sd=(summary["loss_block_change"] / summary["final_loss_sd"]
                                      if summary["final_loss_sd"] else None),
                       gp_rate_shift_max=float(np.max(summary["gp_rate_relative_shift"])),
                       block_multiplier_median=summary.get("block_multiplier_median"))
            for name in ("full_relative", "fluct_relative", "reduced_relative"):
                row["mean_" + name] = mean_if_complete(
                    [None if t["mean"]["errors"] is None else t["mean"]["errors"][name] for t in targets])
                row["median_" + name] = mean_if_complete(
                    [None if t["median_errors"] is None else t["median_errors"][name] for t in targets])
                row["floor_" + name] = float(np.mean([t["projected_reconstruction"][name] for t in targets]))
            row["max_mean_fluct_relative"] = (
                None if any(t["mean"]["errors"] is None for t in targets)
                else max(t["mean"]["errors"]["fluct_relative"] for t in targets))
            row["short_failures"] = {}
            for target in targets:
                for solve in target["short"]:
                    if not solve["complete"]:
                        reason = solve["reason"]
                        row["short_failures"][reason] = row["short_failures"].get(reason, 0) + 1
            records.append(row)
    bases = {case_key(r): r for r in records if r["candidate"] == "original"}
    for row in records:
        base = bases.get(case_key(row))
        if base is None:
            continue
        if base["training_hash"] != row["training_hash"]:
            raise ValueError(f"Unmatched data/basis for {row['path']}")
        for name in ("mean_full_relative", "mean_fluct_relative", "median_reduced_relative"):
            numerator, denominator = row[name], base[name]
            row[name + "_ratio"] = (
                None if numerator is None or denominator is None or denominator <= 0 else numerator / denominator)
    return records, failures


def case_key(row):
    return row["system"], row["schema"], row["rank"], row["seed"], tuple(row["window"]), row["held_out"]


def plots(records, output):
    systems = ["euler", "heat", "burgers_2d", "cylinder"]
    names = sorted({r["candidate"] for r in records})
    cases = sorted({case_key(r) for r in records if not r["held_out"]},
                   key=lambda key: (systems.index(key[0]), key[1:]))
    errors = np.full((len(names), len(cases)), np.nan)
    survival = np.full_like(errors, np.nan)
    for i, name in enumerate(names):
        for j, case in enumerate(cases):
            rows = [r for r in records if r["candidate"] == name and case_key(r) == case]
            if len(rows) != 1:
                if rows:
                    raise ValueError(f"Duplicate candidate/case: {name}, {case}")
                continue
            row = rows[0]
            metric = "mean_fluct_relative" if case[0] == "cylinder" else "mean_full_relative"
            if row[metric] is not None:
                errors[i, j] = 100 * row[metric]
            survival[i, j] = 100 * row["min_survival"]
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 2.4*len(cases)), max(6, .32*len(names))), sharey=True)
    from matplotlib.colors import LogNorm
    image = axes[0].imshow(errors, aspect="auto", norm=LogNorm(vmin=.1, vmax=100), cmap="viridis_r")
    plt.colorbar(image, ax=axes[0], label="Mean-operator forecast error (%)")
    image = axes[1].imshow(survival, aspect="auto", vmin=0, vmax=100, cmap="RdYlGn")
    plt.colorbar(image, ax=axes[1], label="Minimum short-draw survival (%)")
    labels = [case[0].replace("burgers_2d", "Diffusion-reaction") for case in cases]
    if len(cases) > 4:
        labels = [f"{label}\n{case[1]}\nr{case[2]} s{case[3]} {case[4]}"
                  for label, case in zip(labels, cases)]
    for ax, values in zip(axes, (errors, survival)):
        ax.set_xticks(range(len(cases)), labels, rotation=20)
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(cases)):
                val = values[i, j]
                text = "fail/missing" if not np.isfinite(val) else (f"{val:.2g}" if val >= 1000 else f"{val:.1f}")
                ax.text(j, i, text,
                        ha="center", va="center", fontsize=8,
                        bbox=dict(facecolor="white", alpha=.7, edgecolor="none", pad=.5))
    axes[0].set_title("Full-state error; cylinder uses fluctuations")
    axes[1].set_title("Different predeclared amplitude caps")
    fig.tight_layout()
    fig.savefig(output / "all_candidates.png", dpi=160)
    plt.close(fig)


def trajectory_plot(records, output, names):
    systems = ["euler", "heat", "burgers_2d", "cylinder"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, system in zip(axes.flat, systems):
        rows = [r for r in records if r["system"] == system and r["candidate"] in names and not r["held_out"]]
        if len({case_key(r) for r in rows}) > 1:
            raise ValueError("Trajectory comparison needs one case per system")
        for j, name in enumerate(names):
            matches = [r for r in rows if r["candidate"] == name]
            if not matches:
                continue
            row = matches[0]
            with np.load(Path(row["path"]) / "plotdata.npz") as data:
                t, truth = data["t_0"], data["truth_0"]
                mode = int(np.argmax(np.std(truth, axis=1)))
                if j == 0:
                    ax.plot(t, truth[mode], "k-", lw=1.5, label="Truth")
                    ax.plot(data["t_sampled_0"], data["observations_0"][mode], "k*", ms=4, label="Observations")
                    ax.axvline(data["t_sampled_0"][-1], color="k", ls=":", lw=1)
                if "mean_0" in data:
                    ax.plot(t, data["mean_0"][mode], label=name)
                else:
                    ax.plot([], [], label=name + " (mean failed)")
                if "lo_0" in data:
                    ax.fill_between(t, data["lo_0"][mode], data["hi_0"][mode], alpha=.12)
        ax.set(title=system.replace("burgers_2d", "Diffusion-reaction"), xlabel="Time", ylabel="Leading fluctuation mode")
        ax.legend(fontsize=8)
    fig.suptitle("Training and forecast; bands use surviving draws only")
    fig.tight_layout()
    fig.savefig(output / "trajectory_comparison.png", dpi=170)
    plt.close(fig)


def main(args):
    records, failures = collect(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    public_records = [dict(row, path=str(Path(row["path"]).relative_to(args.results))) for row in records]
    public_failures = [dict(row, path=str(Path(row["path"]).relative_to(args.results))) for row in failures]
    report = dict(records=public_records, failed_or_incomplete_jobs=public_failures,
                  artifact_paths="relative to the collected results root")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    keys = sorted(set().union(*(r.keys() for r in public_records)))
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
                         for row in public_records)
    lines = [f"Successful fit-groups: {len(records)}; failed/incomplete jobs: {len(failures)}",
             "Errors are percent; cylinder fluctuation-relative, others full-state; survival is minimum per target."]
    for row in records:
        metric = "mean_fluct_relative" if row["system"] == "cylinder" else "mean_full_relative"
        error = "FAIL" if row[metric] is None else f"{100*row[metric]:.3f}"
        lines.append(f"{row['system']:12} {row['candidate']:29} held={row['held_out']} "
                     f"error={error:>9} survival={100*row['min_survival']:6.1f} "
                     f"long={row['long_survival']} GP-rate-shift={row['gp_rate_shift_max']:.3g}")
    (args.output / "report.txt").write_text("\n".join(lines) + "\n")
    plots(records, args.output)
    if args.compare:
        trajectory_plot(records, args.output, args.compare.split(","))
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", default="")
    main(parser.parse_args())

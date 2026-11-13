"""Reproduce selected cylinder figures from the completed initial prior grid."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CANDIDATES = ("original", "rate-centered-h0.5", "dimensional-clock-d1", "full-support-0.0001")
LABELS = ("Original", "Rate-scaled, centered", "Dimensional + rate-clock damping", "Full feature support, ridge 0.0001")


def main(results, output):
    output.mkdir(parents=True, exist_ok=True)
    summaries, folders, rows = {}, {}, []
    for path in sorted(results.glob("*/request.json")):
        request = json.loads(path.read_text())
        if request["system"] != "cylinder":
            continue
        # The stored window is (start, training duration, prediction duration).
        if (request["config"]["num_modes"], request["config"]["seed"],
                request["schema"]["window"]) != (25, 42, [16.0, 1.0, 3.0]):
            raise ValueError("These figures require the initial r25, seed42, [16,17] cylinder grid")
        name = request["candidate"]
        if name in summaries:
            raise ValueError(f"Duplicate cylinder candidate: {name}")
        summary = json.loads((path.parent / "summary.json").read_text())
        summaries[name], folders[name] = summary, path.parent
        for target in summary["targets"]:
            errors = target["mean"]["errors"]
            rows.append(dict(
                candidate=name, target=target["label"], held_out=target["held_out"],
                mean_completed=target["mean"]["complete"],
                mean_fluctuation_error=None if errors is None else errors["fluct_relative"],
                reconstruction_floor=target["projected_reconstruction"]["fluct_relative"],
                short_survivors=sum(s["complete"] for s in target["short"]),
                short_draws=len(target["short"]),
                long_survivors=sum(s["complete"] for s in target["long"]),
                long_draws=len(target["long"]),
            ))
    if not rows:
        raise ValueError("No cylinder results found")
    with (output / "cylinder_targets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    positions = np.arange(len(CANDIDATES))
    for index, (label, color) in enumerate((("Re 150 (train)", "tab:blue"), ("Re 300 (train)", "tab:orange"))):
        selected = [next(t for t in summaries[name]["targets"] if t["label"] == label) for name in CANDIDATES]
        errors = [np.nan if t["mean"]["errors"] is None else 100*t["mean"]["errors"]["fluct_relative"]
                  for t in selected]
        survival = [100*t["survival"] for t in selected]
        for ax, values in zip(axes, (errors, survival)):
            y = positions + (index-.5)*.33
            ax.barh(y, values, height=.30, color=color, label=label.replace(" (train)", ""))
            for yi, value in zip(y, values):
                text = "mean failed" if not np.isfinite(value) else f"{value:.2f}%"
                ax.text(1 if not np.isfinite(value) else value+1, yi, text,
                        va="center", fontsize=9)
    axes[0].set(yticks=positions, yticklabels=LABELS, xlim=(0, 100),
                xlabel="Mean-operator fluctuation-relative forecast error")
    axes[0].invert_yaxis()
    axes[0].axvline(5, color=".5", ls=":", label="5% accuracy target")
    axes[1].set(xlim=(0, 112), xlabel="Posterior draws completing the short forecast")
    axes[1].axvline(90, color=".5", ls=":", label="90% survival target")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Cylinder: improving sampled survival sacrifices mean accuracy")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"cylinder_tradeoff.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    names = CANDIDATES[:3]
    target_labels = ("Re 150 (train)", "Re 300 (train)", "Re 200 (held out)")
    for row_index, label in enumerate(target_labels):
        counts = []
        for candidate_index, name in enumerate(names):
            targets = summaries[name]["targets"]
            target_index = next(i for i, t in enumerate(targets) if t["label"] == label)
            target = targets[target_index]
            counts.append((sum(s["complete"] for s in target["short"]), len(target["short"])))
            with np.load(folders[name] / "plotdata.npz") as data:
                time = data[f"t_{target_index}"]
                observed = data[f"observations_{target_index}"]
                observed_time = data[f"t_sampled_{target_index}"]
                modes = np.argsort(np.std(observed, axis=1))[-2:][::-1]
                for column, mode in enumerate(modes):
                    ax = axes[row_index, column]
                    offset = np.mean(observed[mode])
                    if candidate_index == 0:
                        ax.plot(time, data[f"truth_{target_index}"][mode]-offset, "k-", lw=1.5, label="Truth")
                        ax.plot(observed_time, observed[mode]-offset, "k*", ms=3,
                                label="Training observations" if not target["held_out"] else "Held-out observations")
                        ax.axvline(1, color=".5", ls=":", lw=1)
                        ax.set(title=f"{label}, mode {mode+1}", ylabel="Modal coefficient - observation mean",
                               xlabel="Time since t = 16")
                    key = f"mean_{target_index}"
                    if key in data:
                        ax.plot(time, data[key][mode]-offset, lw=1.2, label=LABELS[candidate_index])
                    else:
                        ax.plot([], [], label=LABELS[candidate_index] + " (mean failed)")
        if len(set(counts)) == 1:
            annotation = f"All three priors: {counts[0][0]}/{counts[0][1]} sampled forecasts survive"
        else:
            annotation = "Surviving draws: " + "; ".join(
                f"{name} {count}/{total}" for name, (count, total) in zip(("Original", "Rate", "Damped"), counts))
        for ax in axes[row_index]:
            ax.text(.02, .04, annotation,
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels[1] = "Observations (Re200 held out)"
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, fontsize=8)
    fig.suptitle("Good training-Re means do not imply usable posterior trajectories\n"
                 "Two modes with greatest observed variation; no surviving-draw uncertainty bands")
    fig.tight_layout(rect=(0, 0, 1, .89))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"cylinder_mean_trajectories.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.results, args.output)

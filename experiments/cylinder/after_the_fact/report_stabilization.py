"""Reproduce the stabilization-study tables and figures from small JSON/plot exports."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORDER = ["broad-control", "fixed01", "fixed03", "fixed1", "narrow03", "rank29",
         "noise43", "noise44", "longer-broad", "longer-fixed03", "longer-fixed09",
         "warm42", "warm43", "modular42", "modular43"]


def main(root, output):
    output.mkdir(parents=True, exist_ok=True)
    screens = []
    for folder in ("screen", "jacglobal"):
        path = root / folder / "summary.json"
        if path.exists():
            screens.extend(json.loads(path.read_text()))
    fits = {}
    for path in (root / "fits").glob("*/summary.json"):
        if path.parent.name in ORDER:
            data = json.loads(path.read_text())
            if len(data["targets"]) == 3:
                fits[path.parent.name] = data
    names = [name for name in ORDER if name in fits]
    with (output / "joint_results.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "Re", "floor_pct", "mean_RB_error_pct", "sample_mean_error_pct",
                         "median_error_pct", "bounded_pct", "coverage_survivors_pct", "band_width",
                         "long_bounded_pct", "gp_only"])
        for name in names:
            for row in fits[name]["targets"]:
                writer.writerow([name, row["re"], 100 * row["floor"],
                    100 * row["means"]["rao_blackwell"].get("error", np.inf),
                    100 * row["means"]["sample_mean"].get("error", np.inf),
                    100 * row.get("median_error", np.inf), 100 * row["survival"],
                    100 * row.get("coverage_survivors", np.nan), row.get("width_over_fluct", np.nan),
                    100 * row["long_survival"], fits[name].get("gp_only", False)])
    with (output / "conditional_results.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "shift", "prior", "multiplier", "power", "damping", "jac_sd", "Re",
                         "pm_error_pct", "bounded_pct", "median_error_pct", "coverage_survivors_pct"])
        for case in screens:
            for row in case["targets"]:
                writer.writerow([case["id"], case["shift"], case["prior"], case["multiplier"], case["power"],
                    case.get("damping", ""), case.get("jac_sd", ""), row["re"],
                    100 * row.get("pm_error", np.inf), 100 * row["survival"],
                    100 * row.get("median_error", np.inf), 100 * row.get("coverage_survivors", np.nan)])
    (output / "all_results.json").write_text(json.dumps(dict(conditional=screens, joint=fits), indent=2))

    diagnostic_path = root / "diagnostics" / "diag_r25_free.json"
    if diagnostic_path.exists():
        d = json.loads(diagnostic_path.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for key, label in (("truth_design", "Noise-free training states"),
                           ("noisy_design", "Noisy training states")):
            s = np.array(d[key]["singular_values"])
            axes[0].semilogy(np.arange(1, len(s) + 1), s / s[0], label=label)
        axes[0].axhline(1e-3, color="gray", linestyle=":")
        axes[0].set(xlabel="Feature singular-value index", ylabel="Relative singular value",
                    title="377 coefficients per equation; only 28 strong directions")
        axes[0].legend(fontsize=8)
        for k, row in enumerate(d["targets"][:2]):
            vals = list(row["finite_sample_mean_check"].values())
            errors = [100 * v.get("reduced_fluct_error", np.nan) for v in vals]
            axes[1].plot(np.arange(len(vals)), errors, "o-", label=f"Re {row['re']}")
            for j, v in enumerate(vals):
                if not v["bounded"]:
                    axes[1].plot(j, 110, "x", color=f"C{k}")
        axes[1].set_xticks(range(6), ["All 500", "100 #1", "100 #2", "100 #3", "100 #4", "100 #5"])
        axes[1].tick_params(axis="x", labelrotation=25)
        axes[1].set(ylabel="Reduced fluctuation forecast error (%)",
                    title="Averaging noisy operator draws is unreliable (x = failure)")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output / "01_diagnosis.png", dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for k, re in enumerate((150, 300, 200)):
        for prior in ("global", "row", "balanced"):
            subset = [c for c in screens if not c["shift"] and c["prior"] == prior
                      and c["power"] == 1 and "damping" not in c]
            subset.sort(key=lambda c: c["multiplier"])
            x = [c["multiplier"] for c in subset]
            error = [100 * c["targets"][k].get("pm_error", np.nan) for c in subset]
            axes[0, k].plot(x, error, "o-", label=prior)
            axes[1, k].plot(x, [100 * c["targets"][k]["survival"] for c in subset], "o-", label=prior)
        axes[0, k].set(yscale="log", title=f"Re {re}" + (" (held out)" if re == 200 else " (training)"))
        axes[1, k].set(xscale="log", xlabel="Prior SD multiplier", ylim=(-3, 103))
        axes[0, k].grid(alpha=.2)
        axes[1, k].grid(alpha=.2)
    axes[0, 0].set_ylabel("Analytic mean error (%)")
    axes[1, 0].set_ylabel("Bounded draws (%)")
    axes[0, 0].legend()
    fig.suptitle("Conditional screen: GP hyperparameters frozen; 32 draws per setting")
    fig.tight_layout()
    fig.savefig(output / "02_conditional_prior_screen.png", dpi=170)
    plt.close(fig)

    if not names:
        return
    for metric, filename, xlabel in (
            ("error", "03_joint_errors.png", "Full-order fluctuation forecast error (%)"),
            ("survival", "04_joint_survival.png", "Bounded posterior draws (%)")):
        fig, axes = plt.subplots(1, 3, figsize=(15, max(4, .45 * len(names))), sharey=True)
        for k, ax in enumerate(axes):
            for j, name in enumerate(names):
                row = fits[name]["targets"][k]
                if metric == "error":
                    val = 100 * row["means"]["rao_blackwell"].get("error", np.nan)
                    median = 100 * row.get("median_error", np.nan)
                    ax.plot(100 * row["floor"], j, "k|", markersize=12, label="Basis floor" if j == 0 else None)
                    ax.plot(val if np.isfinite(val) else 500, j, "o" if np.isfinite(val) else "x",
                            color="C0", label="Rao-Blackwell mean (x = failure)" if j == 0 else None)
                    if np.isfinite(median):
                        ax.plot(median, j, "s", color="C1", markersize=4,
                                label="Median of bounded draws" if j == 0 else None)
                else:
                    ax.barh(j, 100 * row["survival"], color="C0", alpha=.75)
                    ax.plot(100 * row["long_survival"], j, "ko", markersize=4,
                            label="10x-window survival (50 draws)" if j == 0 else None)
            ax.set(title=f"Re {(150, 300, 200)[k]}" + (" (held out)" if k == 2 else " (training)"),
                   xlabel=xlabel)
            ax.grid(axis="x", alpha=.2)
            if metric == "error":
                ax.set_xscale("log")
            else:
                ax.set_xlim(0, 105)
        axes[0].set_yticks(range(len(names)), names)
        axes[0].invert_yaxis()
        axes[0].legend(fontsize=7, loc="best")
        fig.suptitle("Joint Bayesian refits; modular runs remove operator-to-GP feedback")
        fig.tight_layout()
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)

    selected = [name for name in ("broad-control", "fixed03", "modular42") if name in fits]
    if not selected:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex="col")
    base = np.load(root / "fits" / selected[0] / "plotdata.npz")
    for k in range(3):
        t, truth = base[f"t_{k}"], base[f"truth_{k}"]
        lead = int(np.argmax(np.std(truth[:, t > 1], axis=1)))
        for j, mode in enumerate((0, lead)):
            ax = axes[j, k]
            ax.plot(t, truth[mode], "k-", linewidth=1, label="Projected truth")
            if k < 2:
                ax.plot(base[f"t_samp_{k}"], base[f"obs_{k}"][mode], "k*", markersize=2, label="Observations")
            for i, name in enumerate(selected):
                with np.load(root / "fits" / name / "plotdata.npz") as data:
                    key = f"rao_blackwell_{k}"
                    if key in data.files:
                        ax.plot(data[f"t_{k}"], data[key][mode], "--", color=f"C{i}", label=name)
                    else:
                        ax.plot([], [], "--", color=f"C{i}", label=f"{name}: unbounded")
            ax.axvline(1 if k < 2 else 0, color="gray", linestyle=":")
            ax.set_ylabel(f"POD coefficient {mode+1}")
        axes[0, k].set_title(f"Re {(150, 300, 200)[k]}" + (" (held out)" if k == 2 else ""))
        axes[1, k].set_xlabel("Time since t=16")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "05_mean_trajectories.png", dpi=170)
    plt.close(fig)
    print(f"{len(screens)} conditional screens, {len(fits)} refits; figures and tables in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.root, args.output)

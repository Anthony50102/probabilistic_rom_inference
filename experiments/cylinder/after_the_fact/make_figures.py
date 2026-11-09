"""Figures + table for the cylinder-wake runs (Mac, light: reads the pulled summary.json/plotdata.npz only).

usage: python make_figures.py RESULTS_DIR OUT_DIR
RESULTS_DIR holds <task>_<regime>_r<r>/{summary.json,plotdata.npz}."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
REGIMES = ["dense_low_noise", "sparse_low_noise", "dense_high_noise"]
REG_LABEL = {"dense_low_noise": "dense, 1% noise", "sparse_low_noise": "sparse, 3% noise",
             "dense_high_noise": "dense, 10% noise"}
TASK_LABEL = {"lc": "periodic shedding", "tr": "onset of shedding"}
BLUE, ORANGE, GREEN = "#1f77b4", "#ff7f0e", "#2ca02c"


def load(name):
    d = os.path.join(RES, name)
    if not os.path.exists(os.path.join(d, "summary.json")):
        return None, None
    with open(os.path.join(d, "summary.json")) as f:
        s = json.load(f)
    p = dict(np.load(os.path.join(d, "plotdata.npz")))
    if os.path.exists(os.path.join(d, "rescore_mean.json")):
        with open(os.path.join(d, "rescore_mean.json")) as f:
            s["mean_op"] = json.load(f)
        if os.path.exists(os.path.join(d, "rescore_mean.npz")):
            m = np.load(os.path.join(d, "rescore_mean.npz"))
            p["post_mean"], p["frame_post_mean"] = m["mean"], m["frame_mean"]
    return s, p


def mean_err(s, key="mean_op_fc_fluct_reduced"):
    m = s.get("mean_op", {}) if s else {}
    return 100 * m[key] if key in m else np.nan


def full_errs(s):
    """Full-order forecast errors (% of the shedding part): posterior mean*, median of stable draws, least squares,
    floor. Least squares is converted exactly from its reduced error (orthonormal basis: full^2 = red^2 + floor^2)."""
    if s is None:
        return [np.nan] * 4
    floor = s["projection_error_forecast_fluct"]
    pm = mean_err(s, "mean_op_full_order_fc_fluct")
    med = 100 * s.get("full_order_error_forecast_fluct", np.nan)
    ratio = s.get("mean_op", {}).get("fluct_norm_ratio_reduced_to_full", np.nan)
    ls = s["least_squares"]
    lsf = 100 * np.hypot(ls["pred_error_fluct"] * ratio, floor) if ls.get("stable") else np.nan
    return [pm, med, lsf, 100 * floor]


def mode_figure(task, regime, r, fname):
    s, p = load(f"{task}_{regime}_r{r}")
    if s is None:
        return
    t0 = s["window"][0]
    t, ts = p["t"] + t0, p["t_samp"] + t0
    n = p["truth"].shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.0 * n), sharex=True)
    step = max(1, len(ts) // 80)
    for j, ax in enumerate(axes):
        if "low" in p:
            ax.fill_between(t, p["low"][j], p["high"][j], color=BLUE, alpha=0.25, lw=0, label="ours 5–95%")
            ax.plot(t, p["median"][j], color=BLUE, lw=1.0, ls="-.", label="ours, median of stable draws")
        if "post_mean" in p:
            ax.plot(t, p["post_mean"][j], color=GREEN, lw=1.4, label="ours, posterior mean*")
        if "least_squares" in p:
            ax.plot(t, p["least_squares"][j], color=ORANGE, lw=1.1, ls="--", label="least-squares OpInf")
        ax.plot(t, p["truth"][j], "k-", lw=0.9, label="truth")
        ax.plot(ts[::step], p["obs"][j][::step], "k*", ms=4, label="noisy training data")
        ax.axvline(t0 + float(p["train"]), color="gray", ls=":", lw=1)
        lo, hi = p["truth"][j].min(), p["truth"][j].max()
        pad = 0.6 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(f"mode {j + 1}")
    axes[0].legend(ncol=3, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02))
    axes[-1].set_xlabel("time")
    fig.suptitle(f"Cylinder wake, {TASK_LABEL[task]} ({REG_LABEL[regime]}, r = {r}): training left of the "
                 f"dotted line, forecast right (* scored after the fact)", y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def field_figure(r, fname):
    rows = []
    for task in ("lc", "tr"):
        s, p = load(f"{task}_dense_low_noise_r{r}")
        if s is not None and ("frame_post_mean" in p or "frame_median" in p):
            rows.append((task, s, p))
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 1.9 * len(rows) + 0.6), squeeze=False)
    for i, (task, s, p) in enumerate(rows):
        ny, nx = len(p["ys"]), len(p["xs"])
        ext = [p["xs"][0], p["xs"][-1], p["ys"][0], p["ys"][-1]]
        k = -1
        truth = p["frame_truth"][:, k].reshape(2, ny, nx)[1]
        key = "frame_post_mean" if "frame_post_mean" in p else "frame_median"
        lab = "ours (posterior mean*)" if key == "frame_post_mean" else "ours (median of stable draws)"
        med = p[key][:, k].reshape(2, ny, nx)[1]
        mask = p["inside"]
        vmax = np.abs(truth).max()
        for ax, f, title, cmap, lim in ((axes[i, 0], truth, "truth", "RdBu_r", vmax),
                                        (axes[i, 1], med, lab, "RdBu_r", vmax),
                                        (axes[i, 2], np.abs(med - truth), "|error|", "magma", vmax)):
            f = np.ma.masked_where(mask, f)
            im = ax.imshow(f, origin="lower", extent=ext, cmap=cmap, vmin=-lim if cmap == "RdBu_r" else 0,
                           vmax=lim, aspect="equal")
            ax.set_title(f"{TASK_LABEL[task]}: {title}, t = {s['window'][0] + p['frame_times'][k]:.2f}",
                         fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.8, pad=0.01)
    fig.suptitle(f"Cross-stream velocity v at the end of the forecast (dense, 1% noise, r = {r}; * scored after the fact)", fontsize=10)
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def regime_figure(r, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, task in zip(axes, ("lc", "tr")):
        ours, ls, floor, pm = [], [], [], []
        for reg in REGIMES:
            e = full_errs(load(f"{task}_{reg}_r{r}")[0])
            pm.append(e[0]), ours.append(e[1]), ls.append(e[2]), floor.append(e[3])
        x = np.arange(len(REGIMES))
        ax.bar(x - 0.27, pm, 0.27, color=GREEN, label="ours, posterior mean*")
        ax.bar(x, ours, 0.27, color=BLUE, label="ours, median of stable draws")
        ax.bar(x + 0.27, ls, 0.27, color=ORANGE, label="least-squares OpInf")
        ax.hlines(floor, x - 0.42, x + 0.42, colors="k", lw=2, label="best possible with r modes")
        ax.set_xticks(x, [REG_LABEL[g] for g in REGIMES])
        ax.set_ylabel("full-order forecast error, % of shedding part")
        ax.set_title(f"{TASK_LABEL[task]} (r = {r})")
        top = np.nanmax(ours + ls + pm + [10]) if np.any(np.isfinite(ours + ls + pm)) else 100
        ax.set_ylim(0, min(150, 1.15 * top))
    axes[0].legend(fontsize=8)
    fig.suptitle("Full-order forecast error (* scored after the fact; missing bar = every draw / the mean model blew up)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def sweep_figure(fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for task, rs, mk in (("lc", (5, 9, 13, 17, 21), "o"), ("tr", (5, 9, 13), "s")):
        rows = []
        for r in rs:
            s_, _ = load(f"{task}_dense_low_noise_r{r}")
            if s_ is None:  # declared fit diverged (float32 NaN): use the disclosed float64 rerun
                s_, _ = load(f"{task}_dense_low_noise_r{r}_f64")
            if s_ is not None:
                rows.append((r, s_))
        if not rows:
            continue
        r = [x[0] for x in rows]
        e = np.array([full_errs(x[1]) for x in rows])
        axes[0].plot(r, e[:, 1], mk + "-.", color=BLUE, label=f"ours median of stable draws, {TASK_LABEL[task]}")
        axes[0].plot(r, e[:, 2], mk + "--", color=ORANGE, label=f"least squares, {TASK_LABEL[task]}")
        axes[0].plot(r, e[:, 3], mk + ":", color="k", label=f"best possible, {TASK_LABEL[task]}")
        axes[0].plot(r, e[:, 0], mk + "-", color=GREEN,
                     label=f"ours posterior mean*, {TASK_LABEL[task]}")
        axes[1].plot(r, [x[1]["stability_pct"] for x in rows], mk + "-", color=BLUE, label=TASK_LABEL[task])
        axes[2].plot(r, [x[1]["runtime_s"] / 60 for x in rows], mk + "-", color=BLUE, label=TASK_LABEL[task])
        for rr, s_ in rows:
            if s_.get("precision") == "float64":
                for ax in axes:
                    ax.axvspan(rr - 0.8, rr + 0.8, color="0.9", zorder=0)
                axes[2].annotate("float64 rerun*\n(8 cores)", (rr, s_["runtime_s"] / 60), fontsize=7,
                                 xytext=(-55, -12), textcoords="offset points")
    for ax in axes:
        ax.set_xticks([5, 9, 13, 17, 21])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of modes r")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, frameon=False)
    axes[0].set_ylabel("full-order forecast error, % of shedding part")
    axes[1].set_xlabel("number of modes r"), axes[1].set_ylabel("posterior draws that stay finite (%)")
    axes[1].set_ylim(0, 100), axes[1].legend(fontsize=8)
    axes[2].set_xlabel("number of modes r"), axes[2].set_ylabel("fit time on 4 cores (min)")
    axes[2].legend(fontsize=8)
    fig.suptitle("Scaling with the number of modes (dense, 1% noise; * after the fact; grey: r = 21 declared fit "
                 "diverged in float32, float64 rerun shown)", fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def table(fname):
    names = sorted(n for n in os.listdir(RES) if os.path.exists(os.path.join(RES, n, "summary.json")))
    head = ("run", "pod%", "err%", "errFl%", "floorFl%", "cov%", "stab%", "St true", "St med", "St 5-95",
            "amp ratio", "long ok", "LS err%", "LS errFl%", "LS St", "min", "PM errFl%", "PM fullFl%", "PM St", "PM long")
    lines = ["  ".join(f"{h:>10}" for h in head)]
    for n in names:
        s, _ = load(n)
        ls = s["least_squares"]
        st = s.get("strouhal_draws_5_50_95", [np.nan] * 3)
        vals = (n, 100 * s["pod_energy"], 100 * s["pred_error"], 100 * s.get("pred_error_fluct", np.nan),
                100 * s["projection_error_forecast_fluct"], 100 * s["ci_cov_ext"], s["stability_pct"],
                s["strouhal_true"], s.get("strouhal_median", np.nan), f"{st[0]:.3f}-{st[2]:.3f}",
                s.get("amplitude_median", np.nan) / s["amplitude_true"],
                f"{s['long_horizon']['bounded']}/{s['long_horizon']['draws']}",
                100 * ls.get("pred_error", np.nan), 100 * ls.get("pred_error_fluct", np.nan),
                ls.get("strouhal", np.nan), s["runtime_s"] / 60, mean_err(s),
                mean_err(s, "mean_op_full_order_fc_fluct"), s.get("mean_op", {}).get("mean_op_strouhal", np.nan),
                str(s.get("mean_op", {}).get("mean_op_long_10x_bounded", "-")))
        lines.append("  ".join(f"{v:>10.3f}" if isinstance(v, float) else f"{v:>10}" for v in vals))
    with open(os.path.join(OUT, fname), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


mode_figure("lc", "dense_low_noise", 9, "01_periodic_shedding_modes.png")
mode_figure("tr", "dense_low_noise", 9, "02_onset_of_shedding_modes.png")
field_figure(9, "03_flow_field_end_of_forecast.png")
regime_figure(9, "04_error_by_data_regime.png")
sweep_figure("05_scaling_with_modes.png")
table("table.txt")

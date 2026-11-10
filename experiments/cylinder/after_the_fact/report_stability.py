"""Report of the stability-prior round (Mac, light: reads pulled JSON + plotdata only).
usage: python report_stability.py NEW_DIR FREE_DIR OUT_DIR
NEW_DIR/<name>/{summary.json, rescore_mean.json, stability_extra.json, plotdata.npz}; FREE_DIR = the earlier free runs."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NEW, FREE, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
RS = [5, 9, 13, 17, 21]
L = "lc_dense_low_noise"
VARIANTS = [  # label, directory, name suffix (None = the earlier free run)
    ("free (earlier)", FREE, None), ("f64 control", NEW, "_f64"), ("shift", NEW, "_f64_shift"),
    ("EP", NEW, "_f64_ep"), ("EP + shift", NEW, "_f64_ep_shift")]
COL = {"free (earlier)": "0.4", "f64 control": "#9467bd", "shift": "#2ca02c", "EP": "#ff7f0e", "EP + shift": "#1f77b4"}


def free_name(base, r):
    return f"{base}_r{r}" + ("_f64" if (base == L and r == 21) else "")


def load(root, name):
    d = os.path.join(root, name)
    if os.path.exists(os.path.join(d, "summary.json")):
        s = json.load(open(os.path.join(d, "summary.json")))
        s["plot"] = os.path.join(d, "plotdata.npz")
    elif os.path.exists(os.path.join(d, "summary_fallback.json")):
        # Run killed in summarize() (long horizon) by the 8 h limit; forecast fields rebuilt from 04_unified.npz.
        s = json.load(open(os.path.join(d, "summary_fallback.json")))
        s["long_horizon"] = None
        s["plot"] = os.path.join(d, "plotdata_fallback.npz")
    else:
        return None
    for f, k in (("rescore_mean.json", "pm"), ("stability_extra.json", "extra")):
        p = os.path.join(d, f)
        s[k] = json.load(open(p)) if os.path.exists(p) else {}
    s["dir"] = d
    return s


def get(base, r, variant):
    label, root, suf = variant
    return load(root, free_name(base, r) if suf is None else f"{base}_r{r}{suf}")


pct = lambda x: np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else 100 * x
f1 = lambda x: "—" if x is None or np.isnan(x) else f"{x:.1f}"


def row(s):
    if s is None:
        return None
    pm, ex = s["pm"], s["extra"]
    return dict(floor=pct(pm.get("floor_full_order_fc_fluct")), pm=pct(pm.get("mean_op_full_order_fc_fluct")),
                pm_bounded=pm.get("mean_op_bounded"), pm_long=pm.get("mean_op_long_10x_bounded"),
                declared=pct(s.get("full_order_error_forecast_fluct")), stable=s["stability_pct"],
                long=(f"{s['long_horizon']['bounded']}/{s['long_horizon']['draws']}" if s["long_horizon"] else ">8h*"), cov=pct(s.get("ci_cov_ext")),
                st_med=s.get("strouhal_median"), st_pm=pm.get("mean_op_strouhal"), ep=ex.get("ep_violation_pm"),
                trap=ex.get("trap_cert_pm"), norm=ex.get("op_norm_pm"), runtime=s.get("runtime_s"), plot=s["plot"])


lines = ["Stability-prior round: full-order forecast error, % of the fluctuating part (floor = POD projection).",
         "declared = median of stable draws; PM* = posterior-mean operator; long = draws bounded at 10x window;",
         "cov = 5-95% band coverage in the forecast window; EP viol = ||C vec PM*||/(||C|| ||PM*||);",
         "trap = Q = I certificate of PM* (< 0 would certify a trapping region).",
         ">8h* = the 50-draw long-horizon integration did not finish inside the declared 8 h limit (explicit RK45 on",
         "      stiff draws); that run's other fields were rebuilt from its saved 04_unified.npz with 04_unified's helpers.", ""]
hdr = f"{'run':34s} {'floor':>5s} {'PM*':>6s} {'decl':>6s} {'stable%':>7s} {'long':>6s} {'cov%':>5s} {'St med':>6s} {'St PM*':>6s} {'EP viol':>8s} {'trap':>7s} {'|PM*|':>7s} {'min':>5s}"
lines.append(hdr)
T = {}
for base, rs in ((L, RS), ("lc_sparse_low_noise", [9]), ("lc_dense_high_noise", [9])):
    for r in rs:
        for v in VARIANTS:
            s = get(base, r, v)
            x = row(s)
            T[(base, r, v[0])] = x
            if x is None:
                continue
            nm = f"{base.replace('lc_', '')} r{r} {v[0]}"
            fmt = lambda y, w=6, p=4: f"{y:{w}.{p}f}" if isinstance(y, (int, float)) and y is not None and not np.isnan(y) else f"{'—':>{w}s}"
            lines.append(f"{nm:34s} {f1(x['floor']):>5s} {f1(x['pm']):>6s} {f1(x['declared']):>6s} {x['stable']:7.1f} "
                         f"{x['long']:>6s} {f1(x['cov']):>5s} {fmt(x['st_med'])} {fmt(x['st_pm'])} "
                         f"{(format(x['ep'], '.1e') if x['ep'] is not None else '—'):>8s} {fmt(x['trap'], 7, 3)} "
                         f"{fmt(x['norm'], 7, 1)} {(x['runtime'] or np.nan) / 60:5.0f}")
        lines.append("")

# Declared reading rule and expectations.
g = lambda base, r, v, k: (T.get((base, r, v)) or {}).get(k, np.nan)
nz = lambda y: np.nan if y is None else y
lines.append("Declared reading rule (r9, r13): EP + shift stable >= 80% AND declared median within 3 points of the free PM*.")
for r in (9, 13):
    st, de, pmf = g(L, r, "EP + shift", "stable"), nz(g(L, r, "EP + shift", "declared")), g(L, r, "free (earlier)", "pm")
    ok = st >= 80 and not np.isnan(de) and de <= pmf + 3
    lines.append(f"  r{r}: stable {st:.1f}%, declared {f1(de)} vs free PM* {f1(pmf)} -> {'FIXES the sampling UQ' if ok else 'does NOT fix'}")
lines.append("")
def verdict(pairs, cond):
    """Judge only r values where the needed fits ran (a fit stopped by the 8 h rule is 'not run')."""
    ran = [(r, a, b) for r, a, b in pairs if not (np.isnan(a) or np.isnan(b))]
    gone = [f"r{r}" for r, a, b in pairs if (np.isnan(a) or np.isnan(b))]
    v = "MET" if ran and all(cond(a, b) for _, a, b in ran) else "NOT MET"
    return v + (f" where run (not run: {', '.join(gone)})" if gone else "")


lines.append("Expectations (written before any fit):")
e1 = [(r, g(L, r, "EP + shift", "stable"), g(L, r, "free (earlier)", "stable")) for r in RS]
lines.append("  1. EP + shift stable > free at every r: " + verdict(e1, lambda a, b: a > b)
             + "  (" + ", ".join(f"r{r} {a:.1f} vs {b:.1f}" for r, a, b in e1) + ")")
e2 = [(r, g(L, r, "EP + shift", "pm"), g(L, r, "free (earlier)", "pm")) for r in (5, 9, 13)]
lines.append("  2. EP + shift PM* within 3 points of free PM* at r <= 13: " + verdict(e2, lambda a, b: a <= b + 3)
             + "  (" + ", ".join(f"r{r} {f1(a)} vs {f1(b)}" for r, a, b in e2) + ")")
e3 = [(r, g(L, r, "EP", "pm"), g(L, r, "EP + shift", "pm")) for r in RS]
lines.append("  3. EP alone clearly worse PM* than EP + shift (operationalised before results: worse by > 1 point at every r): "
             + verdict(e3, lambda a, b: a > b + 1)
             + "  (" + ", ".join(f"r{r} {f1(a)} vs {f1(b)}" for r, a, b in e3) + ")")
s13 = g(L, 13, "shift", "stable")
lines.append(f"  4. shift alone stable < 50% at r13: {'MET' if s13 < 50 else 'NOT MET'}  ({s13:.1f}%)")
traps = [(k, v["trap"]) for k, v in T.items() if v is not None and v["trap"] is not None and k[2] != "free (earlier)"]
lines.append(f"  5. Q = I certificate never negative: {'MET' if all(t >= 0 for _, t in traps) else 'NOT MET'}  (min {min(t for _, t in traps):.3g})")
open(os.path.join(OUT, "report.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# Figure 1: stable fraction and errors vs r.
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for lab, _, _ in VARIANTS:
    rs = [r for r in RS if T.get((L, r, lab))]
    if not rs:
        continue
    ax[0].plot(rs, [T[(L, r, lab)]["stable"] for r in rs], "o-", color=COL[lab], label=lab)
    ax[1].plot(rs, [T[(L, r, lab)]["pm"] for r in rs], "o-", color=COL[lab], label=f"{lab}: posterior mean")
    ax[1].plot(rs, [T[(L, r, lab)]["declared"] for r in rs], "s--", color=COL[lab], mfc="none", label=f"{lab}: median of stable draws")
fl = [T[(L, r, "free (earlier)")]["floor"] for r in RS if T.get((L, r, "free (earlier)"))]
ax[1].plot([r for r in RS if T.get((L, r, "free (earlier)"))], fl, "k:", lw=2, label="POD floor")
ax[0].set(xlabel="reduced dimension r", ylabel="% of posterior draws that stay finite", title="Stable posterior draws", xticks=RS, ylim=(-3, 103))
ax[1].set(xlabel="reduced dimension r", ylabel="full-order forecast error (% of fluctuation)", yscale="log", title="Forecast error", xticks=RS)
ax[0].legend(fontsize=8); ax[1].legend(fontsize=6, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_stability_and_error_vs_r.png"), dpi=150); plt.close(fig)

# Figure 2: leading modes, band of stable draws, free vs shift vs EP + shift at r9 and r13; scans as black stars.
for r in (9, 13):
    pair = [(lab, T.get((L, r, lab))) for lab in ("free (earlier)", "shift", "EP + shift")]
    if any(x is None for _, x in pair):
        continue
    fig, ax = plt.subplots(3, 3, figsize=(17, 7.5), sharex=True)
    for j, (lab, x) in enumerate(pair):
        root = FREE if lab == "free (earlier)" else NEW
        nm = free_name(L, r) if lab == "free (earlier)" else f"{L}_r{r}" + {"shift": "_f64_shift", "EP + shift": "_f64_ep_shift"}[lab]
        p = np.load(x["plot"])
        pmp = os.path.join(root, nm, "rescore_mean.npz")
        pm = np.load(pmp)["mean"] if os.path.exists(pmp) else None
        for i, k in enumerate((0, 1, 2)):
            a = ax[i, j]
            a.axvspan(p["t"][0], float(p["train"]), color="0.93")
            a.plot(p["t"], p["truth"][k], "k--", lw=0.9, label="truth")
            if np.all(np.isfinite(p["low"][k])):
                a.fill_between(p["t"], p["low"][k], p["high"][k], color=COL[lab], alpha=0.3, label="5-95% band (stable draws)")
            if pm is not None:
                a.plot(p["t"], pm[k], color=COL[lab], lw=1.2, label="posterior mean")
            a.plot(p["t_samp"], p["obs"][k], "k*", ms=3, label="noisy training data")
            a.set_ylabel(f"mode {k + 1}")
        ax[0, j].set_title(f"{lab}, r = {r}: {x['stable']:.0f}% stable draws,\nposterior mean {f1(x['pm'])}%, median of stable draws {f1(x['declared'])}%", fontsize=10)
        ax[2, j].set_xlabel("time")
    ax[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"02_modes_r{r}.png"), dpi=150); plt.close(fig)
print("figures in", OUT)

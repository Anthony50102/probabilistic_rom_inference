"""scout_modes.py — how many POD modes a cylinder-wake run needs (higher-mode variant, DESIGN.md).

For one FOM file (fom_dolfinx.py output) this reports, without any ROM fit:
  - shedding statistics over t >= 16: Cd_max, Cl_max, Strouhal number;
  - periodicity: spread of the per-cycle lift amplitudes and periods over t >= 16, and the share of the lift's
    fluctuating power at the fundamental frequency and its harmonics;
  - the POD floor on the lc task (train [16, 17], forecast to 19): the forecast-window projection error of the
    truth onto the rank-r basis, relative to the truth's fluctuating part (summary.json's
    projection_error_forecast_fluct). The basis is built as in 04_unified.prepare: 250 samples per time unit,
    uniform random with both ends included, 1% pointwise noise, np.random.seed(42). The noise-free basis
    (truth samples) is reported too.
  - r_2pct: the smallest r with the 1%-noise floor <= 2% (the selection quantity).

    python scout_modes.py FOM.npz OUT.json
"""
import json
import sys

import numpy as np

import config

R_MAX = 48


def _rel_fluct(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b - b.mean(axis=-1, keepdims=True)))


def periodicity(t, cl, t_from):
    keep = t >= t_from
    t, cl = t[keep], cl[keep]
    up = np.flatnonzero((cl[:-1] < 0) & (cl[1:] >= 0))
    tc = t[up] - cl[up] * (t[up + 1] - t[up]) / (cl[up + 1] - cl[up])
    periods = np.diff(tc)
    amps = np.array([0.5 * (cl[a:b + 1].max() - cl[a:b + 1].min()) for a, b in zip(up[:-1], up[1:])])
    # Lift spectrum over a whole number of cycles (from the first to the last upward crossing).
    seg = cl[up[0]:up[-1]] - cl[up[0]:up[-1]].mean()
    dt = t[1] - t[0]
    power = np.abs(np.fft.rfft(seg)) ** 2
    freq = np.fft.rfftfreq(len(seg), dt)
    f0 = 1.0 / np.mean(periods)
    df = freq[1]
    harmonic = np.zeros_like(power, dtype=bool)
    for k in range(1, int(freq[-1] / f0) + 1):
        harmonic |= np.abs(freq - k * f0) <= 1.5 * df
    return dict(cycles=int(len(periods)), period_mean=float(np.mean(periods)),
                period_range_rel=float(np.ptp(periods) / np.mean(periods)),
                amplitude_median=float(np.median(amps)), amplitude_range_rel=float(np.ptp(amps) / np.median(amps)),
                harmonic_power_share=float(power[harmonic].sum() / power[1:].sum()),
                strouhal=float(0.1 / np.mean(periods)))               # D / (U_mean T)


def floors(V, truth, fc):
    out = []
    for r in range(1, V.shape[1] + 1):
        Vr = V[:, :r]
        proj = Vr @ (Vr.T @ truth[:, fc])
        out.append(_rel_fluct(proj, truth[:, fc]))
    return out


def main(path, out_path):
    fom = config.load_fom(path)
    t_end = float(fom["t"][-1])
    res = dict(file=path, t_end=t_end)
    d = np.load(path)
    res["nu"] = float(d["nu"]) if "nu" in d.files else 1e-3
    res["Re"] = 0.1 / res["nu"]
    keep = fom["t_steps"] >= config.LC_START
    res["cd_max"] = float(fom["cd"][keep].max())
    res["cd_mean"] = float(fom["cd"][keep].mean())
    res["cl_max"] = float(fom["cl"][keep].max())
    res["cl_min"] = float(fom["cl"][keep].min())
    res.update(periodicity(fom["t_steps"], fom["cl"], config.LC_START))

    start, train, total = config.LC_START, config.LC_TRAIN, config.LC_TOTAL
    t_full, truth, spline = config.window_states(fom, start, total)
    fc = t_full > train
    np.random.seed(42)
    n = int(round(config.SAMPLES_PER_TIME["dense"] * train))
    t_samp = np.sort(np.random.uniform(0, train, n))
    t_samp[0], t_samp[-1] = 0.0, train
    clean = spline(t_samp).T
    noisy = config.noise(clean, 0.01)
    for name, snaps in (("noisy_1pct", noisy), ("clean", clean)):
        U, s, _ = np.linalg.svd(snaps, full_matrices=False)
        V = U[:, :R_MAX]
        res[f"floor_{name}"] = floors(V, truth, fc)
        res[f"sv_{name}"] = [float(v) for v in s[:R_MAX + 1]]
    f = np.array(res["floor_noisy_1pct"])
    hit = np.flatnonzero(f <= 0.02)
    res["r_2pct"] = int(hit[0] + 1) if len(hit) else None
    hit = np.flatnonzero(f <= 0.03)
    res["r_3pct"] = int(hit[0] + 1) if len(hit) else None
    res["min_floor_noisy"] = float(f.min())
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"Re {res['Re']:g}: Cd_max {res['cd_max']:.4f} Cl_max {res['cl_max']:.4f} St {res['strouhal']:.4f}; "
          f"amp range {res['amplitude_range_rel']:.2%}, period range {res['period_range_rel']:.2%}, "
          f"harmonic share {res['harmonic_power_share']:.4f}; r_2pct {res['r_2pct']} r_3pct {res['r_3pct']}; "
          f"floor r9/13/17/21/25/29 " + " ".join(f"{f[r - 1]:.2%}" for r in (9, 13, 17, 21, 25, 29)))


if __name__ == "__main__":
    main(*sys.argv[1:3])

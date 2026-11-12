"""
05_multi_re.py — one parametric ROM for the cylinder wake at several Reynolds numbers (DESIGN.md, "Parametric
multi-Re ROM").

A Galerkin ROM of incompressible Navier–Stokes on a fixed basis is dq/dt = c + Aq + H[q ⊗ q] + ν(b + Nq): the
viscosity enters linearly and multiplies only constant and linear terms. So one cAHBN ROM with the scalar input
u = ν / ν_ref = Re_ref / Re (Re_ref = 200, constant along each trajectory) covers every Reynolds number. Each
training Re contributes its own lc window ([16, 17] observed, forecast to 19; 250 noisy samples, 1% noise, own GP);
the POD basis is fitted to all training samples together, which is what pushes the rank needed for a 2% floor past
20 (the single-Re wakes need at most 14).

    python 05_multi_re.py --train-re 150,300 --held-re 200 --modes 25 [--feature-shift training_mean]

FOM files: $CYL_FOM_DIR/medium_re{Re}.npz. Outputs in results/comparison/<name>/: the pipeline's NPZ,
summary.json (per Re: floor, posterior-mean (PM*) and draw-median full-order errors, Strouhal, coverage, stable
draws, long-horizon check, least-squares cAHBN reference) and plotdata.npz.
"""
import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis
from core import JaxCompatibleModel
from core.bayesian_opinf import generate_rom_predictions
from core.weakform_opinf import EvalTarget, PreparedRun, run_experiment
from core.weakform_opinf.constraints import _features
import opinf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cyl04", os.path.join(SCRIPT_DIR, "04_unified.py"))
cyl04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cyl04)
_rel, _rel_fluct, _period, _bounded = cyl04._rel, cyl04._rel_fluct, cyl04._period, cyl04._bounded

RE_REF = 200.0
REGIME = dict(density="dense", NOISE_LEVEL=0.01, label="Dense data, low noise", regime="dense_low_noise")
LONG_FACTOR, LONG_DRAWS, PLOT_MODES = 10, 50, 5
FOM_DIR = os.environ.get("CYL_FOM_DIR", os.path.join(SCRIPT_DIR, "data"))


def fom_path(re):
    return os.path.join(FOM_DIR, f"medium_re{re:g}.npz")


def constant_input(u):
    return lambda t, u=u: np.full((1,) + np.shape(t), u)


def _window(re, num_samples, window=None):
    """Truth on the lc window and noisy samples on its training part (global RNG, seeded by the pipeline)."""
    fom = config.load_fom(fom_path(re))
    if fom["geometry"] != "cylinder" or abs(fom["D"] / fom["nu"] - re) > 1e-6 * re:
        raise ValueError(f"{fom_path(re)} is not the cylinder at Re {re}")
    start, train, total = config.task_window("lc", fom) if window is None else window
    if not (0 <= start and 0 < train < total and fom["t"][0] <= start and start + total <= fom["t"][-1]):
        raise ValueError(f"Invalid or unavailable window {(start, train, total)} for Re {re}")
    t_full, X, spline = config.window_states(fom, start, total)
    t_samp = np.sort(np.random.uniform(0, train, num_samples))
    t_samp[0], t_samp[-1] = 0.0, train
    snaps = config.noise(spline(t_samp).T, REGIME["NOISE_LEVEL"])
    grid = (fom["xs"], fom["ys"], fom["inside"])
    return dict(re=re, u=RE_REF / re, nu=fom["nu"], D=fom["D"], window=(start, train, total), t_full=t_full, X=X,
                t_samp=t_samp, snaps=snaps, grid=grid, dt=float(np.median(np.diff(fom["t"]))))


class MultiReSpec:
    name = "05_multi_re"

    def __init__(self, train_re, held_re, window=None, samples=None):
        self.train_re, self.held_re = list(train_re), list(held_re)
        self.window, self.samples = window, samples

    def prepare(self, cfg, schema):
        n = (int(round(config.SAMPLES_PER_TIME[REGIME["density"]] * config.LC_TRAIN))
             if self.samples is None else self.samples)
        runs = [_window(re, n, self.window) for re in self.train_re] + [_window(re, n, self.window) for re in self.held_re]
        train_runs = runs[:len(self.train_re)]
        basis = Basis(num_vectors=cfg.num_modes)
        basis.fit(np.hstack([w["snaps"] for w in train_runs]))
        print(f"  training Re {self.train_re} (held out {self.held_re}); {n} samples each; {cfg.num_modes} modes "
              f"(POD energy {basis.cumulative_energy:.4%})")
        for w in runs:
            w["comp"], w["true_comp"] = basis.compress(w["snaps"]), basis.compress(w["X"])

        w0 = train_runs[0]
        rom = opinf.ROM(
            basis=basis, ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(w0["t_samp"]),
            model=JaxCompatibleModel(operators=cfg.operators,
                                     solver=opinf.lstsq.L2Solver(regularizer=cfg.regularizer)))
        rom.fit(states=w0["snaps"], inputs=np.full((1, len(w0["t_samp"])), w0["u"]))
        trajectories = [dict(t_sampled=w["t_samp"], snapshots_comp=w["comp"],
                             inputs_eval=np.full((1, cfg.num_eval_points), w["u"])) for w in train_runs]
        targets = [EvalTarget(t_pred=w["t_full"], true_comp=w["true_comp"], true_states=w["X"],
                              state0_comp=w["comp"][:, 0], t_full=w["t_full"], input_func=constant_input(w["u"]),
                              t_sampled=w["t_samp"], snapshots_comp=w["comp"],
                              training_index=k if k < len(train_runs) else None,
                              label=f"Re {w['re']:g} ({'train' if k < len(train_runs) else 'held out'})")
                   for k, w in enumerate(runs)]
        for w in runs:
            del w["snaps"]
        return PreparedRun(rom=rom, trajectories=trajectories, basis=basis, eval_targets=targets,
                           training_span=(0.0, w0["window"][1]), snapshots_comp=w0["comp"],
                           t_sampled=w0["t_samp"], extra=dict(runs=runs),
                           npz_fields=dict(train_re=np.array(self.train_re, float),
                                           held_re=np.array(self.held_re, float), re_ref=RE_REF,
                                           window=np.array(w0["window"])))


def _rhs_factory(O, rom, r, u):
    """dq/dt for operator O (opinf cAHBN layout [c | A | H | B | N]) at constant input u, in numpy."""
    ckron = opinf.operators.QuadraticOperator.ckron
    nH = r * (r + 1) // 2
    c, A, H = O[:, 0], O[:, 1:1 + r], O[:, 1 + r:1 + r + nH]
    B, N = O[:, 1 + r + nH], O[:, 2 + r + nH:]
    q = np.random.default_rng(0).normal(size=(r, 3))
    feats = _features(rom, q, np.full((1, 3), u))
    if O.shape[1] != feats.shape[1] or not np.allclose(
            feats @ O.T, np.stack([c + A @ x + H @ ckron(x) + u * (B + N @ x) for x in q.T]), rtol=1e-8, atol=1e-8):
        raise ValueError("Unexpected cAHBN column layout")
    lin, const = A + u * N, c + u * B
    return lambda t, x: const + lin @ x + H @ ckron(x)


def integrate(O, rom, r, u, q0, t, cap):
    """Integrate with a blow-up stop (|q| > cap); None unless the whole grid is reached."""
    rhs = _rhs_factory(O, rom, r, u)
    ev = lambda tt, q: np.max(np.abs(q)) - cap
    ev.terminal = True
    s = solve_ivp(rhs, (t[0], t[-1]), q0, t_eval=t, rtol=1e-6, atol=1e-8, events=ev)
    return s.y if s.y.shape[1] == len(t) else None


def least_squares_reference(prepared, runs, t_pred, train, r):
    """Deterministic cAHBN OpInf on all training trajectories (finite-difference derivatives, L2). The regulariser:
    among the values whose forecasts stay bounded for every training Re, the smallest mean misfit to the noisy
    reduced training snapshots (no truth used)."""
    trains = [w for w, tr in zip(runs, prepared.eval_targets) if tr.training_index is not None]
    Q = np.hstack([w["comp"] for w in trains])
    dQ = np.hstack([opinf.ddt.NonuniformFiniteDifferencer(w["t_samp"]).estimate(w["comp"])[1] for w in trains])
    U = np.hstack([np.full((1, w["comp"].shape[1]), w["u"]) for w in trains])
    fit = t_pred <= train
    best = None
    for lam in cyl04.LS_REGULARIZERS:
        model = opinf.models.ContinuousModel("cAHBN", solver=opinf.lstsq.L2Solver(regularizer=float(lam)))
        model.fit(states=Q, ddts=dQ, inputs=U)
        O = np.asarray(model.operator_matrix)
        sols, errs = [], []
        for w in trains:
            sol = integrate(O, prepared.rom, r, w["u"], w["comp"][:, 0], t_pred, 2 * np.max(np.abs(w["comp"])))
            if sol is None:
                break
            at = np.stack([np.interp(w["t_samp"], t_pred[fit], s) for s in sol[:, fit]])
            sols.append(sol)
            errs.append(_rel(at, w["comp"]))
        if len(sols) == len(trains) and (best is None or np.mean(errs) < best[1]):
            best = (float(lam), float(np.mean(errs)), O)
    return best


def summarize(result, prepared, schema, cfg, out_dir):
    runs = prepared.extra["runs"]
    V = np.asarray(result["basis"].entries)
    r = cfg.num_modes
    O_mean = np.asarray(result["O_samples"], float).mean(0)
    train = schema["window"][1]
    out = dict(train_re=schema["train_re"], held_re=schema["held_re"], re_ref=RE_REF, input="u = Re_ref / Re",
               modes=r, window=schema["window"], noise=REGIME["NOISE_LEVEL"], samples_per_re=schema["samples"],
               pod_energy=float(result["basis"].cumulative_energy), runtime_s=float(result["runtime"]),
               precision=cfg.precision, feature_shift=cfg.feature_shift, quad_constraint=cfg.quad_constraint,
               full_order_dim=int(runs[0]["X"].shape[0]), n_draws=int(len(result["O_samples"])),
               op_norm_pm=float(np.linalg.norm(O_mean)), per_re=[])
    for key in ("train_error", "pred_error", "ci_cov_ext", "stability_pct", "test_pred_error", "test_ci_cov_ext",
                "test_stability_pct"):
        if key in result:
            out[key] = float(result[key])
    ls = least_squares_reference(prepared, runs, runs[0]["t_full"], train, r)
    out["least_squares"] = dict(stable=ls is not None)
    if ls is not None:
        out["least_squares"]["regularizer"] = ls[0]
    plot = dict(V5=V[:, :PLOT_MODES].astype(np.float32), xs=runs[0]["grid"][0], ys=runs[0]["grid"][1],
                inside=runs[0]["grid"][2], train=train)
    for k, (w, tgt, score) in enumerate(zip(runs, result["eval_targets"], result["per_target"])):
        held = tgt.training_index is None
        t, X, Tc = tgt.t_pred, w["X"], tgt.true_comp
        # Training Re: the forecast window (train, end]. Held-out Re: everything after its first sample.
        fc = t > (0.0 if held else train)
        row = dict(re=w["re"], u=w["u"], held_out=held, nu=w["nu"])
        row["floor_full_fluct"] = _rel_fluct((V @ Tc)[:, fc], X[:, fc])
        lead = int(np.argmax(np.std(Tc[:, fc], axis=1)))
        row["lead_mode"] = lead
        row["strouhal_true"] = w["D"] / _period(t[fc], Tc[lead, fc])
        q0 = np.asarray(score["state0_samples"]).mean(0)
        cap = 2 * np.max(np.abs(tgt.snapshots_comp))
        pm = integrate(O_mean, prepared.rom, r, w["u"], q0, t, cap)
        row["pm_bounded"] = pm is not None
        frames = np.array([np.argmin(np.abs(t - (train + f * (t[-1] - train)))) for f in (0.5, 1.0)])
        p = dict(t=t, truth=Tc[:PLOT_MODES], t_samp=tgt.t_sampled, obs=tgt.snapshots_comp[:PLOT_MODES],
                 frame_times=t[frames], frame_truth=X[:, frames].astype(np.float32))
        if pm is not None:
            row["pm_full_fluct"] = _rel_fluct((V @ pm)[:, fc], X[:, fc])
            row["pm_reduced_fluct"] = _rel_fluct(pm[:, fc], Tc[:, fc])
            row["pm_strouhal"] = w["D"] / _period(t[fc], pm[lead, fc])
            row["pm_amplitude"], row["amplitude_true"] = float(np.std(pm[lead, fc])), float(np.std(Tc[lead, fc]))
            t_long = np.arange(t[0], t[0] + LONG_FACTOR * train + 1e-9, t[1] - t[0])
            row["pm_long_bounded"] = integrate(O_mean, prepared.rom, r, w["u"], q0, t_long, cap) is not None
            p.update(pm=pm[:PLOT_MODES], frame_pm=(V @ pm[:, frames]).astype(np.float32))
        sols = np.asarray(score["rom_solves"])
        row.update(stable_pct=float(score["stability_pct"]), ci_cov_ext=float(score["ci_cov_ext"]))
        if len(sols):
            med = np.median(sols, axis=0)
            row["median_full_fluct"] = _rel_fluct((V @ med)[:, fc], X[:, fc])
            inside = (Tc >= np.percentile(sols, 5, 0)) & (Tc <= np.percentile(sols, 95, 0))
            row["band_coverage_fc"] = float(np.mean(inside[:, fc]))
            p.update(median=med[:PLOT_MODES], low=np.percentile(sols, 5, 0)[:PLOT_MODES],
                     high=np.percentile(sols, 95, 0)[:PLOT_MODES])
        if not held:
            t_long = np.arange(t[0], t[0] + LONG_FACTOR * train + 1e-9, t[1] - t[0])
            _, _, long_sols = generate_rom_predictions(
                samples={"O": result["O_samples"]}, rom=result["rom"], snapshots_compressed=tgt.snapshots_comp,
                time_eval=t_long, num_modes=r, num_pulls=LONG_DRAWS, input_func=tgt.input_func,
                state0_samples=np.broadcast_to(tgt.state0_comp, (len(result["O_samples"]), r)))
            row["long_draws_bounded"] = int(sum(_bounded(np.asarray(s), cap) for s in long_sols))
            row["long_draws"] = LONG_DRAWS
        if ls is not None:
            sol = integrate(ls[2], prepared.rom, r, w["u"], tgt.state0_comp, t, cap)
            row["ls_full_fluct"] = None if sol is None else _rel_fluct((V @ sol)[:, fc], X[:, fc])
        out["per_re"].append(row)
        plot.update({f"{key}_{k}": v for key, v in p.items()})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, "plotdata.npz"), **plot)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-re", required=True, help="comma-separated training Reynolds numbers, e.g. 150,300")
    ap.add_argument("--held-re", default="", help="comma-separated held-out Reynolds numbers (scored only)")
    ap.add_argument("--modes", type=int, required=True)
    ap.add_argument("--feature-shift", choices=["none", "training_mean"], default="none")
    ap.add_argument("--smoke", action="store_true", help="plumbing check only: 200 SVI steps, 20 draws")
    args = ap.parse_args()
    parse = lambda s: [float(x) for x in s.split(",") if x]
    train_re, held_re = parse(args.train_re), parse(args.held_re)
    name = ("multire_" + "_".join(f"{x:g}" for x in train_re)
            + ("_held" + "_".join(f"{x:g}" for x in held_re) if held_re else "")
            + f"_lc_dense_low_noise_r{args.modes}_f64" + ("_shift" if args.feature_shift == "training_mean" else "")
            + ("_smoke" if args.smoke else ""))
    probe = config.load_fom(fom_path(train_re[0]))
    window = tuple(float(x) for x in config.task_window("lc", probe))
    num_pred = int(round(window[2] / np.median(np.diff(probe["t"])))) + 1
    del probe
    schema = dict(REGIME, name=name, task="lc", modes=args.modes, precision="float64", quad_constraint="none",
                  feature_shift=args.feature_shift, window=window, num_pred_points=num_pred,
                  train_re=train_re, held_re=held_re,
                  samples=int(round(config.SAMPLES_PER_TIME[REGIME["density"]] * config.LC_TRAIN)))
    cfg = cyl04.make_config(schema)
    cfg = dataclasses.replace(cfg, operators="cAHBN",
                              **(dict(num_steps=200, num_posterior_samples=20) if args.smoke else {}))
    spec = MultiReSpec(train_re, held_re)
    holder = {}
    original = spec.prepare

    def prepare(cfg_, schema_):
        holder["p"] = original(cfg_, schema_)
        return holder["p"]

    spec.prepare = prepare
    t0 = time.time()
    result = run_experiment(spec, cfg, schema, SCRIPT_DIR)
    out_dir = os.path.join(SCRIPT_DIR, "results", "comparison", name)
    out = summarize(result, holder["p"], schema, cfg, out_dir)
    print(json.dumps(out, indent=1))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

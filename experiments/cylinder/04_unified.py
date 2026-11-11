"""
04_unified.py — Marginalised-O × Weak-Form Bayesian OpInf on the cylinder wake (Navier–Stokes, Re = 100).

Thin adapter over ``core.weakform_opinf`` with the shared production settings (as for Euler); only the ROM
structure (cAH, r modes) and the data are set here. One run = one task x regime x r:

    python 04_unified.py --task lc --regime dense_low_noise --modes 9

Besides the pipeline's NPZ it writes, in results/comparison/<task>_<regime>_r<r>/:
  summary.json   the paper-style metrics plus metrics that the mean flow cannot hide (errors relative to the
                 fluctuating part of the flow, Strouhal number and amplitude of the leading shedding mode),
                 the full-order errors, long-horizon stability, and a least-squares OpInf reference;
  plotdata.npz   small arrays for the figures (leading modes, bands, observations, two full-field frames).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis
from core import JaxCompatibleModel
from core.bayesian_opinf import generate_rom_predictions
from core.weakform_opinf import WeakFormConfig, EvalTarget, PreparedRun, run_experiment
import opinf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REGIMES = {
    "dense_low_noise": dict(density="dense", NOISE_LEVEL=0.01, label="Dense data, low noise"),
    "sparse_low_noise": dict(density="sparse", NOISE_LEVEL=0.03, label="Sparse data, low noise"),
    "dense_high_noise": dict(density="dense", NOISE_LEVEL=0.10, label="Dense data, high noise"),
}
LS_REGULARIZERS = np.logspace(-6, 6, 25)
LONG_FACTOR, LONG_DRAWS, PLOT_MODES = 10, 50, 5


def make_config(schema):
    """Shared production algorithm (core defaults, as for Euler); only the ROM structure is set here."""
    return WeakFormConfig(
        operators="cAH", num_modes=schema["modes"], bump_p=6, weakform_mode="ibp", deriv_cov="diag",
        weakform_cov="diag", op_prior_mode="block_hier", num_posterior_samples=500, regularizer=1.0,
        ic_uncertainty=True, ic_scale=1.0, num_pred_points=schema["num_pred_points"], seed=42,
        precision=schema.get("precision", "default"), quad_constraint=schema.get("quad_constraint", "none"),
        feature_shift=schema.get("feature_shift", "none"))


class CylinderSpec:
    name = "04_unified"

    def __init__(self, fom):
        self.fom = fom

    def prepare(self, cfg, schema):
        start, train, total = schema["window"]
        t_full, true_states, spline = config.window_states(self.fom, start, total)
        n = int(round(config.SAMPLES_PER_TIME[schema["density"]] * train / self.fom["time_scale"]))
        t_samp = np.sort(np.random.uniform(0, train, n))
        t_samp[0], t_samp[-1] = 0.0, train
        snaps_samp = config.noise(spline(t_samp).T, schema["NOISE_LEVEL"])

        basis = Basis(num_vectors=cfg.num_modes)
        basis.fit(snaps_samp)
        snaps_comp = basis.compress(snaps_samp)
        true_comp = basis.compress(true_states)
        print(f"  window start {start:.3f}, train {train:.3f}, total {total:.3f}; {n} samples; "
              f"{cfg.num_modes} modes (POD energy {basis.cumulative_energy:.4%})")

        rom = opinf.ROM(
            basis=basis, ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
            model=JaxCompatibleModel(operators=cfg.operators,
                                     solver=opinf.lstsq.L2Solver(regularizer=cfg.regularizer)))
        rom.fit(states=snaps_samp)
        trajectories = [dict(t_sampled=t_samp, snapshots_comp=snaps_comp, inputs_eval=None)]
        target = EvalTarget(t_pred=t_full, true_comp=true_comp, true_states=true_states,
                            state0_comp=snaps_comp[:, 0], t_full=t_full, input_func=None,
                            t_sampled=t_samp, snapshots_comp=snaps_comp, label=schema["label"])
        return PreparedRun(rom=rom, trajectories=trajectories, basis=basis, eval_targets=[target],
                           training_span=(0.0, train), snapshots_comp=snaps_comp, t_sampled=t_samp,
                           extra=dict(t_full=t_full, snaps_samp=snaps_samp),
                           npz_fields=dict(window=np.array(schema["window"])))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _rel_fluct(a, b):
    """Error relative to the fluctuating part of the truth (its deviation from its own time mean)."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b - b.mean(axis=-1, keepdims=True)))


def _period(t, x):
    """Mean period from upward zero crossings of x minus its mean (nan if fewer than three crossings)."""
    x = x - x.mean()
    up = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    if len(up) < 3:
        return float("nan")
    tc = t[up] - x[up] * (t[up + 1] - t[up]) / (x[up + 1] - x[up])
    return float(np.mean(np.diff(tc)))


def _bounded(sol, scale):
    return sol is not None and np.all(np.isfinite(sol)) and np.max(np.abs(sol)) <= scale


def least_squares_reference(prepared, cfg, t_pred, train):
    """Deterministic OpInf (finite-difference derivatives of the noisy reduced snapshots, L2 regularisation).

    The regulariser is chosen without the truth: among the values whose prediction over the whole horizon stays
    bounded (|q| <= 2 x the largest training |q|), the one with the smallest error against the noisy reduced
    training snapshots."""
    t_samp, snaps_comp = prepared.t_sampled, prepared.snapshots_comp
    scale = 2 * np.max(np.abs(snaps_comp))
    fit = t_pred <= train
    best = None
    for lam in LS_REGULARIZERS:
        model = opinf.models.ContinuousModel("cAH", solver=opinf.lstsq.L2Solver(regularizer=float(lam)))
        ddts = opinf.ddt.NonuniformFiniteDifferencer(t_samp).estimate(snaps_comp)[1]
        model.fit(states=snaps_comp, ddts=ddts)
        try:
            sol = model.predict(snaps_comp[:, 0], t_pred, method="RK45")
        except Exception:
            continue
        if sol.shape[1] != len(t_pred) or not _bounded(sol, scale):
            continue
        at_samples = np.stack([np.interp(t_samp, t_pred[fit], s) for s in sol[:, fit]])
        err = _rel(at_samples, snaps_comp)
        if best is None or err < best[1]:
            best = (float(lam), err, sol)
    return best


def summarize(result, prepared, schema, cfg, out_dir):
    tgt, score = result["eval_targets"][0], result["per_target"][0]
    basis = result["basis"]
    V = np.asarray(basis.entries)
    t = tgt.t_pred
    train = schema["window"][1]
    fit, fc = t <= train, t > train
    truth_c, X = tgt.true_comp, tgt.true_states
    out = dict(task=schema["task"], regime=schema["regime"], modes=cfg.num_modes, window=schema["window"],
               num_samples=int(len(tgt.t_sampled)), noise=schema["NOISE_LEVEL"],
               pod_energy=float(basis.cumulative_energy), runtime_s=float(result["runtime"]),
               full_order_dim=int(X.shape[0]), precision=cfg.precision,
               quad_constraint=cfg.quad_constraint, feature_shift=cfg.feature_shift)
    for key in ("train_error", "pred_error", "ci_coverage", "ci_cov_fit", "ci_cov_ext", "ci_width",
                "stability_pct", "n_stable", "n_total"):
        out[key] = float(result[key])

    proj = V @ truth_c
    out["projection_error_forecast"] = _rel(proj[:, fc], X[:, fc])
    out["projection_error_forecast_fluct"] = _rel_fluct(proj[:, fc], X[:, fc])
    lead = int(np.argmax(np.std(truth_c[:, fc], axis=1)))
    out["lead_mode"] = lead
    T_true = _period(t[fc], truth_c[lead, fc])
    D = prepared.fom_D
    out["strouhal_true"] = D / T_true
    out["amplitude_true"] = float(np.std(truth_c[lead, fc]))

    sols = np.asarray(score["rom_solves"])
    plot = dict(t=t, train=train, truth=truth_c[:PLOT_MODES], t_samp=tgt.t_sampled,
                obs=tgt.snapshots_comp[:PLOT_MODES], xs=prepared.fom_grid[0], ys=prepared.fom_grid[1],
                inside=prepared.fom_grid[2])
    frames = np.array([np.argmin(np.abs(t - (train + f * (t[-1] - train)))) for f in (0.5, 1.0)])
    plot["frame_times"] = t[frames]
    plot["frame_truth"] = X[:, frames].astype(np.float32)
    if len(sols):
        med = np.median(sols, axis=0)
        out["pred_error_fluct"] = _rel_fluct(med[:, fc], truth_c[:, fc])
        out["train_error_fluct"] = _rel_fluct(med[:, fit], truth_c[:, fit])
        full_med = V @ med
        out["full_order_error_forecast"] = _rel(full_med[:, fc], X[:, fc])
        out["full_order_error_forecast_fluct"] = _rel_fluct(full_med[:, fc], X[:, fc])
        T_med = _period(t[fc], med[lead, fc])
        out["strouhal_median"] = D / T_med
        out["amplitude_median"] = float(np.std(med[lead, fc]))
        st_draws = np.array([D / _period(t[fc], s[lead, fc]) for s in sols])
        st_draws = st_draws[np.isfinite(st_draws)]
        if len(st_draws):
            out["strouhal_draws_5_50_95"] = [float(v) for v in np.percentile(st_draws, [5, 50, 95])]
        plot.update(median=med[:PLOT_MODES], low=np.percentile(sols, 5, axis=0)[:PLOT_MODES],
                    high=np.percentile(sols, 95, axis=0)[:PLOT_MODES],
                    frame_median=full_med[:, frames].astype(np.float32))

    # Long horizon: LONG_DRAWS posterior draws integrated to LONG_FACTOR x the training window.
    t_long = np.linspace(0, LONG_FACTOR * train, int(LONG_FACTOR * train / (t[1] - t[0])) + 1)
    _, _, long_sols = generate_rom_predictions(
        samples={"O": result["O_samples"]}, rom=result["rom"], snapshots_compressed=tgt.snapshots_comp,
        time_eval=t_long, num_modes=cfg.num_modes, num_pulls=LONG_DRAWS,
        state0_samples=np.broadcast_to(tgt.state0_comp, (len(result["O_samples"]), cfg.num_modes)))
    scale = 2 * np.max(np.abs(tgt.snapshots_comp))
    n_bounded = sum(_bounded(np.asarray(s), scale) for s in long_sols)
    out["long_horizon"] = dict(factor=LONG_FACTOR, draws=LONG_DRAWS, finite=len(long_sols), bounded=int(n_bounded))

    ls = least_squares_reference(prepared, cfg, t, train)
    if ls is None:
        out["least_squares"] = dict(stable=False)
    else:
        lam, _, sol = ls
        out["least_squares"] = dict(
            stable=True, regularizer=lam, train_error=_rel(sol[:, fit], truth_c[:, fit]),
            pred_error=_rel(sol[:, fc], truth_c[:, fc]), pred_error_fluct=_rel_fluct(sol[:, fc], truth_c[:, fc]),
            strouhal=D / _period(t[fc], sol[lead, fc]), amplitude=float(np.std(sol[lead, fc])))
        plot["least_squares"] = sol[:PLOT_MODES]
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, "plotdata.npz"), **plot)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["lc", "tr"], required=True)
    ap.add_argument("--regime", choices=list(REGIMES), required=True)
    ap.add_argument("--modes", type=int, default=9)
    ap.add_argument("--precision", choices=["default", "float64"], default="default",
                    help="float64: after-the-fact rerun of a fit whose float32 SVI diverged (output suffix _f64)")
    ap.add_argument("--quad-constraint", choices=["none", "energy_preserving"], default="none",
                    help="energy_preserving: stability prior of DESIGN.md (output suffix _ep)")
    ap.add_argument("--feature-shift", choices=["none", "training_mean"], default="none",
                    help="training_mean: operator fitted about the mean training state (output suffix _shift)")
    args = ap.parse_args()
    fom = config.load_fom()
    window = config.task_window(args.task, fom)
    name = (f"{args.task}_{args.regime}_r{args.modes}" + ("_f64" if args.precision == "float64" else "")
            + ("_ep" if args.quad_constraint == "energy_preserving" else "")
            + ("_shift" if args.feature_shift == "training_mean" else ""))
    if fom["geometry"] != "cylinder":                         # e.g. pinball_re90_lc_dense_low_noise_r25
        name = f"{fom['geometry']}_re{fom['D'] / fom['nu']:.0f}_{name}"
    schema = dict(REGIMES[args.regime], name=name, task=args.task, regime=args.regime, modes=args.modes,
                  precision=args.precision, quad_constraint=args.quad_constraint, feature_shift=args.feature_shift,
                  window=tuple(float(w) for w in window),
                  num_pred_points=int(round(window[2] / np.median(np.diff(fom["t"])))) + 1)
    cfg = make_config(schema)
    spec = CylinderSpec(fom)
    prepared_holder = {}
    original = spec.prepare

    def prepare(cfg_, schema_):
        prepared = original(cfg_, schema_)
        prepared.fom_grid = (fom["xs"], fom["ys"], fom["inside"])
        prepared.fom_D = fom["D"]
        prepared_holder["p"] = prepared
        return prepared

    spec.prepare = prepare
    t0 = time.time()
    result = run_experiment(spec, cfg, schema, SCRIPT_DIR)
    out_dir = os.path.join(SCRIPT_DIR, "results", "comparison", name)
    out = summarize(result, prepared_holder["p"], schema, cfg, out_dir)
    print(json.dumps(out, indent=1))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

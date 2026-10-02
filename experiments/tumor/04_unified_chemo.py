"""
04_unified_chemo.py — Marginalised-O × Weak-Form Bayesian OpInf (Tumor + Chemo).

Thin experiment adapter over the centralised method in
``core.weakform_opinf``. Tumor growth WITH chemotherapy, an input-driven cABN
reduced model:

    dq̂/dt = ĉ + Â q̂ + B̂ α(t) + N̂ [α(t) ⊗ q̂]

The B̂ (pure-input) term is required because the POD basis is mean-centred, so
the physical forcing −α(t)·u projects to a constant-in-state input. The
quadratic Ĥ term is dropped (unidentifiable from a single trajectory). Because
α(t) is fixed data the dynamics stay linear in O, so the shared closed-form
marginalisation + SVI inference apply unchanged.

Prerequisite:
    python generate_fom_data_chemo.py

Usage:
    python 04_unified_chemo.py                  # all regimes
    python 04_unified_chemo.py dense_low_noise  # one regime
"""

import sys
import os
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import (
    ChemoReducedOrderModel,
)
from chemo_protocol import (
    TRAINING_SPAN, PREDICTION_DAYS, FOM_DATA_PATH, SCHEMAS, OUTPUT_ROOT, INPUT_AWARE_OUTPUT_ROOT,
    prepare_data, array_fingerprint, save_protocol,
)
from chemo_evaluation import evaluate_doses, write_json
from core.bayesian_opinf import generate_rom_predictions
from core.weakform_opinf.pipeline import _score, precision_context
from core.weakform_opinf import (
    WeakFormConfig, EvalTarget, PreparedRun, run_experiment, plot_standard,
)
import opinf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")

# The matched comparison uses the canonical nominal-dose cache.
CHEMO_FOM_PATH = FOM_DATA_PATH


def make_config(schema, profile="historical"):
    """Reported (production) chemo model by default (shared core defaults and the
    tumour closure constant c_gamma = 0.1, selected on development data, README); the
    input-aware model is experimental and keeps the dimensional operator prior,
    slack and weights it was studied with."""
    cfg = WeakFormConfig(
        operators="cABN",
        num_modes=4,
        bump_p=6,
        weakform_mode="ibp",
        deriv_cov="diag",
        weakform_cov="diag",
        op_prior_mode="block_hier",
        gamma2_nd=0.1,
        num_posterior_samples=500,
        regularizer=0.1,
        ic_uncertainty=True,
        ic_scale=1.0,
        num_pred_points=400,
        seed=42,
    )
    if profile == "historical":
        return cfg
    if profile == "input-aware":
        return replace(
            cfg, mll_weight=1., operator_solver="qr", precision="float64",
            gp_input_trend=True, gp_noise_prior="measurement", sigma_O=5.0,
            gamma2=0.035, weak_slack="legacy", weakform_weight=8.0)
    raise ValueError(f"Unknown chemo inference profile: {profile}")


class ChemoSpec:
    """ExperimentSpec adapter: cached TumorTwin chemo FOM → cABN ROM."""

    name = "04_unified_chemo"

    def __init__(self, data=None):
        self.data = data

    def prepare(self, cfg, schema):
        neval = cfg.num_eval_points
        data = self.data if self.data is not None else prepare_data(
            schema, seed=cfg.seed, num_modes=cfg.num_modes)
        t_full, true_states = data["t_full"], data["true_states"]
        t_samp, snaps_noisy = data["t_samp"], data["snaps_noisy"]
        ifn_jax, chemo_meta = data["input_func"], data["chemo_meta"]
        print(f"  Chemo: {len(chemo_meta['dose_days'])} doses, "
              f"sens={chemo_meta['sensitivity']:.2f}, "
              f"decay={chemo_meta['decay_rate']:.2f}")

        basis, snaps_comp, true_comp = data["basis"], data["snaps_comp"], data["true_comp"]
        print(f"  Using {cfg.num_modes} modes  "
              f"(POD energy: {basis.cumulative_energy:.4%})")

        # cABN ROM with per-block Tikhonov ridge (structure only; O marginalised).
        ncols = {"c": 1, "A": cfg.num_modes, "B": 1, "N": cfg.num_modes}
        block_reg = np.concatenate(
            [np.full(ncols[ch], cfg.regularizer) for ch in cfg.operators])
        rom = opinf.ROM(
            basis=basis,
            ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
            model=ChemoReducedOrderModel(
                operator_string=cfg.operators,
                solver=opinf.lstsq.TikhonovSolver(regularizer=np.diag(block_reg))))
        inputs_at_samp = np.array(
            [float(np.asarray(ifn_jax(t)).ravel()[0]) for t in t_samp]
        ).reshape(1, -1)
        rom.fit(states=snaps_noisy, inputs=inputs_at_samp)
        print(f"  Operator shape: {rom.model.operator_matrix.shape} ({cfg.operators})")

        # α(t) tabulated on the model's internal eval grid (training window).
        time_eval = np.linspace(float(t_samp[0]), float(t_samp[-1]), neval)
        inputs_eval = np.array(
            [float(np.asarray(ifn_jax(t)).ravel()[0]) for t in time_eval]
        ).reshape(1, -1)

        trajectory = dict(t_sampled=t_samp, snapshots_comp=snaps_comp, inputs_eval=inputs_eval)
        if cfg.gp_input_trend:
            trajectory["input_table"] = dict(times=ifn_jax.t_grid, values=ifn_jax.alpha_grid)
        if cfg.gp_noise_prior == "measurement":
            trajectory["noise_variances"] = data["noise_variances_comp"]
        trajectories = [trajectory]

        t_pred = data["t_pred"]
        eval_targets = [EvalTarget(
            t_pred=t_pred, true_comp=true_comp, true_states=true_states,
            state0_comp=snaps_comp[:, 0], t_full=t_full, input_func=ifn_jax,
            t_sampled=t_samp, snapshots_comp=snaps_comp,
            label=schema["label"])]

        alpha_pred = np.array(
            [float(np.asarray(ifn_jax(t)).ravel()[0]) for t in t_pred])
        return PreparedRun(
            rom=rom, trajectories=trajectories, basis=basis,
            eval_targets=eval_targets, training_span=TRAINING_SPAN,
            snapshots_comp=snaps_comp, t_sampled=t_samp,
            npz_fields=dict(alpha_pred=alpha_pred,
                            dose_days=np.asarray(chemo_meta["dose_days"]),
                            data_fingerprint=data["fingerprint"],
                            operators=cfg.operators),
            extra=dict(chemo_meta=chemo_meta, alpha_pred=alpha_pred,
                       t_full=t_full))

    def plot(self, result, save_dir=None):
        dose_days = result["extra"]["chemo_meta"]["dose_days"]
        plot_standard(result, save_dir or FIGURE_DIR,
                      prefix=f"04_chemo_{result['schema']['name']}",
                      dose_days=dose_days)


def run_matched(schema, data=None, out_dir=None, profile="historical"):
    data = prepare_data(schema) if data is None else data
    cfg, spec = make_config(schema, profile), ChemoSpec(data)
    default_root = OUTPUT_ROOT if profile == "historical" else INPUT_AWARE_OUTPUT_ROOT
    out_dir = Path(out_dir or Path(default_root) / schema["name"])
    save_protocol(data, out_dir)
    path = out_dir / "bayesian_fit.npz"
    meta_path = out_dir / "bayesian_fit.json"
    metadata = dict(data_fingerprint=data["fingerprint"], config=asdict(cfg), profile=profile)
    if cfg.gp_noise_prior == "measurement":
        metadata["noise_variances"] = data["noise_variances_comp"].tolist()
    stored = None
    if path.exists() and meta_path.exists():
        with meta_path.open() as stream:
            stored = json.load(stream)
        stored_config = dict(stored["config"])
        for name, default in (("operator_solver", "normal"), ("gp_input_trend", False),
                              ("gp_noise_prior", "spectrum"), ("precision", "default"),
                              ("weak_slack", "legacy"),
                              # Checkpoints without the field fixed gamma2, so gamma2_nd was unused.
                              ("gamma2_nd", cfg.gamma2_nd)):
            stored_config.setdefault(name, default)
        if (stored.get("data_fingerprint") != metadata["data_fingerprint"]
                or stored_config != metadata["config"]
                or stored.get("profile", "historical") != profile
                or (cfg.gp_noise_prior == "measurement"
                    and stored.get("noise_variances") != metadata["noise_variances"])):
            raise ValueError(f"Bayesian checkpoint does not match the protocol: {path}")
        prepared = spec.prepare(cfg, schema)
        with np.load(path, allow_pickle=False) as cached:
            operators = cached["O_samples"]
            solves = cached["rom_solves"]
            selected = operators[np.linspace(0, len(operators) - 1, min(200, len(operators)), dtype=int)]
            score = _score(prepared.eval_targets[0], solves, selected, TRAINING_SPAN)
            score["state0_samples"] = cached["state0_samples"]
            result = dict(
                schema=schema, cfg=cfg, O_samples=operators, losses=cached["losses"],
                runtime=float(cached["runtime"]), basis=data["basis"], rom=prepared.rom,
                training_span=TRAINING_SPAN, num_modes=cfg.num_modes,
                eval_targets=prepared.eval_targets, per_target=[score],
                extra=prepared.extra, **{k: v for k, v in score.items()
                                       if k not in ("rom_solves", "state0_samples", "t_pred")})
            for key in ("gp_lengthscale_samples", "gp_variance_samples", "gp_noise_samples"):
                if key in cached:
                    result[key] = cached[key]
        print(f"Reusing Bayesian fit: {path}", flush=True)
    else:
        result = run_experiment(spec, cfg, schema, SCRIPT_DIR, save=False)
        score = result["per_target"][0]
        np.savez_compressed(
            path, O_samples=result["O_samples"], state0_samples=score["state0_samples"],
            rom_solves=score["rom_solves"], losses=result["losses"],
            runtime=result["runtime"], t_pred=data["t_pred"],
            gp_lengthscale_samples=result["gp_lengthscale_samples"],
            gp_variance_samples=result["gp_variance_samples"],
            gp_noise_samples=result["gp_noise_samples"],
            t_samp=data["t_samp"], snaps_comp=data["snaps_comp"],
            basis_entries=data["basis"].entries, basis_shift=data["basis"].shift_)
    result["data"] = data
    result["profile"] = profile
    result["output_dir"] = out_dir
    result["model_id"] = array_fingerprint(
        result["O_samples"], result["per_target"][0]["state0_samples"])
    if stored is not None and stored["model_id"] != result["model_id"]:
        raise ValueError(f"Bayesian checkpoint arrays do not match their fingerprint: {path}")
    if stored is None:
        write_json(meta_path, dict(metadata, model_id=result["model_id"]))
    return result


def evaluate_dose_variation(result, out_dir=None):
    data = result["data"]
    score = result["per_target"][0]

    def predict(scale, input_func):
        with precision_context(result["cfg"]):
            _, _, solves = generate_rom_predictions(
                {"O": result["O_samples"]}, result["rom"], data["snaps_comp"],
                data["t_pred"], result["num_modes"], num_pulls=score["n_total"],
                input_func=input_func, state0_samples=score["state0_samples"])
        if len(solves) == 0:
            return np.empty((0, result["num_modes"], len(data["t_pred"])))
        return solves

    return evaluate_doses(
        data, predict, "04_unified_chemo",
        out_dir or result.get("output_dir", Path(OUTPUT_ROOT) / result["schema"]["name"]),
        nominal_solves=score["rom_solves"], n_total=score["n_total"],
        model_id=result["model_id"])


def main(schema_names=None, dose_variation=False, profile="historical"):
    schemas = SCHEMAS if not schema_names else [
        s for s in SCHEMAS if s["name"] in schema_names]
    if not schemas:
        print(f"Unknown schema(s): {schema_names}")
        print(f"Available: {[s['name'] for s in SCHEMAS]}")
        return

    print("=" * 78)
    print("04_unified_chemo — Marginalised-O × Weak-Form (Tumor + Chemo)")
    print("=" * 78)

    results = []
    for schema in schemas:
        r = run_matched(schema, profile=profile)
        if dose_variation:
            evaluate_dose_variation(r)
        results.append(r)

    print(f"\n\n{'=' * 82}\nSUMMARY — Marg-O × Weak-Form (Tumor + Chemo)\n{'=' * 82}")
    print(f"{'Regime':<28s} {'Noise':>5s} {'Stab':>5s} {'Train':>8s} "
          f"{'Pred':>8s} {'CI_cov':>7s} {'Time':>6s}")
    for r in results:
        s = r["schema"]
        print(f"{s['label']:<28s} {s['NOISE_LEVEL']:>4.0%} "
              f"{r['stability_pct']:>4.0f}% {r['train_error']:>7.2%} "
              f"{r['pred_error']:>7.2%} {r['ci_coverage']:>6.1%} "
              f"{r['runtime']:>5.0f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("schemas", nargs="*")
    parser.add_argument("--dose-variation", action="store_true")
    parser.add_argument("--profile", choices=("historical", "input-aware"), default="historical")
    args = parser.parse_args()
    main(args.schemas or None, dose_variation=args.dose_variation, profile=args.profile)

"""
04_unified.py — Marginalised-O × Weak-Form Bayesian OpInf (Tumor, autonomous).

Thin experiment adapter over the centralised method in
``core.weakform_opinf``. Pure tumor growth (no treatment): an autonomous cA
reduced model learned from cached TumorTwin FOM snapshots.

GP-hyperparameter priors are spectrum-anchored and explored by SVI (no MLE and
no SNR-based POD selection — a fixed number of modes, inline with euler/burgers).

Prerequisite:
    python generate_fom_data.py

Usage:
    python 04_unified.py                  # all regimes
    python 04_unified.py dense_low_noise  # one regime
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis, load_fom_data
from core import JaxCompatibleModel
from core.weakform_opinf import (
    WeakFormConfig, EvalTarget, PreparedRun, run_experiment, plot_standard,
)
import opinf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")
TRAINING_SPAN = config.TRAINING_SPAN

SCHEMAS = [
    {"name": "dense_low_noise",    "NUM_SAMPLES": 80, "NOISE_LEVEL": 0.01,
     "NUM_EVAL_POINTS": 200, "label": "Dense data, low noise"},
    {"name": "dense_medium_noise", "NUM_SAMPLES": 80, "NOISE_LEVEL": 0.03,
     "NUM_EVAL_POINTS": 200, "label": "Dense data, medium noise"},
    {"name": "dense_high_noise",   "NUM_SAMPLES": 80, "NOISE_LEVEL": 0.05,
     "NUM_EVAL_POINTS": 200, "label": "Dense data, high noise"},
]


def make_config(schema):
    return WeakFormConfig(
        operators="cA",
        num_modes=4,
        num_eval_points=schema["NUM_EVAL_POINTS"],
        gamma2=0.035,
        deriv_weight=1.0,
        weakform_weight=8.0,
        mll_weight=0.1,
        sigma_O=5.0,
        window_size=20,
        bump_p=6,
        weakform_mode="ibp",
        deriv_cov="diag",
        weakform_cov="diag",
        op_prior_mode="block_hier",
        num_steps=12000,
        learning_rate=3e-3,
        num_posterior_samples=500,
        regularizer=1.0,
        ic_uncertainty=True,
        ic_scale=1.0,
        num_pred_points=400,
        gp_jitter_rel=1e-3,
        seed=42,
    )


class TumorSpec:
    """ExperimentSpec adapter: cached TumorTwin growth FOM → autonomous cA ROM."""

    name = "04_unified"

    def prepare(self, cfg, schema):
        noise = schema["NOISE_LEVEL"]
        nsamp = schema["NUM_SAMPLES"]
        neval = cfg.num_eval_points

        t_pred_full = np.linspace(TRAINING_SPAN[0], config.PREDICTION_DAYS, neval)
        fom, t_full, true_states, t_samp, snaps_samp = load_fom_data(
            t_pred_full, TRAINING_SPAN, nsamp, noise)

        # Fixed POD modes; basis fit on the (noisy) training snapshots.
        basis = Basis(num_vectors=cfg.num_modes)
        basis.fit(snaps_samp)
        snaps_comp = basis.compress(snaps_samp)
        true_comp = basis.compress(true_states)
        print(f"  Using {cfg.num_modes} modes  "
              f"(POD energy: {basis.cumulative_energy:.4%})")

        rom = opinf.ROM(
            basis=basis,
            ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
            model=JaxCompatibleModel(
                operators=cfg.operators,
                solver=opinf.lstsq.L2Solver(regularizer=cfg.regularizer)))
        rom.fit(states=snaps_samp)

        trajectories = [dict(t_sampled=t_samp, snapshots_comp=snaps_comp,
                             inputs_eval=None)]
        t_pred = np.linspace(TRAINING_SPAN[0], config.PREDICTION_DAYS,
                             cfg.num_pred_points)
        eval_targets = [EvalTarget(
            t_pred=t_pred, true_comp=true_comp, true_states=true_states,
            state0_comp=snaps_comp[:, 0], t_full=t_full, input_func=None,
            label=schema["label"])]

        return PreparedRun(
            rom=rom, trajectories=trajectories, basis=basis,
            eval_targets=eval_targets, training_span=TRAINING_SPAN,
            snapshots_comp=snaps_comp, t_sampled=t_samp,
            extra=dict(t_full=t_full))

    def plot(self, result, save_dir=None):
        plot_standard(result, save_dir or FIGURE_DIR,
                      prefix=f"04_{result['schema']['name']}")


def main(schema_names=None):
    spec = TumorSpec()
    schemas = SCHEMAS if not schema_names else [
        s for s in SCHEMAS if s["name"] in schema_names]
    if not schemas:
        print(f"Unknown schema(s): {schema_names}")
        print(f"Available: {[s['name'] for s in SCHEMAS]}")
        return

    print("=" * 78)
    print("04_unified — Marginalised-O × Weak-Form (Tumor, autonomous)")
    print("=" * 78)

    results = []
    for schema in schemas:
        cfg = make_config(schema)
        r = run_experiment(spec, cfg, schema, SCRIPT_DIR)
        spec.plot(r, FIGURE_DIR)
        results.append(r)

    print(f"\n\n{'=' * 82}\nSUMMARY — Marg-O × Weak-Form (Tumor)\n{'=' * 82}")
    print(f"{'Regime':<28s} {'Noise':>5s} {'Stab':>5s} {'Train':>8s} "
          f"{'Pred':>8s} {'CI_cov':>7s} {'Time':>6s}")
    for r in results:
        s = r["schema"]
        print(f"{s['label']:<28s} {s['NOISE_LEVEL']:>4.0%} "
              f"{r['stability_pct']:>4.0f}% {r['train_error']:>7.2%} "
              f"{r['pred_error']:>7.2%} {r['ci_coverage']:>6.1%} "
              f"{r['runtime']:>5.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)

"""
04_unified.py — Marginalised-O × Weak-Form Bayesian OpInf (Burgers 2D).

Thin experiment adapter over the centralised method in
``core.weakform_opinf``. Single-trajectory autonomous cAH ROM learned from a
noisy PDE solve.
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis
from core import JaxCompatibleModel
from core.utils import generate_trajectory
from core.weakform_opinf import (
    WeakFormConfig, EvalTarget, PreparedRun, run_experiment, plot_standard,
)
import opinf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")
TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 3.0)

SCHEMAS = [
    {"name": "dense_medium_noise", "NUM_SAMPLES": 60, "NOISE_LEVEL": 0.03,
     "NUM_EVAL_POINTS": 200, "label": "Dense data, medium noise"},
]


def make_config(schema):
    return WeakFormConfig(
        operators="cAH",
        num_modes=3,
        num_eval_points=schema["NUM_EVAL_POINTS"],
        gamma2=10.0,
        deriv_weight=1.0,
        weakform_weight=1.0,
        mll_weight=1.0,
        sigma_O=10.0,
        window_size=10,
        bump_p=6,
        weakform_mode="ibp",
        deriv_cov="diag",
        weakform_cov="diag",
        op_prior_mode="block_hier",
        num_steps=8000,
        learning_rate=3e-3,
        num_posterior_samples=500,
        regularizer=1.0,
        gp_jitter_rel=1e-3,
        ic_uncertainty=True,
        ic_scale=1.0,
        num_pred_points=400,
        seed=42,
    )


class BurgersSpec:
    """ExperimentSpec adapter: 2D Burgers PDE solve → autonomous cAH ROM."""

    name = "04_unified"

    def prepare(self, cfg, schema):
        noise = schema["NOISE_LEVEL"]
        nsamp = schema["NUM_SAMPLES"]
        fom, t_full, true_states, t_samp, snaps_samp = generate_trajectory(
            config, config.time_domain, TRAINING_SPAN, nsamp, noise)

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
        t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1],
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
    spec = BurgersSpec()
    schemas = SCHEMAS if not schema_names else [
        s for s in SCHEMAS if s["name"] in schema_names]
    if not schemas:
        print(f"Unknown schema(s): {schema_names}; available "
              f"{[s['name'] for s in SCHEMAS]}")
        return
    print("=" * 78)
    print("04_unified — Marginalised-O × Weak-Form (Burgers 2D)")
    print("=" * 78)
    for schema in schemas:
        cfg = make_config(schema)
        r = run_experiment(spec, cfg, schema, SCRIPT_DIR)
        spec.plot(r, FIGURE_DIR)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)

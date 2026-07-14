"""
04_unified.py — Marginalised-O × Weak-Form Bayesian OpInf (Cubic Heat, multi-IC).

Thin experiment adapter over the centralised method in
``core.weakform_opinf``. Multi-IC, input-driven cAHBN reduced model with one
shared operator learned across 5 training initial conditions and evaluated on
those plus one held-out test IC.

Uses the current-vision method throughout: zero-mean operator prior with
block-hierarchical ARD, spectrum-anchored GP-hyperparameter priors, and SVI —
no MLE, no LS operator-prior center, and no A-block stability shift (the
diagnostic showed the weak-form fit yields a naturally stable operator).
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import (
    Basis, ReducedOrderModel, input_func_factory, input_parameters,
    test_parameters,
)
from step1_generate_data import TrajectorySampler
from core.weakform_opinf import (
    WeakFormConfig, EvalTarget, PreparedRun, run_experiment, plot_standard,
)
import opinf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")
TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 2.0)
NUM_ICS = 5

SCHEMAS = [
    {"name": "sparse_low_noise",    "label": "Sparse data, low noise",
     "NUM_SAMPLES": 20, "NOISE_LEVEL": 0.01, "NUM_EVAL_POINTS": 100},
    {"name": "sparse_medium_noise", "label": "Sparse data, medium noise",
     "NUM_SAMPLES": 20, "NOISE_LEVEL": 0.03, "NUM_EVAL_POINTS": 100},
    {"name": "sparse_high_noise",   "label": "Sparse data, high noise",
     "NUM_SAMPLES": 20, "NOISE_LEVEL": 0.05, "NUM_EVAL_POINTS": 100},
]


def make_config(schema):
    return WeakFormConfig(
        operators="cAHBN",
        num_modes=5,
        num_eval_points=schema["NUM_EVAL_POINTS"],
        gamma2=0.5,
        deriv_weight=1.0,
        weakform_weight=2.0,
        mll_weight=0.1,
        sigma_O=0.5,
        window_size=20,
        bump_p=6,
        weakform_mode="ibp",
        deriv_cov="diag",
        weakform_cov="diag",
        op_prior_mode="block_hier",
        num_steps=10000,
        learning_rate=3e-3,
        num_posterior_samples=500,
        regularizer=1.0,
        gp_jitter_rel=1e-3,
        ic_uncertainty=True,
        ic_scale=1.0,
        num_pred_points=400,
        seed=42,
    )


def _np_input(params):
    """Numpy-returning input function (opinf.predict requires ndarray inputs)."""
    f = input_func_factory(params)
    return lambda t, f=f: np.asarray(f(t))


def _inputs_on_grid(t_samp, params, neval):
    te = np.linspace(float(t_samp[0]), float(t_samp[-1]), neval)
    f = input_func_factory(params)
    return np.array([np.asarray(f(t)) for t in te]).T   # (2, neval)


class HeatSpec:
    """ExperimentSpec adapter: multi-IC cubic-heat FOM → shared cAHBN ROM."""

    name = "04_unified"

    def prepare(self, cfg, schema):
        noise = schema["NOISE_LEVEL"]
        nsamp = schema["NUM_SAMPLES"]
        neval = cfg.num_eval_points
        train_params = list(input_parameters[:NUM_ICS])

        sampler = TrajectorySampler(
            training_span=TRAINING_SPAN, num_samples=nsamp, noiselevel=noise,
            num_regression_points=neval, synced=False)
        all_true, all_ts, all_snaps, _ = sampler.multisample(train_params)

        basis = Basis(num_vectors=cfg.num_modes)
        basis.fit(np.hstack(all_snaps))
        print(f"  Using {cfg.num_modes} modes  "
              f"(POD energy: {basis.cumulative_energy:.4%})")
        all_comp = [basis.compress(s) for s in all_snaps]
        all_true_comp = [basis.compress(s) for s in all_true]

        rom = opinf.ROM(
            basis=basis,
            ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(all_ts[0]),
            model=ReducedOrderModel())
        f0 = input_func_factory(train_params[0])
        rom.fit(states=all_snaps[0], inputs=np.asarray(f0(all_ts[0])))
        print(f"  Operator shape: {rom.model.operator_matrix.shape} ({cfg.operators})")

        trajectories = [
            dict(t_sampled=all_ts[ic], snapshots_comp=all_comp[ic],
                 inputs_eval=_inputs_on_grid(all_ts[ic], train_params[ic], neval))
            for ic in range(NUM_ICS)]

        # Held-out test IC (not used for training the operator).
        test_true, test_ts, test_snaps, _ = sampler.multisample([test_parameters])
        test_comp = basis.compress(test_snaps[0])
        test_true_comp = basis.compress(test_true[0])

        t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1],
                             cfg.num_pred_points)
        t_full = config.time_domain
        eval_targets = []
        for ic in range(NUM_ICS):
            eval_targets.append(EvalTarget(
                t_pred=t_pred, true_comp=all_true_comp[ic],
                true_states=all_true[ic], state0_comp=all_comp[ic][:, 0],
                t_full=t_full, input_func=_np_input(train_params[ic]),
                label=f"Train IC {train_params[ic]}"))
        eval_targets.append(EvalTarget(
            t_pred=t_pred, true_comp=test_true_comp, true_states=test_true[0],
            state0_comp=test_comp[:, 0], t_full=t_full,
            input_func=_np_input(test_parameters),
            label=f"Test IC {test_parameters}"))

        return PreparedRun(
            rom=rom, trajectories=trajectories, basis=basis,
            eval_targets=eval_targets, training_span=TRAINING_SPAN,
            snapshots_comp=all_comp[0], t_sampled=all_ts[0],
            extra=dict(t_full=t_full, num_train_ics=NUM_ICS))

    def plot(self, result, save_dir=None):
        plot_standard(result, save_dir or FIGURE_DIR,
                      prefix=f"04_{result['schema']['name']}")


def main(schema_names=None):
    spec = HeatSpec()
    schemas = SCHEMAS if not schema_names else [
        s for s in SCHEMAS if s["name"] in schema_names]
    if not schemas:
        print(f"Unknown schema(s): {schema_names}; available "
              f"{[s['name'] for s in SCHEMAS]}")
        return
    print("=" * 78)
    print("04_unified — Marginalised-O × Weak-Form (Cubic Heat, multi-IC)")
    print("=" * 78)
    results = []
    for schema in schemas:
        cfg = make_config(schema)
        r = run_experiment(spec, cfg, schema, SCRIPT_DIR)
        spec.plot(r, FIGURE_DIR)
        results.append(r)

    print(f"\n\n{'=' * 82}\nSUMMARY — Marg-O × Weak-Form (Heat, multi-IC)\n{'=' * 82}")
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

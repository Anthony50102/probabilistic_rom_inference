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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import (
    Basis, ChemoReducedOrderModel, load_chemo_fom_data, make_jax_input_func,
)
from core.weakform_opinf import WeakFormConfig, EvalTarget, PreparedRun, run_experiment
from core.plotting import plot_full_order_error
from core.diagnostics import plot_trace
import opinf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")
TRAINING_SPAN = config.TRAINING_SPAN

# Headline chemo FOM cache (matches 05_neural_ode_chemo). Override with
# CHEMO_FOM_PATH=/abs/path for dose/schedule ablations.
CHEMO_FOM_PATH = os.environ.get(
    "CHEMO_FOM_PATH",
    os.path.join(SCRIPT_DIR, "data", "TNBC_demo_001_fom_chemo_sparse5_sens0p5.npz"),
)

# ── Data regimes ─────────────────────────────────────────────────────────────
SCHEMAS = [
    {"name": "dense_low_noise",    "NUM_SAMPLES": 80, "NOISE_LEVEL": 0.01,
     "NUM_EVAL_POINTS": 200, "label": "Dense data, low noise"},
    {"name": "dense_medium_noise", "NUM_SAMPLES": 80, "NOISE_LEVEL": 0.03,
     "NUM_EVAL_POINTS": 200, "label": "Dense data, medium noise"},
    {"name": "dense_high_noise",   "NUM_SAMPLES": 80, "NOISE_LEVEL": 0.05,
     "NUM_EVAL_POINTS": 200, "label": "Dense data, high noise"},
]


def make_config(schema):
    """WeakFormConfig for the chemo experiment (autonomous-tumor hypers +
    cABN operators + weak-form weight 8)."""
    return WeakFormConfig(
        operators="cABN",
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
        regularizer=0.1,
        ic_uncertainty=True,
        ic_scale=1.0,
        num_pred_points=400,
        seed=42,
    )


class ChemoSpec:
    """ExperimentSpec adapter: cached TumorTwin chemo FOM → cABN ROM."""

    name = "04_unified_chemo"

    def prepare(self, cfg, schema):
        noise = schema["NOISE_LEVEL"]
        nsamp = schema["NUM_SAMPLES"]
        neval = cfg.num_eval_points

        t_pred_full = np.linspace(TRAINING_SPAN[0], config.PREDICTION_DAYS, neval)
        fom, t_full, true_states, t_samp, snaps_noisy, ifn, chemo_meta = \
            load_chemo_fom_data(CHEMO_FOM_PATH, t_pred_full, TRAINING_SPAN,
                                nsamp, noise, seed=cfg.seed)
        ifn_jax = make_jax_input_func(ifn, float(t_pred_full[0]),
                                      float(t_pred_full[-1]), n_points=4001)
        print(f"  Chemo: {len(chemo_meta['dose_days'])} doses, "
              f"sens={chemo_meta['sensitivity']:.2f}, "
              f"decay={chemo_meta['decay_rate']:.2f}")

        # Fixed POD modes; fit basis on CLEAN snapshots (noise-free basis).
        snaps_clean = fom.get_states(t_samp)
        basis = Basis(num_vectors=cfg.num_modes)
        basis.fit(snaps_clean)
        snaps_comp = basis.compress(snaps_noisy)
        true_comp = basis.compress(true_states)
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

        trajectories = [dict(t_sampled=t_samp, snapshots_comp=snaps_comp,
                             inputs_eval=inputs_eval)]

        t_pred = np.linspace(TRAINING_SPAN[0], config.PREDICTION_DAYS,
                             cfg.num_pred_points)
        eval_targets = [EvalTarget(
            t_pred=t_pred, true_comp=true_comp, true_states=true_states,
            state0_comp=snaps_comp[:, 0], t_full=t_full, input_func=ifn_jax,
            label=schema["label"])]

        alpha_pred = np.array(
            [float(np.asarray(ifn_jax(t)).ravel()[0]) for t in t_pred])
        return PreparedRun(
            rom=rom, trajectories=trajectories, basis=basis,
            eval_targets=eval_targets, training_span=TRAINING_SPAN,
            snapshots_comp=snaps_comp, t_sampled=t_samp,
            extra=dict(chemo_meta=chemo_meta, alpha_pred=alpha_pred,
                       t_full=t_full))

    def plot(self, result, save_dir=None):
        save_dir = save_dir or FIGURE_DIR
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"04_chemo_{result['schema']['name']}"
        tgt = result["eval_targets"][0]
        sc = result["per_target"][0]
        rom_solves = sc["rom_solves"]
        t_pred = sc["t_pred"]
        span = result["training_span"]
        nmodes = result["num_modes"]
        dose_days = np.asarray(result["extra"]["chemo_meta"]["dose_days"])

        # Reduced-coordinate ROM trajectories
        if len(rom_solves) > 0:
            rom_arr = np.array(rom_solves)
            rom_med = np.median(rom_arr, axis=0)
            q05 = np.percentile(rom_arr, 5, axis=0)
            q95 = np.percentile(rom_arr, 95, axis=0)
            ti = interp1d(tgt.t_full, tgt.true_comp, kind="cubic",
                          fill_value="extrapolate")
            truth = ti(t_pred)
            ncol = min(nmodes, 4)
            nrow = int(np.ceil(nmodes / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow),
                                     squeeze=False)
            for i in range(nmodes):
                ax = axes[i // ncol][i % ncol]
                for dd in dose_days:
                    ax.axvline(dd, color="tab:red", lw=0.5, alpha=0.15, zorder=0)
                ax.plot(t_pred, truth[i], "k-", lw=1.5, label="Truth", zorder=3)
                ax.plot(t_pred, rom_med[i], color="tab:purple", lw=1.8,
                        label="ROM median", zorder=5)
                ax.fill_between(t_pred, q05[i], q95[i], color="tab:purple",
                                alpha=0.18, label="ROM 5–95%", zorder=2)
                ax.axvline(span[1], color="gray", ls=":", lw=0.9, alpha=0.6)
                ax.set_title(f"Mode {i}")
                ax.set_xlabel("time (days)")
                if i == 0:
                    ax.legend(fontsize=7)
            for j in range(nmodes, nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            fig.suptitle(f"Chemo ROM trajectories — {result['schema']['label']}")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"{prefix}_rom_trajectories.png"),
                        dpi=200, bbox_inches="tight")
            plt.close(fig)

        # Loss
        losses = np.asarray(result["losses"])
        if losses.size > 1:
            fig_l, ax_l = plt.subplots(1, 2, figsize=(12, 4))
            ax_l[0].plot(losses, lw=0.8, color="tab:purple")
            ax_l[0].set(xlabel="step", ylabel="-ELBO", title="Loss convergence")
            half = len(losses) // 2
            ax_l[1].plot(range(half, len(losses)), losses[half:], lw=0.8,
                         color="tab:purple")
            ax_l[1].set(xlabel="step", ylabel="-ELBO", title="Loss (last 50%)")
            fig_l.tight_layout()
            fig_l.savefig(os.path.join(save_dir, f"{prefix}_loss.png"),
                          dpi=200, bbox_inches="tight")
            plt.close(fig_l)

        # Full-order error
        if len(rom_solves) > 0:
            try:
                fig_e, _ = plot_full_order_error(
                    rom_solves=np.array(rom_solves), basis=result["basis"],
                    true_states=tgt.true_states, time_domain_full=tgt.t_full,
                    time_domain_eval=t_pred, training_span=tuple(span),
                    suptitle=f"Chemo — {result['schema']['label']}")
                fig_e.savefig(os.path.join(save_dir, f"{prefix}_full_order_error.png"),
                              dpi=200, bbox_inches="tight")
                plt.close(fig_e)
            except Exception as e:
                print(f"  [plot] full_order_error skipped: {e}")

        # Operator traces
        try:
            fig_t, _ = plot_trace({"O": np.asarray(result["O_samples"])},
                                  param_name="O", n_random=6)
            fig_t.savefig(os.path.join(save_dir, f"{prefix}_operator_traces.png"),
                          dpi=200, bbox_inches="tight")
            plt.close(fig_t)
        except Exception as e:
            print(f"  [plot] operator_traces skipped: {e}")


def main(schema_names=None):
    spec = ChemoSpec()
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
        cfg = make_config(schema)
        r = run_experiment(spec, cfg, schema, SCRIPT_DIR)
        spec.plot(r, FIGURE_DIR)
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
    main(sys.argv[1:] if len(sys.argv) > 1 else None)

"""
Shared experiment utilities for model comparison.

Handles data generation, POD reduction, GP warm-start, evaluation,
diagnostic output, and result visualization.
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import opinf
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from numpyro.infer.initialization import init_to_value
from numpyro.optim import ClippedAdam
from jax import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'euler'))

from core import (
    generate_trajectory,
    JaxCompatibleModel,
    compute_gp_derivatives,
    generate_rom_predictions,
    DataScaler,
    SVIResult,
    rbf_eval,
)
from core.bayesian_opinf import (
    fit_gp_hyperparameters_mle,
    _find_operator_samples,
)
import config
from config import Basis

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str = "unnamed"
    operators: str = "cAH"
    num_modes: int = 6
    training_span: tuple = (0, 0.08)
    prediction_span: tuple = (0, 0.15)
    num_samples: int = 250
    noise_level: float = 0.15
    num_eval_points: int = 400
    use_scaled_data: bool = False
    seed: int = 42

    # Inference
    num_svi_steps: int = 10000
    learning_rate: float = 1e-3
    num_posterior_samples: int = 500

    # Model-specific (set by each model)
    gamma: float = 0.5
    gamma2: float = 10.0


@dataclass
class ExperimentData:
    """Pre-processed experiment data."""
    fom: Any
    time_domain_full: np.ndarray
    true_states: np.ndarray
    time_sampled: np.ndarray
    snapshots_sampled: np.ndarray
    basis: Any
    snapshots_comp_sampled: np.ndarray
    full_states_compressed: np.ndarray
    training_data: np.ndarray
    data_scaler: Optional[DataScaler]
    rom: Any
    # MLE warm start
    mle_Ls: np.ndarray
    mle_Vs: np.ndarray
    mle_Ns: np.ndarray
    # Least-squares operator
    O_init: np.ndarray
    time_eval: np.ndarray


def prepare_experiment(cfg: ExperimentConfig) -> ExperimentData:
    """Generate data, fit POD, MLE GPs, and least-squares operator."""
    np.random.seed(cfg.seed)

    print(f"{'='*60}")
    print(f"PREPARING: {cfg.name}")
    print(f"{'='*60}")

    # 1. Generate data
    t0 = time.time()
    (fom, time_domain_full, true_states, time_sampled, snapshots_sampled) = \
        generate_trajectory(config, config.time_domain, cfg.training_span,
                          cfg.num_samples, cfg.noise_level)
    print(f"Data generated in {time.time()-t0:.1f}s")
    print(f"  Samples: {cfg.num_samples}, Noise: {cfg.noise_level}")
    print(f"  Training span: {cfg.training_span}")

    # 2. POD basis
    basis = Basis(num_vectors=cfg.num_modes)
    basis.fit(snapshots_sampled)
    snapshots_comp_sampled = basis.compress(snapshots_sampled)
    full_states_compressed = basis.compress(true_states)
    print(f"  POD modes: {cfg.num_modes}, Energy: {basis.cumulative_energy:.4%}")

    # 3. Scaling
    if cfg.use_scaled_data:
        data_scaler = DataScaler(num_modes=cfg.num_modes)
        data_scaler.fit(snapshots_comp_sampled)
        training_data = data_scaler.transform(snapshots_comp_sampled)
    else:
        data_scaler = None
        training_data = snapshots_comp_sampled

    # 4. ROM structure
    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(time_sampled),
        model=JaxCompatibleModel(operators=cfg.operators,
                                solver=opinf.lstsq.L2Solver(regularizer=1e0))
    )
    rom.fit(states=snapshots_sampled)
    print(f"  Operator shape: {rom.model.operator_matrix.shape}")

    # 5. MLE GP warm start
    mle_Ls, mle_Vs, mle_Ns, _ = fit_gp_hyperparameters_mle(
        time_domain=time_sampled, snapshots=training_data, verbose=False)

    print(f"\n  MLE GP hyperparameters:")
    for i in range(cfg.num_modes):
        T = time_sampled[-1] - time_sampled[0]
        print(f"    Mode {i}: ℓ={mle_Ls[i]:.5f} (T/ℓ={T/mle_Ls[i]:.1f}), "
              f"σ²={mle_Vs[i]:.4f}, ν={mle_Ns[i]:.6f}")

    # 6. Least-squares operator warm start
    time_eval_ls = np.linspace(float(time_sampled[0]), float(time_sampled[-1]),
                               cfg.num_eval_points)

    mu_z_mle, _ = compute_gp_derivatives(
        Ls=mle_Ls, Vs=mle_Vs, time_train=time_sampled,
        time_eval=time_eval_ls, y_train=training_data, Ns=mle_Ns)

    X_mle = np.zeros((cfg.num_modes, cfg.num_eval_points))
    for i in range(cfg.num_modes):
        ell, sig2, nu = mle_Ls[i], mle_Vs[i], mle_Ns[i]
        K = rbf_eval(ell, sig2, time_sampled, time_sampled) + \
            (nu + 1e-5) * np.eye(len(time_sampled))
        K_star = rbf_eval(ell, sig2, time_eval_ls, time_sampled)
        X_mle[i] = K_star @ np.linalg.solve(K, training_data[i])

    X_mle_orig = X_mle
    if cfg.use_scaled_data and data_scaler is not None:
        X_mle_orig = np.array([
            X_mle[i] * data_scaler.stds_[i, 0] + data_scaler.means_[i, 0]
            for i in range(cfg.num_modes)])

    D = np.array(rom.model._assemble_data_matrix(jnp.array(X_mle_orig), inputs=None))
    dXdt_orig = np.array(mu_z_mle)
    if cfg.use_scaled_data and data_scaler is not None:
        dXdt_orig = np.array([
            mu_z_mle[i] * data_scaler.stds_[i, 0]
            for i in range(cfg.num_modes)])

    reg_lambda = 1.0 / max(cfg.gamma, 0.01)**2
    DtD = D.T @ D
    O_lstsq = np.linalg.solve(DtD + reg_lambda * np.eye(DtD.shape[0]),
                               D.T @ dXdt_orig.T)
    O_init = O_lstsq.T

    print(f"  LS operator norm: {np.linalg.norm(O_init):.4f}")
    print(f"  D matrix cond: {np.linalg.cond(D):.1f}")

    return ExperimentData(
        fom=fom,
        time_domain_full=time_domain_full,
        true_states=true_states,
        time_sampled=time_sampled,
        snapshots_sampled=snapshots_sampled,
        basis=basis,
        snapshots_comp_sampled=snapshots_comp_sampled,
        full_states_compressed=full_states_compressed,
        training_data=training_data,
        data_scaler=data_scaler,
        rom=rom,
        mle_Ls=mle_Ls,
        mle_Vs=mle_Vs,
        mle_Ns=mle_Ns,
        O_init=O_init,
        time_eval=time_eval_ls,
    )


def evaluate_results(
    cfg: ExperimentConfig,
    data: ExperimentData,
    svi_result: SVIResult,
    model_description: str = "",
) -> Dict[str, Any]:
    """Evaluate SVI results with text-based diagnostics."""
    samples = svi_result.samples
    num_modes = cfg.num_modes

    print(f"\n{'='*60}")
    print(f"RESULTS: {cfg.name}")
    if model_description:
        print(f"  {model_description}")
    print(f"{'='*60}")

    # 1. Convergence
    losses = svi_result.losses
    print(f"\n--- Convergence ---")
    print(f"  Final loss: {losses[-1]:.2f}")
    print(f"  Initial loss: {losses[0]:.2f}")
    print(f"  Loss ratio (final/initial): {losses[-1]/losses[0]:.4f}")
    # Check if converged (last 10% is within 2% of final)
    tail = losses[int(0.9*len(losses)):]
    tail_std = np.std(tail)
    tail_mean = np.mean(tail)
    print(f"  Last 10% mean: {tail_mean:.2f}, std: {tail_std:.2f} "
          f"(CoV: {tail_std/abs(tail_mean)*100:.2f}%)")

    # 2. GP hyperparameters
    print(f"\n--- GP Hyperparameters (SVI median vs MLE) ---")
    print(f"  {'Mode':>4s}  {'ℓ_SVI':>8s}  {'ℓ_MLE':>8s}  {'ℓ_drift':>8s}  "
          f"{'σ²_SVI':>8s}  {'σ²_MLE':>8s}  {'σ²_drift':>9s}  "
          f"{'ν_SVI':>10s}  {'ν_MLE':>10s}")
    gp_drifts = []
    for i in range(num_modes):
        l_key = f'lengthscale_{i}'
        v_key = f'variance_{i}'
        n_key = f'noise_{i}'

        if l_key in samples:
            L_svi = float(np.median(samples[l_key]))
            V_svi = float(np.median(samples[v_key]))
            N_svi = float(np.median(samples[n_key]))
            l_drift = (L_svi - data.mle_Ls[i]) / data.mle_Ls[i] * 100
            v_drift = (V_svi - data.mle_Vs[i]) / data.mle_Vs[i] * 100
            gp_drifts.append(abs(v_drift))
            print(f"  {i:4d}  {L_svi:8.5f}  {data.mle_Ls[i]:8.5f}  {l_drift:+7.1f}%  "
                  f"{V_svi:8.4f}  {data.mle_Vs[i]:8.4f}  {v_drift:+8.1f}%  "
                  f"{N_svi:10.6f}  {data.mle_Ns[i]:10.6f}")
        else:
            print(f"  {i:4d}  (GP hypers not sampled — fixed at MLE)")
            gp_drifts.append(0.0)

    if gp_drifts:
        mean_v_drift = np.mean(gp_drifts)
        print(f"  Mean |σ² drift|: {mean_v_drift:.1f}%")
        if mean_v_drift > 50:
            print(f"  ⚠ SIGNIFICANT GP DRIFT — possible null basin!")

    # 3. Learned γ₂
    gamma2_vals = []
    for i in range(num_modes):
        g2_key = f'gamma2_{i}'
        if g2_key in samples:
            gamma2_vals.append(float(np.median(samples[g2_key])))
    if gamma2_vals:
        print(f"\n--- Learned ODE slack γ₂ (median) ---")
        for i, g2 in enumerate(gamma2_vals):
            print(f"  Mode {i}: γ₂={g2:.4f}")
        if max(gamma2_vals) > 100:
            print(f"  ⚠ LARGE γ₂ — ODE constraint may be too loose!")

    # 4. Operator analysis
    print(f"\n--- Operator ---")
    O_samples = _find_operator_samples(samples, "O")
    if O_samples.ndim == 2:
        O_samples = O_samples[np.newaxis, ...]
    O_median = np.median(O_samples, axis=0)
    print(f"  Samples shape: {O_samples.shape}")
    print(f"  Frobenius norm (median): {np.linalg.norm(O_median):.4f}")
    print(f"  Frobenius norm (init LS): {np.linalg.norm(data.O_init):.4f}")
    print(f"  Norm ratio (SVI/LS): {np.linalg.norm(O_median)/max(np.linalg.norm(data.O_init), 1e-10):.4f}")

    # Element-wise spread
    O_std = np.std(O_samples, axis=0)
    print(f"  Mean element std: {np.mean(O_std):.6f}")
    print(f"  Max element std: {np.max(O_std):.6f}")

    # 5. ROM stability and accuracy
    print(f"\n--- ROM Predictions ---")
    time_pred = np.linspace(cfg.prediction_span[0], cfg.prediction_span[1], 400)
    Os, Xs, rom_solves = generate_rom_predictions(
        samples=samples, rom=data.rom,
        snapshots_compressed=data.snapshots_comp_sampled,
        time_eval=time_pred,
        num_modes=num_modes, num_pulls=min(200, len(O_samples)),
        data_scaler=data.data_scaler,
    )
    n_stable = len(rom_solves)
    n_total = len(Os)
    stability_pct = n_stable / max(n_total, 1) * 100
    print(f"  Stable solves: {n_stable}/{n_total} ({stability_pct:.1f}%)")

    # Accuracy on training region
    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_median = np.median(rom_arr, axis=0)

        # Find training region indices
        train_mask = time_pred <= cfg.training_span[1]
        pred_mask = time_pred > cfg.training_span[1]

        # Interpolate true solution to time_pred
        from scipy.interpolate import interp1d
        true_interp = interp1d(data.time_domain_full, data.full_states_compressed,
                               kind='cubic', fill_value='extrapolate')
        true_at_pred = true_interp(time_pred)

        train_error = np.linalg.norm(rom_median[:, train_mask] - true_at_pred[:, train_mask]) / \
                      np.linalg.norm(true_at_pred[:, train_mask])
        print(f"  Training region relative error: {train_error:.4%}")

        if np.any(pred_mask):
            pred_error = np.linalg.norm(rom_median[:, pred_mask] - true_at_pred[:, pred_mask]) / \
                        np.linalg.norm(true_at_pred[:, pred_mask])
            print(f"  Prediction region relative error: {pred_error:.4%}")

        # Per-mode errors
        print(f"  Per-mode training errors:")
        for i in range(num_modes):
            mode_err = np.linalg.norm(rom_median[i, train_mask] - true_at_pred[i, train_mask]) / \
                      max(np.linalg.norm(true_at_pred[i, train_mask]), 1e-10)
            print(f"    Mode {i}: {mode_err:.4%}")

        # Uncertainty width
        rom_q05 = np.percentile(rom_arr, 5, axis=0)
        rom_q95 = np.percentile(rom_arr, 95, axis=0)
        mean_width = np.mean(rom_q95 - rom_q05)
        mean_signal = np.mean(np.abs(rom_median))
        print(f"  Mean 90% CI width: {mean_width:.6f}")
        print(f"  Relative CI width: {mean_width/max(mean_signal, 1e-10):.4%}")
    else:
        train_error = float('inf')
        stability_pct = 0

    # 6. Summary score
    print(f"\n--- Summary ---")
    print(f"  Stability: {stability_pct:.1f}%")
    if n_stable > 0:
        print(f"  Training error: {train_error:.4%}")
    print(f"  GP drift: {mean_v_drift:.1f}%" if gp_drifts else "  GP drift: N/A")
    print(f"  Runtime: included in caller")

    return {
        'stability_pct': stability_pct,
        'n_stable': n_stable,
        'n_total': n_total,
        'train_error': train_error if n_stable > 0 else float('inf'),
        'final_loss': losses[-1],
        'mean_v_drift': mean_v_drift if gp_drifts else 0,
        'O_norm': float(np.linalg.norm(O_median)),
    }


def build_init_values(data: ExperimentData, cfg: ExperimentConfig) -> dict:
    """Build init_values dict from MLE GP hypers + LS operator."""
    init_values = {}
    for i in range(cfg.num_modes):
        init_values[f'lengthscale_{i}'] = data.mle_Ls[i]
        init_values[f'variance_{i}'] = data.mle_Vs[i]
        init_values[f'noise_{i}'] = data.mle_Ns[i]
    init_values['O'] = jnp.array(data.O_init)
    return init_values


def run_warm_start_svi(
    model,
    rng_key,
    init_values,
    model_kwargs,
    num_steps=10000,
    learning_rate=1e-3,
    num_samples=500,
    guide_class=None,
):
    """Run SVI with warm-started guide parameters (β=1.0 throughout)."""
    if guide_class is None:
        guide_class = autoguide.AutoNormal

    guide = guide_class(model, init_loc_fn=init_to_value(values=init_values))
    optimizer = ClippedAdam(step_size=learning_rate)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    betas = jnp.ones(num_steps)

    rng_key, init_key = random.split(rng_key)
    svi_state = svi.init(init_key, **model_kwargs)

    @jax.jit
    def _scan_body(svi_state, beta_val):
        svi_state, loss = svi.update(svi_state, **model_kwargs)
        return svi_state, loss

    # Run in segments for progress
    segment_size = max(1, num_steps // 10)
    all_losses = []

    for seg_idx in range(10):
        start = seg_idx * segment_size
        end = min(start + segment_size, num_steps)
        if seg_idx == 9:
            end = num_steps
        if start >= num_steps:
            break

        svi_state, seg_losses = jax.lax.scan(_scan_body, svi_state, betas[start:end])
        seg_losses_np = np.array(seg_losses)
        all_losses.extend(seg_losses_np.tolist())
        cumulative = start + (end - start)
        print(f"  step {cumulative:6d}/{num_steps}  loss={seg_losses_np[-1]:12.2f}")

    params = svi.get_params(svi_state)
    rng_key, sample_key, pred_key = random.split(rng_key, 3)

    posterior_samples = guide.sample_posterior(
        sample_key, params, sample_shape=(num_samples,), **model_kwargs)
    predictive = Predictive(
        model, posterior_samples=posterior_samples, num_samples=num_samples)
    model_output = predictive(pred_key, **model_kwargs)
    samples = {**model_output, **posterior_samples}

    return SVIResult(samples=samples, params=params, losses=all_losses)


# =============================================================================
# Plotting utilities
# =============================================================================


def plot_experiment_results(
    result: Dict[str, Any],
    cfg: Optional[ExperimentConfig] = None,
    save_dir: Optional[str] = None,
    prefix: str = "best_model",
    show: bool = False,
):
    """
    Generate and save diagnostic plots from experiment results.

    Produces three figures:
    1. ROM trajectory plot (3-column: training / prediction / full span)
       with median (dashed purple), 90% CI (shaded), observations (black
       stars), and ground truth (gray).
    2. Operator trace plot — sampled values over sample index + marginal
       histograms for selected operator matrix entries.
    3. Loss convergence plot — ELBO loss over SVI iterations.

    Parameters
    ----------
    result : dict
        Return value from ``run_experiment()`` (keys: samples, losses, O_ls,
        rom, basis, snaps_comp, true_comp, t_full, t_pred, rom_solves,
        and optionally t_samp, training_span).
    cfg : ExperimentConfig, optional
        Experiment configuration (used for spans / num_modes).
    save_dir : str, optional
        Directory for saved figures. Defaults to ``figures/`` next to calling
        script.
    prefix : str
        Filename prefix for saved figures.
    show : bool
        If True, call ``plt.show()`` after plotting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.diagnostics import plot_trace
    from scipy.interpolate import interp1d

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(save_dir, exist_ok=True)

    samples = result["samples"]
    losses = result["losses"]
    rom_solves = result["rom_solves"]
    snaps_comp = result["snaps_comp"]
    true_comp = result["true_comp"]
    t_full = result["t_full"]
    t_pred = result["t_pred"]
    t_samp = result.get("t_samp")
    training_span = result.get("training_span", (0, 0.08))
    num_modes = snaps_comp.shape[0]

    if len(rom_solves) == 0:
        print("⚠ No stable ROM solves — skipping ROM trajectory plot")
    else:
        # ── 1. ROM Trajectory Plot (3-column) ──────────────────────────
        rom_arr = np.array(rom_solves)  # (n_stable, num_modes, n_time)
        rom_med = np.median(rom_arr, axis=0)
        rom_q05 = np.percentile(rom_arr, 5, axis=0)
        rom_q95 = np.percentile(rom_arr, 95, axis=0)

        true_interp = interp1d(
            t_full, true_comp, kind="cubic", fill_value="extrapolate"
        )
        true_at_pred = true_interp(t_pred)

        train_end = training_span[1]
        train_mask = t_pred <= train_end
        pred_mask = t_pred > train_end

        t_train_win = t_pred[train_mask]
        t_pred_win = t_pred[pred_mask]

        fig, ax = plt.subplots(
            num_modes, 3, figsize=(15, 2.5 * num_modes),
            sharey="row", sharex="col",
        )
        if num_modes == 1:
            ax = ax.reshape(1, -1)

        for i in range(num_modes):
            # ── Column 0: Training window ──
            if t_samp is not None:
                ax[i, 0].plot(t_samp, snaps_comp[i], "k*", ms=3, label="Obs")
            ax[i, 0].plot(
                t_train_win, true_at_pred[i, train_mask],
                color="tab:gray", lw=1.5, label="Truth",
            )
            ax[i, 0].plot(
                t_train_win, rom_med[i, train_mask],
                color="tab:purple", ls="--", lw=2, alpha=0.9, label="Median",
            )
            ax[i, 0].fill_between(
                t_train_win,
                rom_q05[i, train_mask], rom_q95[i, train_mask],
                color="tab:purple", alpha=0.15, label="90% CI",
            )
            ax[i, 0].set_ylabel(f"Mode {i}")

            # ── Column 1: Prediction window ──
            ax[i, 1].plot(
                t_pred_win, true_at_pred[i, pred_mask],
                color="tab:gray", lw=1.5,
            )
            ax[i, 1].plot(
                t_pred_win, rom_med[i, pred_mask],
                color="tab:purple", ls="--", lw=2, alpha=0.9,
            )
            ax[i, 1].fill_between(
                t_pred_win,
                rom_q05[i, pred_mask], rom_q95[i, pred_mask],
                color="tab:purple", alpha=0.15,
            )

            # ── Column 2: Full span ──
            if t_samp is not None:
                ax[i, 2].plot(t_samp, snaps_comp[i], "k*", ms=3)
            ax[i, 2].plot(
                t_pred, true_at_pred[i],
                color="tab:gray", lw=1.5,
            )
            ax[i, 2].plot(
                t_pred, rom_med[i],
                color="tab:purple", ls="--", lw=2, alpha=0.9,
            )
            ax[i, 2].fill_between(
                t_pred,
                rom_q05[i], rom_q95[i],
                color="tab:purple", alpha=0.15,
            )
            ax[i, 2].axvline(
                train_end, color="k", ls=":", lw=0.8, alpha=0.5,
            )

            # y-limits from ground truth
            yvals = true_at_pred[i]
            ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
            pad = max(abs(ymax - ymin) * 0.3, 1e-6)
            for j in range(3):
                ax[i, j].set_ylim(ymin - pad, ymax + pad)

        ax[0, 0].set_title("Training Window")
        ax[0, 1].set_title("Prediction Window")
        ax[0, 2].set_title("Full Span")
        ax[0, 0].legend(fontsize=7, loc="upper right")
        for j in range(3):
            ax[-1, j].set_xlabel("Time")

        fig.suptitle("ROM Coefficient Trajectories", fontsize=14)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_rom_trajectories.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  📊 Saved ROM trajectory plot: {path}")
        if show:
            plt.show()
        plt.close(fig)

    # ── 2. Operator Trace Plot ─────────────────────────────────────
    try:
        fig_trace, _ = plot_trace(samples, param_name="O", n_random=6)
        path = os.path.join(save_dir, f"{prefix}_operator_traces.png")
        fig_trace.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  📊 Saved operator trace plot: {path}")
        if show:
            plt.show()
        plt.close(fig_trace)
    except Exception as e:
        print(f"  ⚠ Could not generate operator trace plot: {e}")

    # ── 3. Loss Convergence Plot ───────────────────────────────────
    fig_loss, ax_loss = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss[0].plot(losses, lw=0.8, color="tab:blue")
    ax_loss[0].set_xlabel("SVI Iteration")
    ax_loss[0].set_ylabel("ELBO Loss")
    ax_loss[0].set_title("Loss Convergence")
    ax_loss[0].grid(True, alpha=0.3)

    # Zoomed view of last 50%
    half = len(losses) // 2
    ax_loss[1].plot(range(half, len(losses)), losses[half:], lw=0.8, color="tab:blue")
    ax_loss[1].set_xlabel("SVI Iteration")
    ax_loss[1].set_ylabel("ELBO Loss")
    ax_loss[1].set_title("Loss (last 50%)")
    ax_loss[1].grid(True, alpha=0.3)

    fig_loss.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_loss.png")
    fig_loss.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  📊 Saved loss convergence plot: {path}")
    if show:
        plt.show()
    plt.close(fig_loss)

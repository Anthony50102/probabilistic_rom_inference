"""
Model: Integral Form (Model E)

Strategy: Replace derivative matching with integral form constraints.
Instead of dX/dt ≈ f(X)O^T, enforce:
    X(t_b) - X(t_a) ≈ ∫_a^b f(X(s))O^T ds

This STRUCTURALLY eliminates the null basin:
- State differences ΔX are anchored to observed data (non-zero)
- If O=0, predicted integral=0 but observed ΔX≠0 → mismatch
- Integration averages noise (smooth), unlike differentiation

Uses warm start from MLE + LS for fast convergence.
"""

import sys, os, time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import autoguide
from jax import random

from experiment_utils import (
    ExperimentConfig, prepare_experiment, evaluate_results,
    build_init_values, run_warm_start_svi,
)


def build_integral_model(rom, num_modes, time_sampled, snapshots,
                         gp_ls_prior_loc, gp_ls_prior_scale,
                         gp_var_prior_loc, gp_var_prior_scale,
                         gp_noise_prior_loc, gp_noise_prior_scale,
                         gamma2_prior_scale, num_eval_points,
                         window_size=10, data_scaler=None):
    """
    Build joint model with integral form ODE constraint.

    Instead of pointwise derivative matching, uses trapezoidal quadrature
    over non-overlapping windows to match state differences.
    """
    t_train = jnp.array(time_sampled)
    n_train = len(t_train)
    y_obs = jnp.array(snapshots)
    use_scaled = data_scaler is not None

    time_eval = jnp.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
    n_eval = len(time_eval)
    dt_eval = float(time_eval[1] - time_eval[0])

    op_shape = rom.model.operator_matrix.shape

    # Precompute distance matrices
    sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
    I_train = jnp.eye(n_train)
    sq_diffs_et = (time_eval[:, None] - t_train[None, :]) ** 2

    if use_scaled:
        scale_stds = jnp.array([data_scaler.stds_[i, 0] for i in range(num_modes)])
        scale_means = jnp.array([data_scaler.means_[i, 0] for i in range(num_modes)])

    # Precompute non-overlapping windows for integral constraints
    n_windows = n_eval // window_size
    window_starts = [i * window_size for i in range(n_windows)]
    window_ends = [(i + 1) * window_size - 1 for i in range(n_windows)]
    # Ensure last window includes remaining points
    if window_ends[-1] < n_eval - 1:
        window_ends[-1] = n_eval - 1

    # Trapezoidal quadrature weights for each window
    trap_weights = []
    for ws, we in zip(window_starts, window_ends):
        n_pts = we - ws + 1
        w = jnp.ones(n_pts) * dt_eval
        w = w.at[0].set(0.5 * dt_eval)
        w = w.at[-1].set(0.5 * dt_eval)
        trap_weights.append(w)

    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def model(gamma=1.0, gamma2=1.0, jitter=1e-4, beta=1.0):
        # γ₂ per mode
        gamma2_arr = jnp.stack([
            numpyro.sample(f"gamma2_{i}",
                          dist.LogNormal(jnp.log(gamma2), gamma2_prior_scale))
            for i in range(num_modes)])

        # GP hyperparameters
        ells = jnp.stack([
            numpyro.sample(f"lengthscale_{i}",
                          dist.LogNormal(gp_ls_prior_loc[i], gp_ls_prior_scale))
            for i in range(num_modes)])
        sig2s = jnp.stack([
            numpyro.sample(f"variance_{i}",
                          dist.LogNormal(gp_var_prior_loc[i], gp_var_prior_scale))
            for i in range(num_modes)])
        nus = jnp.stack([
            numpyro.sample(f"noise_{i}",
                          dist.LogNormal(gp_noise_prior_loc[i], gp_noise_prior_scale))
            for i in range(num_modes)])

        Xs_eval_list = []
        for i in range(num_modes):
            # GP marginal likelihood
            K = _rbf_sq(ells[i], sig2s[i], sq_diff_tt)
            K_y = K + (nus[i] + jitter) * I_train
            numpyro.sample(f"obs_{i}",
                          dist.MultivariateNormal(jnp.zeros(n_train), K_y),
                          obs=y_obs[i])

            # Recover X at train and eval points
            L_y = jnp.linalg.cholesky(K_y)
            K_inv_y = jax.scipy.linalg.cho_solve((L_y, True), y_obs[i])
            X_train_i = K @ K_inv_y
            numpyro.deterministic(f"X_{i}", X_train_i)

            K_et = _rbf_sq(ells[i], sig2s[i], sq_diffs_et)
            X_eval_i = K_et @ K_inv_y
            numpyro.deterministic(f"X_eval_{i}", X_eval_i)
            Xs_eval_list.append(X_eval_i)

        Xs_eval = jnp.stack(Xs_eval_list)

        # Operator
        O = numpyro.sample("O",
            dist.Normal(jnp.zeros(op_shape), gamma * jnp.ones(op_shape)))

        # Transform to original space
        if use_scaled:
            Xs_eval_original = Xs_eval * scale_stds[:, None] + scale_means[:, None]
        else:
            Xs_eval_original = Xs_eval

        # Compute f(X) @ O^T at all eval points
        f_Xi_all = rom.model._assemble_data_matrix(
            Xs_eval_original, inputs=None) @ O.T  # (n_eval, num_modes)
        if use_scaled:
            f_Xi_all_scaled = f_Xi_all / scale_stds[None, :]
        else:
            f_Xi_all_scaled = f_Xi_all

        # Integral form constraints
        for i in range(num_modes):
            for w_idx, (ws, we) in enumerate(zip(window_starts, window_ends)):
                # Observed state difference
                delta_X_obs = Xs_eval[i, we] - Xs_eval[i, ws]

                # Predicted integral via trapezoidal rule
                f_window = f_Xi_all_scaled[ws:we+1, i]
                delta_X_pred = jnp.sum(trap_weights[w_idx] * f_window)

                # Constraint: delta_X_obs ≈ delta_X_pred
                g2_eff = jnp.maximum(gamma2_arr[i], 1e-3)
                # Scale constraint noise by window duration for consistency
                window_duration = float(time_eval[we] - time_eval[ws])
                constraint_std = jnp.sqrt(g2_eff) * window_duration

                numpyro.factor(
                    f"integral_constraint_{i}_{w_idx}",
                    dist.Normal(delta_X_pred, constraint_std).log_prob(delta_X_obs))

    return model, np.array(time_eval)


def main():
    cfg = ExperimentConfig(
        name="Integral Form (Model E) + Warm Start",
        gamma=0.5,
        gamma2=1.0,
        num_svi_steps=10000,
        learning_rate=1e-3,
        num_posterior_samples=500,
    )

    WINDOW_SIZE = 10
    GP_LENGTHSCALE_PRIOR_SCALE = 0.25
    GP_VARIANCE_PRIOR_SCALE = 0.25
    GP_NOISE_PRIOR_SCALE = 0.25
    GAMMA2_PRIOR_SCALE = 0.5

    data = prepare_experiment(cfg)
    rng_key = random.PRNGKey(cfg.seed)

    model, time_eval = build_integral_model(
        rom=data.rom,
        num_modes=cfg.num_modes,
        time_sampled=data.time_sampled,
        snapshots=data.training_data,
        gp_ls_prior_loc=np.log(data.mle_Ls),
        gp_ls_prior_scale=GP_LENGTHSCALE_PRIOR_SCALE,
        gp_var_prior_loc=np.log(data.mle_Vs),
        gp_var_prior_scale=GP_VARIANCE_PRIOR_SCALE,
        gp_noise_prior_loc=np.log(data.mle_Ns),
        gp_noise_prior_scale=GP_NOISE_PRIOR_SCALE,
        gamma2_prior_scale=GAMMA2_PRIOR_SCALE,
        num_eval_points=cfg.num_eval_points,
        window_size=WINDOW_SIZE,
        data_scaler=data.data_scaler,
    )

    init_values = build_init_values(data, cfg)
    model_kwargs = dict(gamma=cfg.gamma, gamma2=cfg.gamma2, jitter=1e-5, beta=1.0)

    print(f"\nRunning SVI ({cfg.num_svi_steps} steps, lr={cfg.learning_rate})...")
    print(f"  Window size: {WINDOW_SIZE}")
    print(f"  Integral windows: {cfg.num_eval_points // WINDOW_SIZE}")
    t0 = time.time()

    result = run_warm_start_svi(
        model=model,
        rng_key=rng_key,
        init_values=init_values,
        model_kwargs=model_kwargs,
        num_steps=cfg.num_svi_steps,
        learning_rate=cfg.learning_rate,
        num_samples=cfg.num_posterior_samples,
    )

    runtime = time.time() - t0
    print(f"\nRuntime: {runtime:.1f}s ({runtime/60:.1f}min)")

    metrics = evaluate_results(cfg, data, result,
        model_description=f"Integral form with window_size={WINDOW_SIZE}")

    print(f"\n{'='*60}")
    print(f"DONE: {cfg.name} (runtime: {runtime:.1f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

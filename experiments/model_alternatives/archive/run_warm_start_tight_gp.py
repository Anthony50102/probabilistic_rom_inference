"""
Model: Warm Start + Tight GP Priors

Strategy: MLE warm start (like notebook 05), but with much tighter GP prior
scales to prevent σ² from collapsing. The hypothesis is that notebook 05's
GP variance collapsed (0.33 → 0.02) because the priors were too loose,
letting the GP shed signal into noise.

Key changes from notebook 05:
- GP_VARIANCE_PRIOR_SCALE = 0.15 (was 0.5) — 3x tighter
- GP_NOISE_PRIOR_SCALE = 0.15 (was 0.25) — tighter noise prior
- GP_LENGTHSCALE_PRIOR_SCALE = 0.25 (was 0.5)
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

def build_model(rom, num_modes, time_sampled, snapshots,
                gp_ls_prior_loc, gp_ls_prior_scale,
                gp_var_prior_loc, gp_var_prior_scale,
                gp_noise_prior_loc, gp_noise_prior_scale,
                gamma2_prior_scale, num_eval_points, data_scaler=None):
    """Build joint model with tight GP priors."""
    t_train = jnp.array(time_sampled)
    n_train = len(t_train)
    y_obs = jnp.array(snapshots)
    use_scaled = data_scaler is not None

    time_eval = jnp.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
    n_eval = len(time_eval)

    op_shape = rom.model.operator_matrix.shape

    # Precompute distance matrices
    sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
    I_train = jnp.eye(n_train)
    I_eval = jnp.eye(n_eval)
    diffs_et = time_eval[:, None] - t_train[None, :]
    sq_diffs_et = diffs_et ** 2
    sq_diffs_ee = (time_eval[:, None] - time_eval[None, :]) ** 2

    if use_scaled:
        scale_stds = jnp.array([data_scaler.stds_[i, 0] for i in range(num_modes)])
        scale_means = jnp.array([data_scaler.means_[i, 0] for i in range(num_modes)])

    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def model(gamma=1.0, gamma2=1.0, jitter=1e-4, beta=1.0):
        # γ₂ per mode
        gamma2_arr = jnp.stack([
            numpyro.sample(f"gamma2_{i}",
                          dist.LogNormal(jnp.log(gamma2), gamma2_prior_scale))
            for i in range(num_modes)])

        # GP hyperparameters with TIGHT priors
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

        # GP marginal likelihood
        for i in range(num_modes):
            K = _rbf_sq(ells[i], sig2s[i], sq_diff_tt)
            K_y = K + (nus[i] + jitter) * I_train
            numpyro.sample(f"obs_{i}",
                          dist.MultivariateNormal(jnp.zeros(n_train), K_y),
                          obs=y_obs[i])

            # Recover X via conditional mean
            L_y = jnp.linalg.cholesky(K_y)
            K_inv_y = jax.scipy.linalg.cho_solve((L_y, True), y_obs[i])
            X_i = K @ K_inv_y
            numpyro.deterministic(f"X_{i}", X_i)

            # Interpolation + derivatives at eval points
            ell2 = ells[i] ** 2
            K_et = _rbf_sq(ells[i], sig2s[i], sq_diffs_et)
            X_eval_i = K_et @ K_inv_y
            numpyro.deterministic(f"X_eval_{i}", X_eval_i)

            K_zy = -(diffs_et / ell2) * K_et
            mu_z_i = K_zy @ K_inv_y
            numpyro.deterministic(f"mu_z_{i}", mu_z_i)

            K_ee = _rbf_sq(ells[i], sig2s[i], sq_diffs_ee)
            K_zz = ((1.0 - sq_diffs_ee / ell2) / ell2) * K_ee
            V = jax.scipy.linalg.solve_triangular(L_y, K_zy.T, lower=True)
            A_i = K_zz - V.T @ V
            A_i = 0.5 * (A_i + A_i.T)

            # Store for ODE constraint
            if i == 0:
                Xs_eval = [X_eval_i]
                mu_zs = [mu_z_i]
                As = [A_i]
            else:
                Xs_eval.append(X_eval_i)
                mu_zs.append(mu_z_i)
                As.append(A_i)

        Xs_eval = jnp.stack(Xs_eval)
        mu_zs = jnp.stack(mu_zs)

        # Operator
        O = numpyro.sample("O",
            dist.Normal(jnp.zeros(op_shape), gamma * jnp.ones(op_shape)))

        # Transform to original space
        if use_scaled:
            Xs_eval_original = Xs_eval * scale_stds[:, None] + scale_means[:, None]
        else:
            Xs_eval_original = Xs_eval

        # Operator dynamics
        f_Xi = rom.model._assemble_data_matrix(
            Xs_eval_original, inputs=None) @ O.T
        if use_scaled:
            f_Xi_scaled = f_Xi.T / scale_stds[:, None]
        else:
            f_Xi_scaled = f_Xi.T

        # ODE constraints — diagonal approximation
        for i in range(num_modes):
            g2_eff = jnp.maximum(gamma2_arr[i], 1e-2) + jitter
            deriv_var = jnp.maximum(jnp.diag(As[i]), 0.0) + g2_eff
            ode_dist = dist.Normal(loc=f_Xi_scaled[i], scale=jnp.sqrt(deriv_var))
            numpyro.factor(f"ode_constraint_{i}",
                          jnp.sum(ode_dist.log_prob(mu_zs[i])))

    return model, np.array(time_eval)


def main():
    cfg = ExperimentConfig(
        name="Warm Start + Tight GP Priors",
        gamma=0.5,
        gamma2=10.0,
        num_svi_steps=10000,
        learning_rate=1e-3,
        num_posterior_samples=500,
    )

    # Tight GP prior scales (the key innovation)
    GP_LENGTHSCALE_PRIOR_SCALE = 0.25
    GP_VARIANCE_PRIOR_SCALE = 0.15
    GP_NOISE_PRIOR_SCALE = 0.15
    GAMMA2_PRIOR_SCALE = 0.1

    data = prepare_experiment(cfg)
    rng_key = random.PRNGKey(cfg.seed)

    model, time_eval = build_model(
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
        data_scaler=data.data_scaler,
    )

    init_values = build_init_values(data, cfg)
    model_kwargs = dict(gamma=cfg.gamma, gamma2=cfg.gamma2, jitter=1e-5, beta=1.0)

    print(f"\nRunning SVI ({cfg.num_svi_steps} steps, lr={cfg.learning_rate})...")
    print(f"  GP prior scales: ℓ={GP_LENGTHSCALE_PRIOR_SCALE}, "
          f"σ²={GP_VARIANCE_PRIOR_SCALE}, ν={GP_NOISE_PRIOR_SCALE}")
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
        model_description=f"Tight GP priors: σ²_scale={GP_VARIANCE_PRIOR_SCALE}, "
                         f"ν_scale={GP_NOISE_PRIOR_SCALE}")

    print(f"\n{'='*60}")
    print(f"DONE: {cfg.name} (runtime: {runtime:.1f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

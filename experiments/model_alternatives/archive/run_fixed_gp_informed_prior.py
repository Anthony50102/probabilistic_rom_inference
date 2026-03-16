"""
Model: Fixed GP + Informative Operator Prior

Strategy: The most practical middle-ground between 2-stage and joint.
- Fix GP hyperparameters at MLE values (no sampling — eliminates null basin)
- Use LS operator as informative prior mean (not zero-centered)
- Learn operator via SVI with derivative matching
- γ acts as a trust parameter: how far from LS solution can we go?

This is essentially the 2-stage model but done cleanly in a single SVI,
with the operator prior centered on the LS estimate instead of zero.
Theoretically, this can be argued as "conditioning on the GP posterior"
which is a valid Bayesian decomposition.

Key advantage: no null basin possible (GP is fixed, can't absorb signal).
"""

import sys, os, time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from numpyro.infer.initialization import init_to_value
from numpyro.optim import ClippedAdam
from jax import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'euler'))

from core import (
    compute_gp_derivatives,
    SVIResult,
    rbf_eval,
)
from core.bayesian_opinf import _find_operator_samples

from experiment_utils import (
    ExperimentConfig, prepare_experiment, evaluate_results,
)


def build_fixed_gp_model(rom, num_modes, time_sampled, training_data,
                         mle_Ls, mle_Vs, mle_Ns, O_prior,
                         num_eval_points, data_scaler=None):
    """
    Build model with fixed GP hyperparameters and informative operator prior.

    The GP is not sampled at all — hyperparameters are fixed at MLE.
    Only the operator O is learned via SVI.
    """
    t_train = jnp.array(time_sampled)
    use_scaled = data_scaler is not None
    op_shape = O_prior.shape

    time_eval = jnp.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
    n_eval = len(time_eval)

    if use_scaled:
        scale_stds = jnp.array([data_scaler.stds_[i, 0] for i in range(num_modes)])
        scale_means = jnp.array([data_scaler.means_[i, 0] for i in range(num_modes)])

    # Precompute GP predictions and derivatives (fixed, not sampled)
    X_eval_fixed = np.zeros((num_modes, n_eval))
    mu_z_fixed = np.zeros((num_modes, n_eval))
    deriv_var_fixed = np.zeros((num_modes, n_eval))

    for i in range(num_modes):
        ell, sig2, nu = mle_Ls[i], mle_Vs[i], mle_Ns[i]
        ell2 = ell**2

        # K matrices
        K = rbf_eval(ell, sig2, time_sampled, time_sampled) + (nu + 1e-5) * np.eye(len(time_sampled))
        K_star = rbf_eval(ell, sig2, time_eval, time_sampled)
        L_y = np.linalg.cholesky(K)
        K_inv_y = np.linalg.solve(K, training_data[i])

        # State predictions
        X_eval_fixed[i] = K_star @ K_inv_y

        # Derivative predictions
        diff_zy = np.array(time_eval)[:, None] - np.array(time_sampled)[None, :]
        K_zy = -(diff_zy / ell2) * np.array(K_star)
        mu_z_fixed[i] = K_zy @ K_inv_y

        # Derivative variance (diagonal only)
        sq_diffs_ee = (np.array(time_eval)[:, None] - np.array(time_eval)[None, :]) ** 2
        K_ee = np.array(rbf_eval(ell, sig2, time_eval, time_eval))
        K_zz = ((1 - sq_diffs_ee / ell2) / ell2) * K_ee
        V = np.linalg.solve(L_y, K_zy.T)
        A = K_zz - V.T @ V
        deriv_var_fixed[i] = np.maximum(np.diag(A), 0.0)

    X_eval_jnp = jnp.array(X_eval_fixed)
    mu_z_jnp = jnp.array(mu_z_fixed)
    deriv_var_jnp = jnp.array(deriv_var_fixed)
    O_prior_jnp = jnp.array(O_prior)

    print(f"  Precomputed GP at {n_eval} eval points")
    print(f"  Derivative variance range: [{deriv_var_fixed.min():.6f}, {deriv_var_fixed.max():.6f}]")

    def model(gamma=1.0, gamma2=1.0, jitter=1e-4, beta=1.0):
        # Operator with informative prior centered on LS estimate
        # Use relative scaling: larger entries get proportionally more room
        prior_scale = gamma * jnp.maximum(jnp.abs(O_prior_jnp), 0.5)
        O = numpyro.sample("O", dist.Normal(O_prior_jnp, prior_scale))

        # Per-mode γ₂
        gamma2_arr = jnp.stack([
            numpyro.sample(f"gamma2_{i}", dist.LogNormal(jnp.log(gamma2), 0.5))
            for i in range(num_modes)])

        # Store fixed GP values as deterministic for evaluation
        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", X_eval_jnp[i])
            numpyro.deterministic(f"X_eval_{i}", X_eval_jnp[i])

        # Compute operator dynamics
        if use_scaled:
            Xs_orig = X_eval_jnp * scale_stds[:, None] + scale_means[:, None]
        else:
            Xs_orig = X_eval_jnp

        f_Xi = rom.model._assemble_data_matrix(Xs_orig, inputs=None) @ O.T
        if use_scaled:
            f_Xi_scaled = f_Xi.T / scale_stds[:, None]
        else:
            f_Xi_scaled = f_Xi.T

        # ODE constraints
        for i in range(num_modes):
            g2_eff = jnp.maximum(gamma2_arr[i], 1e-3) + jitter
            total_var = deriv_var_jnp[i] + g2_eff
            ode_dist = dist.Normal(loc=f_Xi_scaled[i], scale=jnp.sqrt(total_var))
            numpyro.factor(f"ode_constraint_{i}",
                          jnp.sum(ode_dist.log_prob(mu_z_jnp[i])))

    return model, np.array(time_eval)


def main():
    cfg = ExperimentConfig(
        name="Fixed GP + Informative Operator Prior",
        gamma=1.0,
        gamma2=1.0,
        num_svi_steps=5000,
        learning_rate=5e-3,
        num_posterior_samples=500,
    )

    data = prepare_experiment(cfg)
    rng_key = random.PRNGKey(cfg.seed)

    model, time_eval = build_fixed_gp_model(
        rom=data.rom,
        num_modes=cfg.num_modes,
        time_sampled=data.time_sampled,
        training_data=data.training_data,
        mle_Ls=data.mle_Ls,
        mle_Vs=data.mle_Vs,
        mle_Ns=data.mle_Ns,
        O_prior=data.O_init,
        num_eval_points=cfg.num_eval_points,
        data_scaler=data.data_scaler,
    )

    # Init at LS operator
    init_values = {'O': jnp.array(data.O_init)}
    model_kwargs = dict(gamma=cfg.gamma, gamma2=cfg.gamma2, jitter=1e-5, beta=1.0)

    guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    optimizer = ClippedAdam(step_size=cfg.learning_rate)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    print(f"\nRunning SVI ({cfg.num_svi_steps} steps, lr={cfg.learning_rate})...")
    print(f"  Fixed GP, only learning O + γ₂")
    t0 = time.time()

    rng_key, init_key = random.split(rng_key)
    svi_state = svi.init(init_key, **model_kwargs)

    betas = jnp.ones(cfg.num_svi_steps)

    @jax.jit
    def _scan_body(svi_state, beta_val):
        svi_state, loss = svi.update(svi_state, **model_kwargs)
        return svi_state, loss

    segment_size = max(1, cfg.num_svi_steps // 10)
    all_losses = []
    for seg_idx in range(10):
        start = seg_idx * segment_size
        end = min(start + segment_size, cfg.num_svi_steps)
        if seg_idx == 9:
            end = cfg.num_svi_steps
        if start >= cfg.num_svi_steps:
            break
        svi_state, seg_losses = jax.lax.scan(_scan_body, svi_state, betas[start:end])
        seg_losses_np = np.array(seg_losses)
        all_losses.extend(seg_losses_np.tolist())
        print(f"  step {end:6d}/{cfg.num_svi_steps}  loss={seg_losses_np[-1]:12.2f}")

    params = svi.get_params(svi_state)
    rng_key, sample_key, pred_key = random.split(rng_key, 3)
    posterior_samples = guide.sample_posterior(
        sample_key, params, sample_shape=(cfg.num_posterior_samples,), **model_kwargs)
    predictive = Predictive(model, posterior_samples=posterior_samples,
                           num_samples=cfg.num_posterior_samples)
    model_output = predictive(pred_key, **model_kwargs)
    samples = {**model_output, **posterior_samples}

    result = SVIResult(samples=samples, params=params, losses=all_losses)

    runtime = time.time() - t0
    print(f"\nRuntime: {runtime:.1f}s ({runtime/60:.1f}min)")

    metrics = evaluate_results(cfg, data, result,
        model_description="Fixed GP (MLE), informative O prior (LS mean)")

    print(f"\n{'='*60}")
    print(f"DONE: {cfg.name} (runtime: {runtime:.1f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

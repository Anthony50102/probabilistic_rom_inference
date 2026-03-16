"""
Model: Fixed GP + Informative Prior — Hyperparameter Sweep

Fast sweep over γ (operator prior scale) and γ₂ (ODE constraint slack)
to find the best configuration for the fixed-GP approach.
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
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'euler'))

from core import compute_gp_derivatives, SVIResult, rbf_eval
from core.bayesian_opinf import _find_operator_samples, generate_rom_predictions

from experiment_utils import ExperimentConfig, prepare_experiment


def build_and_run_fixed_gp(data, cfg, gamma, gamma2, num_steps=5000,
                           lr=5e-3, num_samples=200, rng_key=None):
    """Build and run fixed GP model with given hyperparameters. Returns metrics dict."""
    num_modes = cfg.num_modes
    use_scaled = data.data_scaler is not None
    op_shape = data.O_init.shape
    time_eval = data.time_eval
    n_eval = len(time_eval)

    # Precompute GP predictions and derivatives
    X_eval_fixed = np.zeros((num_modes, n_eval))
    mu_z_fixed = np.zeros((num_modes, n_eval))
    deriv_var_fixed = np.zeros((num_modes, n_eval))

    for i in range(num_modes):
        ell, sig2, nu = data.mle_Ls[i], data.mle_Vs[i], data.mle_Ns[i]
        ell2 = ell**2
        K = rbf_eval(ell, sig2, data.time_sampled, data.time_sampled) + \
            (nu + 1e-5) * np.eye(len(data.time_sampled))
        K_star = rbf_eval(ell, sig2, time_eval, data.time_sampled)
        L_y = np.linalg.cholesky(K)
        K_inv_y = np.linalg.solve(K, data.training_data[i])
        X_eval_fixed[i] = K_star @ K_inv_y

        diff_zy = np.array(time_eval)[:, None] - np.array(data.time_sampled)[None, :]
        K_zy = -(diff_zy / ell2) * np.array(K_star)
        mu_z_fixed[i] = K_zy @ K_inv_y

        sq_diffs_ee = (np.array(time_eval)[:, None] - np.array(time_eval)[None, :]) ** 2
        K_ee = np.array(rbf_eval(ell, sig2, time_eval, time_eval))
        K_zz = ((1 - sq_diffs_ee / ell2) / ell2) * K_ee
        V = np.linalg.solve(L_y, K_zy.T)
        A = K_zz - V.T @ V
        deriv_var_fixed[i] = np.maximum(np.diag(A), 0.0)

    X_eval_jnp = jnp.array(X_eval_fixed)
    mu_z_jnp = jnp.array(mu_z_fixed)
    deriv_var_jnp = jnp.array(deriv_var_fixed)
    O_prior_jnp = jnp.array(data.O_init)

    def model(gamma_=1.0, gamma2_=1.0, jitter=1e-4):
        prior_scale = gamma_ * jnp.maximum(jnp.abs(O_prior_jnp), 0.5)
        O = numpyro.sample("O", dist.Normal(O_prior_jnp, prior_scale))

        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", X_eval_jnp[i])
            numpyro.deterministic(f"X_eval_{i}", X_eval_jnp[i])

        if use_scaled:
            stds = jnp.array([data.data_scaler.stds_[i, 0] for i in range(num_modes)])
            means = jnp.array([data.data_scaler.means_[i, 0] for i in range(num_modes)])
            Xs_orig = X_eval_jnp * stds[:, None] + means[:, None]
        else:
            Xs_orig = X_eval_jnp

        f_Xi = data.rom.model._assemble_data_matrix(Xs_orig, inputs=None) @ O.T
        if use_scaled:
            f_Xi_scaled = f_Xi.T / stds[:, None]
        else:
            f_Xi_scaled = f_Xi.T

        for i in range(num_modes):
            total_var = deriv_var_jnp[i] + gamma2_ + jitter
            ode_dist = dist.Normal(loc=f_Xi_scaled[i], scale=jnp.sqrt(total_var))
            numpyro.factor(f"ode_constraint_{i}",
                          jnp.sum(ode_dist.log_prob(mu_z_jnp[i])))

    if rng_key is None:
        rng_key = random.PRNGKey(42)

    init_values = {'O': jnp.array(data.O_init)}
    model_kwargs = dict(gamma_=gamma, gamma2_=gamma2, jitter=1e-5)

    guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    optimizer = ClippedAdam(step_size=lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key, init_key = random.split(rng_key)
    svi_state = svi.init(init_key, **model_kwargs)

    @jax.jit
    def _step(svi_state, _):
        svi_state, loss = svi.update(svi_state, **model_kwargs)
        return svi_state, loss

    svi_state, losses = jax.lax.scan(_step, svi_state, jnp.arange(num_steps))
    losses = np.array(losses)

    params = svi.get_params(svi_state)
    rng_key, sk, pk = random.split(rng_key, 3)
    posterior = guide.sample_posterior(sk, params, sample_shape=(num_samples,), **model_kwargs)
    pred = Predictive(model, posterior_samples=posterior, num_samples=num_samples)
    out = pred(pk, **model_kwargs)
    samples = {**out, **posterior}

    # Evaluate
    time_pred = np.linspace(cfg.prediction_span[0], cfg.prediction_span[1], 400)
    Os, Xs, rom_solves = generate_rom_predictions(
        samples=samples, rom=data.rom,
        snapshots_compressed=data.snapshots_comp_sampled,
        time_eval=time_pred, num_modes=num_modes, num_pulls=min(100, num_samples),
        data_scaler=data.data_scaler)

    n_stable = len(rom_solves)
    stability_pct = n_stable / max(len(Os), 1) * 100

    train_error = float('inf')
    pred_error = float('inf')
    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_median = np.median(rom_arr, axis=0)
        train_mask = time_pred <= cfg.training_span[1]
        pred_mask = time_pred > cfg.training_span[1]

        from scipy.interpolate import interp1d
        true_interp = interp1d(data.time_domain_full, data.full_states_compressed,
                               kind='cubic', fill_value='extrapolate')
        true_at_pred = true_interp(time_pred)

        train_error = float(np.linalg.norm(rom_median[:, train_mask] - true_at_pred[:, train_mask]) /
                           np.linalg.norm(true_at_pred[:, train_mask]))
        if np.any(pred_mask):
            pred_error = float(np.linalg.norm(rom_median[:, pred_mask] - true_at_pred[:, pred_mask]) /
                              np.linalg.norm(true_at_pred[:, pred_mask]))

    return {
        'gamma': gamma, 'gamma2': gamma2,
        'stability_pct': stability_pct, 'n_stable': n_stable,
        'train_error': train_error, 'pred_error': pred_error,
        'final_loss': float(losses[-1]),
        'O_norm': float(np.linalg.norm(np.median(_find_operator_samples(samples, "O"), axis=0))),
    }


def main():
    cfg = ExperimentConfig(name="Fixed GP Sweep")
    data = prepare_experiment(cfg)

    # Sweep grid
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0]
    gamma2s = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

    print(f"\n{'='*80}")
    print(f"SWEEP: {len(gammas)} x {len(gamma2s)} = {len(gammas)*len(gamma2s)} configurations")
    print(f"{'='*80}")

    results = []
    for g, g2 in product(gammas, gamma2s):
        t0 = time.time()
        r = build_and_run_fixed_gp(data, cfg, gamma=g, gamma2=g2,
                                    num_steps=5000, lr=5e-3, num_samples=200)
        dt = time.time() - t0
        r['runtime'] = dt
        results.append(r)
        print(f"  γ={g:5.2f}  γ₂={g2:6.3f}  →  "
              f"stab={r['stability_pct']:5.1f}%  "
              f"train_err={r['train_error']:7.2%}  "
              f"pred_err={r['pred_error']:7.2%}  "
              f"loss={r['final_loss']:10.1f}  "
              f"O_norm={r['O_norm']:8.1f}  "
              f"({dt:.1f}s)")

    # Find best
    print(f"\n{'='*80}")
    print("TOP 5 BY TRAINING ERROR (stable only):")
    print(f"{'='*80}")
    stable = [r for r in results if r['stability_pct'] > 90]
    stable.sort(key=lambda r: r['train_error'])
    for i, r in enumerate(stable[:5]):
        print(f"  #{i+1}: γ={r['gamma']:.2f}, γ₂={r['gamma2']:.3f}  "
              f"train={r['train_error']:.2%}  pred={r['pred_error']:.2%}  "
              f"stab={r['stability_pct']:.0f}%  O_norm={r['O_norm']:.1f}")

    print(f"\nTOP 5 BY PREDICTION ERROR (stable only):")
    stable.sort(key=lambda r: r['pred_error'])
    for i, r in enumerate(stable[:5]):
        print(f"  #{i+1}: γ={r['gamma']:.2f}, γ₂={r['gamma2']:.3f}  "
              f"train={r['train_error']:.2%}  pred={r['pred_error']:.2%}  "
              f"stab={r['stability_pct']:.0f}%  O_norm={r['O_norm']:.1f}")


if __name__ == "__main__":
    main()

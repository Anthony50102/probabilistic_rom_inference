"""
Comprehensive Model Comparison

Runs all model variants on the Euler problem with the notebook-02 data settings
(250 samples, 3% noise) and compares results.

Models:
A. Fixed GP + Informative Prior (practical baseline)
B. Fixed GP + Full Covariance ODE constraint
C. Integral Form + Warm Start
D. Warm Start Joint + Tight GP (marginal likelihood)
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
    generate_trajectory, JaxCompatibleModel, compute_gp_derivatives,
    generate_rom_predictions, DataScaler, SVIResult, rbf_eval,
)
from core.bayesian_opinf import (
    fit_gp_hyperparameters_mle, _find_operator_samples,
    build_bayesian_opinf_model, grid_search_prior_operator,
    run_svi as core_run_svi,
)
import config
from config import Basis
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)


# ============================================================
# Data preparation
# ============================================================

def prepare_data(noise_level=0.03, num_samples=250, num_modes=6, seed=42):
    """Generate Euler data matching notebook-02 settings."""
    np.random.seed(seed)
    TRAINING_SPAN = (0, 0.08)
    PREDICTION_SPAN = (0, 0.15)
    NUM_EVAL_POINTS = 400

    print(f"{'='*70}")
    print(f"DATA: noise={noise_level}, samples={num_samples}, modes={num_modes}")
    print(f"{'='*70}")

    (fom, t_full, true_states, t_samp, snaps_samp) = \
        generate_trajectory(config, config.time_domain, TRAINING_SPAN, num_samples, noise_level)

    basis = Basis(num_vectors=num_modes)
    basis.fit(snaps_samp)
    snaps_comp = basis.compress(snaps_samp)
    true_comp = basis.compress(true_states)
    print(f"POD modes: {num_modes}, Energy: {basis.cumulative_energy:.4%}")

    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
        model=JaxCompatibleModel(operators="cAH", solver=opinf.lstsq.L2Solver(regularizer=1e0)),
    )
    rom.fit(states=snaps_samp)

    # MLE GPs
    Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(t_samp, snaps_comp, verbose=False)
    print(f"MLE GP: ℓ=[{', '.join(f'{l:.5f}' for l in Ls)}]")
    print(f"        σ²=[{', '.join(f'{v:.4f}' for v in Vs)}]")
    print(f"        ν=[{', '.join(f'{n:.6f}' for n in Ns)}]")

    # Eval time grid
    t_eval = np.linspace(float(t_samp[0]), float(t_samp[-1]), NUM_EVAL_POINTS)

    # GP predictions at eval points
    X_eval = np.zeros((num_modes, NUM_EVAL_POINTS))
    for i in range(num_modes):
        K = rbf_eval(Ls[i], Vs[i], t_samp, t_samp) + (Ns[i] + 1e-5) * np.eye(len(t_samp))
        Ks = rbf_eval(Ls[i], Vs[i], t_eval, t_samp)
        X_eval[i] = Ks @ np.linalg.solve(K, snaps_comp[i])

    # GP derivatives
    mu_z, cov_z = compute_gp_derivatives(Ls, Vs, t_samp, t_eval, snaps_comp, Ns=Ns)

    # LS operator (light regularization)
    D = np.array(rom.model._assemble_data_matrix(jnp.array(X_eval), inputs=None))
    dXdt = np.array(mu_z)
    reg_lambda = 1.0
    DtD = D.T @ D
    O_ls = np.linalg.solve(DtD + reg_lambda * np.eye(DtD.shape[0]), D.T @ dXdt.T).T
    print(f"LS operator norm: {np.linalg.norm(O_ls):.1f}")

    # Derivative variance (diagonal) for reference
    deriv_vars = np.array([np.diag(cov_z[i]) for i in range(num_modes)])
    print(f"Deriv variance range: [{deriv_vars.min():.2f}, {deriv_vars.max():.2f}]")

    return {
        't_full': t_full, 'true_states': true_states, 'true_comp': true_comp,
        't_samp': t_samp, 'snaps_samp': snaps_samp, 'snaps_comp': snaps_comp,
        'basis': basis, 'rom': rom, 'num_modes': num_modes,
        'Ls': Ls, 'Vs': Vs, 'Ns': Ns,
        't_eval': t_eval, 'X_eval': X_eval,
        'mu_z': np.array(mu_z), 'cov_z': np.array(cov_z),
        'deriv_vars': deriv_vars,
        'O_ls': O_ls, 'D': D,
        'training_span': TRAINING_SPAN, 'prediction_span': PREDICTION_SPAN,
    }


def evaluate(d, samples, label, t0):
    """Evaluate model results."""
    dt = time.time() - t0
    num_modes = d['num_modes']
    t_pred = np.linspace(d['prediction_span'][0], d['prediction_span'][1], 400)

    O_samples = _find_operator_samples(samples, "O")
    if O_samples.ndim == 2:
        O_samples = O_samples[np.newaxis, ...]
    O_med = np.median(O_samples, axis=0)

    Os, Xs, rom_solves = generate_rom_predictions(
        samples=samples, rom=d['rom'],
        snapshots_compressed=d['snaps_comp'],
        time_eval=t_pred, num_modes=num_modes, num_pulls=min(200, len(O_samples)))

    n_stable = len(rom_solves)
    n_total = len(Os)
    stab = n_stable / max(n_total, 1) * 100

    train_err = pred_err = float('inf')
    ci_width = 0
    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        train_mask = t_pred <= d['training_span'][1]
        pred_mask = t_pred > d['training_span'][1]

        from scipy.interpolate import interp1d
        true_interp = interp1d(d['t_full'], d['true_comp'], kind='cubic', fill_value='extrapolate')
        true_at = true_interp(t_pred)

        train_err = float(np.linalg.norm(rom_med[:, train_mask] - true_at[:, train_mask]) /
                         np.linalg.norm(true_at[:, train_mask]))
        if np.any(pred_mask):
            pred_err = float(np.linalg.norm(rom_med[:, pred_mask] - true_at[:, pred_mask]) /
                            np.linalg.norm(true_at[:, pred_mask]))

        q05 = np.percentile(rom_arr, 5, axis=0)
        q95 = np.percentile(rom_arr, 95, axis=0)
        ci_width = float(np.mean(q95 - q05))

    # GP drift check
    v_drifts = []
    for i in range(num_modes):
        vk = f'variance_{i}'
        if vk in samples:
            v_svi = float(np.median(samples[vk]))
            drift = abs(v_svi - d['Vs'][i]) / d['Vs'][i] * 100
            v_drifts.append(drift)

    gp_drift = np.mean(v_drifts) if v_drifts else 0

    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  Stability:    {n_stable}/{n_total} ({stab:.0f}%)")
    print(f"  Train error:  {train_err:.2%}")
    print(f"  Pred error:   {pred_err:.2%}")
    print(f"  CI width:     {ci_width:.6f}")
    print(f"  O norm:       {np.linalg.norm(O_med):.1f} (LS: {np.linalg.norm(d['O_ls']):.1f})")
    print(f"  O elem std:   {np.std(O_samples, axis=0).mean():.4f}")
    if v_drifts:
        print(f"  GP σ² drift:  {gp_drift:.1f}%")
    print(f"  Runtime:      {dt:.1f}s")

    return {
        'label': label, 'stability': stab, 'n_stable': n_stable,
        'train_err': train_err, 'pred_err': pred_err,
        'ci_width': ci_width, 'gp_drift': gp_drift, 'runtime': dt,
        'O_norm': float(np.linalg.norm(O_med)),
    }


# ============================================================
# Model A: Fixed GP + Informative Prior + Full Cov
# ============================================================

def run_model_A(d, rng_key):
    """Fixed GP, informative prior, full covariance ODE constraint."""
    print(f"\n{'='*70}")
    print(f"MODEL A: Fixed GP + Informative Prior + Full Covariance")
    print(f"{'='*70}")

    num_modes = d['num_modes']
    t0 = time.time()

    # Use existing build_bayesian_opinf_model with fixed GP
    model = build_bayesian_opinf_model(
        prior_operator=jnp.array(d['O_ls']),
        rom=d['rom'],
        Ls_means=d['Ls'],
        Vs_means=d['Vs'],
        time_domain_sampled=d['t_samp'],
        snapshots=d['snaps_comp'],
        Xs_means=d['X_eval'],
        Ns_means=d['Ns'],
        data_scaler=None,
        sample_X=False,
        relative_gamma=True,
        gamma_floor=0.5,
    )

    rng_key, svi_key = random.split(rng_key)
    result = core_run_svi(
        model=model, rng_key=svi_key,
        time_eval=d['t_eval'],
        gamma=1.0, gamma2=1.0,
        num_steps=5000, learning_rate=0.01,
        num_samples=500, verbose=False,
        guide=autoguide.AutoNormal,
    )

    return evaluate(d, result.samples, "A: Fixed GP + Informative Prior + Full Cov", t0), result


# ============================================================
# Model B: Fixed GP + Informative Prior + MCMC-like wide search
# ============================================================

def run_model_B(d, rng_key):
    """Fixed GP, wider operator prior, tighter ODE constraint."""
    print(f"\n{'='*70}")
    print(f"MODEL B: Fixed GP + Wide Prior + Tight ODE Constraint")
    print(f"{'='*70}")

    num_modes = d['num_modes']
    t0 = time.time()

    model = build_bayesian_opinf_model(
        prior_operator=jnp.array(d['O_ls']),
        rom=d['rom'],
        Ls_means=d['Ls'],
        Vs_means=d['Vs'],
        time_domain_sampled=d['t_samp'],
        snapshots=d['snaps_comp'],
        Xs_means=d['X_eval'],
        Ns_means=d['Ns'],
        data_scaler=None,
        sample_X=False,
        relative_gamma=True,
        gamma_floor=1.0,
    )

    rng_key, svi_key = random.split(rng_key)
    result = core_run_svi(
        model=model, rng_key=svi_key,
        time_eval=d['t_eval'],
        gamma=5.0, gamma2=0.1,
        num_steps=10000, learning_rate=0.005,
        num_samples=500, verbose=False,
        guide=autoguide.AutoNormal,
    )

    return evaluate(d, result.samples, "B: Fixed GP + Wide Prior + Tight ODE", t0), result


# ============================================================
# Model C: Integral Form
# ============================================================

def run_model_C(d, rng_key):
    """Integral form constraint with fixed GP."""
    print(f"\n{'='*70}")
    print(f"MODEL C: Fixed GP + Integral Form Constraint")
    print(f"{'='*70}")

    num_modes = d['num_modes']
    t_eval = d['t_eval']
    n_eval = len(t_eval)
    dt_eval = float(t_eval[1] - t_eval[0])
    t0_clock = time.time()

    X_eval_jnp = jnp.array(d['X_eval'])
    mu_z_jnp = jnp.array(d['mu_z'])
    deriv_var_jnp = jnp.array(d['deriv_vars'])
    O_prior_jnp = jnp.array(d['O_ls'])
    op_shape = O_prior_jnp.shape

    # Integral windows
    WINDOW_SIZE = 10
    n_windows = n_eval // WINDOW_SIZE
    ws_list = [i * WINDOW_SIZE for i in range(n_windows)]
    we_list = [(i + 1) * WINDOW_SIZE - 1 for i in range(n_windows)]
    if we_list[-1] < n_eval - 1:
        we_list[-1] = n_eval - 1

    trap_ws = []
    for ws, we in zip(ws_list, we_list):
        n_pts = we - ws + 1
        w = jnp.ones(n_pts) * dt_eval
        w = w.at[0].set(0.5 * dt_eval)
        w = w.at[-1].set(0.5 * dt_eval)
        trap_ws.append(w)

    def model(gamma_=1.0, gamma2_=1.0, jitter=1e-5):
        prior_scale = gamma_ * jnp.maximum(jnp.abs(O_prior_jnp), 0.5)
        O = numpyro.sample("O", dist.Normal(O_prior_jnp, prior_scale))

        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", X_eval_jnp[i])
            numpyro.deterministic(f"X_eval_{i}", X_eval_jnp[i])

        # Operator dynamics at all eval points
        f_Xi = d['rom'].model._assemble_data_matrix(X_eval_jnp, inputs=None) @ O.T

        # Derivative matching (standard)
        for i in range(num_modes):
            total_var = deriv_var_jnp[i] + gamma2_ + jitter
            ode_dist = dist.Normal(loc=f_Xi[:, i], scale=jnp.sqrt(total_var))
            numpyro.factor(f"ode_constraint_{i}",
                          jnp.sum(ode_dist.log_prob(mu_z_jnp[i])))

        # Integral constraints (added)
        for i in range(num_modes):
            for w_idx, (ws, we) in enumerate(zip(ws_list, we_list)):
                delta_X_obs = X_eval_jnp[i, we] - X_eval_jnp[i, ws]
                f_window = f_Xi[ws:we+1, i]
                delta_X_pred = jnp.sum(trap_ws[w_idx] * f_window)
                window_dur = float(t_eval[we] - t_eval[ws])
                numpyro.factor(
                    f"integral_{i}_{w_idx}",
                    dist.Normal(delta_X_pred, jnp.sqrt(gamma2_) * window_dur).log_prob(delta_X_obs))

    init_values = {'O': jnp.array(d['O_ls'])}
    model_kwargs = dict(gamma_=2.0, gamma2_=0.5, jitter=1e-5)

    guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    optimizer = ClippedAdam(step_size=5e-3)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key, ik = random.split(rng_key)
    svi_state = svi.init(ik, **model_kwargs)

    @jax.jit
    def step(s, _):
        s, l = svi.update(s, **model_kwargs)
        return s, l

    svi_state, losses = jax.lax.scan(step, svi_state, jnp.arange(5000))
    losses = np.array(losses)
    print(f"  Final loss: {losses[-1]:.1f} (init: {losses[0]:.1f})")

    params = svi.get_params(svi_state)
    rng_key, sk, pk = random.split(rng_key, 3)
    post = guide.sample_posterior(sk, params, sample_shape=(500,), **model_kwargs)
    pred = Predictive(model, posterior_samples=post, num_samples=500)
    out = pred(pk, **model_kwargs)
    samples = {**out, **post}

    return evaluate(d, samples, "C: Fixed GP + Derivative + Integral", t0_clock), SVIResult(samples, params, list(losses))


# ============================================================
# Model D: Grid Search Baseline (like notebook 02)
# ============================================================

def run_model_D(d, rng_key):
    """Grid search + SVI, matching notebook 02."""
    print(f"\n{'='*70}")
    print(f"MODEL D: Grid Search + SVI (Notebook 02 style)")
    print(f"{'='*70}")

    t0 = time.time()

    # Grid search for prior operator
    print("  Running grid search...")
    gs_result = grid_search_prior_operator(
        basis=d['basis'],
        time_domain_sampled=d['t_samp'],
        snapshots_sampled=d['snaps_samp'],
        snapshots_compressed=d['snaps_comp'],
        operators="cAH",
        reg_values=np.logspace(-8, 4, 49).tolist(),
        verbose=False,
    )
    print(f"  Best reg: {gs_result.best_reg:.2e}, error: {gs_result.best_error:.4%}")

    model = build_bayesian_opinf_model(
        prior_operator=jnp.array(gs_result.operator),
        rom=d['rom'],
        Ls_means=d['Ls'],
        Vs_means=d['Vs'],
        time_domain_sampled=d['t_samp'],
        snapshots=d['snaps_comp'],
        Xs_means=d['X_eval'],
        Ns_means=d['Ns'],
        data_scaler=None,
        sample_X=False,
        relative_gamma=True,
        gamma_floor=0.5,
    )

    rng_key, svi_key = random.split(rng_key)
    result = core_run_svi(
        model=model, rng_key=svi_key,
        time_eval=d['t_eval'],
        gamma=1.0, gamma2=1.0,
        num_steps=10000, learning_rate=0.01,
        num_samples=500, verbose=False,
        guide=autoguide.AutoNormal,
    )

    return evaluate(d, result.samples, "D: Grid Search + SVI (notebook-02 style)", t0), result


# ============================================================
# Main
# ============================================================

def main():
    d = prepare_data(noise_level=0.03, num_samples=250, num_modes=6)
    rng_key = random.PRNGKey(42)

    results = []

    rng_key, k = random.split(rng_key)
    r_a, _ = run_model_A(d, k)
    results.append(r_a)

    rng_key, k = random.split(rng_key)
    r_b, _ = run_model_B(d, k)
    results.append(r_b)

    rng_key, k = random.split(rng_key)
    r_c, _ = run_model_C(d, k)
    results.append(r_c)

    rng_key, k = random.split(rng_key)
    r_d, _ = run_model_D(d, k)
    results.append(r_d)

    # Summary table
    print(f"\n{'='*90}")
    print(f"COMPARISON SUMMARY (Euler, 3% noise, 250 samples, 6 modes)")
    print(f"{'='*90}")
    print(f"{'Model':<50s}  {'Stab':>5s}  {'Train':>7s}  {'Pred':>7s}  {'CI':>8s}  {'Time':>5s}")
    print(f"{'─'*90}")
    for r in results:
        print(f"{r['label']:<50s}  "
              f"{r['stability']:>4.0f}%  "
              f"{r['train_err']:>6.2%}  "
              f"{r['pred_err']:>6.2%}  "
              f"{r['ci_width']:>8.4f}  "
              f"{r['runtime']:>4.0f}s")

    # Winner
    stable_results = [r for r in results if r['stability'] > 90]
    if stable_results:
        best = min(stable_results, key=lambda r: r['pred_err'])
        print(f"\n  BEST: {best['label']}")
        print(f"        pred_err={best['pred_err']:.2%}, stability={best['stability']:.0f}%")


if __name__ == "__main__":
    main()

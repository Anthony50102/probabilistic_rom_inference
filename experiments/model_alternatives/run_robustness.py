"""
Robustness Test Suite for the Conditional GP + Integral Model

Tests the three paper schemas plus additional stress tests:
  1. Dense data, low noise    (250 samples, 3% noise)  — paper schema
  2. Sparse data, medium noise (55 samples, 5% noise)  — paper schema
  3. Dense data, high noise   (250 samples, 15% noise) — paper schema
  4. Very sparse data          (30 samples, 5% noise)  — stress test
  5. Extremely noisy           (250 samples, 25% noise) — stress test
  6. Sparse + noisy            (55 samples, 15% noise)  — stress test
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
    generate_rom_predictions, SVIResult, rbf_eval,
)
from core.bayesian_opinf import fit_gp_hyperparameters_mle, _find_operator_samples
import config
from config import Basis
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

# Import the model builder from run_conditional_integral
from run_conditional_integral import build_conditional_integral_model


def run_robustness_test(
    noise_level=0.03, num_samples=250, num_modes=6,
    # Model settings (defaults = our best from Experiment 2)
    gamma=2.0, gamma2=2.0, deriv_weight=1.0, integral_weight=1.0,
    mll_weight=0.1, gp_prior_scale=0.1,
    # SVI settings
    num_steps=10000, learning_rate=3e-3,
    num_posterior_samples=300, num_eval_points=None,
    seed=42, label="",
):
    """Run a single robustness test. Returns dict of metrics."""
    np.random.seed(seed)
    rng_key = random.PRNGKey(seed)

    TRAINING_SPAN = (0, 0.08)
    PREDICTION_SPAN = (0, 0.15)

    # Auto-scale eval points for sparse data
    if num_eval_points is None:
        num_eval_points = min(400, max(100, num_samples * 3))

    print(f"\n{'='*70}")
    print(f"ROBUSTNESS TEST: {label}")
    print(f"{'='*70}")
    print(f"Data: noise={noise_level}, samples={num_samples}, modes={num_modes}")
    print(f"Model: γ={gamma}, γ₂={gamma2}, eval_pts={num_eval_points}")
    print(f"SVI: steps={num_steps}, lr={learning_rate}")

    # --- Data ---
    (fom, t_full, true_states, t_samp, snaps_samp) = \
        generate_trajectory(config, config.time_domain, TRAINING_SPAN, num_samples, noise_level)
    basis = Basis(num_vectors=num_modes)
    basis.fit(snaps_samp)
    snaps_comp = basis.compress(snaps_samp)
    true_comp = basis.compress(true_states)
    print(f"POD energy: {basis.cumulative_energy:.4%}")

    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
        model=JaxCompatibleModel(operators="cAH", solver=opinf.lstsq.L2Solver(regularizer=1e0)),
    )
    rom.fit(states=snaps_samp)

    # --- MLE warm start ---
    Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(t_samp, snaps_comp, verbose=False)
    print(f"MLE GP: ℓ=[{min(Ls):.5f}, {max(Ls):.5f}], "
          f"σ²=[{min(Vs):.4f}, {max(Vs):.4f}], "
          f"ν=[{min(Ns):.6f}, {max(Ns):.6f}]")

    # LS operator
    t_eval_ls = np.linspace(float(t_samp[0]), float(t_samp[-1]), num_eval_points)
    X_mle = np.zeros((num_modes, num_eval_points))
    for i in range(num_modes):
        K = rbf_eval(Ls[i], Vs[i], t_samp, t_samp) + (Ns[i]+1e-5)*np.eye(len(t_samp))
        Ks = rbf_eval(Ls[i], Vs[i], t_eval_ls, t_samp)
        X_mle[i] = Ks @ np.linalg.solve(K, snaps_comp[i])

    mu_z_mle, _ = compute_gp_derivatives(Ls, Vs, t_samp, t_eval_ls, snaps_comp, Ns=Ns)
    D = np.array(rom.model._assemble_data_matrix(jnp.array(X_mle), inputs=None))
    DtD = D.T @ D
    O_ls = np.linalg.solve(DtD + np.eye(DtD.shape[0]), D.T @ np.array(mu_z_mle).T).T
    print(f"LS operator norm: {np.linalg.norm(O_ls):.1f}")

    # --- Build model ---
    model, time_eval = build_conditional_integral_model(
        rom=rom, num_modes=num_modes,
        time_sampled=t_samp, snapshots_comp=snaps_comp,
        O_prior=O_ls, mle_Ls=Ls, mle_Vs=Vs, mle_Ns=Ns,
        num_eval_points=num_eval_points, window_size=max(5, num_eval_points // 40),
        deriv_weight=deriv_weight, integral_weight=integral_weight,
        mll_weight=mll_weight, gp_prior_scale=gp_prior_scale,
    )

    # --- Init values ---
    init_values = {'O': jnp.array(O_ls)}
    for i in range(num_modes):
        init_values[f'lengthscale_{i}'] = Ls[i]
        init_values[f'variance_{i}'] = Vs[i]
        init_values[f'noise_{i}'] = Ns[i]

    model_kwargs = dict(gamma=gamma, gamma2=gamma2, jitter=1e-4)

    guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    optimizer = ClippedAdam(step_size=learning_rate)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key, ik = random.split(rng_key)
    t0 = time.time()
    svi_state = svi.init(ik, **model_kwargs)

    @jax.jit
    def _step(s, _):
        s, l = svi.update(s, **model_kwargs)
        return s, l

    seg_size = max(1, num_steps // 5)
    all_losses = []
    for seg in range(5):
        start = seg * seg_size
        end = min(start + seg_size, num_steps)
        if seg == 4:
            end = num_steps
        if start >= num_steps:
            break
        svi_state, seg_losses = jax.lax.scan(_step, svi_state, jnp.arange(end-start))
        seg_np = np.array(seg_losses)
        all_losses.extend(seg_np.tolist())
        print(f"  step {end:6d}/{num_steps}  loss={seg_np[-1]:10.2f}")

    params = svi.get_params(svi_state)
    rng_key, sk, pk = random.split(rng_key, 3)
    post = guide.sample_posterior(sk, params, sample_shape=(num_posterior_samples,), **model_kwargs)
    pred = Predictive(model, posterior_samples=post, num_samples=num_posterior_samples)
    out = pred(pk, **model_kwargs)
    samples = {**out, **post}
    runtime = time.time() - t0

    # --- Evaluate ---
    # GP drift
    gp_drifts = []
    for i in range(num_modes):
        v_svi = float(np.median(samples[f'variance_{i}']))
        v_drift = (v_svi - Vs[i]) / Vs[i] * 100
        gp_drifts.append(abs(v_drift))
    mean_v_drift = np.mean(gp_drifts)

    # GP uncertainty
    gp_cvs = []
    for i in range(num_modes):
        v_std = float(np.std(samples[f'variance_{i}']))
        v_med = float(np.median(samples[f'variance_{i}']))
        gp_cvs.append(v_std / v_med)
    mean_cv = np.mean(gp_cvs)

    # Operator
    O_samp = _find_operator_samples(samples, "O")
    if O_samp.ndim == 2:
        O_samp = O_samp[np.newaxis, ...]
    O_med = np.median(O_samp, axis=0)
    O_std = np.std(O_samp, axis=0)

    # ROM predictions
    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    Os, Xs, rom_solves = generate_rom_predictions(
        samples=samples, rom=rom, snapshots_compressed=snaps_comp,
        time_eval=t_pred, num_modes=num_modes,
        num_pulls=min(200, num_posterior_samples))

    n_stable = len(rom_solves)
    n_total = len(Os)
    stability_pct = n_stable / max(n_total, 1) * 100

    train_error = float('inf')
    pred_error = float('inf')
    ci_coverage = 0.0
    ci_width_rel = 0.0

    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        train_mask = t_pred <= TRAINING_SPAN[1]
        pred_mask = t_pred > TRAINING_SPAN[1]

        from scipy.interpolate import interp1d
        ti = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)

        train_error = float(np.linalg.norm(rom_med[:, train_mask]-ta[:, train_mask])/np.linalg.norm(ta[:, train_mask]))
        pred_error = float(np.linalg.norm(rom_med[:, pred_mask]-ta[:, pred_mask])/np.linalg.norm(ta[:, pred_mask]))

        q05 = np.percentile(rom_arr, 5, axis=0)
        q95 = np.percentile(rom_arr, 95, axis=0)
        ci_width = np.mean(q95 - q05)
        ci_width_rel = float(ci_width / np.mean(np.abs(rom_med)))
        ci_coverage = float(np.mean((ta >= q05) & (ta <= q95)))

    result = {
        'label': label,
        'noise_level': noise_level,
        'num_samples': num_samples,
        'pod_energy': float(basis.cumulative_energy),
        'stability_pct': stability_pct,
        'train_error': train_error,
        'pred_error': pred_error,
        'mean_v_drift': mean_v_drift,
        'mean_cv': mean_cv,
        'O_norm': float(np.linalg.norm(O_med)),
        'O_ls_norm': float(np.linalg.norm(O_ls)),
        'ci_coverage': ci_coverage,
        'ci_width_rel': ci_width_rel,
        'runtime': runtime,
        'final_loss': all_losses[-1],
    }

    print(f"\n--- {label} ---")
    print(f"  POD energy:  {result['pod_energy']:.2%}")
    print(f"  Stability:   {stability_pct:.0f}%")
    print(f"  Train error: {train_error:.4%}")
    print(f"  Pred error:  {pred_error:.4%}")
    print(f"  σ² drift:    {mean_v_drift:.1f}%")
    print(f"  CV(σ²):      {mean_cv:.3f}")
    print(f"  CI coverage: {ci_coverage:.2%}")
    print(f"  CI width:    {ci_width_rel:.2%}")
    print(f"  Runtime:     {runtime:.1f}s")

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("ROBUSTNESS TEST SUITE")
    print("Conditional GP + Integral Constraint Model")
    print("=" * 70)

    schemas = [
        # Paper schemas
        dict(num_samples=250, noise_level=0.03, label="Paper 1: Dense, low noise"),
        dict(num_samples=55,  noise_level=0.05, label="Paper 2: Sparse, medium noise"),
        dict(num_samples=250, noise_level=0.15, label="Paper 3: Dense, high noise"),
        # Stress tests
        dict(num_samples=30,  noise_level=0.05, label="Stress: Very sparse"),
        dict(num_samples=250, noise_level=0.25, label="Stress: Extreme noise"),
        dict(num_samples=55,  noise_level=0.15, label="Stress: Sparse + noisy"),
    ]

    results = []
    for schema in schemas:
        r = run_robustness_test(**schema)
        results.append(r)

    # Final summary
    print(f"\n\n{'='*100}")
    print(f"ROBUSTNESS SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"{'Label':<35s} {'Samp':>4s} {'Noise':>5s} {'POD':>6s} {'Stab':>5s} "
          f"{'Train':>7s} {'Pred':>7s} {'σ²drft':>6s} {'CV(σ²)':>6s} "
          f"{'CI_cov':>6s} {'CI_w':>6s} {'Time':>5s}")
    print(f"{'-'*35} {'-'*4} {'-'*5} {'-'*6} {'-'*5} "
          f"{'-'*7} {'-'*7} {'-'*6} {'-'*6} "
          f"{'-'*6} {'-'*6} {'-'*5}")
    for r in results:
        print(f"{r['label']:<35s} {r['num_samples']:>4d} {r['noise_level']:>5.2f} "
              f"{r['pod_energy']:>5.1%} {r['stability_pct']:>4.0f}% "
              f"{r['train_error']:>6.2%} {r['pred_error']:>6.2%} "
              f"{r['mean_v_drift']:>5.1f}% {r['mean_cv']:>5.3f} "
              f"{r['ci_coverage']:>5.1%} {r['ci_width_rel']:>5.1%} "
              f"{r['runtime']:>4.0f}s")

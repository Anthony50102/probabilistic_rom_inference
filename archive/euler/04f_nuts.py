"""
04f_nuts.py — Run the same model as 04d in scalefree/prior config but use NUTS
instead of SVI.  Tests whether VI's local convergence is what gets us stuck in
the null basin, vs. the posterior shape itself.

Single-chain, modest sample count (NUTS is expensive at d~70 with this model):
  --warmup  300
  --samples 300
"""
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis
from core import (generate_trajectory, JaxCompatibleModel, compute_gp_derivatives,
                  generate_rom_predictions, rbf_eval)
from core.bayesian_opinf import fit_gp_hyperparameters_mle, _find_operator_samples
import opinf

# Re-use 04d's build_model so we are testing the SAME model
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "m04d", os.path.join(os.path.dirname(__file__), "04d_no_warmstart.py"))
m04d = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m04d)

numpyro.set_platform('cpu')
numpyro.set_host_device_count(1)

SCHEMA = next(s for s in m04d.SCHEMAS if s['name'] == 'dense_low_noise')
TRAINING_SPAN = m04d.TRAINING_SPAN
PREDICTION_SPAN = m04d.PREDICTION_SPAN

NUM_WARMUP = 300
NUM_SAMPLES = 300


def main():
    p = dict(m04d.MODEL_PARAMS)
    p['PRIOR'] = 'scalefree'
    p['INIT'] = 'prior'
    print(f"04f — NUTS test, prior=scalefree, init=prior, warmup={NUM_WARMUP}, samples={NUM_SAMPLES}")

    noise = SCHEMA['NOISE_LEVEL']
    nsamp = SCHEMA['NUM_SAMPLES']
    neval = SCHEMA['NUM_EVAL_POINTS']
    nmodes = p['NUM_MODES']

    np.random.seed(p['SEED'])
    rng_key = random.PRNGKey(p['SEED'])

    fom, t_full, true_states, t_samp, snaps_samp = generate_trajectory(
        config, config.time_domain, TRAINING_SPAN, nsamp, noise)
    basis = Basis(num_vectors=nmodes)
    basis.fit(snaps_samp)
    snaps_comp = basis.compress(snaps_samp)
    true_comp = basis.compress(true_states)
    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
        model=JaxCompatibleModel(operators="cAH",
                                 solver=opinf.lstsq.L2Solver(regularizer=1e0)))
    rom.fit(states=snaps_samp)

    Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(t_samp, snaps_comp, verbose=False)
    t_eval_ls = np.linspace(float(t_samp[0]), float(t_samp[-1]), neval)
    X_mle = np.zeros((nmodes, neval))
    for i in range(nmodes):
        K = rbf_eval(Ls[i], Vs[i], t_samp, t_samp) + (Ns[i] + 1e-5) * np.eye(len(t_samp))
        Ks = rbf_eval(Ls[i], Vs[i], t_eval_ls, t_samp)
        X_mle[i] = Ks @ np.linalg.solve(K, snaps_comp[i])
    mu_z_mle, _ = compute_gp_derivatives(Ls, Vs, t_samp, t_eval_ls, snaps_comp, Ns=Ns)
    D = np.array(rom.model._assemble_data_matrix(jnp.array(X_mle), inputs=None))
    O_ls = np.linalg.solve(D.T @ D + np.eye(D.shape[1]), D.T @ np.array(mu_z_mle).T).T

    model, time_eval = m04d.build_model(
        rom=rom, num_modes=nmodes,
        time_sampled=t_samp, snapshots_comp=snaps_comp,
        O_prior=O_ls, mle_Ls=Ls, mle_Vs=Vs, mle_Ns=Ns,
        num_eval_points=neval, window_size=p['WINDOW_SIZE'],
        deriv_weight=p['DERIV_WEIGHT'], integral_weight=p['INTEGRAL_WEIGHT'],
        mll_weight=p['MLL_WEIGHT'], gp_prior_scale=p['GP_PRIOR_SCALE'],
        prior_style='scalefree')

    nuts_kernel = NUTS(model, target_accept_prob=0.8)
    mcmc = MCMC(nuts_kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=1, progress_bar=True)

    rng_key, ik = random.split(rng_key)
    t0 = time.time()
    mcmc.run(ik, gamma=p['GAMMA'], gamma2=p['GAMMA2'], jitter=1e-4)
    runtime = time.time() - t0
    mcmc.print_summary()

    samples = {k: np.array(v) for k, v in mcmc.get_samples().items()}
    O_samp = _find_operator_samples(samples, "O")
    if O_samp.ndim == 2:
        O_samp = O_samp[np.newaxis, ...]
    O_med = np.median(O_samp, axis=0)
    print(f"  Operator norm: median={np.linalg.norm(O_med):.1f}  LS={np.linalg.norm(O_ls):.1f}")
    print(f"  Operator-norm per sample: min={np.min(np.linalg.norm(O_samp.reshape(O_samp.shape[0],-1),axis=1)):.1f}  "
          f"median={np.median(np.linalg.norm(O_samp.reshape(O_samp.shape[0],-1),axis=1)):.1f}  "
          f"max={np.max(np.linalg.norm(O_samp.reshape(O_samp.shape[0],-1),axis=1)):.1f}")

    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    Os, Xs, rom_solves = generate_rom_predictions(
        samples=samples, rom=rom, snapshots_compressed=snaps_comp,
        time_eval=t_pred, num_modes=nmodes,
        num_pulls=min(200, NUM_SAMPLES))
    n_stable, n_total = len(rom_solves), len(Os)
    stab = n_stable / max(n_total, 1) * 100

    train_err = pred_err = float('inf')
    ci_cov = ci_w = float('nan')
    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        tm = t_pred <= TRAINING_SPAN[1]; pm = t_pred > TRAINING_SPAN[1]
        ti = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)
        train_err = float(np.linalg.norm(rom_med[:, tm] - ta[:, tm]) /
                          np.linalg.norm(ta[:, tm]))
        pred_err = float(np.linalg.norm(rom_med[:, pm] - ta[:, pm]) /
                         np.linalg.norm(ta[:, pm]))
        q05 = np.percentile(rom_arr, 5, axis=0); q95 = np.percentile(rom_arr, 95, axis=0)
        ci_w = float(np.mean(q95 - q05))
        ci_cov = float(np.mean((ta >= q05) & (ta <= q95)))

    print(f"\n==== NUTS RESULTS ====")
    print(f"  Stability:    {stab:.1f}% ({n_stable}/{n_total})")
    print(f"  Train error:  {train_err:.2%}")
    print(f"  Pred error:   {pred_err:.2%}")
    print(f"  CI coverage:  {ci_cov:.1%}")
    print(f"  CI width:     {ci_w:.4f}")
    print(f"  Operator norm (median): {np.linalg.norm(O_med):.1f}  (LS: {np.linalg.norm(O_ls):.1f})")
    print(f"  Runtime:      {runtime:.0f}s")

    # Save for the aggregator
    out_dir = os.path.join(os.path.dirname(__file__), "results", "comparison",
                           SCHEMA['name'])
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "04f_nuts_scalefree_prior.npz"),
        train_error=train_err, pred_error=pred_err,
        stability_pct=stab, ci_coverage=ci_cov, ci_width=ci_w,
        runtime=runtime,
        op_norm_median=float(np.linalg.norm(O_med)),
        op_norm_ls=float(np.linalg.norm(O_ls)))


if __name__ == "__main__":
    main()

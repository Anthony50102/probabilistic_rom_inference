"""
Conditional GP + Joint Operator Model with Integral Constraint

Full uncertainty flow WITHOUT explicit latent states:
  GP hypers (ℓ, σ², ν) sampled → GP posterior X_eval, dX/dt computed → operator O sampled

Key insight: X_eval = GP_mean(y, θ_GP) is a deterministic function of (data, GP hypers).
Different GP hyperparameter samples produce different states, derivatives, and operator fits.
This IS joint inference — the physics constraint feeds back to GP hypers through the ELBO.

Why this works:
- No latent X_raw parameters → simpler optimization (from ~1500 params to ~18 + operator)
- No σ²/ν competition (the collapse problem from the latent-X formulation)
- GP marginal likelihood informs hypers, physics informs operator + hypers
- Integral form prevents null basin (state differences ≠ 0 → operator must be non-trivial)
- Full UQ chain: P(O, θ_GP | y) approximated jointly via SVI

Difference from 2-stage:
- GP hypers have priors and are SAMPLED (not point MLE)
- Physics constraint feeds back to GP hypers through joint optimization
- Posterior samples contain (θ_GP, O) pairs → correlated uncertainty
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


def build_conditional_integral_model(
    rom, num_modes, time_sampled, snapshots_comp,
    O_prior, mle_Ls, mle_Vs, mle_Ns,
    num_eval_points=400, window_size=10,
    deriv_weight=1.0, integral_weight=1.0,
    mll_weight=1.0,
    gp_prior_scale=0.3,
):
    """
    Build model where GP states are computed (not sampled) from sampled GP hypers.
    """
    t_train = jnp.array(time_sampled)
    n_train = len(t_train)
    y_obs = jnp.array(snapshots_comp)

    time_eval = np.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
    t_eval = jnp.array(time_eval)
    n_eval = num_eval_points
    dt_eval = float(time_eval[1] - time_eval[0])

    # Precompute squared-difference matrices
    sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
    sq_diffs_et = (t_eval[:, None] - t_train[None, :]) ** 2
    diffs_et = t_eval[:, None] - t_train[None, :]
    sq_diffs_ee = (t_eval[:, None] - t_eval[None, :]) ** 2
    I_train = jnp.eye(n_train)

    op_shape = O_prior.shape
    O_prior_jnp = jnp.array(O_prior)

    mle_log_ells = jnp.array([jnp.log(l) for l in mle_Ls])
    mle_log_sig2s = jnp.array([jnp.log(v) for v in mle_Vs])
    mle_log_nus = jnp.array([jnp.log(n) for n in mle_Ns])

    # Precompute integration windows
    n_windows = n_eval // window_size
    ws_list = [i * window_size for i in range(n_windows)]
    we_list = [(i + 1) * window_size - 1 for i in range(n_windows)]
    if we_list[-1] < n_eval - 1:
        we_list[-1] = n_eval - 1

    trap_weights = []
    window_durations = []
    for ws, we in zip(ws_list, we_list):
        n_pts = we - ws + 1
        w = jnp.ones(n_pts) * dt_eval
        w = w.at[0].set(0.5 * dt_eval)
        w = w.at[-1].set(0.5 * dt_eval)
        trap_weights.append(w)
        window_durations.append(float(time_eval[we] - time_eval[ws]))

    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def _single_gp_conditional(ell, sig2, nu, y_i):
        """Compute GP posterior mean at eval points + derivative mean + variance.

        All outputs are deterministic functions of (ell, sig2, nu, y_i).
        """
        ell2 = ell ** 2
        jitter = jnp.maximum(1e-5, sig2 * 1e-4)

        # Training kernel + noise
        K_tt = _rbf_sq(ell, sig2, sq_diff_tt) + (nu + jitter) * I_train
        L = jnp.linalg.cholesky(K_tt)
        alpha = jax.scipy.linalg.cho_solve((L, True), y_i)

        # GP posterior mean at eval points
        K_et = _rbf_sq(ell, sig2, sq_diffs_et)
        X_eval = K_et @ alpha

        # Derivative cross-covariance K'(t_eval, t_train)
        K_zy = -(diffs_et / ell2) * K_et
        mu_z = K_zy @ alpha

        # Derivative auto-covariance K''(t_eval, t_eval)
        K_ee = _rbf_sq(ell, sig2, sq_diffs_ee)
        K_zz = ((1.0 - sq_diffs_ee / ell2) / ell2) * K_ee

        # Conditional variance: diag(K'' - K' @ K^{-1} @ K'^T)
        V = jax.scipy.linalg.cho_solve((L, True), K_zy.T)
        deriv_var = jnp.maximum(jnp.diag(K_zz) - jnp.sum(K_zy * V.T, axis=1), 0.0)

        # Marginal log-likelihood: -0.5 * (y^T K^{-1} y + log|K| + n*log(2π))
        mll = -0.5 * (jnp.dot(y_i, alpha) +
                       2.0 * jnp.sum(jnp.log(jnp.diag(L))) +
                       n_train * jnp.log(2.0 * jnp.pi))

        return X_eval, mu_z, deriv_var, mll

    _batch_gp_conditional = jax.vmap(_single_gp_conditional)

    def model(gamma=2.0, gamma2=0.5, jitter=1e-4):
        # --- GP hyperparameters with informative priors centered at MLE ---
        ells = jnp.stack([
            numpyro.sample(f"lengthscale_{i}",
                          dist.LogNormal(mle_log_ells[i], gp_prior_scale))
            for i in range(num_modes)
        ])
        sig2s = jnp.stack([
            numpyro.sample(f"variance_{i}",
                          dist.LogNormal(mle_log_sig2s[i], gp_prior_scale))
            for i in range(num_modes)
        ])
        nus = jnp.stack([
            numpyro.sample(f"noise_{i}",
                          dist.LogNormal(mle_log_nus[i], gp_prior_scale))
            for i in range(num_modes)
        ])

        # --- Compute GP conditional (deterministic given hypers + data) ---
        Xs_eval, mu_zs, deriv_vars, mlls = _batch_gp_conditional(
            ells, sig2s, nus, y_obs)

        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", Xs_eval[i])
            numpyro.deterministic(f"X_eval_{i}", Xs_eval[i])

        # --- GP marginal log-likelihood (data fidelity for learning hypers) ---
        if mll_weight > 0:
            numpyro.factor("gp_mll", mll_weight * jnp.sum(mlls))

        # --- Operator with informative prior ---
        prior_scale = gamma * jnp.maximum(jnp.abs(O_prior_jnp), 0.5)
        O = numpyro.sample("O", dist.Normal(O_prior_jnp, prior_scale))

        # --- Compute operator dynamics: f(X_eval) @ O^T ---
        f_Xi = rom.model._assemble_data_matrix(Xs_eval, inputs=None) @ O.T

        # --- CONSTRAINT 1: Derivative matching ---
        if deriv_weight > 0:
            for i in range(num_modes):
                total_var = deriv_vars[i] + gamma2 + jitter
                numpyro.factor(f"ode_constraint_{i}",
                    deriv_weight * jnp.sum(
                        dist.Normal(f_Xi[:, i], jnp.sqrt(total_var)).log_prob(mu_zs[i])))

        # --- CONSTRAINT 2: Integral form ---
        if integral_weight > 0:
            for i in range(num_modes):
                for w_idx, (ws, we) in enumerate(zip(ws_list, we_list)):
                    delta_X_obs = Xs_eval[i, we] - Xs_eval[i, ws]
                    delta_X_pred = jnp.sum(trap_weights[w_idx] * f_Xi[ws:we+1, i])
                    constraint_std = jnp.sqrt(gamma2) * window_durations[w_idx]
                    numpyro.factor(f"integral_{i}_{w_idx}",
                        integral_weight * dist.Normal(
                            delta_X_pred, constraint_std).log_prob(delta_X_obs))

    return model, time_eval


def run_experiment(
    noise_level=0.15, num_samples=250, num_modes=6,
    gamma=2.0, gamma2=0.5, window_size=10,
    deriv_weight=1.0, integral_weight=1.0,
    mll_weight=1.0,
    gp_prior_scale=0.3,
    num_steps=8000, learning_rate=3e-3,
    num_posterior_samples=500, num_eval_points=400,
    seed=42, label="",
):
    """Run the conditional GP + integral form experiment."""
    np.random.seed(seed)
    rng_key = random.PRNGKey(seed)

    TRAINING_SPAN = (0, 0.08)
    PREDICTION_SPAN = (0, 0.15)

    print(f"\n{'='*70}")
    print(f"CONDITIONAL GP + INTEGRAL MODEL {label}")
    print(f"{'='*70}")
    print(f"Data: noise={noise_level}, samples={num_samples}, modes={num_modes}")
    print(f"Model: γ={gamma}, γ₂={gamma2}, window={window_size}")
    print(f"       deriv_w={deriv_weight}, integral_w={integral_weight}, mll_w={mll_weight}")
    print(f"       gp_prior_scale={gp_prior_scale}")
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
    print(f"\nMLE GP hyperparameters:")
    for i in range(num_modes):
        T = t_samp[-1] - t_samp[0]
        print(f"  Mode {i}: ℓ={Ls[i]:.5f} (T/ℓ={T/Ls[i]:.0f}), σ²={Vs[i]:.4f}, ν={Ns[i]:.6f}")

    # LS operator (same warm start as before)
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
        num_eval_points=num_eval_points, window_size=window_size,
        deriv_weight=deriv_weight, integral_weight=integral_weight,
        mll_weight=mll_weight,
        gp_prior_scale=gp_prior_scale,
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

    # Run with progress logging
    seg_size = max(1, num_steps // 10)
    all_losses = []
    for seg in range(10):
        start = seg * seg_size
        end = min(start + seg_size, num_steps)
        if seg == 9:
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
    print(f"\n--- Results (runtime: {runtime:.1f}s) ---")

    # GP hyperparameter drift
    print(f"\nGP hyperparameter drift (SVI median vs MLE):")
    gp_drifts = []
    for i in range(num_modes):
        l_svi = float(np.median(samples[f'lengthscale_{i}']))
        v_svi = float(np.median(samples[f'variance_{i}']))
        n_svi = float(np.median(samples[f'noise_{i}']))
        l_drift = (l_svi - Ls[i]) / Ls[i] * 100
        v_drift = (v_svi - Vs[i]) / Vs[i] * 100
        n_drift = (n_svi - Ns[i]) / max(Ns[i], 1e-10) * 100
        gp_drifts.append(abs(v_drift))
        print(f"  Mode {i}: ℓ={l_svi:.5f} ({l_drift:+.1f}%), "
              f"σ²={v_svi:.4f} ({v_drift:+.1f}%), "
              f"ν={n_svi:.6f} ({n_drift:+.1f}%)")
    mean_v_drift = np.mean(gp_drifts)
    print(f"  Mean |σ² drift|: {mean_v_drift:.1f}%")
    if mean_v_drift > 50:
        print(f"  WARNING: Significant GP drift!")

    # GP hyperparameter uncertainty
    print(f"\nGP hyperparameter uncertainty (posterior std / median):")
    for i in range(num_modes):
        l_std = float(np.std(samples[f'lengthscale_{i}']))
        l_med = float(np.median(samples[f'lengthscale_{i}']))
        v_std = float(np.std(samples[f'variance_{i}']))
        v_med = float(np.median(samples[f'variance_{i}']))
        n_std = float(np.std(samples[f'noise_{i}']))
        n_med = float(np.median(samples[f'noise_{i}']))
        print(f"  Mode {i}: CV(ℓ)={l_std/l_med:.3f}, CV(σ²)={v_std/v_med:.3f}, CV(ν)={n_std/n_med:.3f}")

    # Operator
    O_samp = _find_operator_samples(samples, "O")
    if O_samp.ndim == 2:
        O_samp = O_samp[np.newaxis, ...]
    O_med = np.median(O_samp, axis=0)
    O_std = np.std(O_samp, axis=0)
    print(f"\nOperator: norm={np.linalg.norm(O_med):.1f} (LS: {np.linalg.norm(O_ls):.1f})")
    print(f"  Norm ratio (SVI/LS): {np.linalg.norm(O_med)/max(np.linalg.norm(O_ls), 1e-10):.3f}")
    print(f"  Mean elem std: {np.mean(O_std):.4f}, Max: {np.max(O_std):.4f}")

    if np.linalg.norm(O_med) < 0.1 * np.linalg.norm(O_ls):
        print(f"  ALERT: Operator collapsed to near-zero!")

    # ROM predictions
    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    Os, Xs, rom_solves = generate_rom_predictions(
        samples=samples, rom=rom, snapshots_compressed=snaps_comp,
        time_eval=t_pred, num_modes=num_modes,
        num_pulls=min(200, num_posterior_samples))

    n_stable = len(rom_solves)
    n_total = len(Os)
    print(f"\nStability: {n_stable}/{n_total} ({n_stable/max(n_total,1)*100:.0f}%)")

    train_error = float('inf')
    pred_error = float('inf')

    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        train_mask = t_pred <= TRAINING_SPAN[1]
        pred_mask = t_pred > TRAINING_SPAN[1]

        from scipy.interpolate import interp1d
        ti = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)

        train_error = np.linalg.norm(rom_med[:, train_mask]-ta[:, train_mask])/np.linalg.norm(ta[:, train_mask])
        pred_error = np.linalg.norm(rom_med[:, pred_mask]-ta[:, pred_mask])/np.linalg.norm(ta[:, pred_mask])
        print(f"Training error:    {train_error:.4%}")
        print(f"Prediction error:  {pred_error:.4%}")

        print(f"\nPer-mode errors (training):")
        for i in range(num_modes):
            mode_err = np.linalg.norm(rom_med[i, train_mask]-ta[i, train_mask])/max(np.linalg.norm(ta[i, train_mask]), 1e-10)
            print(f"  Mode {i}: {mode_err:.4%}")

        q05 = np.percentile(rom_arr, 5, axis=0)
        q95 = np.percentile(rom_arr, 95, axis=0)
        ci_width = np.mean(q95 - q05)
        print(f"\n90% CI width: {ci_width:.6f}")
        print(f"Relative CI:  {ci_width/np.mean(np.abs(rom_med)):.4%}")

        in_ci = np.mean((ta >= q05) & (ta <= q95))
        print(f"CI coverage:  {in_ci:.2%} (target: 90%)")

    print(f"\nConvergence: loss {all_losses[0]:.0f} → {all_losses[-1]:.0f}")
    print(f"Runtime: {runtime:.1f}s")

    return {
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': n_stable/max(n_total,1)*100,
        'n_stable': n_stable, 'n_total': n_total,
        'mean_v_drift': mean_v_drift,
        'O_norm': float(np.linalg.norm(O_med)),
        'O_ls_norm': float(np.linalg.norm(O_ls)),
        'runtime': runtime,
        'samples': samples, 'losses': all_losses,
        # Data needed for plotting
        'rom_solves': rom_solves, 'snaps_comp': snaps_comp,
        'true_comp': true_comp, 't_full': t_full, 't_pred': t_pred,
        't_samp': t_samp, 'training_span': TRAINING_SPAN,
    }


if __name__ == "__main__":
    from experiment_utils import plot_experiment_results

    print("=" * 70)
    print("EXPERIMENT: Conditional GP + Integral Form — 3 Regimes")
    print("=" * 70)

    regimes = [
        {
            "name": "dense_low_noise",
            "label": "Dense data, low noise (250 samples, 3%)",
            "noise_level": 0.03,
            "num_samples": 250,
            "num_eval_points": 400,
        },
        {
            "name": "sparse_medium_noise",
            "label": "Sparse data, medium noise (55 samples, 5%)",
            "noise_level": 0.05,
            "num_samples": 55,
            "num_eval_points": 200,
        },
        {
            "name": "dense_high_noise",
            "label": "Dense data, high noise (250 samples, 15%)",
            "noise_level": 0.15,
            "num_samples": 250,
            "num_eval_points": 400,
        },
    ]

    # Shared best hyperparameters from experiment log
    shared_kwargs = dict(
        gamma=2.0, gamma2=2.0,
        deriv_weight=1.0, integral_weight=1.0,
        mll_weight=0.1, gp_prior_scale=0.1,
        num_steps=10000, learning_rate=3e-3,
        num_posterior_samples=500,
    )

    results = []
    for regime in regimes:
        print(f"\n{'='*70}")
        print(f"REGIME: {regime['label']}")
        print(f"{'='*70}")

        r = run_experiment(
            noise_level=regime["noise_level"],
            num_samples=regime["num_samples"],
            num_eval_points=regime["num_eval_points"],
            label=regime["label"],
            **shared_kwargs,
        )
        plot_experiment_results(r, prefix=f"cond_integral_{regime['name']}")
        r["regime"] = regime["name"]
        results.append(r)

    print(f"\n\n{'='*90}")
    print(f"SUMMARY TABLE — Conditional GP + Integral Form")
    print(f"{'='*90}")
    print(f"{'Regime':<35s} {'Stab':>5s} {'Train':>8s} {'Pred':>8s} {'O_norm':>7s} {'σ²_drift':>8s} {'Time':>6s}")
    print(f"{'-'*35} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*6}")
    for r in results:
        print(f"{r['regime']:<35s} {r['stability_pct']:>4.0f}% "
              f"{r['train_error']:>7.2%} {r['pred_error']:>7.2%} "
              f"{r['O_norm']:>7.1f} {r['mean_v_drift']:>7.1f}% "
              f"{r['runtime']:>5.0f}s")

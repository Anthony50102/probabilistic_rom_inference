"""
Joint Latent-X Model with Integral Form Constraint

Full uncertainty flow: observations → GP → derivatives → operator
- Latent states X sampled via non-centered parameterization (X = L @ X_raw)
- GP hyperparameters (ℓ, σ², ν) sampled with informative priors from MLE warm start
- Pointwise observation likelihood: y|X ~ N(X, √ν)
- Dual physics constraints:
  1. Derivative: dX/dt ≈ f(X)O^T  (local accuracy, uses GP derivative uncertainty)
  2. Integral:  X(tb)-X(ta) ≈ ∫ f(X)O^T ds  (global consistency, prevents null basin)
- Operator O ~ N(O_ls, γ·|O_ls|) with informative prior from LS warm start

Why this should work:
- Pointwise observation y|X ~ N(X,√ν) is O(n_train) in log-prob magnitude
- Physics constraints are O(n_eval) — naturally balanced, unlike marginal likelihood
- Integral form structurally prevents null basin (state diffs ≠ 0 → O must be nontrivial)
- Non-centered parameterization handles the GP funnel geometry
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


def build_latent_integral_model(
    rom, num_modes, time_sampled, snapshots_comp,
    O_prior, mle_Ls, mle_Vs, mle_Ns,
    num_eval_points=400, window_size=10,
    deriv_weight=1.0, integral_weight=1.0,
    gp_prior_scale=0.3,
    noise_mode="fixed",  # "fixed", "tight", or "free"
    noise_prior_scale=0.05,  # only used when noise_mode="tight"
):
    """
    Build joint latent-X model with integral form constraint.

    The GP hyperparameters, latent states X, and operator O are all sampled.
    Uncertainty flows: observations → GP states → derivatives → operator.
    """
    t_train = jnp.array(time_sampled)
    n_train = len(t_train)
    T = float(t_train[-1] - t_train[0])
    y_obs = jnp.array(snapshots_comp)

    # Evaluation grid (denser than training for ODE constraints)
    time_eval = np.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
    t_eval = jnp.array(time_eval)
    n_eval = num_eval_points
    dt_eval = float(time_eval[1] - time_eval[0])

    # Precompute squared-difference matrices (constant across SVI steps)
    sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
    sq_diffs_et = (t_eval[:, None] - t_train[None, :]) ** 2
    diffs_et = t_eval[:, None] - t_train[None, :]
    sq_diffs_ee = (t_eval[:, None] - t_eval[None, :]) ** 2
    I_train = jnp.eye(n_train)
    I_eval = jnp.eye(n_eval)

    # Operator shape
    op_shape = O_prior.shape
    O_prior_jnp = jnp.array(O_prior)

    # MLE values for prior centers
    mle_log_ells = jnp.array([jnp.log(l) for l in mle_Ls])
    mle_log_sig2s = jnp.array([jnp.log(v) for v in mle_Vs])
    mle_log_nus = jnp.array([jnp.log(n) for n in mle_Ns])
    mle_nus_jnp = jnp.array(mle_Ns)
    _noise_mode = noise_mode
    _noise_prior_scale = noise_prior_scale

    # Precompute integration windows
    n_windows = n_eval // window_size
    ws_list = [i * window_size for i in range(n_windows)]
    we_list = [(i + 1) * window_size - 1 for i in range(n_windows)]
    if we_list[-1] < n_eval - 1:
        we_list[-1] = n_eval - 1

    # Precompute trapezoidal weights and window durations (outside model!)
    trap_weights = []
    window_durations = []
    for ws, we in zip(ws_list, we_list):
        n_pts = we - ws + 1
        w = jnp.ones(n_pts) * dt_eval
        w = w.at[0].set(0.5 * dt_eval)
        w = w.at[-1].set(0.5 * dt_eval)
        trap_weights.append(w)
        window_durations.append(float(time_eval[we] - time_eval[ws]))

    # RBF kernel from precomputed squared diffs
    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    # Single-mode GP forward: kernel → Cholesky → X = L @ X_raw
    def _single_gp_forward(ell, sig2, X_raw, jitter):
        eff_jitter = jnp.maximum(jitter, sig2 * 1e-4)
        K = _rbf_sq(ell, sig2, sq_diff_tt) + eff_jitter * I_train
        L = jnp.linalg.cholesky(K)
        X = L @ X_raw
        return L, X

    # Single-mode interpolation + derivative computation
    def _single_interp_deriv(ell, sig2, L, X):
        ell2 = ell ** 2
        K_et = _rbf_sq(ell, sig2, sq_diffs_et)
        K_inv_X = jax.scipy.linalg.cho_solve((L, True), X)
        X_eval = K_et @ K_inv_X

        # Derivative cross-covariance
        K_zy = -(diffs_et / ell2) * K_et
        mu_z = K_zy @ K_inv_X

        # Derivative auto-covariance
        K_ee = _rbf_sq(ell, sig2, sq_diffs_ee)
        K_zz = ((1.0 - sq_diffs_ee / ell2) / ell2) * K_ee
        K_inv_Kzy_T = jax.scipy.linalg.cho_solve((L, True), K_zy.T)
        A = K_zz - K_zy @ K_inv_Kzy_T
        A = 0.5 * (A + A.T)

        # Only need diagonal for efficiency
        deriv_var = jnp.maximum(jnp.diag(A), 0.0)

        return X_eval, mu_z, deriv_var

    _batch_gp_forward = jax.vmap(_single_gp_forward, in_axes=(0, 0, 0, None))
    _batch_interp_deriv = jax.vmap(_single_interp_deriv)

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
        if _noise_mode == "fixed":
            nus = mle_nus_jnp
        elif _noise_mode == "tight":
            nus = jnp.stack([
                numpyro.sample(f"noise_{i}",
                              dist.LogNormal(mle_log_nus[i], _noise_prior_scale))
                for i in range(num_modes)
            ])
        else:  # "free"
            nus = jnp.stack([
                numpyro.sample(f"noise_{i}",
                              dist.LogNormal(mle_log_nus[i], gp_prior_scale))
                for i in range(num_modes)
            ])

        # --- Latent states (non-centered parameterization) ---
        X_raws = jnp.stack([
            numpyro.sample(f"X_raw_{i}", dist.Normal(jnp.zeros(n_train), jnp.ones(n_train)))
            for i in range(num_modes)
        ])

        # GP forward: K → L → X = L @ X_raw
        Ls, Xs = _batch_gp_forward(ells, sig2s, X_raws, jitter)

        # Observation likelihood (pointwise — this is the key difference from marginal)
        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", Xs[i])
            numpyro.sample(f"obs_{i}", dist.Normal(Xs[i], jnp.sqrt(nus[i])), obs=y_obs[i])

        # Interpolate to eval grid + compute derivatives
        Xs_eval, mu_zs, deriv_vars = _batch_interp_deriv(ells, sig2s, Ls, Xs)

        for i in range(num_modes):
            numpyro.deterministic(f"X_eval_{i}", Xs_eval[i])

        # --- Operator with informative prior ---
        prior_scale = gamma * jnp.maximum(jnp.abs(O_prior_jnp), 0.5)
        O = numpyro.sample("O", dist.Normal(O_prior_jnp, prior_scale))

        # --- Compute operator dynamics: f(X_eval) @ O^T ---
        f_Xi = rom.model._assemble_data_matrix(Xs_eval, inputs=None) @ O.T

        # --- CONSTRAINT 1: Derivative matching (diagonal GP uncertainty) ---
        if deriv_weight > 0:
            for i in range(num_modes):
                total_var = deriv_vars[i] + gamma2 + jitter
                numpyro.factor(f"ode_constraint_{i}",
                    deriv_weight * jnp.sum(
                        dist.Normal(f_Xi[:, i], jnp.sqrt(total_var)).log_prob(mu_zs[i])))

        # --- CONSTRAINT 2: Integral form (prevents null basin) ---
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
    gp_prior_scale=0.3,
    noise_mode="fixed", noise_prior_scale=0.05,
    num_steps=8000, learning_rate=3e-3,
    num_posterior_samples=500, num_eval_points=400,
    seed=42, label="",
):
    """Run the latent-X + integral form experiment."""
    np.random.seed(seed)
    rng_key = random.PRNGKey(seed)

    TRAINING_SPAN = (0, 0.08)
    PREDICTION_SPAN = (0, 0.15)

    print(f"\n{'='*70}")
    print(f"LATENT-X + INTEGRAL FORM MODEL {label}")
    print(f"{'='*70}")
    print(f"Data: noise={noise_level}, samples={num_samples}, modes={num_modes}")
    print(f"Model: γ={gamma}, γ₂={gamma2}, window={window_size}")
    print(f"       deriv_w={deriv_weight}, integral_w={integral_weight}")
    print(f"       gp_prior_scale={gp_prior_scale}, noise_mode={noise_mode}")
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
    model, time_eval = build_latent_integral_model(
        rom=rom, num_modes=num_modes,
        time_sampled=t_samp, snapshots_comp=snaps_comp,
        O_prior=O_ls, mle_Ls=Ls, mle_Vs=Vs, mle_Ns=Ns,
        num_eval_points=num_eval_points, window_size=window_size,
        deriv_weight=deriv_weight, integral_weight=integral_weight,
        gp_prior_scale=gp_prior_scale,
        noise_mode=noise_mode, noise_prior_scale=noise_prior_scale,
    )

    # --- Init values ---
    init_values = {'O': jnp.array(O_ls)}
    for i in range(num_modes):
        init_values[f'lengthscale_{i}'] = Ls[i]
        init_values[f'variance_{i}'] = Vs[i]
        if noise_mode != "fixed":
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
        l_drift = (l_svi - Ls[i]) / Ls[i] * 100
        v_drift = (v_svi - Vs[i]) / Vs[i] * 100
        gp_drifts.append(abs(v_drift))
        if noise_mode != "fixed" and f'noise_{i}' in samples:
            n_svi = float(np.median(samples[f'noise_{i}']))
            n_drift = (n_svi - Ns[i]) / max(Ns[i], 1e-10) * 100
            print(f"  Mode {i}: ℓ={l_svi:.5f} ({l_drift:+.1f}%), "
                  f"σ²={v_svi:.4f} ({v_drift:+.1f}%), "
                  f"ν={n_svi:.6f} ({n_drift:+.1f}%)")
        else:
            print(f"  Mode {i}: ℓ={l_svi:.5f} ({l_drift:+.1f}%), "
                  f"σ²={v_svi:.4f} ({v_drift:+.1f}%), "
                  f"ν={Ns[i]:.6f} (fixed)")
    mean_v_drift = np.mean(gp_drifts)
    print(f"  Mean |σ² drift|: {mean_v_drift:.1f}%")
    if mean_v_drift > 50:
        print(f"  WARNING: Significant GP drift — possible null basin!")

    # GP hyperparameter uncertainty (key diagnostic for UQ flow)
    print(f"\nGP hyperparameter uncertainty (posterior std / median):")
    for i in range(num_modes):
        l_std = float(np.std(samples[f'lengthscale_{i}']))
        l_med = float(np.median(samples[f'lengthscale_{i}']))
        v_std = float(np.std(samples[f'variance_{i}']))
        v_med = float(np.median(samples[f'variance_{i}']))
        nu_str = ""
        if noise_mode != "fixed" and f'noise_{i}' in samples:
            n_std = float(np.std(samples[f'noise_{i}']))
            n_med = float(np.median(samples[f'noise_{i}']))
            nu_str = f", CV(ν)={n_std/n_med:.3f}"
        print(f"  Mode {i}: CV(ℓ)={l_std/l_med:.3f}, CV(σ²)={v_std/v_med:.3f}{nu_str}")

    # Operator
    O_samp = _find_operator_samples(samples, "O")
    if O_samp.ndim == 2:
        O_samp = O_samp[np.newaxis, ...]
    O_med = np.median(O_samp, axis=0)
    O_std = np.std(O_samp, axis=0)
    print(f"\nOperator: norm={np.linalg.norm(O_med):.1f} (LS: {np.linalg.norm(O_ls):.1f})")
    print(f"  Norm ratio (SVI/LS): {np.linalg.norm(O_med)/max(np.linalg.norm(O_ls), 1e-10):.3f}")
    print(f"  Mean elem std: {np.mean(O_std):.4f}, Max: {np.max(O_std):.4f}")

    # Null basin check
    if np.linalg.norm(O_med) < 0.1 * np.linalg.norm(O_ls):
        print(f"  ALERT: Operator collapsed to near-zero (null basin)!")

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

        # Per-mode errors
        print(f"\nPer-mode errors (training):")
        for i in range(num_modes):
            mode_err = np.linalg.norm(rom_med[i, train_mask]-ta[i, train_mask])/max(np.linalg.norm(ta[i, train_mask]), 1e-10)
            print(f"  Mode {i}: {mode_err:.4%}")

        # Confidence intervals
        q05 = np.percentile(rom_arr, 5, axis=0)
        q95 = np.percentile(rom_arr, 95, axis=0)
        ci_width = np.mean(q95 - q05)
        print(f"\n90% CI width: {ci_width:.6f}")
        print(f"Relative CI:  {ci_width/np.mean(np.abs(rom_med)):.4%}")

        # Coverage
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
    }


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT: Joint Latent-X + Integral Form Constraint")
    print("Goal: Full UQ flow with null-basin prevention")
    print("=" * 70)

    # Configuration sweep — testing key parameters
    configs = [
        # Config A: Baseline — moderate priors, both constraints
        dict(gamma=2.0, gamma2=0.5, deriv_weight=1.0, integral_weight=1.0,
             gp_prior_scale=0.3, num_steps=8000, learning_rate=3e-3,
             label="[A] Baseline dual constraint"),

        # Config B: Tighter GP priors (less GP freedom)
        dict(gamma=2.0, gamma2=0.5, deriv_weight=1.0, integral_weight=1.0,
             gp_prior_scale=0.1, num_steps=8000, learning_rate=3e-3,
             label="[B] Tight GP priors (scale=0.1)"),

        # Config C: Integral-only (no derivative constraint)
        dict(gamma=2.0, gamma2=0.5, deriv_weight=0.0, integral_weight=2.0,
             gp_prior_scale=0.3, num_steps=8000, learning_rate=3e-3,
             label="[C] Integral-only (no derivative)"),

        # Config D: Stronger integral weight
        dict(gamma=2.0, gamma2=0.5, deriv_weight=1.0, integral_weight=3.0,
             gp_prior_scale=0.3, num_steps=8000, learning_rate=3e-3,
             label="[D] Strong integral (w=3)"),

        # Config E: Looser GP priors (more GP freedom)
        dict(gamma=2.0, gamma2=0.5, deriv_weight=1.0, integral_weight=1.0,
             gp_prior_scale=0.5, num_steps=8000, learning_rate=3e-3,
             label="[E] Loose GP priors (scale=0.5)"),

        # Config F: More SVI steps + lower LR
        dict(gamma=2.0, gamma2=0.5, deriv_weight=1.0, integral_weight=1.0,
             gp_prior_scale=0.3, num_steps=15000, learning_rate=1e-3,
             label="[F] Long SVI (15k steps, lr=1e-3)"),
    ]

    results = []
    for cfg in configs:
        label = cfg.pop('label')
        r = run_experiment(label=label, **cfg)
        r['label'] = label
        results.append(r)
        # Don't store bulky samples in summary
        r_summary = {k: v for k, v in r.items() if k != 'samples'}
        print(f"\n>>> {label}: stab={r['stability_pct']:.0f}%, "
              f"train={r['train_error']:.2%}, pred={r['pred_error']:.2%}, "
              f"O_norm={r['O_norm']:.1f}, v_drift={r['mean_v_drift']:.1f}%")

    # Final comparison table
    print(f"\n\n{'='*90}")
    print(f"SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"{'Config':<40s} {'Stab':>5s} {'Train':>8s} {'Pred':>8s} {'O_norm':>7s} {'σ²_drift':>8s} {'Time':>6s}")
    print(f"{'-'*40} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*6}")
    for r in results:
        print(f"{r['label']:<40s} {r['stability_pct']:>4.0f}% "
              f"{r['train_error']:>7.2%} {r['pred_error']:>7.2%} "
              f"{r['O_norm']:>7.1f} {r['mean_v_drift']:>7.1f}% "
              f"{r['runtime']:>5.0f}s")

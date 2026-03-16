"""
Model: Joint GP + Integral Form (The Target Model)

This is the theoretically elegant model we want for the paper:
- GP hyperparameters are jointly learned (not fixed at MLE)
- Operator is jointly learned
- Integral form constraint prevents the null basin
- Warm started from MLE + LS for fast convergence

The integral form structurally eliminates the null basin because:
state differences ΔX are non-zero → operator must be non-trivial.
Unlike derivative matching, where large derivative variance makes
the constraint vacuous, the integral constraint directly compares
predicted state changes to observed state changes.
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
from core.bayesian_opinf import (
    fit_gp_hyperparameters_mle, _find_operator_samples,
)
import config
from config import Basis
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)


def build_joint_integral_model(
    rom, num_modes, time_sampled, snapshots,
    gp_ls_prior_loc, gp_var_prior_loc, gp_noise_prior_loc,
    gp_prior_scale=0.25,
    num_eval_points=400,
    window_size=10,
    integral_weight=1.0,
    deriv_weight=1.0,
):
    """
    Joint model: GP hyperparameters + operator learned together,
    with integral form constraint to prevent null basin.
    """
    t_train = jnp.array(time_sampled)
    n_train = len(t_train)
    y_obs = jnp.array(snapshots)
    op_shape = rom.model.operator_matrix.shape

    time_eval = jnp.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
    n_eval = len(time_eval)
    dt_eval = float(time_eval[1] - time_eval[0])

    # Precompute distance matrices
    sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
    I_train = jnp.eye(n_train)
    diffs_et = time_eval[:, None] - t_train[None, :]
    sq_diffs_et = diffs_et ** 2
    sq_diffs_ee = (time_eval[:, None] - time_eval[None, :]) ** 2

    # Integral windows
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

    _gp_ls_loc = jnp.array(gp_ls_prior_loc)
    _gp_var_loc = jnp.array(gp_var_prior_loc)
    _gp_noise_loc = jnp.array(gp_noise_prior_loc)

    def _rbf(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def model(gamma=1.0, gamma2=1.0, jitter=1e-4, beta=1.0):
        # GP hyperparameters (jointly learned)
        ells = jnp.stack([
            numpyro.sample(f"lengthscale_{i}",
                          dist.LogNormal(_gp_ls_loc[i], gp_prior_scale))
            for i in range(num_modes)])
        sig2s = jnp.stack([
            numpyro.sample(f"variance_{i}",
                          dist.LogNormal(_gp_var_loc[i], gp_prior_scale))
            for i in range(num_modes)])
        nus = jnp.stack([
            numpyro.sample(f"noise_{i}",
                          dist.LogNormal(_gp_noise_loc[i], gp_prior_scale))
            for i in range(num_modes)])

        Xs_eval_all = []
        mu_zs_all = []
        deriv_vars_all = []

        for i in range(num_modes):
            # Observation likelihood via GP marginal
            K = _rbf(ells[i], sig2s[i], sq_diff_tt)
            K_y = K + (nus[i] + jitter) * I_train
            numpyro.sample(f"obs_{i}",
                          dist.MultivariateNormal(jnp.zeros(n_train), K_y),
                          obs=y_obs[i])

            # Conditional mean: recover X
            L_y = jnp.linalg.cholesky(K_y)
            K_inv_y = jax.scipy.linalg.cho_solve((L_y, True), y_obs[i])
            X_train_i = K @ K_inv_y
            numpyro.deterministic(f"X_{i}", X_train_i)

            # Interpolation to eval grid
            ell2 = ells[i] ** 2
            K_et = _rbf(ells[i], sig2s[i], sq_diffs_et)
            X_eval_i = K_et @ K_inv_y
            numpyro.deterministic(f"X_eval_{i}", X_eval_i)
            Xs_eval_all.append(X_eval_i)

            # Derivative posterior
            K_zy = -(diffs_et / ell2) * K_et
            mu_z_i = K_zy @ K_inv_y
            numpyro.deterministic(f"mu_z_{i}", mu_z_i)
            mu_zs_all.append(mu_z_i)

            # Derivative variance (diagonal)
            K_ee = _rbf(ells[i], sig2s[i], sq_diffs_ee)
            K_zz = ((1.0 - sq_diffs_ee / ell2) / ell2) * K_ee
            V = jax.scipy.linalg.solve_triangular(L_y, K_zy.T, lower=True)
            A_diag = jnp.maximum(jnp.diag(K_zz - V.T @ V), 0.0)
            deriv_vars_all.append(A_diag)

        Xs_eval = jnp.stack(Xs_eval_all)  # (num_modes, n_eval)
        mu_zs = jnp.stack(mu_zs_all)
        deriv_vars = jnp.stack(deriv_vars_all)

        # Operator
        O = numpyro.sample("O",
            dist.Normal(jnp.zeros(op_shape), gamma * jnp.ones(op_shape)))

        # Operator dynamics
        f_Xi = rom.model._assemble_data_matrix(Xs_eval, inputs=None) @ O.T  # (n_eval, num_modes)

        # CONSTRAINT 1: Derivative matching (pointwise, diagonal cov)
        if deriv_weight > 0:
            for i in range(num_modes):
                total_var = deriv_vars[i] + gamma2 + jitter
                numpyro.factor(f"deriv_{i}",
                    deriv_weight * jnp.sum(
                        dist.Normal(f_Xi[:, i], jnp.sqrt(total_var)).log_prob(mu_zs[i])))

        # CONSTRAINT 2: Integral form (structurally prevents null basin)
        if integral_weight > 0:
            for i in range(num_modes):
                for w_idx, (ws, we) in enumerate(zip(ws_list, we_list)):
                    delta_X_obs = Xs_eval[i, we] - Xs_eval[i, ws]
                    delta_X_pred = jnp.sum(trap_weights[w_idx] * f_Xi[ws:we+1, i])
                    window_dur = window_durations[w_idx]
                    numpyro.factor(
                        f"integral_{i}_{w_idx}",
                        integral_weight * dist.Normal(
                            delta_X_pred, jnp.sqrt(gamma2) * window_dur
                        ).log_prob(delta_X_obs))

    return model, np.array(time_eval)


def main():
    np.random.seed(42)
    NOISE_LEVEL = 0.03
    NUM_SAMPLES = 250
    NUM_MODES = 6
    TRAINING_SPAN = (0, 0.08)
    PREDICTION_SPAN = (0, 0.15)
    NUM_EVAL_POINTS = 400

    print(f"{'='*70}")
    print(f"Joint GP + Integral Form — The Target Model")
    print(f"{'='*70}")

    # Data
    (fom, t_full, true_states, t_samp, snaps_samp) = \
        generate_trajectory(config, config.time_domain, TRAINING_SPAN, NUM_SAMPLES, NOISE_LEVEL)
    basis = Basis(num_vectors=NUM_MODES)
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

    # MLE warm start
    Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(t_samp, snaps_comp, verbose=True)

    t_eval = np.linspace(float(t_samp[0]), float(t_samp[-1]), NUM_EVAL_POINTS)
    X_eval = np.zeros((NUM_MODES, NUM_EVAL_POINTS))
    for i in range(NUM_MODES):
        K = rbf_eval(Ls[i], Vs[i], t_samp, t_samp) + (Ns[i] + 1e-5) * np.eye(len(t_samp))
        Ks = rbf_eval(Ls[i], Vs[i], t_eval, t_samp)
        X_eval[i] = Ks @ np.linalg.solve(K, snaps_comp[i])

    D = np.array(rom.model._assemble_data_matrix(jnp.array(X_eval), inputs=None))
    mu_z, _ = compute_gp_derivatives(Ls, Vs, t_samp, t_eval, snaps_comp, Ns=Ns)
    dXdt = np.array(mu_z)
    DtD = D.T @ D
    O_ls = np.linalg.solve(DtD + 1.0 * np.eye(DtD.shape[0]), D.T @ dXdt.T).T
    print(f"LS operator norm: {np.linalg.norm(O_ls):.1f}")

    # === Sweep over configurations ===
    configs = [
        # (name, gamma, gamma2, gp_scale, window, deriv_w, int_w, steps)
        ("Integral only (w=10)",     2.0, 0.5, 0.25, 10, 0.0, 1.0, 10000),
        ("Integral only (w=20)",     2.0, 0.5, 0.25, 20, 0.0, 1.0, 10000),
        ("Dual (w=10)",              2.0, 0.5, 0.25, 10, 1.0, 1.0, 10000),
        ("Dual (w=10, tight GP)",    2.0, 0.5, 0.15, 10, 1.0, 1.0, 10000),
        ("Dual (w=10, loose GP)",    2.0, 0.5, 0.50, 10, 1.0, 1.0, 10000),
        ("Strong integral (w=10)",   2.0, 0.5, 0.25, 10, 0.5, 2.0, 10000),
    ]

    rng_key = random.PRNGKey(42)
    all_results = []

    for name, gamma, gamma2, gp_scale, window, dw, iw, steps in configs:
        print(f"\n{'─'*70}")
        print(f"Config: {name}")
        print(f"  γ={gamma}, γ₂={gamma2}, gp_scale={gp_scale}, window={window}")
        print(f"  deriv_weight={dw}, integral_weight={iw}, steps={steps}")

        model, t_ev = build_joint_integral_model(
            rom=rom, num_modes=NUM_MODES,
            time_sampled=t_samp, snapshots=snaps_comp,
            gp_ls_prior_loc=np.log(Ls),
            gp_var_prior_loc=np.log(Vs),
            gp_noise_prior_loc=np.log(Ns),
            gp_prior_scale=gp_scale,
            num_eval_points=NUM_EVAL_POINTS,
            window_size=window,
            integral_weight=iw, deriv_weight=dw,
        )

        init_values = {'O': jnp.array(O_ls)}
        for i in range(NUM_MODES):
            init_values[f'lengthscale_{i}'] = Ls[i]
            init_values[f'variance_{i}'] = Vs[i]
            init_values[f'noise_{i}'] = Ns[i]

        model_kwargs = dict(gamma=gamma, gamma2=gamma2, jitter=1e-5, beta=1.0)
        guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
        optimizer = ClippedAdam(step_size=1e-3)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        rng_key, ik = random.split(rng_key)
        t0 = time.time()
        svi_state = svi.init(ik, **model_kwargs)

        @jax.jit
        def _step(s, _):
            s, l = svi.update(s, **model_kwargs)
            return s, l

        svi_state, losses = jax.lax.scan(_step, svi_state, jnp.arange(steps))
        losses_np = np.array(losses)

        params = svi.get_params(svi_state)
        rng_key, sk, pk = random.split(rng_key, 3)
        post = guide.sample_posterior(sk, params, sample_shape=(500,), **model_kwargs)
        pred = Predictive(model, posterior_samples=post, num_samples=500)
        out = pred(pk, **model_kwargs)
        samples = {**out, **post}
        dt = time.time() - t0

        # Evaluate
        O_samp = _find_operator_samples(samples, "O")
        if O_samp.ndim == 2:
            O_samp = O_samp[np.newaxis, ...]

        t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
        Os, Xs, rom_solves = generate_rom_predictions(
            samples=samples, rom=rom,
            snapshots_compressed=snaps_comp,
            time_eval=t_pred, num_modes=NUM_MODES, num_pulls=200)

        n_stable = len(rom_solves)
        stab = n_stable / max(len(Os), 1) * 100

        train_err = pred_err = float('inf')
        if n_stable > 0:
            rom_arr = np.array(rom_solves)
            rom_med = np.median(rom_arr, axis=0)
            train_mask = t_pred <= TRAINING_SPAN[1]
            pred_mask = t_pred > TRAINING_SPAN[1]
            from scipy.interpolate import interp1d
            true_interp = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
            true_at = true_interp(t_pred)
            train_err = float(np.linalg.norm(rom_med[:, train_mask] - true_at[:, train_mask]) /
                             np.linalg.norm(true_at[:, train_mask]))
            if np.any(pred_mask):
                pred_err = float(np.linalg.norm(rom_med[:, pred_mask] - true_at[:, pred_mask]) /
                                np.linalg.norm(true_at[:, pred_mask]))

        # GP drift
        v_drifts = []
        for i in range(NUM_MODES):
            v_svi = float(np.median(samples[f'variance_{i}']))
            drift = abs(v_svi - Vs[i]) / Vs[i] * 100
            v_drifts.append(drift)
        gp_drift = np.mean(v_drifts)

        print(f"  Loss: {losses_np[0]:.0f} → {losses_np[-1]:.0f}")
        print(f"  Stability: {n_stable}/{len(Os)} ({stab:.0f}%)")
        print(f"  Train err: {train_err:.2%}, Pred err: {pred_err:.2%}")
        print(f"  O norm: {np.linalg.norm(np.median(O_samp, axis=0)):.1f}")
        print(f"  GP σ² drift: {gp_drift:.1f}%")
        print(f"  Runtime: {dt:.1f}s")

        all_results.append({
            'name': name, 'stability': stab, 'train_err': train_err,
            'pred_err': pred_err, 'gp_drift': gp_drift, 'runtime': dt,
        })

    # Summary
    print(f"\n{'='*90}")
    print(f"JOINT MODEL COMPARISON")
    print(f"{'='*90}")
    print(f"{'Config':<35s}  {'Stab':>5s}  {'Train':>7s}  {'Pred':>7s}  {'GP drift':>8s}  {'Time':>5s}")
    print(f"{'─'*90}")
    for r in all_results:
        print(f"{r['name']:<35s}  "
              f"{r['stability']:>4.0f}%  "
              f"{r['train_err']:>6.2%}  "
              f"{r['pred_err']:>6.2%}  "
              f"{r['gp_drift']:>7.1f}%  "
              f"{r['runtime']:>4.0f}s")


if __name__ == "__main__":
    main()

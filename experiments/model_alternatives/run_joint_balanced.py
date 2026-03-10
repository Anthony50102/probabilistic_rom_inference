"""
Model: Joint GP + Balanced Likelihoods

Key insight from previous experiments: the GP marginal likelihood (~120K)
overwhelms the ODE constraint (~9K), so the optimizer focuses entirely on
fitting observations and ignores the operator.

Solution: scale the observation likelihood by a factor β_obs < 1 to balance
the two terms. This is theoretically justified as a "tempered posterior"
or "power likelihood" — a well-known technique in Bayesian inference for
combining multiple data sources with different scales.

Also try: precomputing the observation likelihood (since it only depends on
GP hyperparameters) and using it as a regularizer on the joint ELBO.
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


def build_balanced_joint_model(
    rom, num_modes, time_sampled, snapshots,
    gp_ls_loc, gp_var_loc, gp_noise_loc,
    gp_prior_scale=0.25,
    num_eval_points=400,
    window_size=10,
    obs_weight=0.1,
    deriv_weight=1.0,
    integral_weight=1.0,
):
    """Joint model with reweighted observation likelihood."""
    t_train = jnp.array(time_sampled)
    n_train = len(t_train)
    y_obs = jnp.array(snapshots)
    op_shape = rom.model.operator_matrix.shape

    time_eval = jnp.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
    n_eval = len(time_eval)
    dt_eval = float(time_eval[1] - time_eval[0])

    sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
    I_train = jnp.eye(n_train)
    diffs_et = time_eval[:, None] - t_train[None, :]
    sq_diffs_et = diffs_et ** 2
    sq_diffs_ee = (time_eval[:, None] - time_eval[None, :]) ** 2

    n_windows = n_eval // window_size
    ws_list = [i * window_size for i in range(n_windows)]
    we_list = [(i + 1) * window_size - 1 for i in range(n_windows)]
    if we_list[-1] < n_eval - 1:
        we_list[-1] = n_eval - 1

    trap_weights = []
    window_durs = []
    for ws, we in zip(ws_list, we_list):
        n_pts = we - ws + 1
        w = jnp.ones(n_pts) * dt_eval
        w = w.at[0].set(0.5 * dt_eval)
        w = w.at[-1].set(0.5 * dt_eval)
        trap_weights.append(w)
        window_durs.append(float(time_eval[we] - time_eval[ws]))

    _gp_ls_loc = jnp.array(gp_ls_loc)
    _gp_var_loc = jnp.array(gp_var_loc)
    _gp_noise_loc = jnp.array(gp_noise_loc)

    def _rbf(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def model(gamma=1.0, gamma2=1.0, jitter=1e-4, beta=1.0):
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
            K = _rbf(ells[i], sig2s[i], sq_diff_tt)
            K_y = K + (nus[i] + jitter) * I_train

            # DOWNWEIGHTED observation likelihood
            obs_ll = dist.MultivariateNormal(jnp.zeros(n_train), K_y).log_prob(y_obs[i])
            numpyro.factor(f"obs_{i}", obs_weight * obs_ll)

            L_y = jnp.linalg.cholesky(K_y)
            K_inv_y = jax.scipy.linalg.cho_solve((L_y, True), y_obs[i])
            X_train_i = K @ K_inv_y
            numpyro.deterministic(f"X_{i}", X_train_i)

            ell2 = ells[i] ** 2
            K_et = _rbf(ells[i], sig2s[i], sq_diffs_et)
            X_eval_i = K_et @ K_inv_y
            numpyro.deterministic(f"X_eval_{i}", X_eval_i)
            Xs_eval_all.append(X_eval_i)

            K_zy = -(diffs_et / ell2) * K_et
            mu_z_i = K_zy @ K_inv_y
            numpyro.deterministic(f"mu_z_{i}", mu_z_i)
            mu_zs_all.append(mu_z_i)

            K_ee = _rbf(ells[i], sig2s[i], sq_diffs_ee)
            K_zz = ((1.0 - sq_diffs_ee / ell2) / ell2) * K_ee
            V = jax.scipy.linalg.solve_triangular(L_y, K_zy.T, lower=True)
            A_diag = jnp.maximum(jnp.diag(K_zz - V.T @ V), 0.0)
            deriv_vars_all.append(A_diag)

        Xs_eval = jnp.stack(Xs_eval_all)
        mu_zs = jnp.stack(mu_zs_all)
        deriv_vars = jnp.stack(deriv_vars_all)

        O = numpyro.sample("O",
            dist.Normal(jnp.zeros(op_shape), gamma * jnp.ones(op_shape)))

        f_Xi = rom.model._assemble_data_matrix(Xs_eval, inputs=None) @ O.T

        # Derivative constraint
        if deriv_weight > 0:
            for i in range(num_modes):
                total_var = deriv_vars[i] + gamma2 + jitter
                numpyro.factor(f"deriv_{i}",
                    deriv_weight * jnp.sum(
                        dist.Normal(f_Xi[:, i], jnp.sqrt(total_var)).log_prob(mu_zs[i])))

        # Integral constraint
        if integral_weight > 0:
            for i in range(num_modes):
                for w_idx, (ws, we) in enumerate(zip(ws_list, we_list)):
                    delta_X_obs = Xs_eval[i, we] - Xs_eval[i, ws]
                    delta_X_pred = jnp.sum(trap_weights[w_idx] * f_Xi[ws:we+1, i])
                    numpyro.factor(
                        f"integral_{i}_{w_idx}",
                        integral_weight * dist.Normal(
                            delta_X_pred, jnp.sqrt(gamma2) * window_durs[w_idx]
                        ).log_prob(delta_X_obs))

    return model, np.array(time_eval)


def main():
    np.random.seed(42)
    NOISE = 0.03
    N_SAMP = 250
    N_MODES = 6
    T_SPAN = (0, 0.08)
    P_SPAN = (0, 0.15)
    N_EVAL = 400

    print(f"{'='*70}")
    print(f"Joint GP + Balanced Likelihoods")
    print(f"{'='*70}")

    (fom, t_full, true_states, t_samp, snaps_samp) = \
        generate_trajectory(config, config.time_domain, T_SPAN, N_SAMP, NOISE)
    basis = Basis(num_vectors=N_MODES)
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

    Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(t_samp, snaps_comp, verbose=True)

    t_eval = np.linspace(float(t_samp[0]), float(t_samp[-1]), N_EVAL)
    X_eval = np.zeros((N_MODES, N_EVAL))
    for i in range(N_MODES):
        K = rbf_eval(Ls[i], Vs[i], t_samp, t_samp) + (Ns[i]+1e-5)*np.eye(len(t_samp))
        Ks = rbf_eval(Ls[i], Vs[i], t_eval, t_samp)
        X_eval[i] = Ks @ np.linalg.solve(K, snaps_comp[i])

    D = np.array(rom.model._assemble_data_matrix(jnp.array(X_eval), inputs=None))
    mu_z, _ = compute_gp_derivatives(Ls, Vs, t_samp, t_eval, snaps_comp, Ns=Ns)
    O_ls = np.linalg.solve(D.T@D + np.eye(D.shape[1]), D.T @ np.array(mu_z).T).T
    print(f"LS operator norm: {np.linalg.norm(O_ls):.1f}")

    # === Sweep over obs_weight ===
    configs = [
        # (name, obs_w, deriv_w, int_w, gamma, gamma2, gp_scale, steps)
        ("obs=1.0 (unbalanced)", 1.0,  1.0, 1.0, 2.0, 0.5, 0.25, 10000),
        ("obs=0.1",              0.1,  1.0, 1.0, 2.0, 0.5, 0.25, 10000),
        ("obs=0.01",             0.01, 1.0, 1.0, 2.0, 0.5, 0.25, 10000),
        ("obs=0.01, strong int", 0.01, 1.0, 3.0, 2.0, 0.5, 0.25, 10000),
        ("obs=0.001",            0.001,1.0, 1.0, 2.0, 0.5, 0.25, 10000),
        ("obs=0.01, tight GP",   0.01, 1.0, 1.0, 2.0, 0.5, 0.15, 10000),
        ("obs=0.01, more steps", 0.01, 1.0, 1.0, 2.0, 0.5, 0.25, 20000),
    ]

    rng_key = random.PRNGKey(42)
    all_results = []

    for name, obs_w, dw, iw, gamma, gamma2, gp_scale, steps in configs:
        print(f"\n{'─'*70}")
        print(f"Config: {name}")
        print(f"  obs_w={obs_w}, deriv_w={dw}, int_w={iw}, γ={gamma}, γ₂={gamma2}")

        model, t_ev = build_balanced_joint_model(
            rom=rom, num_modes=N_MODES,
            time_sampled=t_samp, snapshots=snaps_comp,
            gp_ls_loc=np.log(Ls), gp_var_loc=np.log(Vs), gp_noise_loc=np.log(Ns),
            gp_prior_scale=gp_scale, num_eval_points=N_EVAL,
            window_size=10, obs_weight=obs_w,
            deriv_weight=dw, integral_weight=iw,
        )

        init_values = {'O': jnp.array(O_ls)}
        for i in range(N_MODES):
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

        t_pred = np.linspace(P_SPAN[0], P_SPAN[1], 400)
        Os, Xs, rom_solves = generate_rom_predictions(
            samples=samples, rom=rom, snapshots_compressed=snaps_comp,
            time_eval=t_pred, num_modes=N_MODES, num_pulls=200)

        n_stable = len(rom_solves)
        stab = n_stable / max(len(Os), 1) * 100

        train_err = pred_err = float('inf')
        if n_stable > 0:
            rom_arr = np.array(rom_solves)
            rom_med = np.median(rom_arr, axis=0)
            train_mask = t_pred <= T_SPAN[1]
            pred_mask = t_pred > T_SPAN[1]
            from scipy.interpolate import interp1d
            ti = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
            ta = ti(t_pred)
            train_err = float(np.linalg.norm(rom_med[:, train_mask]-ta[:, train_mask])/np.linalg.norm(ta[:, train_mask]))
            if np.any(pred_mask):
                pred_err = float(np.linalg.norm(rom_med[:, pred_mask]-ta[:, pred_mask])/np.linalg.norm(ta[:, pred_mask]))

        v_drifts = [abs(float(np.median(samples[f'variance_{i}']))-Vs[i])/Vs[i]*100 for i in range(N_MODES)]
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
    print(f"BALANCED LIKELIHOOD COMPARISON")
    print(f"{'='*90}")
    print(f"{'Config':<30s}  {'Stab':>5s}  {'Train':>7s}  {'Pred':>7s}  {'GP drift':>8s}  {'Time':>5s}")
    print(f"{'─'*90}")
    for r in all_results:
        print(f"{r['name']:<30s}  "
              f"{r['stability']:>4.0f}%  "
              f"{r['train_err']:>6.2%}  "
              f"{r['pred_err']:>6.2%}  "
              f"{r['gp_drift']:>7.1f}%  "
              f"{r['runtime']:>4.0f}s")

    print(f"\nReference: Fixed GP + Dual Constraint = 2.45% train, 10.17% pred, 100% stable, 8s")


if __name__ == "__main__":
    main()

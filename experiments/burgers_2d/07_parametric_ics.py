"""
07 — Parametric Initial Conditions (Heat-Style Stacked Trajectories)
    2D Diffusion-Reaction Equation: ∂u/∂t = κ∇²u − βu²

Heat-style parametric variation: multiple training trajectories differ only
in their initial condition u₀(·; μ). A single global POD basis is fit to
the stacked snapshots and a single set of ROM operators (c, A, H) is
learned jointly across all trajectories via a shared posterior O. The
parameter μ never enters the operators — only the IC.

Evaluation is on a held-out μ: project its IC through the global basis and
integrate each posterior operator sample forward in time.

Usage:
    python 07_parametric_ics.py
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from numpyro.infer.initialization import init_to_value
from numpyro.optim import ClippedAdam
from jax import random
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
import config_parametric as config
from config_parametric import Basis
from core import JaxCompatibleModel, compute_gp_derivatives, rbf_eval
from core.bayesian_opinf import fit_gp_hyperparameters_mle, _find_operator_samples
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

# ── Data regime ──────────────────────────────────────────────────────────────
NUM_SAMPLES = 60
NOISE_LEVEL = 0.03
NUM_EVAL_POINTS = 200

# ── Model hyperparameters ───────────────────────────────────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=4,
    GAMMA=2.0,
    GAMMA2=10.0,
    DERIV_WEIGHT=1.0,
    INTEGRAL_WEIGHT=1.0,
    MLL_WEIGHT=0.1,
    GP_PRIOR_SCALE=0.1,
    WINDOW_SIZE=10,
    NUM_STEPS=10000,
    LEARNING_RATE=3e-3,
    NUM_POSTERIOR_SAMPLES=500,
    REGULARIZER=1.0,
    SEED=42,
)

TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 3.0)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# Per-trajectory data generation
# =============================================================================
def _generate_one(fom, mu, full_time_domain, training_span, num_samples, noise_level, rng):
    """Solve FOM for IC(μ), sub-sample training snapshots, add noise."""
    ic = config.initial_conditions(*mu)
    true_states = fom.solve(ic, full_time_domain)

    t_samp = np.sort(rng.uniform(training_span[0], training_span[1], size=num_samples))
    t_samp[0] = training_span[0]
    t_samp[-1] = training_span[1]

    clean = fom.solve(ic, t_samp)
    # Use fom.noise() — does not depend on np.random seed state via rng
    snaps_samp = fom.noise(clean, noise_level)
    return true_states, t_samp, snaps_samp


# =============================================================================
# Model builder — shared O across trajectories
# =============================================================================
def build_parametric_model(
    rom, num_modes,
    t_samp_list, snaps_comp_list,
    O_prior,
    mle_Ls_list, mle_Vs_list, mle_Ns_list,
    num_eval_points=200, window_size=10,
    deriv_weight=1.0, integral_weight=1.0,
    mll_weight=0.1, gp_prior_scale=0.1,
):
    """Build NumPyro model with shared operators across M trajectories.

    Each trajectory contributes its own GP hyperparameters, GP conditionals,
    derivative-matching and integral constraints. All trajectories share a
    single operator draw ``O``.
    """
    M = len(t_samp_list)
    assert len(snaps_comp_list) == M == len(mle_Ls_list)

    # Precompute per-trajectory kernel utilities and targets
    per_traj = []
    time_eval_list = []
    for m in range(M):
        t_samp = t_samp_list[m]
        snaps_comp = snaps_comp_list[m]
        t_train = jnp.array(t_samp)
        n_train = len(t_train)
        y_obs = jnp.array(snaps_comp)

        time_eval = np.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
        t_eval = jnp.array(time_eval)
        dt_eval = float(time_eval[1] - time_eval[0])

        sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
        sq_diffs_et = (t_eval[:, None] - t_train[None, :]) ** 2
        diffs_et = t_eval[:, None] - t_train[None, :]
        sq_diffs_ee = (t_eval[:, None] - t_eval[None, :]) ** 2
        I_train = jnp.eye(n_train)

        n_windows = num_eval_points // window_size
        ws_list = [i * window_size for i in range(n_windows)]
        we_list = [(i + 1) * window_size - 1 for i in range(n_windows)]
        if we_list[-1] < num_eval_points - 1:
            we_list[-1] = num_eval_points - 1

        trap_weights = []
        window_durations = []
        for ws, we in zip(ws_list, we_list):
            n_pts = we - ws + 1
            w = jnp.ones(n_pts) * dt_eval
            w = w.at[0].set(0.5 * dt_eval)
            w = w.at[-1].set(0.5 * dt_eval)
            trap_weights.append(w)
            window_durations.append(float(time_eval[we] - time_eval[ws]))

        per_traj.append(dict(
            t_train=t_train, n_train=n_train, y_obs=y_obs,
            sq_diff_tt=sq_diff_tt, sq_diffs_et=sq_diffs_et,
            diffs_et=diffs_et, sq_diffs_ee=sq_diffs_ee, I_train=I_train,
            ws_list=ws_list, we_list=we_list,
            trap_weights=trap_weights, window_durations=window_durations,
            log_ells=jnp.array([jnp.log(l) for l in mle_Ls_list[m]]),
            log_sig2s=jnp.array([jnp.log(v) for v in mle_Vs_list[m]]),
            log_nus=jnp.array([jnp.log(n) for n in mle_Ns_list[m]]),
        ))
        time_eval_list.append(time_eval)

    O_prior_jnp = jnp.array(O_prior)

    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def _single_gp_cond_factory(pt):
        sq_diff_tt = pt['sq_diff_tt']
        sq_diffs_et = pt['sq_diffs_et']
        diffs_et = pt['diffs_et']
        sq_diffs_ee = pt['sq_diffs_ee']
        I_train = pt['I_train']
        n_train = pt['n_train']

        def _single_gp_conditional(ell, sig2, nu, y_i):
            ell2 = ell ** 2
            jitter = jnp.maximum(1e-5, sig2 * 1e-4)
            K_tt = _rbf_sq(ell, sig2, sq_diff_tt) + (nu + jitter) * I_train
            L = jnp.linalg.cholesky(K_tt)
            alpha = jax.scipy.linalg.cho_solve((L, True), y_i)
            K_et = _rbf_sq(ell, sig2, sq_diffs_et)
            X_eval = K_et @ alpha
            K_zy = -(diffs_et / ell2) * K_et
            mu_z = K_zy @ alpha
            K_ee = _rbf_sq(ell, sig2, sq_diffs_ee)
            K_zz = ((1.0 - sq_diffs_ee / ell2) / ell2) * K_ee
            V = jax.scipy.linalg.cho_solve((L, True), K_zy.T)
            deriv_var = jnp.maximum(jnp.diag(K_zz) - jnp.sum(K_zy * V.T, axis=1), 0.0)
            mll = -0.5 * (jnp.dot(y_i, alpha) +
                          2.0 * jnp.sum(jnp.log(jnp.diag(L))) +
                          n_train * jnp.log(2.0 * jnp.pi))
            return X_eval, mu_z, deriv_var, mll

        return jax.vmap(_single_gp_conditional)

    batch_gp_per_traj = [_single_gp_cond_factory(pt) for pt in per_traj]

    def model(gamma=2.0, gamma2=10.0, jitter=1e-4):
        # Shared operator
        prior_scale = gamma * jnp.maximum(jnp.abs(O_prior_jnp), 0.5)
        O = numpyro.sample("O", dist.Normal(O_prior_jnp, prior_scale))

        for m in range(M):
            pt = per_traj[m]
            ells = jnp.stack([
                numpyro.sample(f"lengthscale_{m}_{i}",
                               dist.LogNormal(pt['log_ells'][i], gp_prior_scale))
                for i in range(num_modes)
            ])
            sig2s = jnp.stack([
                numpyro.sample(f"variance_{m}_{i}",
                               dist.LogNormal(pt['log_sig2s'][i], gp_prior_scale))
                for i in range(num_modes)
            ])
            nus = jnp.stack([
                numpyro.sample(f"noise_{m}_{i}",
                               dist.LogNormal(pt['log_nus'][i], gp_prior_scale))
                for i in range(num_modes)
            ])

            Xs_eval, mu_zs, deriv_vars, mlls = batch_gp_per_traj[m](
                ells, sig2s, nus, pt['y_obs'])

            for i in range(num_modes):
                numpyro.deterministic(f"X_{m}_{i}", Xs_eval[i])

            if mll_weight > 0:
                numpyro.factor(f"gp_mll_{m}", mll_weight * jnp.sum(mlls))

            f_Xi = rom.model._assemble_data_matrix(Xs_eval, inputs=None) @ O.T

            if deriv_weight > 0:
                for i in range(num_modes):
                    total_var = deriv_vars[i] + gamma2 + jitter
                    numpyro.factor(
                        f"ode_{m}_{i}",
                        deriv_weight * jnp.sum(
                            dist.Normal(f_Xi[:, i], jnp.sqrt(total_var))
                                .log_prob(mu_zs[i])))

            if integral_weight > 0:
                for i in range(num_modes):
                    for w_idx, (ws, we) in enumerate(zip(pt['ws_list'], pt['we_list'])):
                        delta_X_obs = Xs_eval[i, we] - Xs_eval[i, ws]
                        delta_X_pred = jnp.sum(
                            pt['trap_weights'][w_idx] * f_Xi[ws:we+1, i])
                        constraint_std = (jnp.sqrt(gamma2) *
                                          pt['window_durations'][w_idx])
                        numpyro.factor(
                            f"int_{m}_{i}_{w_idx}",
                            integral_weight * dist.Normal(
                                delta_X_pred, constraint_std).log_prob(delta_X_obs))

    return model, time_eval_list


# =============================================================================
# ROM prediction (scipy)
# =============================================================================
def _generate_rom_predictions_scipy(samples, rom, ic_comp, t_pred,
                                     num_modes, num_pulls=200):
    """Integrate posterior operator samples forward from the given compressed IC."""
    Os, rom_solves = [], []
    O_samples = _find_operator_samples(samples, "O")
    if O_samples.ndim == 2:
        O_samples = O_samples[np.newaxis, ...]

    n_total = len(O_samples)
    n_use = min(num_pulls, n_total)
    indices = np.linspace(0, n_total - 1, n_use, dtype=int)

    for idx in indices:
        O = O_samples[idx]
        Os.append(np.array(O))
        rom.model._extract_operators(np.array(O))

        def rhs(t, state, _rom=rom):
            return np.array(_rom.model.rhs(t, state, None))

        try:
            sol = solve_ivp(rhs, [t_pred[0], t_pred[-1]],
                            np.array(ic_comp),
                            t_eval=t_pred, method='RK45', max_step=0.01)
            if sol.success and np.all(np.isfinite(sol.y)):
                rom_solves.append(sol.y)
        except Exception:
            pass

    return (np.array(Os),
            np.array(rom_solves) if rom_solves else np.empty((0, num_modes, len(t_pred))))


# =============================================================================
# Run experiment
# =============================================================================
def run_experiment():
    p = MODEL_PARAMS
    np.random.seed(p['SEED'])
    rng = np.random.default_rng(p['SEED'])
    rng_key = random.PRNGKey(p['SEED'])
    num_modes = p['NUM_MODES']

    fom = config.FullOrderModel()
    full_time = config.time_domain

    print(f"\n{'='*70}")
    print(f"  Parametric ICs — {len(config.TRAINING_MUS)} training μ's, "
          f"test μ={config.TEST_MU}")
    print(f"  samples={NUM_SAMPLES} noise={NOISE_LEVEL:.0%} "
          f"NUM_MODES={num_modes}")
    print(f"{'='*70}")

    # ── 1. Generate training data for each μ ────────────────────────────
    true_states_list, t_samp_list, snaps_samp_list = [], [], []
    for m, mu in enumerate(config.TRAINING_MUS):
        # Seed per-trajectory so noise is reproducible and different per μ.
        local_rng = np.random.default_rng(p['SEED'] + 100 * (m + 1))
        ts, t_samp, snaps = _generate_one(
            fom, mu, full_time, TRAINING_SPAN,
            NUM_SAMPLES, NOISE_LEVEL, local_rng)
        true_states_list.append(ts)
        t_samp_list.append(t_samp)
        snaps_samp_list.append(snaps)
        print(f"  μ{m}={mu}: noisy snaps shape={snaps.shape} "
              f"range=[{snaps.min():.3f}, {snaps.max():.3f}]")

    # ── 2. Global POD basis on stacked snapshots ────────────────────────
    stacked = np.concatenate(snaps_samp_list, axis=1)
    basis = Basis(num_vectors=num_modes)
    basis.fit(stacked)
    print(f"  Global POD energy ({num_modes} modes): {basis.cumulative_energy:.4%}")

    snaps_comp_list = [basis.compress(s) for s in snaps_samp_list]
    true_comp_list = [basis.compress(ts) for ts in true_states_list]

    # ── 3. ROM skeleton (operators get overwritten during prediction) ───
    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp_list[0]),
        model=JaxCompatibleModel(operators="cAH",
                                 solver=opinf.lstsq.L2Solver(regularizer=p['REGULARIZER'])),
    )
    # Initial fit on first trajectory — only to populate operator shapes.
    rom.fit(states=snaps_samp_list[0])

    # ── 4. Per-trajectory GP hyperparameter MLE ─────────────────────────
    mle_Ls_list, mle_Vs_list, mle_Ns_list = [], [], []
    X_mle_list, mu_z_mle_list = [], []
    for m in range(len(config.TRAINING_MUS)):
        Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(
            t_samp_list[m], snaps_comp_list[m], verbose=False)
        mle_Ls_list.append(Ls); mle_Vs_list.append(Vs); mle_Ns_list.append(Ns)

        t_eval_ls = np.linspace(float(t_samp_list[m][0]),
                                float(t_samp_list[m][-1]), NUM_EVAL_POINTS)
        X_mle = np.zeros((num_modes, NUM_EVAL_POINTS))
        for i in range(num_modes):
            K = rbf_eval(Ls[i], Vs[i], t_samp_list[m], t_samp_list[m]) + \
                (Ns[i] + 1e-5) * np.eye(len(t_samp_list[m]))
            Ks = rbf_eval(Ls[i], Vs[i], t_eval_ls, t_samp_list[m])
            X_mle[i] = Ks @ np.linalg.solve(K, snaps_comp_list[m][i])

        mu_z_mle, _ = compute_gp_derivatives(
            Ls, Vs, t_samp_list[m], t_eval_ls, snaps_comp_list[m], Ns=Ns)
        X_mle_list.append(X_mle)
        mu_z_mle_list.append(np.array(mu_z_mle))

        T = t_samp_list[m][-1] - t_samp_list[m][0]
        print(f"  μ{m}: mode-0 ℓ={Ls[0]:.4f} (T/ℓ={T/Ls[0]:.0f}) "
              f"σ²={Vs[0]:.4f} ν={Ns[0]:.5f}")

    # ── 5. Stacked LS warm-start for shared O ───────────────────────────
    D_blocks = [
        np.array(rom.model._assemble_data_matrix(jnp.array(X_mle), inputs=None))
        for X_mle in X_mle_list
    ]
    D_big = np.concatenate(D_blocks, axis=0)
    Z_big = np.concatenate([z.T for z in mu_z_mle_list], axis=0)  # (M*N_eval, r)
    DtD = D_big.T @ D_big
    O_ls = np.linalg.solve(DtD + np.eye(DtD.shape[0]), D_big.T @ Z_big).T
    print(f"  Stacked LS operator: norm={np.linalg.norm(O_ls):.2f} "
          f"shape={O_ls.shape}")

    # ── 6. Build & run SVI ───────────────────────────────────────────────
    model, _ = build_parametric_model(
        rom=rom, num_modes=num_modes,
        t_samp_list=t_samp_list, snaps_comp_list=snaps_comp_list,
        O_prior=O_ls,
        mle_Ls_list=mle_Ls_list, mle_Vs_list=mle_Vs_list, mle_Ns_list=mle_Ns_list,
        num_eval_points=NUM_EVAL_POINTS, window_size=p['WINDOW_SIZE'],
        deriv_weight=p['DERIV_WEIGHT'], integral_weight=p['INTEGRAL_WEIGHT'],
        mll_weight=p['MLL_WEIGHT'], gp_prior_scale=p['GP_PRIOR_SCALE'],
    )

    init_values = {'O': jnp.array(O_ls)}
    for m in range(len(config.TRAINING_MUS)):
        for i in range(num_modes):
            init_values[f'lengthscale_{m}_{i}'] = mle_Ls_list[m][i]
            init_values[f'variance_{m}_{i}'] = mle_Vs_list[m][i]
            init_values[f'noise_{m}_{i}'] = mle_Ns_list[m][i]

    model_kwargs = dict(gamma=p['GAMMA'], gamma2=p['GAMMA2'], jitter=1e-4)
    guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    optimizer = ClippedAdam(step_size=p['LEARNING_RATE'])
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key, ik = random.split(rng_key)
    t0 = time.time()
    svi_state = svi.init(ik, **model_kwargs)

    @jax.jit
    def _step(s, _):
        s, l = svi.update(s, **model_kwargs)
        return s, l

    num_steps = p['NUM_STEPS']
    seg = max(1, num_steps // 10)
    all_losses = []
    for si in range(10):
        start = si * seg
        end = min(start + seg, num_steps) if si < 9 else num_steps
        if start >= num_steps:
            break
        svi_state, seg_losses = jax.lax.scan(_step, svi_state, jnp.arange(end - start))
        all_losses.extend(np.array(seg_losses).tolist())
        print(f"    step {end:6d}/{num_steps}  loss={all_losses[-1]:10.2f}")

    params = svi.get_params(svi_state)
    rng_key, sk, pk = random.split(rng_key, 3)
    n_post = p['NUM_POSTERIOR_SAMPLES']
    post = guide.sample_posterior(sk, params, sample_shape=(n_post,), **model_kwargs)
    pred = Predictive(model, posterior_samples=post, num_samples=n_post)
    out = pred(pk, **model_kwargs)
    samples = {**out, **post}
    runtime = time.time() - t0

    # ── 7. Evaluate on TEST_MU ───────────────────────────────────────────
    test_ic = config.initial_conditions(*config.TEST_MU)
    test_true = fom.solve(test_ic, full_time)
    test_true_comp = basis.compress(test_true)
    test_ic_comp = basis.compress(test_ic.reshape(-1, 1))[:, 0]

    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    Os, rom_solves = _generate_rom_predictions_scipy(
        samples, rom, test_ic_comp, t_pred, num_modes,
        num_pulls=min(200, n_post))

    n_stable = len(rom_solves)
    n_total = len(Os)
    stab_pct = 100 * n_stable / max(n_total, 1)

    train_err = pred_err = float('inf')
    ci_cov = ci_w = float('nan')
    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        train_mask = t_pred <= TRAINING_SPAN[1]
        pred_mask = t_pred > TRAINING_SPAN[1]
        ti = interp1d(full_time, test_true_comp, kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)
        train_err = float(np.linalg.norm(rom_med[:, train_mask] - ta[:, train_mask]) /
                          np.linalg.norm(ta[:, train_mask]))
        pred_err = float(np.linalg.norm(rom_med[:, pred_mask] - ta[:, pred_mask]) /
                         np.linalg.norm(ta[:, pred_mask]))
        q05 = np.percentile(rom_arr, 5, axis=0)
        q95 = np.percentile(rom_arr, 95, axis=0)
        ci_w = float(np.mean(q95 - q05))
        ci_cov = float(np.mean((ta >= q05) & (ta <= q95)))

    print(f"\n  Results ({runtime:.0f}s) [TEST μ={config.TEST_MU}]:")
    print(f"    Stability: {n_stable}/{n_total} ({stab_pct:.0f}%)")
    print(f"    Train err: {train_err:.4%}  |  Pred err: {pred_err:.4%}")
    print(f"    CI coverage: {ci_cov:.2%}  | mean width: {ci_w:.4f}")

    return dict(
        samples=samples, rom_solves=rom_solves, Os=Os,
        t_pred=t_pred, full_time=full_time,
        test_true=test_true, test_true_comp=test_true_comp,
        test_mu=config.TEST_MU,
        training_mus=config.TRAINING_MUS,
        t_samp_list=t_samp_list,
        snaps_comp_list=snaps_comp_list,
        true_comp_list=true_comp_list,
        basis=basis, fom=fom,
        num_modes=num_modes, training_span=TRAINING_SPAN,
        runtime=runtime, losses=all_losses,
        train_error=train_err, pred_error=pred_err,
        stability_pct=stab_pct, ci_coverage=ci_cov, ci_width=ci_w,
        n_stable=n_stable, n_total=n_total,
        O_ls=O_ls,
    )


# =============================================================================
# Plotting
# =============================================================================
def plot_results(result, save_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)
    prefix = "07_parametric_ics"

    rom_solves = result['rom_solves']
    num_modes = result['num_modes']
    t_pred = result['t_pred']
    t_full = result['full_time']
    true_comp = result['test_true_comp']
    training_span = result['training_span']

    # ── 1. Per-mode trajectory plot for TEST μ ──────────────────────────
    if len(rom_solves) > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        q05 = np.percentile(rom_arr, 5, axis=0)
        q95 = np.percentile(rom_arr, 95, axis=0)
        ti = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)

        fig, ax = plt.subplots(num_modes, 1, figsize=(10, 2.5 * num_modes), sharex=True)
        if num_modes == 1:
            ax = [ax]
        for i in range(num_modes):
            ax[i].axvspan(training_span[0], training_span[1], color='gray', alpha=0.10)
            ax[i].plot(t_pred, ta[i], color='tab:gray', lw=2, label='FOM (test μ)')
            ax[i].plot(t_pred, rom_med[i], color='tab:purple', ls='--', lw=2,
                       label='ROM median')
            ax[i].fill_between(t_pred, q05[i], q95[i], color='tab:purple',
                               alpha=0.15, label='ROM 5–95%')
            ax[i].axvline(training_span[1], color='k', ls=':', lw=0.8, alpha=0.5)
            ax[i].set_ylabel(f'Mode {i+1}')
            if i == 0:
                ax[i].legend(loc='upper right', fontsize=9)
        ax[-1].set_xlabel('Time')
        fig.suptitle(f'Parametric ROM @ test μ={result["test_mu"]} '
                     f'({result["n_stable"]}/{result["n_total"]} stable)',
                     fontsize=13)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_test_modes.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)

    # ── 2. Training trajectories: GP mean vs data (one mode per row) ────
    M = len(result['training_mus'])
    fig, ax = plt.subplots(num_modes, 1, figsize=(10, 2.5 * num_modes), sharex=True)
    if num_modes == 1:
        ax = [ax]
    cmap = plt.get_cmap('tab10')
    for i in range(num_modes):
        for m in range(M):
            c = cmap(m)
            ax[i].plot(t_full, result['true_comp_list'][m][i], color=c, lw=1.5,
                       label=f'μ={result["training_mus"][m]}' if i == 0 else None)
            ax[i].plot(result['t_samp_list'][m],
                       result['snaps_comp_list'][m][i],
                       '.', color=c, ms=4, alpha=0.5)
        ax[i].axvspan(*training_span, color='gray', alpha=0.08)
        ax[i].set_ylabel(f'Mode {i+1}')
        if i == 0:
            ax[i].legend(loc='upper right', fontsize=8, ncol=2)
    ax[-1].set_xlabel('Time')
    fig.suptitle('Training trajectories (global POD compression)', fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_training_modes.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)

    # ── 3. 2D contour for TEST μ: True / ROM median / width ─────────────
    fom = result['fom']
    basis = result['basis']
    test_true = result['test_true']
    if len(rom_solves) > 0 and fom is not None:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)

        snapshot_times = [t for t in [0.0, 0.5, 1.0, 1.5, 2.0]
                          if t <= t_pred[-1]]
        fig, axes = plt.subplots(3, len(snapshot_times),
                                 figsize=(3.5 * len(snapshot_times), 9.5))
        x, y = fom.spatial_domain

        for col, ts in enumerate(snapshot_times):
            ti_idx = np.argmin(np.abs(t_full - ts))
            u_true = fom.reconstruct_2d(test_true[:, ti_idx])

            tp_idx = np.argmin(np.abs(t_pred - ts))
            u_rom = fom.reconstruct_2d(basis.decompress(rom_med[:, tp_idx]))

            fields = np.stack([
                fom.reconstruct_2d(basis.decompress(rom_arr[s, :, tp_idx]))
                for s in range(rom_arr.shape[0])
            ], axis=0)
            u_width = np.percentile(fields, 95, 0) - np.percentile(fields, 5, 0)

            vmin = min(u_true.min(), u_rom.min())
            vmax = max(u_true.max(), u_rom.max())
            levels = np.linspace(vmin, vmax, 30)

            im0 = axes[0, col].contourf(x, y, u_true, levels=levels,
                                        cmap='RdBu_r', extend='both')
            axes[0, col].set_aspect('equal')
            axes[0, col].set_title(f't = {ts:.1f}', fontsize=12)
            plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

            im1 = axes[1, col].contourf(x, y, u_rom, levels=levels,
                                        cmap='RdBu_r', extend='both')
            axes[1, col].set_aspect('equal')
            plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

            w_levels = np.linspace(0.0, max(u_width.max(), 1e-12), 30)
            im2 = axes[2, col].contourf(x, y, u_width, levels=w_levels,
                                        cmap='viridis', extend='max')
            axes[2, col].set_aspect('equal')
            plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)

            for row in range(3):
                if col > 0:
                    axes[row, col].set_yticklabels([])

        axes[0, 0].set_ylabel('True', fontsize=12)
        axes[1, 0].set_ylabel('ROM median', fontsize=12)
        axes[2, 0].set_ylabel('ROM width\n(q95 − q05)', fontsize=12)
        fig.suptitle(f'2D Field — test μ={result["test_mu"]}', fontsize=14)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_2d_contours.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)

    # ── 4. Loss ─────────────────────────────────────────────────────────
    losses = result['losses']
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(losses, lw=0.8); ax[0].set_title('ELBO'); ax[0].grid(alpha=0.3)
    half = len(losses) // 2
    ax[1].plot(range(half, len(losses)), losses[half:], lw=0.8)
    ax[1].set_title('ELBO (last 50%)'); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_loss.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)


def save_predictions(result, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, "results", "comparison",
                                "parametric_ics")
    os.makedirs(save_dir, exist_ok=True)
    rom_arr = (np.array(result['rom_solves']) if len(result['rom_solves']) > 0
               else np.empty((0, result['num_modes'], len(result['t_pred']))))
    path = os.path.join(save_dir, "07_parametric_ics.npz")
    np.savez(path,
        rom_solves=rom_arr,
        t_pred=result['t_pred'],
        train_error=result['train_error'],
        pred_error=result['pred_error'],
        stability_pct=result['stability_pct'],
        ci_coverage=result['ci_coverage'],
        ci_width=result['ci_width'],
        runtime=result['runtime'],
        test_mu=np.array(result['test_mu']),
        training_mus=np.array(result['training_mus']),
    )
    print(f"  💾 Saved predictions: {path}")


def main():
    print("=" * 70)
    print("07 — Parametric ICs — 2D Diffusion-Reaction")
    print("=" * 70)
    r = run_experiment()
    plot_results(r)
    save_predictions(r)


if __name__ == "__main__":
    main()

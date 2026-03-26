"""
04 — Conditional GP + Dual Constraint (Integral + Derivative Form) — Heat Equation

Bayesian Operator Inference with full uncertainty propagation for multi-trajectory,
input-dependent PDEs:
  θ_GP^(ic) ~ LogNormal(MLE, σ)    — per-IC GP hyperparameters are sampled
  X^(ic)(t) = K_* K⁻¹ y^(ic)      — states computed analytically from θ_GP
  O ~ N(O_ls, γ|O_ls|)             — SHARED operator with informative prior
  γ₂ = fixed hyperparameter        — constraint noise scale (fixed)

Physics constraints (likelihood factors in ELBO):
  1. Derivative:  dX/dt ≈ f(X, u)O^T   (weighted by GP derivative variance)
  2. Integral:    ∫f(X, u)O^T ds ≈ ΔX  (prevents null basin, robust to noise)
  3. GP MLL:      Σ_ic log p(y|θ_GP)    (data fidelity for hyperparameters)

Operators: cAHBN (constant + linear + quadratic + input + state×input)

Data regimes (same hyperparameters for all):
  1. Dense data, low noise    (65 samples, 1% noise)
  2. Sparse data, medium noise (20 samples, 5% noise)
  3. Dense data, high noise   (65 samples, 10% noise)

Usage:
    python 04_conditional_integral.py                  # run all 3 regimes
    python 04_conditional_integral.py dense_low_noise  # run one regime
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
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import (
    Basis, ReducedOrderModel, input_func_factory,
    input_parameters, test_parameters,
)
from step1_generate_data import TrajectorySampler
from core import compute_gp_derivatives, rbf_eval
from core.bayesian_opinf import fit_gp_hyperparameters_mle, _find_operator_samples
from core.diagnostics import plot_trace
from core.plotting import plot_full_order_error
from heat_plotter import _generate_rom_solves
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

# ── Data regime definitions ──────────────────────────────────────────────────
SCHEMAS = [
    {
        "name": "sparse_low_noise",
        "label": "Sparse data, low noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.01,
        "NUM_EVAL_POINTS": 100,
    },
    {
        "name": "sparse_medium_noise",
        "label": "Sparse data, medium noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.03,
        "NUM_EVAL_POINTS": 100,
    },
    {
        "name": "sparse_high_noise",
        "label": "Sparse data, high noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.05,
        "NUM_EVAL_POINTS": 100,
    },
]

# ── Shared model hyperparameters (same for ALL regimes) ──────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=5,
    NUM_ICS=5,
    GAMMA=2.0,
    GAMMA2=0.5,
    DERIV_WEIGHT=1.0,
    INTEGRAL_WEIGHT=2.0,
    MLL_WEIGHT=0.1,
    GP_PRIOR_SCALE=0.1,
    WINDOW_SIZE=20,
    NUM_STEPS=10000,
    LEARNING_RATE=3e-3,
    NUM_POSTERIOR_SAMPLES=500,
    SEED=42,
)

TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 2.0)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# Model builder
# =============================================================================
def build_model(
    rom, num_modes, num_ics,
    all_time_sampled, all_snapshots_comp, all_inputs_eval,
    O_prior, all_mle_Ls, all_mle_Vs, all_mle_Ns,
    num_eval_points=150, window_size=10,
    deriv_weight=1.0, integral_weight=1.0,
    mll_weight=0.1, gp_prior_scale=0.1,
):
    """Build multi-IC conditional integral NumPyro model."""
    O_prior_jnp = jnp.array(O_prior)

    # Precompute per-IC kernel matrices
    ic_data = []
    for ic in range(num_ics):
        t_train = jnp.array(all_time_sampled[ic])
        n_train = len(t_train)
        y_obs = jnp.array(all_snapshots_comp[ic])
        inputs_eval = jnp.array(all_inputs_eval[ic])

        time_eval = np.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
        t_eval = jnp.array(time_eval)
        dt_eval = float(time_eval[1] - time_eval[0])

        sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
        sq_diffs_et = (t_eval[:, None] - t_train[None, :]) ** 2
        diffs_et = t_eval[:, None] - t_train[None, :]
        sq_diffs_ee = (t_eval[:, None] - t_eval[None, :]) ** 2
        I_train = jnp.eye(n_train)

        mle_log_ells = jnp.array([jnp.log(all_mle_Ls[ic][j]) for j in range(num_modes)])
        mle_log_sig2s = jnp.array([jnp.log(all_mle_Vs[ic][j]) for j in range(num_modes)])
        mle_log_nus = jnp.array([jnp.log(all_mle_Ns[ic][j]) for j in range(num_modes)])

        # Integration windows
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

        ic_data.append(dict(
            t_train=t_train, n_train=n_train, y_obs=y_obs,
            inputs_eval=inputs_eval,
            t_eval=t_eval, time_eval=time_eval, dt_eval=dt_eval,
            sq_diff_tt=sq_diff_tt, sq_diffs_et=sq_diffs_et,
            diffs_et=diffs_et, sq_diffs_ee=sq_diffs_ee, I_train=I_train,
            mle_log_ells=mle_log_ells, mle_log_sig2s=mle_log_sig2s,
            mle_log_nus=mle_log_nus,
            ws_list=ws_list, we_list=we_list,
            trap_weights=trap_weights, window_durations=window_durations,
        ))

    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def _single_gp_conditional(ell, sig2, nu, y_i, sq_diff_tt, sq_diffs_et,
                                diffs_et, sq_diffs_ee, I_train, n_train):
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

    def model(gamma=2.0, gamma2=0.5, jitter=1e-4):

        # Shared operator
        prior_scale = gamma * jnp.maximum(jnp.abs(O_prior_jnp), 0.5)
        O = numpyro.sample("O", dist.Normal(O_prior_jnp, prior_scale))

        total_mll = 0.0

        for ic in range(num_ics):
            d = ic_data[ic]

            # Per-IC GP hyperparameters
            ells = jnp.stack([
                numpyro.sample(f"lengthscale_{ic}_{j}",
                              dist.LogNormal(d['mle_log_ells'][j], gp_prior_scale))
                for j in range(num_modes)
            ])
            sig2s = jnp.stack([
                numpyro.sample(f"variance_{ic}_{j}",
                              dist.LogNormal(d['mle_log_sig2s'][j], gp_prior_scale))
                for j in range(num_modes)
            ])
            nus = jnp.stack([
                numpyro.sample(f"noise_{ic}_{j}",
                              dist.LogNormal(d['mle_log_nus'][j], gp_prior_scale))
                for j in range(num_modes)
            ])

            # GP conditional per mode
            ic_mll = 0.0
            Xs_eval_list, mu_zs_list, deriv_vars_list = [], [], []
            for j in range(num_modes):
                X_j, mu_j, dv_j, mll_j = _single_gp_conditional(
                    ells[j], sig2s[j], nus[j], d['y_obs'][j],
                    d['sq_diff_tt'], d['sq_diffs_et'], d['diffs_et'],
                    d['sq_diffs_ee'], d['I_train'], d['n_train'])
                Xs_eval_list.append(X_j)
                mu_zs_list.append(mu_j)
                deriv_vars_list.append(dv_j)
                ic_mll = ic_mll + mll_j

            Xs_eval = jnp.stack(Xs_eval_list)
            mu_zs = jnp.stack(mu_zs_list)
            deriv_vars = jnp.stack(deriv_vars_list)
            total_mll = total_mll + ic_mll

            for j in range(num_modes):
                numpyro.deterministic(f"X_{ic}_{j}", Xs_eval[j])

            # Data matrix WITH inputs
            f_Xi = rom.model._assemble_data_matrix(
                Xs_eval, inputs=d['inputs_eval']) @ O.T

            # CONSTRAINT 1: Derivative matching
            if deriv_weight > 0:
                for j in range(num_modes):
                    total_var = deriv_vars[j] + gamma2 + jitter
                    numpyro.factor(f"ode_{ic}_{j}",
                        deriv_weight * jnp.sum(
                            dist.Normal(f_Xi[:, j], jnp.sqrt(total_var)).log_prob(mu_zs[j])))

            # CONSTRAINT 2: Integral form
            if integral_weight > 0:
                for j in range(num_modes):
                    for w_idx, (ws, we) in enumerate(zip(d['ws_list'], d['we_list'])):
                        delta_X_obs = Xs_eval[j, we] - Xs_eval[j, ws]
                        delta_X_pred = jnp.sum(d['trap_weights'][w_idx] * f_Xi[ws:we+1, j])
                        constraint_std = jnp.sqrt(gamma2) * d['window_durations'][w_idx]
                        numpyro.factor(f"integral_{ic}_{j}_{w_idx}",
                            integral_weight * dist.Normal(
                                delta_X_pred, constraint_std).log_prob(delta_X_obs))

        # GP marginal log-likelihood
        if mll_weight > 0:
            numpyro.factor("gp_mll", mll_weight * total_mll)

    return model


# =============================================================================
# Run experiment
# =============================================================================
def run_experiment(schema):
    """Run one data regime. Returns results dict."""
    p = MODEL_PARAMS
    np.random.seed(p['SEED'])
    rng_key = random.PRNGKey(p['SEED'])

    noise_level = schema['NOISE_LEVEL']
    num_samples = schema['NUM_SAMPLES']
    num_eval_points = schema['NUM_EVAL_POINTS']
    num_modes = p['NUM_MODES']
    num_ics = p['NUM_ICS']
    train_params = input_parameters[:num_ics]

    print(f"\n{'='*70}")
    print(f"  {schema['label']}  ({num_samples} samples, {noise_level:.0%} noise)")
    print(f"{'='*70}")

    # ── Data generation ──────────────────────────────────────────────────
    sampler = TrajectorySampler(
        training_span=TRAINING_SPAN,
        num_samples=num_samples,
        noiselevel=noise_level,
        num_regression_points=num_eval_points,
        synced=False,
    )
    (all_true_states, all_time_sampled, all_snapshots,
     all_training_inputs) = sampler.multisample(train_params)

    snapshots_train = np.hstack(all_snapshots)
    basis = Basis(num_vectors=num_modes)
    basis.fit(snapshots_train)
    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    all_snapshots_comp = [basis.compress(s) for s in all_snapshots]
    all_true_comp = [basis.compress(s) for s in all_true_states]

    # Build ROM with inputs
    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(all_time_sampled[0]),
        model=ReducedOrderModel(),
    )
    first_input_func = input_func_factory(train_params[0])
    first_inputs = first_input_func(all_time_sampled[0])
    rom.fit(states=all_snapshots[0], inputs=first_inputs)
    print(f"  Operator shape: {rom.model.operator_matrix.shape}")

    # Per-IC eval inputs
    all_inputs_eval = []
    for ic in range(num_ics):
        t_eval_ic = np.linspace(
            float(all_time_sampled[ic][0]),
            float(all_time_sampled[ic][-1]),
            num_eval_points,
        )
        in_func = input_func_factory(train_params[ic])
        all_inputs_eval.append(in_func(t_eval_ic))

    # ── MLE warm start (per-IC) ──────────────────────────────────────────
    all_mle_Ls, all_mle_Vs, all_mle_Ns = [], [], []
    for ic in range(num_ics):
        print(f"  GP MLE — IC {ic} ({train_params[ic]})")
        Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(
            all_time_sampled[ic], all_snapshots_comp[ic], verbose=False)
        all_mle_Ls.append(Ls)
        all_mle_Vs.append(Vs)
        all_mle_Ns.append(Ns)
        for j in range(num_modes):
            print(f"    Mode {j}: ℓ={Ls[j]:.5f}, σ²={Vs[j]:.4f}, ν={Ns[j]:.6f}")

    # ── LS operator (using all ICs) ──────────────────────────────────────
    D_blocks, dXdt_blocks = [], []
    for ic in range(num_ics):
        t_eval_ic = np.linspace(
            float(all_time_sampled[ic][0]),
            float(all_time_sampled[ic][-1]),
            num_eval_points,
        )
        X_mle = np.zeros((num_modes, num_eval_points))
        for j in range(num_modes):
            ell, sig2, nu = all_mle_Ls[ic][j], all_mle_Vs[ic][j], all_mle_Ns[ic][j]
            K = rbf_eval(ell, sig2, all_time_sampled[ic], all_time_sampled[ic]) + \
                (nu + 1e-5) * np.eye(len(all_time_sampled[ic]))
            Ks = rbf_eval(ell, sig2, t_eval_ic, all_time_sampled[ic])
            X_mle[j] = Ks @ np.linalg.solve(K, all_snapshots_comp[ic][j])

        mu_z_ic, _ = compute_gp_derivatives(
            all_mle_Ls[ic], all_mle_Vs[ic],
            all_time_sampled[ic], t_eval_ic,
            all_snapshots_comp[ic], Ns=all_mle_Ns[ic])

        in_func = input_func_factory(train_params[ic])
        inputs_ic = in_func(t_eval_ic)
        D_ic = np.array(rom.model._assemble_data_matrix(
            jnp.array(X_mle), inputs=jnp.array(inputs_ic)))
        D_blocks.append(D_ic)
        dXdt_blocks.append(np.array(mu_z_ic).T)

    D_all = np.vstack(D_blocks)
    dXdt_all = np.vstack(dXdt_blocks)
    DtD = D_all.T @ D_all
    O_ls = np.linalg.solve(DtD + np.eye(DtD.shape[0]), D_all.T @ dXdt_all).T
    print(f"  LS operator norm: {np.linalg.norm(O_ls):.1f}, shape: {O_ls.shape}")

    # ── Build & run SVI ──────────────────────────────────────────────────
    model = build_model(
        rom=rom, num_modes=num_modes, num_ics=num_ics,
        all_time_sampled=all_time_sampled,
        all_snapshots_comp=all_snapshots_comp,
        all_inputs_eval=all_inputs_eval,
        O_prior=O_ls,
        all_mle_Ls=all_mle_Ls, all_mle_Vs=all_mle_Vs, all_mle_Ns=all_mle_Ns,
        num_eval_points=num_eval_points, window_size=p['WINDOW_SIZE'],
        deriv_weight=p['DERIV_WEIGHT'], integral_weight=p['INTEGRAL_WEIGHT'],
        mll_weight=p['MLL_WEIGHT'], gp_prior_scale=p['GP_PRIOR_SCALE'],
    )

    init_values = {'O': jnp.array(O_ls)}
    for ic in range(num_ics):
        for j in range(num_modes):
            init_values[f'lengthscale_{ic}_{j}'] = all_mle_Ls[ic][j]
            init_values[f'variance_{ic}_{j}'] = all_mle_Vs[ic][j]
            init_values[f'noise_{ic}_{j}'] = all_mle_Ns[ic][j]

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
    seg_size = max(1, num_steps // 10)
    all_losses = []
    for seg in range(10):
        start = seg * seg_size
        end = min(start + seg_size, num_steps)
        if seg == 9:
            end = num_steps
        if start >= num_steps:
            break
        svi_state, seg_losses = jax.lax.scan(_step, svi_state, jnp.arange(end - start))
        seg_np = np.array(seg_losses)
        all_losses.extend(seg_np.tolist())
        print(f"    step {end:6d}/{num_steps}  loss={seg_np[-1]:10.2f}")

    params = svi.get_params(svi_state)
    rng_key, sk, pk = random.split(rng_key, 3)
    n_post = p['NUM_POSTERIOR_SAMPLES']
    post = guide.sample_posterior(sk, params, sample_shape=(n_post,), **model_kwargs)
    pred = Predictive(model, posterior_samples=post, num_samples=n_post)
    out = pred(pk, **model_kwargs)
    samples = {**out, **post}
    runtime = time.time() - t0

    # ── Evaluate all training ICs + test IC ───────────────────────────────
    print(f"\n  Results ({runtime:.0f}s):")

    O_samp = _find_operator_samples(samples, "O")
    if O_samp.ndim == 2:
        O_samp = O_samp[np.newaxis, ...]
    O_med = np.median(O_samp, axis=0)
    O_std = np.std(O_samp, axis=0)
    print(f"    Operator norm: {np.linalg.norm(O_med):.1f} (LS: {np.linalg.norm(O_ls):.1f})")

    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    max_samp = min(200, len(O_samp))

    # Generate test IC data
    test_sampler = TrajectorySampler(
        training_span=TRAINING_SPAN,
        num_samples=num_samples,
        noiselevel=noise_level,
        num_regression_points=num_eval_points,
        synced=False,
    )
    (test_true_list, test_t_list, test_snap_list,
     test_inp_list) = test_sampler.multisample([test_parameters])
    test_true_comp = basis.compress(test_true_list[0])
    test_snaps_comp = basis.compress(test_snap_list[0])
    test_t_samp = test_t_list[0]

    # Evaluate all ICs
    eval_params = list(train_params) + [test_parameters]
    eval_snaps_comp = all_snapshots_comp + [test_snaps_comp]
    eval_true_comp = all_true_comp + [test_true_comp]
    eval_t_samp = all_time_sampled + [test_t_samp]
    eval_labels = [f"Train IC {i} {train_params[i]}" for i in range(num_ics)] + \
                  [f"Test IC {test_parameters}"]

    all_rom_solves = []
    all_n_stable = []
    all_train_errors = []
    all_pred_errors = []
    train_mask = t_pred <= TRAINING_SPAN[1]
    pred_mask = t_pred > TRAINING_SPAN[1]

    for ic_idx, (params, true_c) in enumerate(zip(eval_params, eval_true_comp)):
        q0 = eval_snaps_comp[ic_idx][:, 0]
        ic_input_func = input_func_factory(params)

        ic_solves = _generate_rom_solves(
            operator_samples=O_samp, rom=rom, q0=q0,
            time_eval=t_pred, input_func=ic_input_func,
            max_samples=max_samp,
        )
        all_rom_solves.append(ic_solves)
        all_n_stable.append(len(ic_solves))

        ti = interp1d(config.time_domain, true_c,
                      kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)

        if len(ic_solves) > 0:
            rom_med = np.median(ic_solves, axis=0)
            te = float(np.linalg.norm(rom_med[:, train_mask] - ta[:, train_mask]) /
                      np.linalg.norm(ta[:, train_mask]))
            pe = float(np.linalg.norm(rom_med[:, pred_mask] - ta[:, pred_mask]) /
                      np.linalg.norm(ta[:, pred_mask]))
        else:
            te, pe = float('inf'), float('inf')

        all_train_errors.append(te)
        all_pred_errors.append(pe)
        label = eval_labels[ic_idx]
        print(f"    {label}: {all_n_stable[-1]}/{max_samp} stable, "
              f"train={te:.4%}, pred={pe:.4%}")

    # Aggregate over training ICs
    train_ic_stable = sum(all_n_stable[:num_ics])
    train_ic_total = max_samp * num_ics
    train_errors_fin = [e for e in all_train_errors[:num_ics] if np.isfinite(e)]
    pred_errors_fin = [e for e in all_pred_errors[:num_ics] if np.isfinite(e)]
    train_error = np.mean(train_errors_fin) if train_errors_fin else float('inf')
    pred_error = np.mean(pred_errors_fin) if pred_errors_fin else float('inf')
    stability_pct = train_ic_stable / max(train_ic_total, 1) * 100

    print(f"\n    Overall: {train_ic_stable}/{train_ic_total} ({stability_pct:.0f}%)")
    print(f"    Avg train: {train_error:.4%}  |  Avg pred: {pred_error:.4%}")
    print(f"    Test IC: {all_n_stable[-1]}/{max_samp} stable, "
          f"train={all_train_errors[-1]:.4%}, pred={all_pred_errors[-1]:.4%}")

    # CI coverage
    ci_width = ci_coverage = float('nan')
    if train_ic_stable > 0:
        all_in_ci, all_widths = [], []
        for ic_idx in range(num_ics):
            if all_n_stable[ic_idx] > 0:
                ti = interp1d(config.time_domain, eval_true_comp[ic_idx],
                              kind='cubic', fill_value='extrapolate')
                ta = ti(t_pred)
                q05 = np.percentile(all_rom_solves[ic_idx], 5, axis=0)
                q95 = np.percentile(all_rom_solves[ic_idx], 95, axis=0)
                all_widths.append(np.mean(q95 - q05))
                all_in_ci.append(np.mean((ta >= q05) & (ta <= q95)))
        if all_widths:
            ci_width = np.mean(all_widths)
            ci_coverage = np.mean(all_in_ci)
            print(f"    CI coverage: {ci_coverage:.2%} (target: 90%)")

    return {
        'schema': schema,
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'n_stable': train_ic_stable, 'n_total': train_ic_total,
        'test_train_error': all_train_errors[-1],
        'test_pred_error': all_pred_errors[-1],
        'test_n_stable': all_n_stable[-1],
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'runtime': runtime, 'losses': all_losses,
        'samples': samples,
        # Per-IC data for multi-trajectory plotting
        'all_rom_solves': all_rom_solves,
        'all_snaps_comp': eval_snaps_comp,
        'all_true_comp': eval_true_comp,
        'all_t_samp': eval_t_samp,
        'all_n_stable': all_n_stable,
        'eval_labels': eval_labels,
        't_full': config.time_domain,
        't_pred': t_pred,
        'training_span': TRAINING_SPAN,
        'num_modes': p['NUM_MODES'],
        'basis': basis,
        'all_true_states_full': all_true_states,
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_results(result, save_dir=None):
    """Generate multi-IC ROM trajectory, operator trace, and loss plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    schema = result['schema']
    prefix = f"04_{schema['name']}"
    samples = result['samples']
    losses = result['losses']
    t_pred = result['t_pred']
    t_full = result['t_full']
    training_span = result['training_span']
    num_modes = result['num_modes']
    all_rom_solves = result['all_rom_solves']
    all_snaps_comp = result['all_snaps_comp']
    all_true_comp = result['all_true_comp']
    all_t_samp = result['all_t_samp']
    all_n_stable = result['all_n_stable']
    eval_labels = result['eval_labels']
    max_samp = min(200, len(_find_operator_samples(samples, "O")))

    # ── 1. Multi-trajectory ROM plot ─────────────────────────────────
    n_ics_total = len(all_rom_solves)
    has_any_stable = any(ns > 0 for ns in all_n_stable)

    if not has_any_stable:
        print("  ⚠ No stable ROM solves — skipping ROM trajectory plot")
    else:
        fig, axes = plt.subplots(
            n_ics_total, num_modes,
            figsize=(4 * num_modes, 2.5 * n_ics_total),
            sharex=True, squeeze=False,
        )

        for row in range(n_ics_total):
            rom_solves = all_rom_solves[row]
            n_stable = all_n_stable[row]
            label = eval_labels[row]

            true_interp = interp1d(t_full, all_true_comp[row],
                                   kind='cubic', fill_value='extrapolate')
            true_at_pred = true_interp(t_pred)

            for col in range(num_modes):
                ax = axes[row, col]

                ax.axvspan(training_span[0], training_span[1],
                           color='gray', alpha=0.10, zorder=0)

                ax.plot(t_pred, true_at_pred[col],
                        color='tab:gray', lw=2,
                        label='True' if (row == 0 and col == 0) else None)

                ax.plot(all_t_samp[row], all_snaps_comp[row][col],
                        'k*', ms=4, zorder=5,
                        label='Data' if (row == 0 and col == 0) else None)

                if n_stable > 0:
                    ax.plot(
                        t_pred,
                        np.median(rom_solves[:, col, :], axis=0),
                        color='tab:purple', ls='--', lw=2, alpha=0.9,
                        label='Median' if (row == 0 and col == 0) else None,
                    )
                    ax.fill_between(
                        t_pred,
                        np.percentile(rom_solves[:, col, :], 5, axis=0),
                        np.percentile(rom_solves[:, col, :], 95, axis=0),
                        color='tab:purple', alpha=0.15,
                        label='90% CI' if (row == 0 and col == 0) else None,
                    )

                ax.axvline(training_span[1], color='k', ls=':', lw=0.8, alpha=0.5)

                yvals = true_at_pred[col]
                ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
                pad = max(abs(ymax - ymin) * 0.3, 1e-6)
                ax.set_ylim(ymin - pad, ymax + pad)

                if row == 0:
                    ax.set_title(f'Mode {col + 1}')
                if col == 0:
                    ax.set_ylabel(f'{label}\n({n_stable}/{max_samp} stable)',
                                  fontsize=8)
                if row == n_ics_total - 1:
                    ax.set_xlabel('Time')

        handles, labels_leg = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels_leg, loc='upper center',
                       ncol=len(handles), fontsize=9,
                       bbox_to_anchor=(0.5, 1.02))

        fig.suptitle(f"ROM Predictions — {schema['label']}", fontsize=14, y=1.05)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_rom_trajectories.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)

    # ── 2. Operator Trace Plot ───────────────────────────────────────
    try:
        fig_trace, _ = plot_trace(samples, param_name="O", n_random=6)
        path = os.path.join(save_dir, f"{prefix}_operator_traces.png")
        fig_trace.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig_trace)
    except Exception as e:
        print(f"  ⚠ Operator trace plot failed: {e}")

    # ── 3. Loss Convergence Plot ─────────────────────────────────────
    fig_loss, ax_loss = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss[0].plot(losses, lw=0.8, color='tab:blue')
    ax_loss[0].set_xlabel('SVI Iteration')
    ax_loss[0].set_ylabel('ELBO Loss')
    ax_loss[0].set_title('Loss Convergence')
    ax_loss[0].grid(True, alpha=0.3)
    half = len(losses) // 2
    ax_loss[1].plot(range(half, len(losses)), losses[half:], lw=0.8, color='tab:blue')
    ax_loss[1].set_xlabel('SVI Iteration')
    ax_loss[1].set_ylabel('ELBO Loss')
    ax_loss[1].set_title('Loss (last 50%)')
    ax_loss[1].grid(True, alpha=0.3)
    fig_loss.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_loss.png")
    fig_loss.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig_loss)

    # ── 4. Full-Order Error Plot (first training IC) ─────────────────
    basis = result.get('basis')
    all_true_states_full = result.get('all_true_states_full')
    if basis is not None and all_true_states_full is not None and len(all_rom_solves) > 0:
        first_ic_solves = all_rom_solves[0]
        if len(first_ic_solves) > 0:
            first_true_full = all_true_states_full[0]  # (n_dof, n_time_full)
            fig_foe, axes_foe = plot_full_order_error(
                rom_solves=first_ic_solves,
                basis=basis,
                true_states=first_true_full,
                time_domain_full=t_full,
                time_domain_eval=t_pred,
                training_span=training_span,
                error_type='relative',
            )
            fig_foe.suptitle(f'Full-Order Error (IC 0) — {schema["label"]}', fontsize=14)
            path = os.path.join(save_dir, f"{prefix}_full_order_error.png")
            fig_foe.savefig(path, dpi=200, bbox_inches='tight')
            print(f"  📊 Saved: {path}")
            plt.close(fig_foe)


# =============================================================================
# save_predictions
# =============================================================================
def save_predictions(result, save_dir=None):
    """Save predictions for cross-method comparison."""
    schema = result['schema']
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, "results", "comparison", schema['name'])
    os.makedirs(save_dir, exist_ok=True)

    all_rom_solves = result['all_rom_solves']
    method_name = "04_conditional_integral"

    save_dict = {
        't_pred': result['t_pred'],
        'train_error': result['train_error'],
        'pred_error': result['pred_error'],
        'stability_pct': result['stability_pct'],
        'ci_coverage': result.get('ci_coverage', float('nan')),
        'ci_width': result.get('ci_width', float('nan')),
        'runtime': result['runtime'],
        'n_ics': len(all_rom_solves),
    }
    for ic_idx, solves in enumerate(all_rom_solves):
        if len(solves) > 0:
            save_dict[f'rom_solves_{ic_idx}'] = np.array(solves)
        else:
            save_dict[f'rom_solves_{ic_idx}'] = np.empty((0, result['num_modes'], len(result['t_pred'])))

    path = os.path.join(save_dir, f"{method_name}.npz")
    np.savez(path, **save_dict)
    print(f"  💾 Saved predictions: {path}")


# =============================================================================
# Main
# =============================================================================
def main(schema_names=None):
    """Run selected (or all) data regimes."""
    schemas = SCHEMAS
    if schema_names:
        schemas = [s for s in SCHEMAS if s['name'] in schema_names]
        if not schemas:
            print(f"Unknown schema(s): {schema_names}")
            print(f"Available: {[s['name'] for s in SCHEMAS]}")
            return

    print("=" * 70)
    print("04 — Conditional GP + Dual Constraint — Cubic Heat Equation")
    print("=" * 70)
    print(f"Regimes: {len(schemas)}")
    for s in schemas:
        print(f"  • {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  noise={s['NOISE_LEVEL']:.0%}")
    print(f"Model:  γ={MODEL_PARAMS['GAMMA']}, γ₂={MODEL_PARAMS['GAMMA2']}, "
          f"lr={MODEL_PARAMS['LEARNING_RATE']}, steps={MODEL_PARAMS['NUM_STEPS']}")
    print(f"ICs:    {MODEL_PARAMS['NUM_ICS']} training + 1 test ({test_parameters})")

    results = []
    for schema in schemas:
        r = run_experiment(schema)
        plot_results(r)
        save_predictions(r)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"SUMMARY — Conditional GP + Dual Constraint (Heat)")
    print(f"{'='*90}")
    print(f"{'Regime':<28s} {'Samp':>4s} {'Noise':>5s} {'Stab':>5s} "
          f"{'Train':>8s} {'Pred':>8s} {'TestPr':>8s} {'CI_cov':>7s} {'Time':>6s}")
    print(f"{'-'*28} {'-'*4} {'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
    for r in results:
        s = r['schema']
        print(f"{s['label']:<28s} {s['NUM_SAMPLES']:>4d} "
              f"{s['NOISE_LEVEL']:>4.0%} {r['stability_pct']:>4.0f}% "
              f"{r['train_error']:>7.2%} {r['pred_error']:>7.2%} "
              f"{r['test_pred_error']:>7.2%} "
              f"{r['ci_coverage']:>6.1%} {r['runtime']:>5.0f}s")


if __name__ == "__main__":
    schema_names = sys.argv[1:] if len(sys.argv) > 1 else None
    main(schema_names)

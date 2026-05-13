"""
tune_04.py — Efficient hyperparameter sweep for Heat 04_conditional_integral

Same caching strategy as Euler: all model weights as kwargs → single JIT for Phase 1+2.
"""
import sys, os, time, itertools
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro.infer.svi import SVIState
from numpyro.infer.initialization import init_to_value
from numpyro.optim import ClippedAdam
from jax import random
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import (Basis, ReducedOrderModel, input_func_factory,
                    input_parameters, test_parameters)
from step1_generate_data import TrajectorySampler
from core import compute_gp_derivatives, rbf_eval
from core.bayesian_opinf import fit_gp_hyperparameters_mle, _find_operator_samples
from heat_plotter import _generate_rom_solves
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 2.0)
SWEEP_STEPS = 5000
SWEEP_SAMPLES = 100
SWEEP_PULLS = 8

SCHEMA = {"name": "dense_low_noise", "label": "Dense data, low noise",
          "NUM_SAMPLES": 65, "NOISE_LEVEL": 0.01, "NUM_EVAL_POINTS": 150}

DEFAULTS = dict(NUM_MODES=5, NUM_ICS=5, GAMMA=2.0, GAMMA2=0.5, DERIV_WEIGHT=1.0,
                INTEGRAL_WEIGHT=1.0, MLL_WEIGHT=0.1, GP_PRIOR_SCALE=0.1,
                WINDOW_SIZE=10, NUM_STEPS=10000, LEARNING_RATE=3e-3,
                NUM_POSTERIOR_SAMPLES=500, SEED=42)


# =============================================================================
# Model builder — weights as kwargs
# =============================================================================
def build_tuning_model(rom, num_modes, num_ics,
                       all_time_sampled, all_snapshots_comp, all_inputs_eval,
                       O_prior, all_mle_Ls, all_mle_Vs, all_mle_Ns,
                       num_eval_points=150, window_size=10):
    """Build multi-IC model with all weights as kwargs."""
    O_prior_jnp = jnp.array(O_prior)

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

        n_windows = num_eval_points // window_size
        ws_list = [i * window_size for i in range(n_windows)]
        we_list = [(i + 1) * window_size - 1 for i in range(n_windows)]
        if we_list[-1] < num_eval_points - 1:
            we_list[-1] = num_eval_points - 1

        trap_weights, window_durations = [], []
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
        mll = -0.5 * (jnp.dot(y_i, alpha) + 2.0 * jnp.sum(jnp.log(jnp.diag(L))) +
                       n_train * jnp.log(2.0 * jnp.pi))
        return X_eval, mu_z, deriv_var, mll

    def model(gamma=2.0, gamma2=0.5, mll_weight=0.1,
              deriv_weight=1.0, integral_weight=1.0, gp_prior_scale=0.1):
        prior_scale = gamma * jnp.maximum(jnp.abs(O_prior_jnp), 0.5)
        O = numpyro.sample("O", dist.Normal(O_prior_jnp, prior_scale))
        total_mll = 0.0

        for ic in range(num_ics):
            d = ic_data[ic]
            ells = jnp.stack([numpyro.sample(f"lengthscale_{ic}_{j}",
                              dist.LogNormal(d['mle_log_ells'][j], gp_prior_scale))
                              for j in range(num_modes)])
            sig2s = jnp.stack([numpyro.sample(f"variance_{ic}_{j}",
                               dist.LogNormal(d['mle_log_sig2s'][j], gp_prior_scale))
                               for j in range(num_modes)])
            nus = jnp.stack([numpyro.sample(f"noise_{ic}_{j}",
                             dist.LogNormal(d['mle_log_nus'][j], gp_prior_scale))
                             for j in range(num_modes)])

            Xs_list, mu_list, dv_list = [], [], []
            ic_mll = 0.0
            for j in range(num_modes):
                X_j, mu_j, dv_j, mll_j = _single_gp_conditional(
                    ells[j], sig2s[j], nus[j], d['y_obs'][j],
                    d['sq_diff_tt'], d['sq_diffs_et'], d['diffs_et'],
                    d['sq_diffs_ee'], d['I_train'], d['n_train'])
                Xs_list.append(X_j); mu_list.append(mu_j); dv_list.append(dv_j)
                ic_mll += mll_j

            Xs_eval = jnp.stack(Xs_list)
            mu_zs = jnp.stack(mu_list)
            deriv_vars = jnp.stack(dv_list)
            total_mll += ic_mll

            for j in range(num_modes):
                numpyro.deterministic(f"X_{ic}_{j}", Xs_eval[j])

            f_Xi = rom.model._assemble_data_matrix(Xs_eval, inputs=d['inputs_eval']) @ O.T

            for j in range(num_modes):
                total_var = deriv_vars[j] + gamma2 + 1e-4
                numpyro.factor(f"ode_{ic}_{j}",
                    deriv_weight * jnp.sum(
                        dist.Normal(f_Xi[:, j], jnp.sqrt(total_var)).log_prob(mu_zs[j])))

            for j in range(num_modes):
                for w_idx, (ws, we) in enumerate(zip(d['ws_list'], d['we_list'])):
                    delta_X_obs = Xs_eval[j, we] - Xs_eval[j, ws]
                    delta_X_pred = jnp.sum(d['trap_weights'][w_idx] * f_Xi[ws:we+1, j])
                    constraint_std = jnp.sqrt(gamma2) * d['window_durations'][w_idx]
                    numpyro.factor(f"integral_{ic}_{j}_{w_idx}",
                        integral_weight * dist.Normal(
                            delta_X_pred, constraint_std).log_prob(delta_X_obs))

        numpyro.factor("gp_mll", mll_weight * total_mll)

    return model


# =============================================================================
# Data preparation
# =============================================================================
def prepare_data():
    p = DEFAULTS
    np.random.seed(p['SEED'])
    num_modes, num_ics = p['NUM_MODES'], p['NUM_ICS']
    train_params = input_parameters[:num_ics]
    nep = SCHEMA['NUM_EVAL_POINTS']

    print("Generating trajectories...", flush=True)
    sampler = TrajectorySampler(training_span=TRAINING_SPAN,
                                num_samples=SCHEMA['NUM_SAMPLES'],
                                noiselevel=SCHEMA['NOISE_LEVEL'],
                                num_regression_points=nep, synced=False)
    (all_true_states, all_time_sampled, all_snapshots,
     all_training_inputs) = sampler.multisample(train_params)

    basis = Basis(num_vectors=num_modes)
    basis.fit(np.hstack(all_snapshots))
    all_snaps_comp = [basis.compress(s) for s in all_snapshots]
    all_true_comp = [basis.compress(s) for s in all_true_states]
    print(f"  POD energy: {basis.cumulative_energy:.4%}", flush=True)

    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(all_time_sampled[0]),
        model=ReducedOrderModel(),
    )
    first_input_func = input_func_factory(train_params[0])
    rom.fit(states=all_snapshots[0], inputs=first_input_func(all_time_sampled[0]))

    all_inputs_eval = []
    for ic in range(num_ics):
        t_eval_ic = np.linspace(float(all_time_sampled[ic][0]),
                                float(all_time_sampled[ic][-1]), nep)
        all_inputs_eval.append(input_func_factory(train_params[ic])(t_eval_ic))

    print("Fitting MLE...", flush=True)
    all_mle_Ls, all_mle_Vs, all_mle_Ns = [], [], []
    for ic in range(num_ics):
        Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(
            all_time_sampled[ic], all_snaps_comp[ic], verbose=False)
        all_mle_Ls.append(Ls); all_mle_Vs.append(Vs); all_mle_Ns.append(Ns)

    D_blocks, dXdt_blocks = [], []
    for ic in range(num_ics):
        t_eval_ic = np.linspace(float(all_time_sampled[ic][0]),
                                float(all_time_sampled[ic][-1]), nep)
        X_mle = np.zeros((num_modes, nep))
        for j in range(num_modes):
            ell, sig2, nu = all_mle_Ls[ic][j], all_mle_Vs[ic][j], all_mle_Ns[ic][j]
            K = rbf_eval(ell, sig2, all_time_sampled[ic], all_time_sampled[ic]) + \
                (nu+1e-5)*np.eye(len(all_time_sampled[ic]))
            Ks = rbf_eval(ell, sig2, t_eval_ic, all_time_sampled[ic])
            X_mle[j] = Ks @ np.linalg.solve(K, all_snaps_comp[ic][j])
        mu_z_ic, _ = compute_gp_derivatives(
            all_mle_Ls[ic], all_mle_Vs[ic], all_time_sampled[ic],
            t_eval_ic, all_snaps_comp[ic], Ns=all_mle_Ns[ic])
        in_func = input_func_factory(train_params[ic])
        D_ic = np.array(rom.model._assemble_data_matrix(
            jnp.array(X_mle), inputs=jnp.array(in_func(t_eval_ic))))
        D_blocks.append(D_ic)
        dXdt_blocks.append(np.array(mu_z_ic).T)

    D_all = np.vstack(D_blocks)
    dXdt_all = np.vstack(dXdt_blocks)
    DtD = D_all.T @ D_all
    O_ls = np.linalg.solve(DtD + np.eye(DtD.shape[0]), D_all.T @ dXdt_all).T
    print(f"  LS operator norm: {np.linalg.norm(O_ls):.1f}", flush=True)

    # Test IC
    test_sampler = TrajectorySampler(training_span=TRAINING_SPAN,
                                     num_samples=SCHEMA['NUM_SAMPLES'],
                                     noiselevel=SCHEMA['NOISE_LEVEL'],
                                     num_regression_points=nep, synced=False)
    (test_true, test_t, test_snap, _) = test_sampler.multisample([test_parameters])
    test_true_comp = basis.compress(test_true[0])
    test_snaps_comp = basis.compress(test_snap[0])

    return dict(rom=rom, basis=basis, num_modes=num_modes, num_ics=num_ics,
                all_time_sampled=all_time_sampled, all_snaps_comp=all_snaps_comp,
                all_true_comp=all_true_comp, all_inputs_eval=all_inputs_eval,
                all_mle_Ls=all_mle_Ls, all_mle_Vs=all_mle_Vs, all_mle_Ns=all_mle_Ns,
                O_ls=O_ls, train_params=train_params,
                test_true_comp=test_true_comp, test_snaps_comp=test_snaps_comp,
                num_eval_points=SCHEMA['NUM_EVAL_POINTS'])


# =============================================================================
# Runner with JIT caching
# =============================================================================
def create_runner(model_fn, init_values, learning_rate):
    guide = autoguide.AutoNormal(model_fn, init_loc_fn=init_to_value(values=init_values))
    optimizer = ClippedAdam(step_size=learning_rate)
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    @jax.jit
    def _step(carry, _):
        s = carry[0]
        s, l = svi.update(s, gamma=carry[1], gamma2=carry[2],
                          mll_weight=carry[3], integral_weight=carry[4],
                          deriv_weight=carry[5], gp_prior_scale=carry[6])
        return (s, carry[1], carry[2], carry[3], carry[4], carry[5], carry[6]), l

    return svi, guide, _step


def _make_kwargs(params):
    return dict(gamma=jnp.float32(params['GAMMA']),
                gamma2=jnp.float32(params['GAMMA2']),
                mll_weight=jnp.float32(params['MLL_WEIGHT']),
                integral_weight=jnp.float32(params['INTEGRAL_WEIGHT']),
                deriv_weight=jnp.float32(params['DERIV_WEIGHT']),
                gp_prior_scale=jnp.float32(params['GP_PRIOR_SCALE']))


def _make_carry(svi_state, params):
    return (svi_state,
            jnp.float32(params['GAMMA']), jnp.float32(params['GAMMA2']),
            jnp.float32(params['MLL_WEIGHT']), jnp.float32(params['INTEGRAL_WEIGHT']),
            jnp.float32(params['DERIV_WEIGHT']), jnp.float32(params['GP_PRIOR_SCALE']))


def run_with_runner(svi, guide, model_fn, step_fn, data, params,
                    num_steps=SWEEP_STEPS, num_samples=SWEEP_SAMPLES,
                    num_pulls=SWEEP_PULLS, initial_state=None):
    rng_key = random.PRNGKey(params['SEED'])
    mkw = _make_kwargs(params)

    rng_key, ik = random.split(rng_key)
    t0 = time.time()

    if initial_state is not None:
        svi_state = SVIState(initial_state.optim_state, None, ik)
    else:
        svi_state = svi.init(ik, **mkw)

    carry = _make_carry(svi_state, params)
    carry, losses = jax.lax.scan(step_fn, carry, jnp.arange(num_steps))
    svi_state = carry[0]

    svi_params = svi.get_params(svi_state)
    rng_key, sk = random.split(rng_key)

    # Manual posterior sampling (skip expensive guide.sample_posterior + Predictive)
    O_loc = svi_params['O_auto_loc']
    O_scale = jax.nn.softplus(svi_params['O_auto_scale'])
    O_samp = np.array(O_loc + O_scale * random.normal(sk, shape=(num_samples,) + O_loc.shape))

    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    max_samp = min(num_pulls, len(O_samp))
    train_mask = t_pred <= TRAINING_SPAN[1]
    pred_mask = t_pred > TRAINING_SPAN[1]

    train_params_list = data['train_params']
    eval_params = list(train_params_list) + [test_parameters]
    eval_snaps_comp = data['all_snaps_comp'] + [data['test_snaps_comp']]
    eval_true_comp = data['all_true_comp'] + [data['test_true_comp']]

    all_n_stable, all_train_errors, all_pred_errors, all_solves = [], [], [], []
    indices = np.linspace(0, len(O_samp) - 1, max_samp, dtype=int)
    for ic_idx, (ic_params, true_c) in enumerate(zip(eval_params, eval_true_comp)):
        q0 = eval_snaps_comp[ic_idx][:, 0]
        ic_input_func = input_func_factory(ic_params)
        ic_solves = _generate_rom_solves(
            operator_samples=O_samp[indices], rom=data['rom'], q0=q0,
            time_eval=t_pred, input_func=ic_input_func, max_samples=max_samp)
        all_n_stable.append(len(ic_solves))
        all_solves.append(ic_solves)

        ti = interp1d(config.time_domain, true_c, kind='cubic', fill_value='extrapolate')
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

    num_ics = data['num_ics']
    train_errors_fin = [e for e in all_train_errors[:num_ics] if np.isfinite(e)]
    pred_errors_fin = [e for e in all_pred_errors[:num_ics] if np.isfinite(e)]
    train_error = np.mean(train_errors_fin) if train_errors_fin else float('inf')
    pred_error = np.mean(pred_errors_fin) if pred_errors_fin else float('inf')
    stability_pct = sum(all_n_stable[:num_ics]) / max(max_samp * num_ics, 1) * 100
    runtime = time.time() - t0

    # CI coverage (reuse stored solves)
    ci_coverage = 0.0
    if any(n > 0 for n in all_n_stable[:num_ics]):
        all_in_ci = []
        for ic_idx in range(num_ics):
            if all_n_stable[ic_idx] > 0 and len(all_solves[ic_idx]) > 0:
                ti = interp1d(config.time_domain, eval_true_comp[ic_idx],
                              kind='cubic', fill_value='extrapolate')
                ta = ti(t_pred)
                q05 = np.percentile(all_solves[ic_idx], 5, axis=0)
                q95 = np.percentile(all_solves[ic_idx], 95, axis=0)
                all_in_ci.append(np.mean((ta >= q05) & (ta <= q95)))
        if all_in_ci:
            ci_coverage = float(np.mean(all_in_ci))

    score = 0.3 * train_error + 0.3 * pred_error + 0.4 * (1.0 - ci_coverage) \
            if stability_pct >= 50 else 999.0

    return dict(train_error=train_error, pred_error=pred_error,
                stability_pct=stability_pct, ci_coverage=ci_coverage,
                runtime=runtime, score=score, final_loss=float(losses[-1]))


# =============================================================================
# Sweep helpers
# =============================================================================
def print_table(rows, title, keys):
    print(f"\n{'='*100}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*100}", flush=True)
    hdr = "  ".join(f"{k:>12s}" for k in keys)
    hdr += f"  {'Train':>8s} {'Pred':>8s} {'Stab':>5s} {'CI':>7s} {'Score':>8s} {'Time':>5s}"
    print(hdr); print("-" * len(hdr))
    for row in sorted(rows, key=lambda r: r['m']['score']):
        v = "  ".join(f"{row['p'][k]:>12g}" for k in keys)
        m = row['m']
        te = f"{m['train_error']:.4%}" if m['train_error'] < 10 else "  >1000%"
        pe = f"{m['pred_error']:.4%}" if m['pred_error'] < 10 else "  >1000%"
        v += f"  {te:>8s} {pe:>8s} {m['stability_pct']:>4.0f}% {m['ci_coverage']:>6.1%}"
        v += f" {m['score']:>8.4f} {m['runtime']:>4.0f}s"
        if row.get('best'): v += "  ◀ BEST"
        print(v)
    print(flush=True)


def run_sweep(runner_args, data, combos, title, keys, initial_state=None):
    svi, guide, model_fn, step_fn = runner_args
    rows = []
    for i, p in enumerate(combos, 1):
        label = ", ".join(f"{k}={p[k]}" for k in keys)
        print(f"  [{i}/{len(combos)}] {label}", end="", flush=True)
        m = run_with_runner(svi, guide, model_fn, step_fn, data, p,
                           initial_state=initial_state)
        print(f"  → score={m['score']:.4f} train={m['train_error']:.4%}"
              f" pred={m['pred_error']:.4%} ci={m['ci_coverage']:.1%}"
              f" ({m['runtime']:.0f}s)", flush=True)
        rows.append({'p': p, 'm': m})
    rows.sort(key=lambda r: r['m']['score'])
    if rows: rows[0]['best'] = True
    print_table(rows, title, keys)
    return rows


# =============================================================================
# Main
# =============================================================================
def main(phases=None):
    if phases is None:
        phases = [1, 2, 3]

    print("=" * 70, flush=True)
    print("  tune_04 — Hyperparameter Sweep for Heat 04", flush=True)
    print(f"  Phases: {phases} | Sweep: {SWEEP_STEPS} steps, {SWEEP_SAMPLES} samples", flush=True)
    print("=" * 70, flush=True)

    data = prepare_data()

    model_fn = build_tuning_model(
        rom=data['rom'], num_modes=data['num_modes'], num_ics=data['num_ics'],
        all_time_sampled=data['all_time_sampled'],
        all_snapshots_comp=data['all_snaps_comp'],
        all_inputs_eval=data['all_inputs_eval'],
        O_prior=data['O_ls'],
        all_mle_Ls=data['all_mle_Ls'], all_mle_Vs=data['all_mle_Vs'],
        all_mle_Ns=data['all_mle_Ns'],
        num_eval_points=data['num_eval_points'],
        window_size=DEFAULTS['WINDOW_SIZE'])

    init_values = {'O': jnp.array(data['O_ls'])}
    for ic in range(data['num_ics']):
        for j in range(data['num_modes']):
            init_values[f'lengthscale_{ic}_{j}'] = data['all_mle_Ls'][ic][j]
            init_values[f'variance_{ic}_{j}'] = data['all_mle_Vs'][ic][j]
            init_values[f'noise_{ic}_{j}'] = data['all_mle_Ns'][ic][j]

    svi, guide, step_fn = create_runner(model_fn, init_values, DEFAULTS['LEARNING_RATE'])
    runner = (svi, guide, model_fn, step_fn)

    print("  Initializing SVI (one-time)...", flush=True)
    init_key = random.PRNGKey(DEFAULTS['SEED'])
    default_mkw = _make_kwargs(DEFAULTS)
    initial_svi_state = svi.init(init_key, **default_mkw)
    print("  SVI initialized.", flush=True)

    print("\n--- Baseline ---", flush=True)
    bl = run_with_runner(svi, guide, model_fn, step_fn, data, DEFAULTS,
                        initial_state=initial_svi_state)
    print(f"  score={bl['score']:.4f} train={bl['train_error']:.4%}"
          f" pred={bl['pred_error']:.4%} ci={bl['ci_coverage']:.1%}", flush=True)

    best = dict(DEFAULTS)
    best_score = bl['score']

    def mkp(**kw):
        p = dict(best); p.update(kw); return p

    if 1 in phases:
        print(f"\n{'#'*70}\n  PHASE 1: GAMMA × GAMMA2\n{'#'*70}", flush=True)
        combos = [mkp(GAMMA=g, GAMMA2=g2)
                  for g, g2 in itertools.product(
                      [1.0, 2.0, 4.0],
                      [0.1, 0.25, 0.5, 1.0, 2.0])]
        rows = run_sweep(runner, data, combos, "Phase 1: GAMMA × GAMMA2",
                        ['GAMMA', 'GAMMA2'], initial_state=initial_svi_state)
        if rows[0]['m']['score'] < best_score:
            best['GAMMA'] = rows[0]['p']['GAMMA']
            best['GAMMA2'] = rows[0]['p']['GAMMA2']
            best_score = rows[0]['m']['score']

    if 2 in phases:
        print(f"\n{'#'*70}\n  PHASE 2: MLL_WEIGHT × INTEGRAL_WEIGHT\n{'#'*70}", flush=True)
        combos = [mkp(MLL_WEIGHT=mw, INTEGRAL_WEIGHT=iw)
                  for mw, iw in itertools.product(
                      [0.01, 0.05, 0.1, 0.5, 1.0],
                      [0.25, 0.5, 1.0, 2.0, 5.0])]
        rows = run_sweep(runner, data, combos, "Phase 2: MLL × INTEGRAL",
                        ['MLL_WEIGHT', 'INTEGRAL_WEIGHT'],
                        initial_state=initial_svi_state)
        if rows[0]['m']['score'] < best_score:
            best['MLL_WEIGHT'] = rows[0]['p']['MLL_WEIGHT']
            best['INTEGRAL_WEIGHT'] = rows[0]['p']['INTEGRAL_WEIGHT']
            best_score = rows[0]['m']['score']

    if 3 in phases:
        print(f"\n{'#'*70}\n  PHASE 3: One-at-a-time\n{'#'*70}", flush=True)

        combos = [mkp(GP_PRIOR_SCALE=v) for v in [0.05, 0.1, 0.2, 0.5]]
        rows = run_sweep(runner, data, combos, "Phase 3a: GP_PRIOR_SCALE",
                        ['GP_PRIOR_SCALE'], initial_state=initial_svi_state)
        if rows[0]['m']['score'] < best_score:
            best['GP_PRIOR_SCALE'] = rows[0]['p']['GP_PRIOR_SCALE']
            best_score = rows[0]['m']['score']

        for lr in [1e-3, 2e-3, 3e-3, 5e-3, 1e-2]:
            if lr == best['LEARNING_RATE']: continue
            print(f"  LR={lr}...", end="", flush=True)
            s2, g2, sf2 = create_runner(model_fn, init_values, lr)
            m = run_with_runner(s2, g2, model_fn, sf2, data, mkp(LEARNING_RATE=lr))
            print(f"  → score={m['score']:.4f}", flush=True)
            if m['score'] < best_score:
                best['LEARNING_RATE'] = lr; best_score = m['score']
                svi, guide, step_fn = s2, g2, sf2
                runner = (svi, guide, model_fn, step_fn)

        for ws in [5, 10, 15, 20]:
            if ws == best['WINDOW_SIZE']: continue
            print(f"  WINDOW_SIZE={ws}...", end="", flush=True)
            mfn = build_tuning_model(
                rom=data['rom'], num_modes=data['num_modes'], num_ics=data['num_ics'],
                all_time_sampled=data['all_time_sampled'],
                all_snapshots_comp=data['all_snaps_comp'],
                all_inputs_eval=data['all_inputs_eval'],
                O_prior=data['O_ls'],
                all_mle_Ls=data['all_mle_Ls'], all_mle_Vs=data['all_mle_Vs'],
                all_mle_Ns=data['all_mle_Ns'],
                num_eval_points=data['num_eval_points'], window_size=ws)
            s2, g2, sf2 = create_runner(mfn, init_values, best['LEARNING_RATE'])
            m = run_with_runner(s2, g2, mfn, sf2, data, mkp(WINDOW_SIZE=ws))
            print(f"  → score={m['score']:.4f}", flush=True)
            if m['score'] < best_score:
                best['WINDOW_SIZE'] = ws; best_score = m['score']

    # Final Summary
    print(f"\n\n{'='*70}\n  FINAL SUMMARY\n{'='*70}", flush=True)
    print(f"  Baseline score: {bl['score']:.4f}", flush=True)
    print(f"  Best score:     {best_score:.4f}", flush=True)
    print(f"\n  Best hyperparameters:", flush=True)
    for k in ['GAMMA', 'GAMMA2', 'MLL_WEIGHT', 'INTEGRAL_WEIGHT', 'DERIV_WEIGHT',
              'GP_PRIOR_SCALE', 'WINDOW_SIZE', 'LEARNING_RATE']:
        d = DEFAULTS[k]; b = best[k]
        changed = " ← CHANGED" if b != d else ""
        print(f"    {k:>20s}: {b:>10g}  (was {d:>10g}){changed}", flush=True)

    print(f"\n  Copy-paste MODEL_PARAMS:", flush=True)
    print(f"  MODEL_PARAMS = dict(")
    for k in ['NUM_MODES', 'NUM_ICS', 'GAMMA', 'GAMMA2', 'DERIV_WEIGHT',
              'INTEGRAL_WEIGHT', 'MLL_WEIGHT', 'GP_PRIOR_SCALE',
              'WINDOW_SIZE', 'NUM_STEPS', 'LEARNING_RATE',
              'NUM_POSTERIOR_SAMPLES', 'SEED']:
        print(f"      {k}={best[k]},")
    print(f"  )")


if __name__ == "__main__":
    phase_args = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(phase_args)

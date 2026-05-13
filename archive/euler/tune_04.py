"""
tune_04.py — Efficient hyperparameter sweep for Euler 04_conditional_integral

Three-phase search with JIT caching:
  Phase 1: GAMMA × GAMMA2 grid
  Phase 2: MLL_WEIGHT × INTEGRAL_WEIGHT
  Phase 3: One-at-a-time (LR, GP_PRIOR_SCALE, WINDOW_SIZE)

Key optimization: all model weights passed as kwargs (not closure params)
→ single JIT compilation reused across Phase 1 AND Phase 2.

Usage:
    python tune_04.py              # all phases
    python tune_04.py 1            # phase 1 only
"""
import sys, os, io, time, contextlib, itertools
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
from config import Basis
from core import (generate_trajectory, JaxCompatibleModel,
                  compute_gp_derivatives, rbf_eval)
from core.bayesian_opinf import fit_gp_hyperparameters_mle, _find_operator_samples
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

TRAINING_SPAN = (0, 0.08)
PREDICTION_SPAN = (0, 0.15)
SWEEP_STEPS = 3000
SWEEP_SAMPLES = 100
SWEEP_PULLS = 10

SCHEMA = {"name": "dense_low_noise", "label": "Dense data, low noise",
          "NUM_SAMPLES": 250, "NOISE_LEVEL": 0.01, "NUM_EVAL_POINTS": 400}

DEFAULTS = dict(NUM_MODES=6, GAMMA=8.0, GAMMA2=1.0, DERIV_WEIGHT=1.0,
                INTEGRAL_WEIGHT=1.0, MLL_WEIGHT=0.1, GP_PRIOR_SCALE=0.1,
                WINDOW_SIZE=10, NUM_STEPS=10000, LEARNING_RATE=3e-3,
                NUM_POSTERIOR_SAMPLES=500, SEED=42)

# Model kwargs that go into the scan carry (order matters!)
CARRY_KEYS = ['GAMMA', 'GAMMA2', 'MLL_WEIGHT', 'INTEGRAL_WEIGHT',
              'DERIV_WEIGHT', 'GP_PRIOR_SCALE']


# =============================================================================
# Model builder — weights as model kwargs, not closure params
# =============================================================================
def build_tuning_model(rom, num_modes, time_sampled, snapshots_comp,
                       O_prior, mle_Ls, mle_Vs, mle_Ns,
                       num_eval_points=400, window_size=10):
    """Build model where ALL weights/scales are kwargs (enables JIT reuse)."""
    t_train = jnp.array(time_sampled)
    n_train = len(t_train)
    y_obs = jnp.array(snapshots_comp)

    time_eval = np.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
    t_eval = jnp.array(time_eval)
    dt_eval = float(time_eval[1] - time_eval[0])

    sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
    sq_diffs_et = (t_eval[:, None] - t_train[None, :]) ** 2
    diffs_et = t_eval[:, None] - t_train[None, :]
    sq_diffs_ee = (t_eval[:, None] - t_eval[None, :]) ** 2
    I_train = jnp.eye(n_train)

    O_prior_jnp = jnp.array(O_prior)
    mle_log_ells = jnp.array([jnp.log(l) for l in mle_Ls])
    mle_log_sig2s = jnp.array([jnp.log(v) for v in mle_Vs])
    mle_log_nus = jnp.array([jnp.log(n) for n in mle_Ns])

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

    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

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

    _batch_gp = jax.vmap(_single_gp_conditional)

    def model(gamma=2.0, gamma2=10.0, mll_weight=0.1,
              deriv_weight=1.0, integral_weight=1.0, gp_prior_scale=0.1):
        ells = jnp.stack([numpyro.sample(f"lengthscale_{i}",
                          dist.LogNormal(mle_log_ells[i], gp_prior_scale))
                          for i in range(num_modes)])
        sig2s = jnp.stack([numpyro.sample(f"variance_{i}",
                           dist.LogNormal(mle_log_sig2s[i], gp_prior_scale))
                           for i in range(num_modes)])
        nus = jnp.stack([numpyro.sample(f"noise_{i}",
                         dist.LogNormal(mle_log_nus[i], gp_prior_scale))
                         for i in range(num_modes)])

        Xs_eval, mu_zs, deriv_vars, mlls = _batch_gp(ells, sig2s, nus, y_obs)
        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", Xs_eval[i])

        numpyro.factor("gp_mll", mll_weight * jnp.sum(mlls))

        prior_scale = gamma * jnp.maximum(jnp.abs(O_prior_jnp), 0.5)
        O = numpyro.sample("O", dist.Normal(O_prior_jnp, prior_scale))
        f_Xi = rom.model._assemble_data_matrix(Xs_eval, inputs=None) @ O.T

        for i in range(num_modes):
            total_var = deriv_vars[i] + gamma2 + 1e-4
            numpyro.factor(f"ode_constraint_{i}",
                deriv_weight * jnp.sum(
                    dist.Normal(f_Xi[:, i], jnp.sqrt(total_var)).log_prob(mu_zs[i])))

        for i in range(num_modes):
            for w_idx, (ws, we) in enumerate(zip(ws_list, we_list)):
                delta_X_obs = Xs_eval[i, we] - Xs_eval[i, ws]
                delta_X_pred = jnp.sum(trap_weights[w_idx] * f_Xi[ws:we+1, i])
                constraint_std = jnp.sqrt(gamma2) * window_durations[w_idx]
                numpyro.factor(f"integral_{i}_{w_idx}",
                    integral_weight * dist.Normal(
                        delta_X_pred, constraint_std).log_prob(delta_X_obs))

    return model, time_eval


# =============================================================================
# Data preparation (ONE TIME)
# =============================================================================
def prepare_data():
    p = DEFAULTS
    np.random.seed(p['SEED'])
    print("Generating FOM data...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        fom, t_full, true_states, t_samp, snaps_samp = \
            generate_trajectory(config, config.time_domain, TRAINING_SPAN,
                                SCHEMA['NUM_SAMPLES'], SCHEMA['NOISE_LEVEL'])

    basis = Basis(num_vectors=p['NUM_MODES'])
    basis.fit(snaps_samp)
    snaps_comp = basis.compress(snaps_samp)
    true_comp = basis.compress(true_states)
    print(f"  POD energy: {basis.cumulative_energy:.4%}", flush=True)

    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
        model=JaxCompatibleModel(operators="cAH",
                                 solver=opinf.lstsq.L2Solver(regularizer=1e0)),
    )
    rom.fit(states=snaps_samp)

    print("Fitting MLE...", flush=True)
    Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(t_samp, snaps_comp, verbose=False)
    nep = SCHEMA['NUM_EVAL_POINTS']
    t_eval_ls = np.linspace(float(t_samp[0]), float(t_samp[-1]), nep)
    X_mle = np.zeros((p['NUM_MODES'], nep))
    for i in range(p['NUM_MODES']):
        K = rbf_eval(Ls[i], Vs[i], t_samp, t_samp) + (Ns[i]+1e-5)*np.eye(len(t_samp))
        Ks = rbf_eval(Ls[i], Vs[i], t_eval_ls, t_samp)
        X_mle[i] = Ks @ np.linalg.solve(K, snaps_comp[i])
    mu_z_mle, _ = compute_gp_derivatives(Ls, Vs, t_samp, t_eval_ls, snaps_comp, Ns=Ns)
    D = np.array(rom.model._assemble_data_matrix(jnp.array(X_mle), inputs=None))
    DtD = D.T @ D
    O_ls = np.linalg.solve(DtD + np.eye(DtD.shape[0]), D.T @ np.array(mu_z_mle).T).T
    print(f"  LS operator norm: {np.linalg.norm(O_ls):.1f}", flush=True)

    ti = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
    return dict(rom=rom, num_modes=p['NUM_MODES'], t_samp=t_samp,
                snaps_comp=snaps_comp, Ls=Ls, Vs=Vs, Ns=Ns, O_ls=O_ls,
                num_eval_points=nep, truth_interp=ti)


# =============================================================================
# Runner with JIT caching
# =============================================================================
def create_runner(model_fn, init_values, learning_rate):
    """Create SVI + guide + JIT step. Returns (svi, guide, step_fn).
    step_fn carry: (svi_state, gamma, gamma2, mll_w, int_w, deriv_w, gp_ps)"""
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
    """Make model kwargs dict from params (as jnp floats for tracing)."""
    return {k.lower(): jnp.float32(params[k]) for k in
            ['GAMMA', 'GAMMA2', 'MLL_WEIGHT', 'INTEGRAL_WEIGHT',
             'DERIV_WEIGHT', 'GP_PRIOR_SCALE']}


def _make_carry(svi_state, params):
    """Build carry tuple for scan."""
    return (svi_state,
            jnp.float32(params['GAMMA']), jnp.float32(params['GAMMA2']),
            jnp.float32(params['MLL_WEIGHT']), jnp.float32(params['INTEGRAL_WEIGHT']),
            jnp.float32(params['DERIV_WEIGHT']), jnp.float32(params['GP_PRIOR_SCALE']))


def run_with_runner(svi, guide, model_fn, step_fn, data, params,
                    num_steps=SWEEP_STEPS, num_samples=SWEEP_SAMPLES,
                    num_pulls=SWEEP_PULLS, initial_state=None):
    """Run SVI using cached step function. Returns metrics dict.
    If initial_state provided, reuse it (skip expensive svi.init)."""
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

    # Manual posterior sampling (0.3s vs 40s for guide.sample_posterior)
    O_loc = svi_params['O_auto_loc']
    O_scale = jax.nn.softplus(svi_params['O_auto_scale'])
    O_samp = O_loc + O_scale * random.normal(sk, shape=(num_samples,) + O_loc.shape)
    O_samp = np.array(O_samp)

    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    rom = data['rom']
    snaps_comp = data['snaps_comp']

    rom_solves = []
    n_use = min(num_pulls, len(O_samp))
    indices = np.linspace(0, len(O_samp) - 1, n_use, dtype=int)
    for idx in indices:
        O_i = np.array(O_samp[idx])
        rom.model._extract_operators(O_i)
        try:
            rom.model.predict(state0=snaps_comp[:, 0], t=t_pred)
            result = rom.model.predict_result_
            sol = result.y if hasattr(result, 'y') else np.array(result.ys).T
            if sol.shape[1] >= len(t_pred) and np.all(np.isfinite(sol)):
                rom_solves.append(sol)
        except Exception:
            pass

    n_stable = len(rom_solves)
    n_total = n_use
    runtime = time.time() - t0
    train_mask = t_pred <= TRAINING_SPAN[1]
    pred_mask = t_pred > TRAINING_SPAN[1]

    if n_stable == 0:
        return dict(train_error=float('inf'), pred_error=float('inf'),
                    stability_pct=0.0, ci_coverage=0.0, runtime=runtime,
                    score=999.0, final_loss=float(losses[-1]))

    rom_arr = np.array(rom_solves)
    rom_med = np.median(rom_arr, axis=0)
    ta = data['truth_interp'](t_pred)
    stability_pct = n_stable / max(n_total, 1) * 100

    train_error = float(np.linalg.norm(rom_med[:, train_mask] - ta[:, train_mask]) /
                        np.linalg.norm(ta[:, train_mask]))
    pred_error = float(np.linalg.norm(rom_med[:, pred_mask] - ta[:, pred_mask]) /
                       np.linalg.norm(ta[:, pred_mask]))
    q05 = np.percentile(rom_arr, 5, axis=0)
    q95 = np.percentile(rom_arr, 95, axis=0)
    ci_coverage = float(np.mean((ta >= q05) & (ta <= q95)))

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
    print(hdr)
    print("-" * len(hdr))
    for row in sorted(rows, key=lambda r: r['m']['score']):
        v = "  ".join(f"{row['p'][k]:>12g}" for k in keys)
        m = row['m']
        te = f"{m['train_error']:.4%}" if m['train_error'] < 10 else "  >1000%"
        pe = f"{m['pred_error']:.4%}" if m['pred_error'] < 10 else "  >1000%"
        v += f"  {te:>8s} {pe:>8s} {m['stability_pct']:>4.0f}% {m['ci_coverage']:>6.1%}"
        v += f" {m['score']:>8.4f} {m['runtime']:>4.0f}s"
        if row.get('best'):
            v += "  ◀ BEST"
        print(v)
    print(flush=True)


def run_sweep(runner_args, data, combos, title, keys, initial_state=None):
    """Run sweep using cached runner. combos = list of param dicts."""
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
    if rows:
        rows[0]['best'] = True
    print_table(rows, title, keys)
    return rows


# =============================================================================
# Main
# =============================================================================
def main(phases=None):
    if phases is None:
        phases = [1, 2, 3]

    print("=" * 70, flush=True)
    print("  tune_04 — Hyperparameter Sweep for Euler 04", flush=True)
    print(f"  Phases: {phases} | Sweep: {SWEEP_STEPS} steps, {SWEEP_SAMPLES} samples", flush=True)
    print(f"  Defaults: γ={DEFAULTS['GAMMA']}, γ₂={DEFAULTS['GAMMA2']}, "
          f"mll={DEFAULTS['MLL_WEIGHT']}, int={DEFAULTS['INTEGRAL_WEIGHT']}", flush=True)
    print("=" * 70, flush=True)

    data = prepare_data()

    # Build model ONCE (window_size=default) — shared across Phase 1+2
    model_fn, time_eval = build_tuning_model(
        rom=data['rom'], num_modes=data['num_modes'],
        time_sampled=data['t_samp'], snapshots_comp=data['snaps_comp'],
        O_prior=data['O_ls'], mle_Ls=data['Ls'], mle_Vs=data['Vs'],
        mle_Ns=data['Ns'], num_eval_points=data['num_eval_points'],
        window_size=DEFAULTS['WINDOW_SIZE'])

    init_values = {'O': jnp.array(data['O_ls'])}
    for i in range(data['num_modes']):
        init_values[f'lengthscale_{i}'] = data['Ls'][i]
        init_values[f'variance_{i}'] = data['Vs'][i]
        init_values[f'noise_{i}'] = data['Ns'][i]

    # Create runner (JIT'd step) — shared across Phase 1+2
    svi, guide, step_fn = create_runner(model_fn, init_values, DEFAULTS['LEARNING_RATE'])
    runner = (svi, guide, model_fn, step_fn)

    # Init SVI state ONCE — reuse to avoid expensive re-tracing
    print("  Initializing SVI (one-time)...", flush=True)
    init_key = random.PRNGKey(DEFAULTS['SEED'])
    default_mkw = _make_kwargs(DEFAULTS)
    initial_svi_state = svi.init(init_key, **default_mkw)
    print("  SVI initialized.", flush=True)

    # Baseline
    print("\n--- Baseline (defaults) ---", flush=True)
    bl = run_with_runner(svi, guide, model_fn, step_fn, data, DEFAULTS,
                        initial_state=initial_svi_state)
    print(f"  score={bl['score']:.4f} train={bl['train_error']:.4%}"
          f" pred={bl['pred_error']:.4%} ci={bl['ci_coverage']:.1%}", flush=True)

    best = dict(DEFAULTS)
    best_score = bl['score']

    def mkp(**kw):
        p = dict(best)
        p.update(kw)
        return p

    # ── Phase 1: GAMMA × GAMMA2 ─────────────────────────────────
    if 1 in phases:
        print(f"\n{'#'*70}\n  PHASE 1: GAMMA × GAMMA2\n{'#'*70}", flush=True)
        combos = [mkp(GAMMA=g, GAMMA2=g2)
                  for g, g2 in itertools.product(
                      [0.5, 1.0, 2.0, 4.0, 8.0],
                      [0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0])]
        rows = run_sweep(runner, data, combos, "Phase 1: GAMMA × GAMMA2",
                        ['GAMMA', 'GAMMA2'], initial_state=initial_svi_state)
        if rows[0]['m']['score'] < best_score:
            best['GAMMA'] = rows[0]['p']['GAMMA']
            best['GAMMA2'] = rows[0]['p']['GAMMA2']
            best_score = rows[0]['m']['score']
            print(f"  ✓ Winner: GAMMA={best['GAMMA']}, GAMMA2={best['GAMMA2']}  "
                  f"score={best_score:.4f}", flush=True)

    # ── Phase 2: MLL_WEIGHT × INTEGRAL_WEIGHT ────────────────────
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

    # ── Phase 3: One-at-a-time ───────────────────────────────────
    if 3 in phases:
        print(f"\n{'#'*70}\n  PHASE 3: One-at-a-time sweeps\n{'#'*70}", flush=True)

        # GP_PRIOR_SCALE — cached (same runner)
        combos = [mkp(GP_PRIOR_SCALE=v) for v in [0.05, 0.1, 0.2, 0.5]]
        rows = run_sweep(runner, data, combos, "Phase 3a: GP_PRIOR_SCALE",
                        ['GP_PRIOR_SCALE'], initial_state=initial_svi_state)
        if rows[0]['m']['score'] < best_score:
            best['GP_PRIOR_SCALE'] = rows[0]['p']['GP_PRIOR_SCALE']
            best_score = rows[0]['m']['score']

        # LEARNING_RATE — needs new runner (different optimizer)
        for lr in [1e-3, 2e-3, 3e-3, 5e-3, 1e-2]:
            if lr == best['LEARNING_RATE']:
                continue
            print(f"  LR={lr}...", end="", flush=True)
            svi2, guide2, step2 = create_runner(model_fn, init_values, lr)
            p = mkp(LEARNING_RATE=lr)
            m = run_with_runner(svi2, guide2, model_fn, step2, data, p)
            print(f"  → score={m['score']:.4f}", flush=True)
            if m['score'] < best_score:
                best['LEARNING_RATE'] = lr
                best_score = m['score']
                svi, guide, step_fn = svi2, guide2, step2
                runner = (svi, guide, model_fn, step_fn)

        # WINDOW_SIZE — needs new model (different # integral windows)
        for ws in [5, 10, 15, 20]:
            if ws == best['WINDOW_SIZE']:
                continue
            print(f"  WINDOW_SIZE={ws}...", end="", flush=True)
            mfn, _ = build_tuning_model(
                rom=data['rom'], num_modes=data['num_modes'],
                time_sampled=data['t_samp'], snapshots_comp=data['snaps_comp'],
                O_prior=data['O_ls'], mle_Ls=data['Ls'], mle_Vs=data['Vs'],
                mle_Ns=data['Ns'], num_eval_points=data['num_eval_points'],
                window_size=ws)
            s2, g2, sf2 = create_runner(mfn, init_values, best['LEARNING_RATE'])
            p = mkp(WINDOW_SIZE=ws)
            m = run_with_runner(s2, g2, mfn, sf2, data, p)
            print(f"  → score={m['score']:.4f}", flush=True)
            if m['score'] < best_score:
                best['WINDOW_SIZE'] = ws
                best_score = m['score']

    # ── Final Summary ────────────────────────────────────────────
    print(f"\n\n{'='*70}\n  FINAL SUMMARY\n{'='*70}", flush=True)
    print(f"  Baseline score: {bl['score']:.4f}  (train={bl['train_error']:.4%}"
          f" pred={bl['pred_error']:.4%} ci={bl['ci_coverage']:.1%})", flush=True)
    print(f"  Best score:     {best_score:.4f}", flush=True)
    print(f"\n  Best hyperparameters:", flush=True)
    for k in ['GAMMA', 'GAMMA2', 'MLL_WEIGHT', 'INTEGRAL_WEIGHT', 'DERIV_WEIGHT',
              'GP_PRIOR_SCALE', 'WINDOW_SIZE', 'LEARNING_RATE']:
        d = DEFAULTS[k]
        b = best[k]
        changed = " ← CHANGED" if b != d else ""
        print(f"    {k:>20s}: {b:>10g}  (was {d:>10g}){changed}", flush=True)

    print(f"\n  Copy-paste MODEL_PARAMS:", flush=True)
    print(f"  MODEL_PARAMS = dict(")
    print(f"      NUM_MODES={best['NUM_MODES']},")
    print(f"      GAMMA={best['GAMMA']},")
    print(f"      GAMMA2={best['GAMMA2']},")
    print(f"      DERIV_WEIGHT={best['DERIV_WEIGHT']},")
    print(f"      INTEGRAL_WEIGHT={best['INTEGRAL_WEIGHT']},")
    print(f"      MLL_WEIGHT={best['MLL_WEIGHT']},")
    print(f"      GP_PRIOR_SCALE={best['GP_PRIOR_SCALE']},")
    print(f"      WINDOW_SIZE={best['WINDOW_SIZE']},")
    print(f"      NUM_STEPS={best['NUM_STEPS']},")
    print(f"      LEARNING_RATE={best['LEARNING_RATE']},")
    print(f"      NUM_POSTERIOR_SAMPLES={best['NUM_POSTERIOR_SAMPLES']},")
    print(f"      SEED={best['SEED']},")
    print(f"  )")


if __name__ == "__main__":
    phase_args = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(phase_args)

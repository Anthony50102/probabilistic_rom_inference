"""
04d — Conditional GP + Dual Constraint  with optional warm-start / informative priors

Tests whether the MLE warm-start and MLE-anchored priors are necessary.
Two knobs (CLI):
  PRIOR ∈ {informative, broad}
    informative : ℓ ~ LogNormal(log ℓ_MLE, 0.1)   (current)
                  O ~ N(O_ls, γ·max(|O_ls|, 0.5))
    broad       : ℓ ~ LogNormal(log(T/20), 1.0)   (paper Stage-1 style)
                  σ² ~ LogNormal(log var(q̂), 0.5)
                  ν  ~ LogNormal(-8.0, 1.0)
                  O  ~ N(0, 10·I)                  (paper Stage-3)
  INIT ∈ {mle, prior}
    mle   : guide initialised at MLE hypers and O_ls   (current)
    prior : init_loc_fn=None → numpyro picks from prior

Usage:
    python 04d_no_warmstart.py informative mle    dense_low_noise   # = 04 baseline
    python 04d_no_warmstart.py informative prior  dense_low_noise   # priors stay tight, no init
    python 04d_no_warmstart.py broad       mle    dense_low_noise   # paper priors w/ init
    python 04d_no_warmstart.py broad       prior  dense_low_noise   # "just run VI"
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
from numpyro.infer.initialization import init_to_value, init_to_sample, init_to_median
from numpyro.optim import ClippedAdam
from jax import random
from scipy.interpolate import interp1d

# Optional MLL_WEIGHT override (for option-i test)
_MLL_OVERRIDE = os.environ.get("MLL_WEIGHT_OVERRIDE")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis
from core import (
    generate_trajectory, JaxCompatibleModel, compute_gp_derivatives,
    generate_rom_predictions, rbf_eval,
)
from core.bayesian_opinf import fit_gp_hyperparameters_mle, _find_operator_samples
from core.diagnostics import plot_trace
from core.plotting import plot_full_order_error
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

# ── Data regime definitions ──────────────────────────────────────────────────
SCHEMAS = [
    {
        "name": "dense_low_noise",
        "label": "Dense data, low noise",
        "NUM_SAMPLES": 250,
        "NOISE_LEVEL": 0.01,
        "NUM_EVAL_POINTS": 400,
    },
    {
        "name": "sparse_low_noise",
        "label": "Sparse data, low noise",
        "NUM_SAMPLES": 55,
        "NOISE_LEVEL": 0.03,
        "NUM_EVAL_POINTS": 200,
    },
    {
        "name": "dense_high_noise",
        "label": "Dense data, high noise",
        "NUM_SAMPLES": 250,
        "NOISE_LEVEL": 0.10,
        "NUM_EVAL_POINTS": 400,
    },
]

# ── Shared model hyperparameters (same for ALL regimes) ──────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=6,
    GAMMA=2.0,
    GAMMA2=10.0,
    DERIV_WEIGHT=1.0,
    INTEGRAL_WEIGHT=1.0,
    MLL_WEIGHT=0.1 if _MLL_OVERRIDE is None else float(_MLL_OVERRIDE),
    GP_PRIOR_SCALE=0.1,
    WINDOW_SIZE=20,
    NUM_STEPS=10000,
    LEARNING_RATE=3e-3,
    NUM_POSTERIOR_SAMPLES=500,
    SEED=42,
    PRIOR='informative',   # informative | broad
    INIT='mle',            # mle | prior
)

PRIOR_CHOICES = ('informative', 'broad', 'scalefree', 'reparam')
INIT_CHOICES = ('mle', 'prior', 'gpmle', 'sample', 'map', 'lbfgs', 'temper')

TRAINING_SPAN = (0, 0.08)
PREDICTION_SPAN = (0, 0.15)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# Model builder
# =============================================================================
def build_model(
    rom, num_modes, time_sampled, snapshots_comp,
    O_prior, mle_Ls, mle_Vs, mle_Ns,
    num_eval_points=400, window_size=10,
    deriv_weight=1.0, integral_weight=1.0,
    mll_weight=0.1, gp_prior_scale=0.1,
    prior_style='informative',
):
    """Build the conditional integral NumPyro model.

    prior_style='informative' (default): tight LogNormal priors centred at
        the MLE hypers, operator prior centred at O_ls with scale gamma|O_ls|.
    prior_style='broad': paper-style priors that do NOT depend on MLE or LS.
        ℓ ~ LogNormal(log(T/20), 1.0),  σ² ~ LogNormal(log var(y), 0.5),
        ν ~ LogNormal(-8.0, 1.0),       O ~ N(0, 10 I).
    """
    t_train = jnp.array(time_sampled)
    n_train = len(t_train)
    y_obs = jnp.array(snapshots_comp)

    time_eval = np.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
    t_eval = jnp.array(time_eval)
    dt_eval = float(time_eval[1] - time_eval[0])

    # Precompute kernel distance matrices
    sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2
    sq_diffs_et = (t_eval[:, None] - t_train[None, :]) ** 2
    diffs_et = t_eval[:, None] - t_train[None, :]
    sq_diffs_ee = (t_eval[:, None] - t_eval[None, :]) ** 2
    I_train = jnp.eye(n_train)

    O_prior_jnp = jnp.array(O_prior)
    mle_log_ells = jnp.array([jnp.log(l) for l in mle_Ls])
    mle_log_sig2s = jnp.array([jnp.log(v) for v in mle_Vs])
    mle_log_nus = jnp.array([jnp.log(n) for n in mle_Ns])

    # Broad-prior parameters (paper Algorithm 1 style)
    T_span = float(t_train[-1] - t_train[0])
    broad_log_ell_loc = jnp.log(T_span / 20.0)
    broad_log_sig2_locs = jnp.array(
        [float(jnp.log(jnp.var(jnp.array(snapshots_comp[i])) + 1e-12))
         for i in range(num_modes)])

    # Precompute integration windows
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

    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def _single_gp_conditional(ell, sig2, nu, y_i):
        """GP posterior: mean, derivative mean/var, MLL — all deterministic given hypers."""
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

    _batch_gp_conditional = jax.vmap(_single_gp_conditional)

    def model(gamma=2.0, gamma2=10.0, jitter=1e-4):

        # GP hyperparameters
        if prior_style == 'informative':
            ells = jnp.stack([
                numpyro.sample(f"lengthscale_{i}",
                              dist.LogNormal(mle_log_ells[i], gp_prior_scale))
                for i in range(num_modes)])
            sig2s = jnp.stack([
                numpyro.sample(f"variance_{i}",
                              dist.LogNormal(mle_log_sig2s[i], gp_prior_scale))
                for i in range(num_modes)])
            nus = jnp.stack([
                numpyro.sample(f"noise_{i}",
                              dist.LogNormal(mle_log_nus[i], gp_prior_scale))
                for i in range(num_modes)])
        else:  # 'broad' or 'scalefree' share paper-style GP-hyper priors
            ells = jnp.stack([
                numpyro.sample(f"lengthscale_{i}",
                              dist.LogNormal(broad_log_ell_loc, 1.0))
                for i in range(num_modes)])
            sig2s = jnp.stack([
                numpyro.sample(f"variance_{i}",
                              dist.LogNormal(broad_log_sig2_locs[i], 0.5))
                for i in range(num_modes)])
            nus = jnp.stack([
                numpyro.sample(f"noise_{i}",
                              dist.LogNormal(-8.0, 1.0))
                for i in range(num_modes)])

        # GP conditional (deterministic given hypers + data)
        Xs_eval, mu_zs, deriv_vars, mlls = _batch_gp_conditional(ells, sig2s, nus, y_obs)

        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", Xs_eval[i])

        # GP marginal log-likelihood
        if mll_weight > 0:
            numpyro.factor("gp_mll", mll_weight * jnp.sum(mlls))

        # Operator prior
        if prior_style == 'informative':
            prior_scale = gamma * jnp.maximum(jnp.abs(O_prior_jnp), 0.5)
            O = numpyro.sample("O", dist.Normal(O_prior_jnp, prior_scale))
        elif prior_style == 'broad':  # paper's N(0, 10 I)
            O = numpyro.sample("O", dist.Normal(jnp.zeros_like(O_prior_jnp),
                                                10.0 * jnp.ones_like(O_prior_jnp)))
        elif prior_style == 'scalefree':  # std comparable to LS operator norm (~975)
            O = numpyro.sample("O", dist.Normal(jnp.zeros_like(O_prior_jnp),
                                                1000.0 * jnp.ones_like(O_prior_jnp)))
        else:  # 'reparam': O = s * U,  s ~ HalfCauchy(1000),  U ~ N(0, I)
            s = numpyro.sample("O_scale", dist.HalfCauchy(scale=1000.0))
            U = numpyro.sample("O_dir",
                dist.Normal(jnp.zeros_like(O_prior_jnp), jnp.ones_like(O_prior_jnp)))
            O = numpyro.deterministic("O", s * U)

        # Operator dynamics: f(X_eval) @ O^T
        f_Xi = rom.model._assemble_data_matrix(Xs_eval, inputs=None) @ O.T

        # CONSTRAINT 1: Derivative matching
        if deriv_weight > 0:
            for i in range(num_modes):
                total_var = deriv_vars[i] + gamma2 + jitter
                numpyro.factor(f"ode_constraint_{i}",
                    deriv_weight * jnp.sum(
                        dist.Normal(f_Xi[:, i], jnp.sqrt(total_var)).log_prob(mu_zs[i])))

        # CONSTRAINT 2: Integral form
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

    print(f"\n{'='*70}")
    print(f"  {schema['label']}  ({num_samples} samples, {noise_level:.0%} noise)")
    print(f"{'='*70}")

    # ── Data generation ──────────────────────────────────────────────────
    (fom, t_full, true_states, t_samp, snaps_samp) = \
        generate_trajectory(config, config.time_domain, TRAINING_SPAN, num_samples, noise_level)
    basis = Basis(num_vectors=num_modes)
    basis.fit(snaps_samp)
    snaps_comp = basis.compress(snaps_samp)
    true_comp = basis.compress(true_states)
    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
        model=JaxCompatibleModel(operators="cAH",
                                 solver=opinf.lstsq.L2Solver(regularizer=1e0)),
    )
    rom.fit(states=snaps_samp)

    # ── MLE warm start ───────────────────────────────────────────────────
    Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(t_samp, snaps_comp, verbose=False)
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
    print(f"  LS operator norm: {np.linalg.norm(O_ls):.1f}")

    # ── Build & run SVI ──────────────────────────────────────────────────
    model, time_eval = build_model(
        rom=rom, num_modes=num_modes,
        time_sampled=t_samp, snapshots_comp=snaps_comp,
        O_prior=O_ls, mle_Ls=Ls, mle_Vs=Vs, mle_Ns=Ns,
        num_eval_points=num_eval_points, window_size=p['WINDOW_SIZE'],
        deriv_weight=p['DERIV_WEIGHT'], integral_weight=p['INTEGRAL_WEIGHT'],
        mll_weight=p['MLL_WEIGHT'], gp_prior_scale=p['GP_PRIOR_SCALE'],
        prior_style=p['PRIOR'],
    )

    if p['INIT'] == 'mle' and p['PRIOR'] != 'reparam':
        init_values = {'O': jnp.array(O_ls)}
        for i in range(num_modes):
            init_values[f'lengthscale_{i}'] = Ls[i]
            init_values[f'variance_{i}'] = Vs[i]
            init_values[f'noise_{i}'] = Ns[i]
        guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    elif p['INIT'] == 'mle' and p['PRIOR'] == 'reparam':
        # reparam splits O into (scale, direction); warm-start at LS by setting
        # s = ||O_ls|| and U = O_ls / ||O_ls|| (unit-norm direction)
        O_ls_jnp = jnp.array(O_ls)
        s_init = float(jnp.linalg.norm(O_ls_jnp))
        U_init = O_ls_jnp / (s_init + 1e-12)
        init_values = {'O_scale': s_init, 'O_dir': U_init}
        for i in range(num_modes):
            init_values[f'lengthscale_{i}'] = Ls[i]
            init_values[f'variance_{i}'] = Vs[i]
            init_values[f'noise_{i}'] = Ns[i]
        guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    elif p['INIT'] == 'gpmle':
        # (B0) GP hypers from MLE; O left to numpyro default (= prior median)
        init_values = {}
        for i in range(num_modes):
            init_values[f'lengthscale_{i}'] = Ls[i]
            init_values[f'variance_{i}'] = Vs[i]
            init_values[f'noise_{i}'] = Ns[i]
        guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    elif p['INIT'] == 'sample':
        # (B1) Random draw from prior — for scalefree this puts O far from 0
        guide = autoguide.AutoNormal(model, init_loc_fn=init_to_sample)
    elif p['INIT'] == 'map':
        # (B2) Cheap MAP optimisation warm-up:
        # ~500 Adam steps maximising log_joint (point estimate of every site),
        # then init the AutoNormal guide at that MAP point.  Init the MAP run
        # itself from a random prior sample so we don't start at the null basin.
        from numpyro.infer.autoguide import AutoDelta
        rng_key, mk = random.split(rng_key)
        delta_guide = AutoDelta(model, init_loc_fn=init_to_sample)
        map_optim = ClippedAdam(step_size=1e-2)
        map_svi = SVI(model, delta_guide, map_optim, loss=Trace_ELBO())
        map_state = map_svi.init(mk, **dict(gamma=p['GAMMA'], gamma2=p['GAMMA2'], jitter=1e-4))

        @jax.jit
        def _map_step(s, _):
            return map_svi.update(s, **dict(gamma=p['GAMMA'], gamma2=p['GAMMA2'], jitter=1e-4))

        N_MAP = 500
        t_map0 = time.time()
        map_state, map_losses = jax.lax.scan(_map_step, map_state, jnp.arange(N_MAP))
        ml = np.array(map_losses)
        map_params = map_svi.get_params(map_state)
        # AutoDelta stores params with '_auto_loc' suffix; strip to get site values
        init_values = {}
        for k, v in map_params.items():
            if k.endswith('_auto_loc'):
                init_values[k[:-len('_auto_loc')]] = v
        print(f"    MAP warm-up: {N_MAP} steps   loss {ml[0]:.2f} → {ml[-1]:.2f}   ({time.time()-t_map0:.1f}s)")
        if 'O' in init_values:
            print(f"      ||O||_MAP = {float(jnp.linalg.norm(init_values['O'])):.2f}")
        if 'O_scale' in init_values:
            print(f"      s_MAP    = {float(init_values['O_scale']):.2f}")
        guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    elif p['INIT'] == 'lbfgs':
        # (C1) L-BFGS MAP via jaxopt
        import jaxopt
        from numpyro.infer.autoguide import AutoDelta
        rng_key, sk = random.split(rng_key)
        init_dict = {}
        for i in range(num_modes):
            init_dict[f'lengthscale_{i}'] = jnp.asarray(Ls[i])
            init_dict[f'variance_{i}'] = jnp.asarray(Vs[i])
            init_dict[f'noise_{i}'] = jnp.asarray(Ns[i])
        if p['PRIOR'] == 'reparam':
            init_dict['O_scale'] = jnp.asarray(100.0)
            init_dict['O_dir'] = jax.random.normal(sk, shape=O_ls.shape) * 0.01
        else:
            init_dict['O'] = jax.random.normal(sk, shape=O_ls.shape) * 10.0
        delta_guide = AutoDelta(model, init_loc_fn=init_to_value(values=init_dict))
        map_svi = SVI(model, delta_guide, ClippedAdam(1e-3), loss=Trace_ELBO())
        rng_key, ik = random.split(rng_key)
        ds = map_svi.init(ik, **dict(gamma=p['GAMMA'], gamma2=p['GAMMA2'], jitter=1e-4))
        params0 = map_svi.get_params(ds)
        flat0, unravel = jax.flatten_util.ravel_pytree(params0)
        rng_key, lk = random.split(rng_key)
        def _loss(flat):
            params = unravel(flat)
            return Trace_ELBO().loss(lk, params, model, delta_guide,
                                     **dict(gamma=p['GAMMA'], gamma2=p['GAMMA2'], jitter=1e-4))
        t_map0 = time.time()
        solver = jaxopt.LBFGS(fun=_loss, maxiter=300, tol=1e-4)
        res = solver.run(flat0)
        params_opt = unravel(res.params)
        init_values = {k[:-len('_auto_loc')]: v for k, v in params_opt.items()
                       if k.endswith('_auto_loc')}
        print(f"    L-BFGS: {int(res.state.iter_num)} iters  loss → {float(res.state.value):.2f}   ({time.time()-t_map0:.1f}s)")
        if 'O' in init_values:
            print(f"      ||O||_LBFGS = {float(jnp.linalg.norm(init_values['O'])):.2f}")
        if 'O_scale' in init_values:
            print(f"      s_LBFGS    = {float(init_values['O_scale']):.2f}")
        guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    elif p['INIT'] == 'temper':
        # (C2) Likelihood-tempered MAP: 100x deriv_weight + integral_weight
        # to overcome the null mode's flat gradient.  Start from GP-MLE hypers
        # and small random O.
        from numpyro.infer.autoguide import AutoDelta
        rng_key, sk = random.split(rng_key)
        TEMP = 100.0
        tempered_model, _ = build_model(
            rom=rom, num_modes=num_modes,
            time_sampled=t_samp, snapshots_comp=snaps_comp,
            O_prior=O_ls, mle_Ls=Ls, mle_Vs=Vs, mle_Ns=Ns,
            num_eval_points=num_eval_points, window_size=p['WINDOW_SIZE'],
            deriv_weight=p['DERIV_WEIGHT'] * TEMP,
            integral_weight=p['INTEGRAL_WEIGHT'] * TEMP,
            mll_weight=p['MLL_WEIGHT'], gp_prior_scale=p['GP_PRIOR_SCALE'],
            prior_style=p['PRIOR'])
        init_dict = {}
        for i in range(num_modes):
            init_dict[f'lengthscale_{i}'] = jnp.asarray(Ls[i])
            init_dict[f'variance_{i}'] = jnp.asarray(Vs[i])
            init_dict[f'noise_{i}'] = jnp.asarray(Ns[i])
        if p['PRIOR'] == 'reparam':
            init_dict['O_scale'] = jnp.asarray(100.0)
            init_dict['O_dir'] = jax.random.normal(sk, shape=O_ls.shape) * 0.01
        else:
            init_dict['O'] = jax.random.normal(sk, shape=O_ls.shape) * 10.0
        delta_guide = AutoDelta(tempered_model, init_loc_fn=init_to_value(values=init_dict))
        map_svi = SVI(tempered_model, delta_guide, ClippedAdam(step_size=5e-3), loss=Trace_ELBO())
        rng_key, ik = random.split(rng_key)
        map_state = map_svi.init(ik, **dict(gamma=p['GAMMA'], gamma2=p['GAMMA2'], jitter=1e-4))

        @jax.jit
        def _map_step(s, _):
            return map_svi.update(s, **dict(gamma=p['GAMMA'], gamma2=p['GAMMA2'], jitter=1e-4))

        N_MAP = 3000
        t_map0 = time.time()
        map_state, map_losses = jax.lax.scan(_map_step, map_state, jnp.arange(N_MAP))
        ml = np.array(map_losses)
        map_params = map_svi.get_params(map_state)
        init_values = {k[:-len('_auto_loc')]: v for k, v in map_params.items()
                       if k.endswith('_auto_loc')}
        print(f"    Tempered MAP (x{TEMP:.0f}): {N_MAP} steps   loss {ml[0]:.2f} → {ml[-1]:.2f}   ({time.time()-t_map0:.1f}s)")
        if 'O' in init_values:
            print(f"      ||O||_MAP = {float(jnp.linalg.norm(init_values['O'])):.2f}")
        if 'O_scale' in init_values:
            print(f"      s_MAP    = {float(init_values['O_scale']):.2f}")
        guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=init_values))
    else:
        # 'prior' — NumPyro defaults to median-of-prior initialisation
        # (note: for N(0,σ²I) priors this IS the null-basin location)
        guide = autoguide.AutoNormal(model)

    print(f"  Prior: {p['PRIOR']}   Init: {p['INIT']}   MLL_WEIGHT: {p['MLL_WEIGHT']}")

    model_kwargs = dict(gamma=p['GAMMA'], gamma2=p['GAMMA2'], jitter=1e-4)
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

    # ── Evaluate ─────────────────────────────────────────────────────────
    O_samp = _find_operator_samples(samples, "O")
    if O_samp.ndim == 2:
        O_samp = O_samp[np.newaxis, ...]
    O_med = np.median(O_samp, axis=0)
    O_std = np.std(O_samp, axis=0)

    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    Os, Xs, rom_solves = generate_rom_predictions(
        samples=samples, rom=rom, snapshots_compressed=snaps_comp,
        time_eval=t_pred, num_modes=num_modes,
        num_pulls=min(200, n_post))

    n_stable = len(rom_solves)
    n_total = len(Os)
    stability_pct = n_stable / max(n_total, 1) * 100

    train_error = pred_error = float('inf')
    ci_coverage = ci_width = float('nan')

    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        train_mask = t_pred <= TRAINING_SPAN[1]
        pred_mask = t_pred > TRAINING_SPAN[1]

        ti = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        ta = ti(t_pred)

        train_error = float(np.linalg.norm(rom_med[:, train_mask] - ta[:, train_mask]) /
                           np.linalg.norm(ta[:, train_mask]))
        pred_error = float(np.linalg.norm(rom_med[:, pred_mask] - ta[:, pred_mask]) /
                          np.linalg.norm(ta[:, pred_mask]))

        q05 = np.percentile(rom_arr, 5, axis=0)
        q95 = np.percentile(rom_arr, 95, axis=0)
        ci_width = float(np.mean(q95 - q05))
        ci_coverage = float(np.mean((ta >= q05) & (ta <= q95)))

    print(f"\n  Results ({runtime:.0f}s):")
    print(f"    Stability: {n_stable}/{n_total} ({stability_pct:.0f}%)")
    print(f"    Train error: {train_error:.4%}  |  Pred error: {pred_error:.4%}")
    print(f"    CI coverage: {ci_coverage:.2%} (target: 90%)")
    print(f"    Operator norm: {np.linalg.norm(O_med):.1f} (LS: {np.linalg.norm(O_ls):.1f})")
    print(f"    Convergence: loss {all_losses[0]:.0f} → {all_losses[-1]:.0f}")

    return {
        'schema': schema,
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'n_stable': n_stable, 'n_total': n_total,
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'runtime': runtime, 'losses': all_losses,
        'samples': samples, 'rom_solves': rom_solves,
        'snaps_comp': snaps_comp, 'true_comp': true_comp,
        't_full': t_full, 't_pred': t_pred, 't_samp': t_samp,
        'training_span': TRAINING_SPAN, 'num_modes': num_modes,
        'true_states': true_states, 'basis': basis,
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_results(result, save_dir=None):
    """Generate ROM trajectory, operator trace, and loss convergence plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    schema = result['schema']
    prefix = f"04d_{MODEL_PARAMS['PRIOR']}_{MODEL_PARAMS['INIT']}_{schema['name']}"
    samples = result['samples']
    losses = result['losses']
    rom_solves = result['rom_solves']
    snaps_comp = result['snaps_comp']
    true_comp = result['true_comp']
    t_full = result['t_full']
    t_pred = result['t_pred']
    t_samp = result['t_samp']
    training_span = result['training_span']
    num_modes = result['num_modes']

    # ── 1. ROM Trajectory Plot (3-column) ────────────────────────────
    if len(rom_solves) > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        rom_q05 = np.percentile(rom_arr, 5, axis=0)
        rom_q95 = np.percentile(rom_arr, 95, axis=0)

        true_interp = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        true_at_pred = true_interp(t_pred)

        train_end = training_span[1]
        train_mask = t_pred <= train_end
        pred_mask = t_pred > train_end
        t_train_win = t_pred[train_mask]
        t_pred_win = t_pred[pred_mask]

        fig, ax = plt.subplots(num_modes, 3, figsize=(15, 2.5 * num_modes),
                               sharey='row', sharex='col')
        if num_modes == 1:
            ax = ax.reshape(1, -1)

        for i in range(num_modes):
            # Training window
            ax[i, 0].plot(t_samp, snaps_comp[i], 'k*', ms=3, label='Obs')
            ax[i, 0].plot(t_train_win, true_at_pred[i, train_mask],
                         color='tab:gray', lw=1.5, label='Truth')
            ax[i, 0].plot(t_train_win, rom_med[i, train_mask],
                         color='tab:purple', ls='--', lw=2, alpha=0.9, label='Median')
            ax[i, 0].fill_between(t_train_win,
                                  rom_q05[i, train_mask], rom_q95[i, train_mask],
                                  color='tab:purple', alpha=0.15, label='90% CI')
            ax[i, 0].set_ylabel(f'Mode {i}')

            # Prediction window
            ax[i, 1].plot(t_pred_win, true_at_pred[i, pred_mask], color='tab:gray', lw=1.5)
            ax[i, 1].plot(t_pred_win, rom_med[i, pred_mask],
                         color='tab:purple', ls='--', lw=2, alpha=0.9)
            ax[i, 1].fill_between(t_pred_win,
                                  rom_q05[i, pred_mask], rom_q95[i, pred_mask],
                                  color='tab:purple', alpha=0.15)

            # Full span
            ax[i, 2].plot(t_samp, snaps_comp[i], 'k*', ms=3)
            ax[i, 2].plot(t_pred, true_at_pred[i], color='tab:gray', lw=1.5)
            ax[i, 2].plot(t_pred, rom_med[i], color='tab:purple', ls='--', lw=2, alpha=0.9)
            ax[i, 2].fill_between(t_pred, rom_q05[i], rom_q95[i],
                                  color='tab:purple', alpha=0.15)
            ax[i, 2].axvline(train_end, color='k', ls=':', lw=0.8, alpha=0.5)

            # y-limits from truth
            yvals = true_at_pred[i]
            ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
            pad = max(abs(ymax - ymin) * 0.3, 1e-6)
            for j in range(3):
                ax[i, j].set_ylim(ymin - pad, ymax + pad)

        ax[0, 0].set_title('Training Window')
        ax[0, 1].set_title('Prediction Window')
        ax[0, 2].set_title('Full Span')
        ax[0, 0].legend(fontsize=7, loc='upper right')
        for j in range(3):
            ax[-1, j].set_xlabel('Time')

        fig.suptitle(f"ROM Trajectories — {schema['label']}", fontsize=14)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_rom_trajectories.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)

    # ── 2. Notebook-style ROM Trajectory Plot ───────────────────────
    if len(rom_solves) > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        rom_q05 = np.percentile(rom_arr, 5, axis=0)
        rom_q95 = np.percentile(rom_arr, 95, axis=0)

        true_interp = interp1d(t_full, true_comp, kind='cubic', fill_value='extrapolate')
        true_at_pred = true_interp(t_pred)

        n_stable = result['n_stable']
        n_total = result['n_total']

        fig, ax = plt.subplots(num_modes, 1, figsize=(10, 2.5 * num_modes), sharex=True)
        if num_modes == 1:
            ax = [ax]
        for i in range(num_modes):
            ax[i].axvspan(training_span[0], training_span[1], color='gray', alpha=0.10, zorder=0)
            ax[i].plot(t_pred, true_at_pred[i], color='tab:gray', lw=2, label='True solution')
            ax[i].plot(t_samp, snaps_comp[i], 'k*', ms=5, label='Training data', zorder=5)
            ax[i].plot(t_pred, rom_med[i], color='tab:purple', linestyle='--', alpha=0.9, lw=2, label='ROM median')
            ax[i].fill_between(t_pred, rom_q05[i], rom_q95[i], color='tab:purple', alpha=0.15, label='ROM 5-95%')
            ax[i].axvline(training_span[1], color='k', ls=':', lw=0.8, alpha=0.5)
            ax[i].set_ylabel(f'Mode {i+1}')
            yvals = true_at_pred[i]
            ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
            pad = max(abs(ymax - ymin) * 0.3, 1e-6)
            ax[i].set_ylim(ymin - pad, ymax + pad)
            if i == 0:
                ax[i].legend(loc='upper right', fontsize=9)
        ax[-1].set_xlabel('Time')
        fig.suptitle(f'Conditional Integral — {schema["label"]}  ({n_stable}/{n_total} stable)', fontsize=14)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_rom_notebook.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)

    # ── 3. Full-Order Error Plot ─────────────────────────────────────
    basis = result.get('basis')
    true_states = result.get('true_states')
    if len(rom_solves) > 0 and basis is not None and true_states is not None:
        rom_arr = np.array(rom_solves)
        fig_foe, axes_foe = plot_full_order_error(
            rom_solves=rom_arr,
            basis=basis,
            true_states=true_states,
            time_domain_full=t_full,
            time_domain_eval=t_pred,
            training_span=training_span,
            error_type='relative',
        )
        fig_foe.suptitle(f'Full-Order Error — {schema["label"]}', fontsize=14)
        path = os.path.join(save_dir, f"{prefix}_full_order_error.png")
        fig_foe.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig_foe)

    # ── 4. Operator Trace Plot ───────────────────────────────────────
    try:
        fig_trace, _ = plot_trace(samples, param_name="O", n_random=6)
        path = os.path.join(save_dir, f"{prefix}_operator_traces.png")
        fig_trace.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig_trace)
    except Exception as e:
        print(f"  ⚠ Operator trace plot failed: {e}")

    # ── 5. Loss Convergence Plot ─────────────────────────────────────
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


# =============================================================================
# Save predictions for cross-method comparison
# =============================================================================
def save_predictions(result, save_dir=None):
    """Save predictions for cross-method comparison."""
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, "results", "comparison", result['schema']['name'])
    os.makedirs(save_dir, exist_ok=True)

    rom_solves = result['rom_solves']
    rom_arr = np.array(rom_solves) if len(rom_solves) > 0 else np.empty((0, result['num_modes'], len(result['t_pred'])))

    method_name = f"04d_{MODEL_PARAMS['PRIOR']}_{MODEL_PARAMS['INIT']}"
    path = os.path.join(save_dir, f"{method_name}.npz")
    np.savez(path,
        rom_solves=rom_arr,
        t_pred=result['t_pred'],
        train_error=result['train_error'],
        pred_error=result['pred_error'],
        stability_pct=result['stability_pct'],
        ci_coverage=result.get('ci_coverage', float('nan')),
        ci_width=result.get('ci_width', float('nan')),
        runtime=result['runtime'],
    )
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
    print("04 — Conditional GP + Dual Constraint — Compressible Euler")
    print("=" * 70)
    print(f"Regimes: {len(schemas)}")
    for s in schemas:
        print(f"  • {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  noise={s['NOISE_LEVEL']:.0%}")
    print(f"Model:  γ={MODEL_PARAMS['GAMMA']}, γ₂={MODEL_PARAMS['GAMMA2']}, "
          f"lr={MODEL_PARAMS['LEARNING_RATE']}, steps={MODEL_PARAMS['NUM_STEPS']}")

    results = []
    for schema in schemas:
        r = run_experiment(schema)
        plot_results(r)
        save_predictions(r)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"SUMMARY — Conditional GP + Dual Constraint (Euler)")
    print(f"{'='*80}")
    print(f"{'Regime':<28s} {'Samp':>4s} {'Noise':>5s} {'Stab':>5s} "
          f"{'Train':>8s} {'Pred':>8s} {'CI_cov':>7s} {'Time':>6s}")
    print(f"{'-'*28} {'-'*4} {'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
    for r in results:
        s = r['schema']
        print(f"{s['label']:<28s} {s['NUM_SAMPLES']:>4d} "
              f"{s['NOISE_LEVEL']:>4.0%} {r['stability_pct']:>4.0f}% "
              f"{r['train_error']:>7.2%} {r['pred_error']:>7.2%} "
              f"{r['ci_coverage']:>6.1%} {r['runtime']:>5.0f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    # consume leading positional flags: prior choice, init choice
    while args and args[0] in PRIOR_CHOICES + INIT_CHOICES:
        if args[0] in PRIOR_CHOICES:
            MODEL_PARAMS['PRIOR'] = args[0]
        else:
            MODEL_PARAMS['INIT'] = args[0]
        args = args[1:]
    schema_names = args if args else None
    main(schema_names)

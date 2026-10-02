"""Unified marginalised-O × weak-form Bayesian OpInf model builder.

One implementation shared by every experiment. It operates on a **list of
trajectories** (a single operator shared across initial conditions); the
single-trajectory experiments are just the ``len(trajectories) == 1`` case.

Because the chemo input α(t) enters only as fixed data in the design matrix,
the reduced dynamics stay linear in O whether or not inputs are present, so the
closed-form operator marginalisation is identical for autonomous and
input-driven ROMs.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from . import gp as _gp
from . import weakform as _wf
from . import evidence as _ev


def _block_id_from_rom(rom):
    """Per-operator-column block ids (0..n_blocks-1) from the ROM operators."""
    col_blocks = []
    for bid, op in enumerate(rom.model.operators):
        e = op.entries
        ncols = e.shape[1] if (e is not None and getattr(e, "ndim", 0) == 2) else 1
        col_blocks.extend([bid] * ncols)
    block_id = np.asarray(col_blocks, dtype=int)
    m_total = rom.model.operator_matrix.shape[1]
    assert block_id.shape[0] == m_total, \
        f"block_id {block_id.shape[0]} != m {m_total}"
    return block_id, m_total, len(rom.model.operators)


def _column_degrees(rom, num_modes, num_inputs):
    """State and input polynomial degree of every operator column, read off by
    doubling the states (inputs) in the ROM's own data-matrix assembly."""
    rng = np.random.default_rng(0)
    q = rng.uniform(0.5, 1.5, size=(num_modes, 4))
    u = None if num_inputs == 0 else rng.uniform(0.5, 1.5, size=(num_inputs, 4))

    def feats(qq, uu):
        return np.asarray(rom.model._assemble_data_matrix(
            jnp.asarray(qq), inputs=None if uu is None else jnp.asarray(uu)), dtype=float)

    base = feats(q, u)
    dq = np.log2(feats(2 * q, u) / base)
    du = np.zeros_like(dq) if u is None else np.log2(feats(q, 2 * u) / base)
    pq, pu = np.round(np.median(dq, axis=0)), np.round(np.median(du, axis=0))
    if not (np.allclose(dq, pq[None], atol=1e-3) and np.allclose(du, pu[None], atol=1e-3)):
        raise ValueError("The nondimensional operator prior needs monomial operator columns")
    return pq.astype(int), pu.astype(int)


def data_scales(trajectories):
    """Reference time T (mean training window) and state size S (RMS training
    POD coefficient) that nondimensionalise the operator prior and the slack."""
    T = float(np.mean([float(tr["t_sampled"][-1] - tr["t_sampled"][0]) for tr in trajectories]))
    Y = np.concatenate([np.asarray(tr["snapshots_comp"], dtype=float) for tr in trajectories], axis=1)
    return T, float(np.sqrt(np.mean(Y ** 2)))


def closure_slack(trajectories, cfg):
    """Closure-error variance γ² and its provenance.

    cfg.gamma2 None: γ² = gamma2_nd (S/T)², i.e. a closure error of a fixed
    fraction of the reference rate S/T at which the reduced state changes, so the
    slack is invariant to the units of time and state. A float is used as given.
    """
    if cfg.gamma2 is not None:
        return float(cfg.gamma2), dict(rule="fixed", gamma2=float(cfg.gamma2), weak_slack=cfg.weak_slack)
    T, S = data_scales(trajectories)
    if not all(np.isfinite(x) and x > 0 for x in (T, S)):
        raise ValueError(f"The nondimensional slack needs positive scales (T={T}, S={S})")
    gamma2 = float(cfg.gamma2_nd) * (S / T) ** 2
    return gamma2, dict(rule="nondimensional", gamma2=gamma2, gamma2_nd=float(cfg.gamma2_nd),
                        S=S, T=T, weak_slack=cfg.weak_slack)


def nondimensional_column_scale(rom, trajectories, num_modes):
    """Operator prior scale s_j = S^(1-p_j) U^(-q_j) / T of every column.

    With time measured in units of the training window T, the reduced state in
    units of its RMS size S and the input in units of its RMS size U, every
    operator entry is O(1); s_j converts that unit scale back to the data's
    units (p_j, q_j = state/input degree of column j). Returns (s, info).
    """
    T, S = data_scales(trajectories)
    inputs = [tr.get("inputs_eval") for tr in trajectories]
    if all(u is None for u in inputs):
        p, U = 0, 1.0
    elif any(u is None for u in inputs):
        raise ValueError("Either every trajectory or none must supply inputs_eval")
    else:
        stacked = np.concatenate([np.atleast_2d(np.asarray(u, dtype=float)) for u in inputs], axis=1)
        p, U = stacked.shape[0], float(np.sqrt(np.mean(stacked ** 2)))
    if not all(np.isfinite(x) and x > 0 for x in (T, S, U)):
        raise ValueError(f"The nondimensional operator prior needs positive scales (T={T}, S={S}, U={U})")
    pq, pu = _column_degrees(rom, num_modes, p)
    scale = S ** (1.0 - pq) * U ** (-pu.astype(float)) / T
    info = dict(rule="nondimensional", T=T, S=S, U=U,
                state_degree=pq.tolist(), input_degree=pu.tolist())
    return scale, info


def build_model(rom, trajectories, cfg):
    """Build the marginalised-O + weak-form NumPyro model.

    Parameters
    ----------
    rom : opinf.ROM
        Provides the operator structure and JAX-traceable data-matrix assembly.
    trajectories : list of dict
        Each dict has keys:
            't_sampled'      (n_i,)          training times
            'snapshots_comp' (num_modes, n_i) noisy POD coefficients
            'inputs_eval'    (p, num_eval) or None   input α(t) on the eval grid
            'input_table'    optional dict(times, values), scalar-input table
            'noise_variances' optional (num_modes,) measurement variances
        All trajectories must share ``num_eval_points`` (derived from cfg).
        Input trends and measurement-noise priors require their corresponding
        trajectory fields when selected in cfg.
    cfg : WeakFormConfig

    Returns
    -------
    model : numpyro model over GP hyperparameters θ only.
    posterior_O_fn : jitted closure (theta_stacked, gamma2, sigma_O, tau_block)
        → (μ_O, C_O) with C_O C_Oᵀ = Σ_O, stacked over modes. Under the
        nondimensional prior (cfg.sigma_O None) ``tau_block`` holds the block
        multipliers κ_b (latent ``log_tau_block`` = log κ_b) and ``sigma_O`` is
        unused; ``tau_block=None`` means the prior centre. ``gamma2=None`` (as
        for the model) means the resolved slack ``prior_info["gamma2"]``.
    time_evals : list of np.ndarray, per-trajectory eval grids.
    prior_info : dict of representative prior locations (diagnostics), the
        resolved closure slack ``gamma2`` and its provenance ``closure_slack``.
    """
    num_modes = cfg.num_modes
    num_traj = len(trajectories)
    deriv_is_diag = (cfg.deriv_cov == "diag")
    weakform_is_diag = (cfg.weakform_cov == "diag")
    evidence_fn = (_ev.per_mode_evidence_qr if cfg.operator_solver == "qr"
                   else _ev.per_mode_evidence)
    posterior_fn = (_ev.per_mode_posterior_qr if cfg.operator_solver == "qr"
                    else _ev.per_mode_posterior)

    block_id, m_total, n_blocks = _block_id_from_rom(rom)
    block_id_jnp = jnp.asarray(block_id)
    nondim = cfg.sigma_O is None
    if nondim:
        col_scale, operator_prior = nondimensional_column_scale(rom, trajectories, num_modes)
        operator_prior["block_scale"] = [float(col_scale[block_id == b][0]) for b in range(n_blocks)]
        col_scale_jnp = jnp.asarray(col_scale)

        def prior_prec_from_tau(kappa_block):
            var = kappa_block[block_id_jnp] ** 2 * col_scale_jnp ** 2
            return 1.0 / var, jnp.sum(jnp.log(var))

        log_tau_center = 0.0
    else:
        operator_prior = dict(rule="sigma_O", sigma_O=float(cfg.sigma_O))
        prior_prec_from_tau = _ev.make_prior_prec_from_tau(block_id_jnp)
        log_tau_center = jnp.log(cfg.sigma_O)
        inv_prec_vec = jnp.full(m_total, 1.0 / (cfg.sigma_O ** 2))

    gamma2_resolved, slack_info = closure_slack(trajectories, cfg)
    # The historical dimensional slack kept an absolute floor on the derivative variance.
    deriv_floor = 1e-4 if cfg.gamma2 is not None else 0.0

    # ── Per-trajectory precompute: eval grid, GP conditional, test funcs ──
    traj_ctx = []
    time_evals = []
    for tr in trajectories:
        t_samp = np.asarray(tr["t_sampled"])
        y_obs = jnp.asarray(tr["snapshots_comp"])
        num_eval = cfg.num_eval_points
        time_eval = np.linspace(float(t_samp[0]), float(t_samp[-1]), num_eval)
        time_evals.append(time_eval)

        make = _gp.trajectory_gp_conditional(tr, cfg)
        _single, _batch = make(time_eval)
        tf = _wf.build_test_functions(time_eval, cfg)
        weak_slack = {"grid": tf["quad_psi_sq"], "support": tf["int_psi"] ** 2,
                      "legacy": tf["int_psi_sq"]}[cfg.weak_slack]
        noise_kwargs = {}
        if cfg.gp_noise_prior == "measurement":
            if "noise_variances" not in tr:
                raise ValueError("Measurement-noise priors require trajectory noise_variances")
            noise_kwargs["noise_variances"] = tr["noise_variances"]
        locs = _gp.spectrum_anchored_prior_locs(
            tr["snapshots_comp"], t_samp, num_modes, cfg, **noise_kwargs)

        inputs_eval = tr.get("inputs_eval", None)
        inputs_eval = None if inputs_eval is None else jnp.asarray(inputs_eval)

        traj_ctx.append(dict(
            y_obs=y_obs, batch_gp=_batch, tf=tf, locs=locs, weak_slack=weak_slack,
            inputs_eval=inputs_eval, n_eval=num_eval))

    def _build_blocks(ctx, ells, sig2s, nus, gamma2):
        """Per-trajectory (A_D, y_D, Sigma_D, A_W, y_W, Sigma_W) + mll."""
        Xs, mu_zs, K_posts_Z, K_posts_X, mlls = ctx["batch_gp"](
            ells, sig2s, nus, ctx["y_obs"])
        f_X = rom.model._assemble_data_matrix(Xs, inputs=ctx["inputs_eval"])
        n_eval = f_X.shape[0]
        I_eval = jnp.eye(n_eval)

        # Derivative block
        if deriv_is_diag:
            deriv_var = jnp.maximum(jax.vmap(jnp.diagonal)(K_posts_Z), 0.0)
            # precision vector per mode: weight / (Σ_z,ii + γ²)
            Sigma_D = cfg.deriv_weight / (deriv_var + gamma2 + deriv_floor)  # (r, n)
        else:
            Sigma_D = (K_posts_Z + gamma2 * I_eval[None]) / (cfg.deriv_weight + 1e-30)

        # Weak-form block
        tf = ctx["tf"]
        wpsi, wpsi_dot = tf["wpsi"], tf["wpsi_dot"]
        A_weak = wpsi @ f_X
        diag_slack = gamma2 * jnp.diag(ctx["weak_slack"])
        if cfg.weakform_mode == "ibp":
            weak_obs = -(Xs @ wpsi_dot.T)
            def _sig_w(Kx):
                return (wpsi_dot @ Kx @ wpsi_dot.T + diag_slack) / (cfg.weakform_weight + 1e-30)
            Sigma_W = jax.vmap(_sig_w)(K_posts_X)
        else:
            weak_obs = mu_zs @ wpsi.T
            def _sig_w(Kz):
                return (wpsi @ Kz @ wpsi.T + diag_slack) / (cfg.weakform_weight + 1e-30)
            Sigma_W = jax.vmap(_sig_w)(K_posts_Z)
        if weakform_is_diag:
            Sigma_W = jax.vmap(lambda S: jnp.diag(jnp.diag(S)))(Sigma_W)

        return (f_X, mu_zs, Sigma_D, A_weak, weak_obs, Sigma_W), Xs, jnp.sum(mlls)

    def _sample_hypers():
        """Sample per-trajectory, per-mode GP hypers. Returns list per traj of
        (ells, sig2s, nus) stacks and the deterministic Xs bookkeeping keys."""
        theta = []
        for ic, ctx in enumerate(traj_ctx):
            locs = ctx["locs"]
            ells = jnp.stack([
                numpyro.sample(f"lengthscale_{ic}_{i}",
                               dist.LogNormal(locs["log_ell_loc"],
                                              locs["log_ell_scale"]))
                for i in range(num_modes)])
            sig2s = jnp.stack([
                numpyro.sample(f"variance_{ic}_{i}",
                               dist.LogNormal(locs["log_sig2_locs"][i],
                                              cfg.sig2_prior_scale))
                for i in range(num_modes)])
            nus = jnp.stack([
                numpyro.sample(f"noise_{ic}_{i}",
                               dist.LogNormal(locs["log_nu_locs"][i],
                                              cfg.nu_prior_scale))
                for i in range(num_modes)])
            theta.append((ells, sig2s, nus))
        return theta

    def model(gamma2=None):
        gamma2 = gamma2_resolved if gamma2 is None else gamma2
        theta = _sample_hypers()

        if cfg.op_prior_mode == "block_hier":
            log_tau = numpyro.sample(
                "log_tau_block",
                dist.Normal(log_tau_center * jnp.ones(n_blocks),
                            cfg.hier_tau_scale))
            prior_prec, log_prior_cov = prior_prec_from_tau(jnp.exp(log_tau))
        elif nondim:
            prior_prec, log_prior_cov = prior_prec_from_tau(jnp.ones(n_blocks))
        else:
            prior_prec = inv_prec_vec
            log_prior_cov = -jnp.sum(jnp.log(inv_prec_vec))

        traj_blocks = []
        mll_total = 0.0
        for ic, ctx in enumerate(traj_ctx):
            ells, sig2s, nus = theta[ic]
            blocks, Xs, mll = _build_blocks(ctx, ells, sig2s, nus, gamma2)
            f_X, mu_zs, Sigma_D, A_W, weak_obs, Sigma_W = blocks
            traj_blocks.append((f_X, mu_zs, Sigma_D, A_W, weak_obs, Sigma_W))
            mll_total = mll_total + mll
            for i in range(num_modes):
                numpyro.deterministic(f"X_{ic}_{i}", Xs[i])

        if cfg.mll_weight > 0:
            numpyro.factor("gp_mll", cfg.mll_weight * mll_total)

        total_evidence = 0.0
        for i in range(num_modes):
            log_p_i, _, _ = evidence_fn(
                traj_blocks, i, m_total, prior_prec, log_prior_cov,
                deriv_is_diag)
            total_evidence = total_evidence + log_p_i
        numpyro.factor("marg_O_evidence", total_evidence)

    @jax.jit
    def posterior_O_fn(theta_stacked, gamma2, sigma_O_val, tau_block=None):
        """Closed-form O posterior given θ. ``theta_stacked`` is
        (ells, sig2s, nus) each shaped (num_traj, num_modes)."""
        gamma2 = gamma2_resolved if gamma2 is None else gamma2
        if nondim:
            prior_prec_vec, _ = prior_prec_from_tau(
                jnp.ones(n_blocks) if tau_block is None else tau_block)
        elif tau_block is None:
            inv_sO2 = 1.0 / (sigma_O_val ** 2 + 1e-12)
            prior_prec_vec = inv_sO2 * jnp.ones(m_total)
        else:
            prior_prec_vec, _ = prior_prec_from_tau(tau_block)

        ells_all, sig2s_all, nus_all = theta_stacked
        traj_blocks = []
        for ic, ctx in enumerate(traj_ctx):
            blocks, _, _ = _build_blocks(
                ctx, ells_all[ic], sig2s_all[ic], nus_all[ic], gamma2)
            f_X, mu_zs, Sigma_D, A_W, weak_obs, Sigma_W = blocks
            traj_blocks.append((f_X, mu_zs, Sigma_D, A_W, weak_obs, Sigma_W))

        mu_all, C_all = [], []
        for i in range(num_modes):
            mi, Ci = posterior_fn(
                traj_blocks, i, m_total, prior_prec_vec, deriv_is_diag)
            mu_all.append(mi)
            C_all.append(Ci)
        return jnp.stack(mu_all), jnp.stack(C_all)

    prior_info = dict(
        ell=float(np.exp(traj_ctx[0]["locs"]["log_ell_loc"])),
        sig2=[float(np.exp(traj_ctx[0]["locs"]["log_sig2_locs"][i]))
              for i in range(num_modes)],
        nu=[float(np.exp(traj_ctx[0]["locs"]["log_nu_locs"][i]))
            for i in range(num_modes)],
        num_traj=num_traj, m_total=m_total, n_blocks=n_blocks,
        operator_prior=operator_prior, gamma2=gamma2_resolved, closure_slack=slack_info,
    )
    return model, posterior_O_fn, time_evals, prior_info

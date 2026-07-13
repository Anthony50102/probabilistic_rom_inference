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
        All trajectories must share ``num_eval_points`` (derived from cfg).
    cfg : WeakFormConfig

    Returns
    -------
    model : numpyro model over GP hyperparameters θ only.
    posterior_O_fn : jitted closure (theta_stacked, gamma2, sigma_O, tau_block)
        → (μ_O, C_O) with C_O C_Oᵀ = Σ_O, stacked over modes.
    time_evals : list of np.ndarray, per-trajectory eval grids.
    prior_info : dict of representative prior locations (diagnostics).
    """
    num_modes = cfg.num_modes
    num_traj = len(trajectories)
    deriv_is_diag = (cfg.deriv_cov == "diag")
    weakform_is_diag = (cfg.weakform_cov == "diag")

    block_id, m_total, n_blocks = _block_id_from_rom(rom)
    block_id_jnp = jnp.asarray(block_id)
    prior_prec_from_tau = _ev.make_prior_prec_from_tau(block_id_jnp)
    inv_prec_vec = jnp.full(m_total, 1.0 / (cfg.sigma_O ** 2))

    # ── Per-trajectory precompute: eval grid, GP conditional, test funcs ──
    traj_ctx = []
    time_evals = []
    for tr in trajectories:
        t_samp = np.asarray(tr["t_sampled"])
        y_obs = jnp.asarray(tr["snapshots_comp"])
        num_eval = cfg.num_eval_points
        time_eval = np.linspace(float(t_samp[0]), float(t_samp[-1]), num_eval)
        time_evals.append(time_eval)

        make = _gp.make_gp_conditional(t_samp)
        _single, _batch = make(time_eval)
        tf = _wf.build_test_functions(time_eval, cfg)
        locs = _gp.spectrum_anchored_prior_locs(
            tr["snapshots_comp"], t_samp, num_modes, cfg)

        inputs_eval = tr.get("inputs_eval", None)
        inputs_eval = None if inputs_eval is None else jnp.asarray(inputs_eval)

        traj_ctx.append(dict(
            y_obs=y_obs, batch_gp=_batch, tf=tf, locs=locs,
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
            Sigma_D = cfg.deriv_weight / (deriv_var + gamma2 + 1e-4)  # (r, n)
        else:
            Sigma_D = (K_posts_Z + gamma2 * I_eval[None]) / (cfg.deriv_weight + 1e-30)

        # Weak-form block
        tf = ctx["tf"]
        wpsi, wpsi_dot = tf["wpsi"], tf["wpsi_dot"]
        A_weak = wpsi @ f_X
        diag_slack = gamma2 * jnp.diag(tf["int_psi_sq"])
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

    def model(gamma2=cfg.gamma2):
        theta = _sample_hypers()

        if cfg.op_prior_mode == "block_hier":
            log_tau = numpyro.sample(
                "log_tau_block",
                dist.Normal(jnp.log(cfg.sigma_O) * jnp.ones(n_blocks),
                            cfg.hier_tau_scale))
            prior_prec, log_prior_cov = prior_prec_from_tau(jnp.exp(log_tau))
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
            log_p_i, _, _ = _ev.per_mode_evidence(
                traj_blocks, i, m_total, prior_prec, log_prior_cov,
                deriv_is_diag)
            total_evidence = total_evidence + log_p_i
        numpyro.factor("marg_O_evidence", total_evidence)

    @jax.jit
    def posterior_O_fn(theta_stacked, gamma2, sigma_O_val, tau_block=None):
        """Closed-form O posterior given θ. ``theta_stacked`` is
        (ells, sig2s, nus) each shaped (num_traj, num_modes)."""
        inv_sO2 = 1.0 / (sigma_O_val ** 2 + 1e-12)
        if tau_block is None:
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
            mi, Ci = _ev.per_mode_posterior(
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
    )
    return model, posterior_O_fn, time_evals, prior_info

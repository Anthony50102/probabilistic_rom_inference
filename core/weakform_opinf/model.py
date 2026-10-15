"""Unified marginalised-O × weak-form Bayesian OpInf model builder.

One implementation shared by every experiment. It operates on a **list of
trajectories** (a single operator shared across initial conditions); the
single-trajectory experiments are just the ``len(trajectories) == 1`` case.
``pooled=True`` instead gives every trajectory its own operator around a shared
population operator, with the spread between them learned (partial pooling).

Because the chemo input α(t) enters only as fixed data in the design matrix,
the reduced dynamics stay linear in O whether or not inputs are present, so the
closed-form operator marginalisation is identical for autonomous and
input-driven ROMs.

The derivative and weak-form rows are correlated functionals of one GP fit per
mode, so treating them as independent observations overcounts their
information. The default (``cfg.closure='tempered'``) enters each mode's rows
as a power likelihood with exponent α_i = min(1, df_i / (n_e + K)), df_i the
effective degrees of freedom of the mode's data-only GP smoother: no constant is
selected per experiment and the cost is one extra GP-only fit.
"""

from __future__ import annotations

import time

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


def structured_rows(row_maps, block_id, col_scale, column_info, rom, trajectories, num_modes):
    """Prior blocks and nondimensional scales of structured operator rows O_i = P_i θ_i.

    A coefficient θ_ij that sets one operator column (O_ic = w θ_ij) keeps that
    column's block and has scale s_c/|w|. One that sets a fixed combination of
    state-only columns of one block multiplies the feature g_ij = Σ_c w_c f_c(q)
    (e.g. a Galerkin-projected term with a learned rate); its scale is S/(T g_j),
    g_j the RMS of g_ij over the observed training coefficients and the modes:
    the same nondimensionalisation, with the feature's own training size in place
    of S^p. Returns (P, block (r, k), scale (r, k), info).
    """
    P = np.asarray(row_maps, dtype=float)
    m = len(block_id)
    if P.ndim != 3 or P.shape[:2] != (num_modes, m) or not 0 < P.shape[2] <= m:
        raise ValueError(f"row_maps must have shape (num_modes, {m}, k) with k <= {m}; got {P.shape}")
    k = P.shape[2]
    if any(np.linalg.matrix_rank(Pi) < k for Pi in P):
        raise ValueError("Every row map must have full column rank")
    input_degree = np.asarray(column_info["input_degree"])
    T, S = data_scales(trajectories)
    blocks, scale = np.zeros((num_modes, k), dtype=int), np.zeros((num_modes, k))
    combined = {}
    for i in range(num_modes):
        for j in range(k):
            used = np.flatnonzero(P[i, :, j])
            if len(used) == 0:
                raise ValueError(f"Structured coefficient {j} of mode {i} sets no operator column")
            if len(set(block_id[used])) > 1:
                raise ValueError(f"Structured coefficient {j} of mode {i} spans several operator blocks")
            blocks[i, j] = block_id[used[0]]
            if len(used) == 1:
                scale[i, j] = col_scale[used[0]] / abs(P[i, used[0], j])
            elif np.any(input_degree[used] != 0):
                raise ValueError("Combined structured columns must be state-only")
            else:
                combined.setdefault(j, []).append(i)
    rms = {}
    if combined:
        Y = np.concatenate([np.asarray(tr["snapshots_comp"], dtype=float) for tr in trajectories], axis=1)
        inputs = [tr.get("inputs_eval") for tr in trajectories]
        ones = None if inputs[0] is None else jnp.ones((np.atleast_2d(inputs[0]).shape[0], Y.shape[1]))
        F = np.asarray(rom.model._assemble_data_matrix(jnp.asarray(Y), inputs=ones), dtype=float)
        for j, modes in combined.items():
            if len({blocks[i, j] for i in modes}) > 1:
                raise ValueError(f"Combined structured coefficient {j} must use one block in every mode")
            rms[j] = float(np.sqrt(np.mean([np.mean((F @ P[i, :, j]) ** 2) for i in modes])))
            if not (np.isfinite(rms[j]) and rms[j] > 0):
                raise ValueError(f"Combined structured feature {j} vanishes on the training data")
            scale[modes, j] = S / (T * rms[j])
    info = dict(coefficients=k, block=blocks.tolist(), scale=scale.tolist(),
                combined_feature_rms={str(j): v for j, v in rms.items()})
    return P, blocks, scale, info


def _prior_locs(trajectory, cfg):
    """Spectrum-anchored GP-hyperparameter prior locations of one trajectory."""
    kwargs = {}
    if cfg.gp_noise_prior == "measurement":
        if "noise_variances" not in trajectory:
            raise ValueError("Measurement-noise priors require trajectory noise_variances")
        kwargs["noise_variances"] = trajectory["noise_variances"]
    return _gp.spectrum_anchored_prior_locs(
        trajectory["snapshots_comp"], np.asarray(trajectory["t_sampled"]), cfg.num_modes, cfg, **kwargs)


def _sample_gp_hypers(locs, cfg):
    """Per-trajectory, per-mode GP hyperparameters (ℓ, σ², ν) drawn from their
    priors; a list over trajectories of (ells, sig2s, nus) stacks, plus the
    input-trend sd multipliers κ when ``cfg.gp_trend_scale == 'learned'``."""
    theta = []
    for ic, loc in enumerate(locs):
        ells = jnp.stack([
            numpyro.sample(f"lengthscale_{ic}_{i}",
                           dist.LogNormal(loc["log_ell_loc"], loc["log_ell_scale"]))
            for i in range(cfg.num_modes)])
        sig2s = jnp.stack([
            numpyro.sample(f"variance_{ic}_{i}",
                           dist.LogNormal(loc["log_sig2_locs"][i], cfg.sig2_prior_scale))
            for i in range(cfg.num_modes)])
        nus = jnp.stack([
            numpyro.sample(f"noise_{ic}_{i}",
                           dist.LogNormal(loc["log_nu_locs"][i], cfg.nu_prior_scale))
            for i in range(cfg.num_modes)])
        if cfg.gp_trend_scale == "learned":
            kappas = jnp.stack([
                numpyro.sample(f"trend_scale_{ic}_{i}", dist.LogNormal(0.0, cfg.hier_tau_scale))
                for i in range(cfg.num_modes)])
            theta.append((ells, sig2s, nus, kappas))
        else:
            theta.append((ells, sig2s, nus))
    return theta


def data_only_gp_fit(trajectories, time_evals, cfg):
    """GP hyperparameters fitted to the data alone (float64): the MAP of their
    priors times the GP marginal likelihood, without the operator evidence.
    Returns a list over trajectories of (ells, sig2s, nus) arrays and the
    optimiser's convergence record (with the trend multipliers κ when learned)."""
    from jax.flatten_util import ravel_pytree
    from numpyro.infer import init_to_median
    from numpyro.infer.util import initialize_model
    from scipy.optimize import minimize

    with jax.experimental.enable_x64():
        locs = [_prior_locs(tr, cfg) for tr in trajectories]
        batches = [_gp.trajectory_gp_conditional(tr, cfg)(te)[1]
                   for tr, te in zip(trajectories, time_evals)]
        observations = [jnp.asarray(tr["snapshots_comp"]) for tr in trajectories]

        def gp_model():
            total = 0.0
            for th, batch, y in zip(_sample_gp_hypers(locs, cfg), batches, observations):
                total = total + jnp.sum(batch(*th[:3], y, *th[3:])[4])
            if cfg.mll_weight > 0:
                numpyro.factor("gp_mll", cfg.mll_weight * total)

        init = initialize_model(jax.random.PRNGKey(0), gp_model,
                                init_strategy=init_to_median(num_samples=15))
        z0, unravel = ravel_pytree(init.param_info.z)
        value_and_grad = jax.jit(jax.value_and_grad(lambda z: init.potential_fn(unravel(z))))

        def objective(x):
            value, grad = value_and_grad(jnp.asarray(x))
            return float(value), np.asarray(grad, dtype=float)

        fit = minimize(objective, np.asarray(z0, dtype=float), jac=True, method="L-BFGS-B",
                       options={"maxiter": 5000})
        # Positive hyperparameters are optimised as logarithms.
        hyp = {k: np.exp(np.asarray(v, dtype=float)) for k, v in unravel(jnp.asarray(fit.x)).items()}
    kinds = ("lengthscale", "variance", "noise") + (("trend_scale",) if cfg.gp_trend_scale == "learned" else ())
    theta = [tuple(np.array([hyp[f"{kind}_{ic}_{i}"] for i in range(cfg.num_modes)])
                   for kind in kinds)
             for ic in range(len(trajectories))]
    return theta, dict(converged=bool(fit.success), iterations=int(fit.nit))


def tempering_exponents(trajectories, time_evals, num_rows, cfg):
    """Power-likelihood exponents α_i = min(1, df_i / (n_e + K)) per trajectory
    and mode, and their provenance.

    The n_e derivative and K weak-form rows of mode i are linear functionals of
    one GP posterior mean, which carries df_i = tr K (K + νI)⁻¹ independent
    pieces of information; weighting each row by α_i makes the rows count as
    df_i observations instead of n_e + K. Uses the data-only GP fit, so the
    exponents are fixed before the joint inference."""
    started = time.time()
    theta, fit = data_only_gp_fit(trajectories, time_evals, cfg)
    df = [_gp.effective_dof(tr, cfg, *th) for tr, th in zip(trajectories, theta)]
    alpha = [np.minimum(1.0, d / n) for d, n in zip(df, num_rows)]
    info = dict(rule="tempered", gamma2=0.0, alpha=[a.tolist() for a in alpha],
                df=[d.tolist() for d in df], rows=[int(n) for n in num_rows],
                data_only_gp=dict(fit, seconds=round(time.time() - started, 2),
                                  **{k: [th[j].tolist() for th in theta]
                                     for j, k in enumerate(("ell", "sig2", "nu", "trend_scale")[:len(theta[0])])}))
    return alpha, info


def power_likelihood_correction(u, alpha):
    """log of L^α / N(r; 0, u/α) for Gaussian rows of variance u (summed over
    the trailing axis and modes): −½ Σ [(α − 1) log 2πu + log α]. Added to the
    Gaussian evidence of the tempered rows it gives the power likelihood's
    evidence exactly; the residual r cancels."""
    alpha = jnp.broadcast_to(alpha, u.shape)
    return -0.5 * jnp.sum((alpha - 1.0) * jnp.log(2 * jnp.pi * u) + jnp.log(alpha))


def build_model(rom, trajectories, cfg, row_maps=None, prior_mean=None, pooled=False):
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
    row_maps : optional (num_modes, m, k) array
        Known structure of the operator rows: row i is O_i = row_maps[i] θ_i and
        the Gaussian prior and posterior are over the k coefficients θ_i (e.g.
        structural zeros, or a projected term whose rate is learned); see
        :func:`structured_rows` for their prior scales. Needs the nondimensional
        prior. None (default): θ_i = O_i.
    prior_mean : optional (num_modes, m) array
        Centre M of the Gaussian operator prior (default zero), e.g. an
        operator fitted to other trajectories in the same basis and input
        units. The prior scales are unchanged, so the block multipliers κ_b
        set how far O may move from M. Exact for the linear-Gaussian
        evidence: the rows are fitted to y − A Mᵢ and Mᵢ is added back to the
        posterior mean. Not available with row maps.
    pooled : bool
        Partial pooling: trajectory k has its own operator O⁽ᵏ⁾ = M + δ⁽ᵏ⁾
        around a shared population operator M, with M ~ N(0, (κ_M,b s_j)²)
        and δ⁽ᵏ⁾ ~ N(0, (κ_δ,b s_j)²) column-wise (s_j the nondimensional
        scales of all trajectories together). ``log_tau_block`` then holds
        log κ_M (first n_blocks entries) and log κ_δ (last n_blocks), so the
        spread κ_δ between trajectories is learned from them, not set by
        hand. Exact: every mode's rows are fitted in the stacked unknowns
        [M, δ⁽⁰⁾, …, δ⁽ᴷ⁻¹⁾] (trajectory k's design A [I, 0, …, I, …, 0]).
        Needs the nondimensional block-hierarchical prior; not available with
        row maps or a prior mean.

    Returns
    -------
    model : numpyro model over GP hyperparameters θ only.
    posterior_O_fn : jitted closure (theta_stacked, gamma2, sigma_O, tau_block)
        → (μ_O, C_O) with C_O C_Oᵀ = Σ_O, stacked over modes. Under the
        nondimensional prior (cfg.sigma_O None) ``tau_block`` holds the block
        multipliers κ_b (latent ``log_tau_block`` = log κ_b) and ``sigma_O`` is
        unused; ``tau_block=None`` means the prior centre. ``gamma2=None`` (as
        for the model) means the resolved slack ``prior_info["gamma2"]``.
        With row maps, μ_O = P_i μ_θ and C_O = [P_i C_θ, 0] (zero-padded to m
        columns), so draws μ_O + C_O z keep the structure exactly.
        Pooled: the operator of trajectory 0 (the target), or with the static
        keyword ``which`` the population operator M (``"mean"``) or trajectory
        k's (an int); C_O is a square (m × m) factor of its covariance.
    time_evals : list of np.ndarray, per-trajectory eval grids.
    prior_info : dict of representative prior locations (diagnostics), the
        closure variance ``gamma2`` (0 under the tempered rule) and the rule's
        record ``closure`` (tempering exponents, effective degrees of freedom
        and data-only GP fit, or the slack's provenance).
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

    structured = row_maps is not None
    if structured:
        if not nondim:
            raise ValueError("Structured operator rows need the nondimensional operator prior (sigma_O=None)")
        P_np, theta_block, theta_scale, operator_prior["structured_rows"] = structured_rows(
            row_maps, block_id, col_scale, operator_prior, rom, trajectories, num_modes)
        P_rows, k_rows = jnp.asarray(P_np), P_np.shape[2]
        theta_block_jnp, theta_scale_jnp = jnp.asarray(theta_block), jnp.asarray(theta_scale)

    centred = prior_mean is not None
    if centred:
        if structured:
            raise ValueError("A prior mean is not supported with structured operator rows")
        mean_np = np.asarray(prior_mean, dtype=float)
        if mean_np.shape != (num_modes, m_total) or not np.isfinite(mean_np).all():
            raise ValueError(f"prior_mean must be a finite ({num_modes}, {m_total}) array")
        mean_O = jnp.asarray(mean_np)
        operator_prior["mean"] = dict(rule="given", max_abs=float(np.abs(mean_np).max()),
                                      row_norms=np.linalg.norm(mean_np, axis=1).tolist())

    def _centre(blocks):
        """Rows y ≈ A Oᵢ refitted as y − A Mᵢ ≈ A (Oᵢ − Mᵢ), so the zero-mean evidence applies."""
        if not centred:
            return blocks
        A_D, y_D, S_D, A_W, y_W, S_W = blocks
        return A_D, y_D - mean_O @ A_D.T, S_D, A_W, y_W - mean_O @ A_W.T, S_W

    pooled = bool(pooled)
    m_fit = m_total
    if pooled:
        if not (nondim and cfg.op_prior_mode == "block_hier"):
            raise ValueError("Partial pooling needs the nondimensional block-hierarchical operator prior "
                             "(sigma_O=None, op_prior_mode='block_hier')")
        if structured or centred:
            raise ValueError("Partial pooling is not supported with structured operator rows or a prior mean")
        m_fit = (num_traj + 1) * m_total
        operator_prior["pooled"] = dict(
            rule="partial pooling", trajectories=num_traj, target_trajectory=0,
            operators="O_k = M + delta_k; M ~ N(0, (kappa_M,b s_j)^2), delta_k ~ N(0, (kappa_delta,b s_j)^2)",
            log_tau_block=f"entries 0-{n_blocks - 1}: log kappa_M; {n_blocks}-{2 * n_blocks - 1}: log kappa_delta")

    def _pooled_blocks(traj_blocks):
        """Trajectory k's rows in the stacked unknowns [M, δ⁽⁰⁾, …]: its design A placed as A [I, 0, …, I, …, 0]."""
        placed = []
        for k, (A_D, y_D, S_D, A_W, y_W, S_W) in enumerate(traj_blocks):
            def place(A):
                zeros = jnp.zeros_like(A)
                return jnp.concatenate([A] + [A if j == k else zeros for j in range(num_traj)], axis=1)
            placed.append((place(A_D), y_D, S_D, place(A_W), y_W, S_W))
        return placed

    def _pooled_prior(kappa):
        """Prior precision and log|prior cov| of [M, δ⁽⁰⁾, …, δ⁽ᴷ⁻¹⁾] from (κ_M, κ_δ) per block."""
        prec_M, log_cov_M = prior_prec_from_tau(kappa[:n_blocks])
        prec_d, log_cov_d = prior_prec_from_tau(kappa[n_blocks:])
        return jnp.concatenate([prec_M] + [prec_d] * num_traj), log_cov_M + num_traj * log_cov_d

    def _pooled_selector(which):
        """Map from the stacked unknowns to M (``"mean"``) or to trajectory k's operator M + δ⁽ᵏ⁾."""
        S = np.zeros((m_total, m_fit))
        S[:, :m_total] = np.eye(m_total)
        if which != "mean":
            if isinstance(which, bool) or not isinstance(which, (int, np.integer)) or not 0 <= which < num_traj:
                raise ValueError(f"which must be 'mean' or a trajectory index below {num_traj}; got {which!r}")
            S[:, (which + 1) * m_total:(which + 2) * m_total] = np.eye(m_total)
        return jnp.asarray(S)

    def _structured_mode(traj_blocks, i, kappa_block):
        """Mode i's design blocks in its coefficients θ_i, their prior precision and log|prior cov|."""
        blocks = [(A_D @ P_rows[i], y_D, S_D, A_W @ P_rows[i], y_W, S_W)
                  for A_D, y_D, S_D, A_W, y_W, S_W in traj_blocks]
        var = kappa_block[theta_block_jnp[i]] ** 2 * theta_scale_jnp[i] ** 2
        return blocks, 1.0 / var, jnp.sum(jnp.log(var))

    tempered = cfg.closure == "tempered"
    if tempered:
        gamma2_resolved, closure_info = 0.0, None
    else:
        gamma2_resolved, closure_info = closure_slack(trajectories, cfg)
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
        locs = _prior_locs(tr, cfg)

        inputs_eval = tr.get("inputs_eval", None)
        inputs_eval = None if inputs_eval is None else jnp.asarray(inputs_eval)

        traj_ctx.append(dict(
            y_obs=y_obs, batch_gp=_batch, tf=tf, locs=locs, weak_slack=weak_slack,
            inputs_eval=inputs_eval, n_eval=num_eval, n_train=len(t_samp)))

    if tempered:
        alphas, closure_info = tempering_exponents(
            trajectories, time_evals, [ctx["n_eval"] + ctx["tf"]["wpsi"].shape[0] for ctx in traj_ctx], cfg)
        for ctx, alpha in zip(traj_ctx, alphas):
            ctx["alpha"] = jnp.asarray(alpha)[:, None]

    def _tempered_blocks(ctx, Xs, f_X, mu_zs, K_posts_Z, K_posts_X, sig2s, ells):
        """Diagonal derivative and IBP weak rows with GP variances v, each a
        power likelihood N(r; 0, v)^α: Gaussian rows of variance v/α plus the
        normaliser correction. v carries a round-off floor (n ε σ²/ℓ² for
        derivatives, n ε σ² ‖w ψ̇_k‖₁² for weak rows; ε of the working dtype)."""
        eps = jnp.finfo(Xs.dtype).eps
        alpha = ctx["alpha"]
        wpsi, wpsi_dot = ctx["tf"]["wpsi"], ctx["tf"]["wpsi_dot"]
        var_D = (jnp.maximum(jax.vmap(jnp.diagonal)(K_posts_Z), 0.0)
                 + (ctx["n_train"] * eps * sig2s / ells ** 2)[:, None])
        var_W = (jnp.maximum(jnp.einsum("kj,ijl,kl->ik", wpsi_dot, K_posts_X, wpsi_dot), 0.0)
                 + ctx["n_train"] * eps * sig2s[:, None] * jnp.sum(jnp.abs(wpsi_dot), axis=1)[None] ** 2)
        weak_weight = cfg.weakform_weight + 1e-30
        Sigma_D = cfg.deriv_weight * alpha / var_D
        Sigma_W = jax.vmap(jnp.diag)(var_W / (alpha * weak_weight))
        # The evidence whitens the weak block with an added 1e-8 I (evidence._whiten_block).
        correction = (power_likelihood_correction(var_D / cfg.deriv_weight, alpha)
                      + power_likelihood_correction(var_W / weak_weight + 1e-8 * alpha, alpha))
        blocks = (f_X, mu_zs, Sigma_D, wpsi @ f_X, -(Xs @ wpsi_dot.T), Sigma_W)
        return blocks, correction

    def _build_blocks(ctx, ells, sig2s, nus, gamma2, trend_scales=None):
        """Per-trajectory (A_D, y_D, Sigma_D, A_W, y_W, Sigma_W), states, mll
        and the power-likelihood correction (0 for the slack rule)."""
        extra = () if trend_scales is None else (trend_scales,)
        Xs, mu_zs, K_posts_Z, K_posts_X, mlls = ctx["batch_gp"](
            ells, sig2s, nus, ctx["y_obs"], *extra)
        f_X = rom.model._assemble_data_matrix(Xs, inputs=ctx["inputs_eval"])
        if tempered:
            blocks, correction = _tempered_blocks(ctx, Xs, f_X, mu_zs, K_posts_Z, K_posts_X, sig2s, ells)
            return _centre(blocks), Xs, jnp.sum(mlls), correction
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

        return _centre((f_X, mu_zs, Sigma_D, A_weak, weak_obs, Sigma_W)), Xs, jnp.sum(mlls), 0.0

    def _sample_hypers():
        """Sample per-trajectory, per-mode GP hypers (list per traj of (ells, sig2s, nus))."""
        return _sample_gp_hypers([ctx["locs"] for ctx in traj_ctx], cfg)

    def model(gamma2=None):
        gamma2 = gamma2_resolved if gamma2 is None else gamma2
        theta = _sample_hypers()

        if cfg.op_prior_mode == "block_hier":
            log_tau = numpyro.sample(
                "log_tau_block",
                dist.Normal(log_tau_center * jnp.ones(2 * n_blocks if pooled else n_blocks),
                            cfg.hier_tau_scale))
            kappa_block = jnp.exp(log_tau)
            if pooled:
                prior_prec, log_prior_cov = _pooled_prior(kappa_block)
            else:
                prior_prec, log_prior_cov = prior_prec_from_tau(kappa_block)
        elif nondim:
            kappa_block = jnp.ones(n_blocks)
            prior_prec, log_prior_cov = prior_prec_from_tau(kappa_block)
        else:
            prior_prec = inv_prec_vec
            log_prior_cov = -jnp.sum(jnp.log(inv_prec_vec))

        traj_blocks = []
        mll_total = 0.0
        correction = 0.0
        for ic, ctx in enumerate(traj_ctx):
            ells, sig2s, nus, *trend = theta[ic]
            blocks, Xs, mll, corr = _build_blocks(ctx, ells, sig2s, nus, gamma2, *trend)
            traj_blocks.append(blocks)
            mll_total = mll_total + mll
            correction = correction + corr
            for i in range(num_modes):
                numpyro.deterministic(f"X_{ic}_{i}", Xs[i])

        if cfg.mll_weight > 0:
            numpyro.factor("gp_mll", cfg.mll_weight * mll_total)

        if pooled:
            traj_blocks = _pooled_blocks(traj_blocks)
        total_evidence = 0.0
        for i in range(num_modes):
            if structured:
                blocks_i, prec_i, log_cov_i = _structured_mode(traj_blocks, i, kappa_block)
                log_p_i, _, _ = evidence_fn(
                    blocks_i, i, k_rows, prec_i, log_cov_i, deriv_is_diag)
            else:
                log_p_i, _, _ = evidence_fn(
                    traj_blocks, i, m_fit, prior_prec, log_prior_cov,
                    deriv_is_diag)
            total_evidence = total_evidence + log_p_i
        numpyro.factor("marg_O_evidence", total_evidence + correction)

    @jax.jit
    def posterior_O_fn(theta_stacked, gamma2, sigma_O_val, tau_block=None):
        """Closed-form O posterior given θ. ``theta_stacked`` is
        (ells, sig2s, nus[, trend_scales]) each shaped (num_traj, num_modes)."""
        gamma2 = gamma2_resolved if gamma2 is None else gamma2
        if nondim:
            kappa_block = jnp.ones(n_blocks) if tau_block is None else tau_block
            prior_prec_vec, _ = prior_prec_from_tau(kappa_block)
        elif tau_block is None:
            inv_sO2 = 1.0 / (sigma_O_val ** 2 + 1e-12)
            prior_prec_vec = inv_sO2 * jnp.ones(m_total)
        else:
            prior_prec_vec, _ = prior_prec_from_tau(tau_block)

        ells_all, sig2s_all, nus_all, *trend_all = theta_stacked
        traj_blocks = []
        for ic, ctx in enumerate(traj_ctx):
            blocks, _, _, _ = _build_blocks(
                ctx, ells_all[ic], sig2s_all[ic], nus_all[ic], gamma2, *(t[ic] for t in trend_all))
            traj_blocks.append(blocks)

        mu_all, C_all = [], []
        for i in range(num_modes):
            if structured:
                blocks_i, prec_i, _ = _structured_mode(traj_blocks, i, kappa_block)
                theta_i, root_i = posterior_fn(
                    blocks_i, i, k_rows, prec_i, deriv_is_diag)
                mi = P_rows[i] @ theta_i
                Ci = jnp.pad(P_rows[i] @ root_i, ((0, 0), (0, m_total - k_rows)))
            else:
                mi, Ci = posterior_fn(
                    traj_blocks, i, m_total, prior_prec_vec, deriv_is_diag)
                if centred:
                    mi = mi + mean_O[i]
            mu_all.append(mi)
            C_all.append(Ci)
        return jnp.stack(mu_all), jnp.stack(C_all)

    def pooled_posterior_O_fn(theta_stacked, gamma2, sigma_O_val, tau_block=None, which=0):
        """Closed-form posterior of trajectory ``which``'s operator (0: the target) or of M (``"mean"``) given θ
        and (κ_M, κ_δ); the stacked unknowns' factor F (F Fᵀ = Σ) is reduced to a square root of S Σ Sᵀ by QR."""
        gamma2 = gamma2_resolved if gamma2 is None else gamma2
        kappa = jnp.ones(2 * n_blocks) if tau_block is None else tau_block
        prior_prec_vec, _ = _pooled_prior(kappa)
        ells_all, sig2s_all, nus_all, *trend_all = theta_stacked
        traj_blocks = []
        for ic, ctx in enumerate(traj_ctx):
            blocks, _, _, _ = _build_blocks(
                ctx, ells_all[ic], sig2s_all[ic], nus_all[ic], gamma2, *(t[ic] for t in trend_all))
            traj_blocks.append(blocks)
        traj_blocks = _pooled_blocks(traj_blocks)
        S = _pooled_selector(which)
        mu_all, C_all = [], []
        for i in range(num_modes):
            mu_i, factor_i = posterior_fn(traj_blocks, i, m_fit, prior_prec_vec, deriv_is_diag)
            _, R_i = jnp.linalg.qr((S @ factor_i).T, mode="reduced")
            mu_all.append(S @ mu_i)
            C_all.append(R_i.T)
        return jnp.stack(mu_all), jnp.stack(C_all)

    if pooled:
        posterior_O_fn = jax.jit(pooled_posterior_O_fn, static_argnames=("which",))

    prior_info = dict(
        ell=float(np.exp(traj_ctx[0]["locs"]["log_ell_loc"])),
        sig2=[float(np.exp(traj_ctx[0]["locs"]["log_sig2_locs"][i]))
              for i in range(num_modes)],
        nu=[float(np.exp(traj_ctx[0]["locs"]["log_nu_locs"][i]))
            for i in range(num_modes)],
        num_traj=num_traj, m_total=m_total, n_blocks=n_blocks,
        operator_prior=operator_prior, gamma2=gamma2_resolved, closure=closure_info,
    )
    if cfg.gp_trend_scale == "learned":
        prior_info["trend_scale"] = dict(rule="learned", sd="kappa_i * sd(y_i)",
                                         log_kappa_prior=["normal", 0.0, float(cfg.hier_tau_scale)])
    return model, posterior_O_fn, time_evals, prior_info

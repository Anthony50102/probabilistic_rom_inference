"""
04 — Multi-dose ablation for Bayesian Operator Inference (chemo).

Three variants test where the dose-extrapolation failure of single-dose training
comes from:

  V1: POD on 1.0x only, operator training on 1.0x only       (baseline)
  V2: POD on {0.8, 1.0, 1.2}x, operator training on 1.0x      (multi-dose subspace)
  V3: POD on {0.8, 1.0, 1.2}x, operator training on {0.8, 1.0, 1.2}x  (full multi-dose)

Diagnostic logic:
  V2 ≈ V1  → POD subspace not the bottleneck → operator non-identifiability is.
  V2 ≈ V3  → POD was the whole problem; bilinear theory holds once subspace covers.
  V2 mid   → both contribute.

Held constant across variants: noise regime (default dense_high_noise),
POD truncation rank (set by SNR check on V1's POD, reused for V2/V3),
SVI settings, posterior sample count, seed, gamma2 (fixed from V1's
empirical Bayes value).

Usage:
    python 04_conditional_integral_chemo_multidose.py
    python 04_conditional_integral_chemo_multidose.py --schema dense_medium_noise
    python 04_conditional_integral_chemo_multidose.py --steps 6000 --variants V1 V2 V3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Sequence

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
    Basis, TumorTwinFOM, load_chemo_fom_data,
    ChemoReducedOrderModel, make_jax_input_func,
)
from core import compute_gp_derivatives, rbf_eval
from core.bayesian_opinf import fit_gp_hyperparameters_mle, _find_operator_samples
import opinf

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)


# =============================================================================
# Constants
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIGURE_DIR = os.path.join(SCRIPT_DIR, 'figures')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

TRAINING_SPAN = (5.0, 70.0)
PREDICTION_DAYS = 110.0

# Filename pattern: 1.0x is the base file; other doses use _dose<scale>p<frac>
FOM_BASE = 'TNBC_demo_001_fom_chemo_sparse5_sens0p5'

SCHEMAS = {
    "dense_low_noise":    dict(NUM_SAMPLES=200, NOISE_LEVEL=0.01, NUM_EVAL_POINTS=400),
    "dense_medium_noise": dict(NUM_SAMPLES=200, NOISE_LEVEL=0.03, NUM_EVAL_POINTS=400),
    "dense_high_noise":   dict(NUM_SAMPLES=200, NOISE_LEVEL=0.05, NUM_EVAL_POINTS=400),
}

MODEL_PARAMS = dict(
    NUM_MODES=4,
    GAMMA=1.0,
    DERIV_WEIGHT=1.0,
    INTEGRAL_WEIGHT=8.0,
    MLL_WEIGHT=0.1,
    GP_PRIOR_SCALE=0.03,
    WINDOW_SIZE=20,
    NUM_STEPS=8000,
    LEARNING_RATE=3e-3,
    NUM_POSTERIOR_SAMPLES=300,
    SEED=42,
)

DEFAULT_POD_DOSES_MULTI = (0.8, 1.0, 1.2)
DEFAULT_TRAIN_DOSES_MULTI = (0.8, 1.0, 1.2)
DEFAULT_EVAL_DOSES = (0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5)


def _dose_path(scale: float) -> str:
    """Filename for a chemo FOM file at given dose scale."""
    if abs(scale - 1.0) < 1e-9:
        return os.path.join(DATA_DIR, f'{FOM_BASE}.npz')
    tag = f'dose{scale:g}'.replace('.', 'p')
    return os.path.join(DATA_DIR, f'{FOM_BASE}_{tag}.npz')


# =============================================================================
# Per-trajectory data container
# =============================================================================
@dataclass
class TrajData:
    """All per-trajectory data needed for POD, GP, LS, and SVI stages."""
    dose: float
    fom: object
    t_full: np.ndarray
    true_states: np.ndarray
    t_samp: np.ndarray
    snaps_noisy: np.ndarray   # raw (n_dof, n_samp)
    snaps_clean: np.ndarray   # raw (n_dof, n_samp)
    chemo_meta: dict
    ifn: object
    ifn_jax: object
    # Filled later
    snaps_comp: np.ndarray = None       # (r, n_samp) noisy projected
    true_comp: np.ndarray = None        # (r, n_pred) clean projected
    inputs_at_samp: np.ndarray = None   # (1, n_samp)
    inputs_at_eval: np.ndarray = None   # (1, n_eval)
    Ls: np.ndarray = None
    Vs: np.ndarray = None
    Ns: np.ndarray = None
    X_mle_at_samp: np.ndarray = None
    X_mle_at_eval: np.ndarray = None
    mu_z_eval: np.ndarray = None
    q0_clean: np.ndarray = None         # (r,) clean projected initial state


def load_trajectory(dose: float, schema: dict, t_pred: np.ndarray, seed: int) -> TrajData:
    """Load a chemo FOM at a given dose and prep noisy + clean snapshots."""
    path = _dose_path(dose)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing FOM file for dose×{dose}: {path}")
    fom, t_full, true_states, t_samp, snaps_noisy, ifn, chemo_meta = \
        load_chemo_fom_data(path, t_pred, TRAINING_SPAN,
                            schema['NUM_SAMPLES'], schema['NOISE_LEVEL'],
                            seed=seed)
    snaps_clean = fom.get_states(t_samp)
    ifn_jax = make_jax_input_func(ifn, float(t_pred[0]), float(t_pred[-1]),
                                  n_points=4001)
    return TrajData(
        dose=dose, fom=fom, t_full=t_full, true_states=true_states,
        t_samp=t_samp, snaps_noisy=snaps_noisy, snaps_clean=snaps_clean,
        chemo_meta=chemo_meta, ifn=ifn, ifn_jax=ifn_jax,
    )


# =============================================================================
# POD basis fitting (single- or multi-trajectory)
# =============================================================================
def fit_pod_basis(traj_list: list[TrajData], num_modes: int) -> Basis:
    """Fit a POD basis on concatenated CLEAN snapshots from given trajectories."""
    snaps_concat = np.hstack([td.snaps_clean for td in traj_list])
    basis = Basis(num_vectors=num_modes)
    basis.fit(snaps_concat)
    return basis


def select_rank_via_snr(traj_list: list[TrajData], max_modes: int,
                       snr_threshold: float = 10.0) -> int:
    """Fit a generous-rank POD on the given trajectory list, project the FIRST
    trajectory's noisy snapshots, fit GP MLE per mode, and truncate at the first
    SNR failure. Returns the effective rank.
    """
    basis_probe = fit_pod_basis(traj_list, max_modes)
    # Use the first trajectory's noisy snapshots for SNR (mirrors V1 logic).
    snaps_comp_probe = basis_probe.compress(traj_list[0].snaps_noisy)
    Ls_p, Vs_p, Ns_p, _ = fit_gp_hyperparameters_mle(
        traj_list[0].t_samp, snaps_comp_probe, verbose=False)

    eff = 0
    for j in range(max_modes):
        snr = Vs_p[j] / max(Ns_p[j], 1e-10)
        passed = snr > snr_threshold
        tag = "✓" if passed else "✗"
        print(f"    Probe mode {j}: σ²={Vs_p[j]:.4f}, ν={Ns_p[j]:.6f}, "
              f"SNR={snr:.1f} {tag}")
        if not passed:
            break
        eff = j + 1
    return max(eff, 2)


# =============================================================================
# Per-trajectory GP MLE and LS prior
# =============================================================================
def fit_per_traj_gp(traj: TrajData, basis: Basis, num_modes: int):
    """Project noisy + true snaps, sample input function on t_samp / t_eval,
    fit GP MLE per mode, and store everything on the trajectory.
    """
    traj.snaps_comp = basis.compress(traj.snaps_noisy)
    traj.true_comp = basis.compress(traj.true_states)
    # Clean projected initial state for ROM evaluation (avoids noisy q0 bias).
    q0_full = traj.fom.get_states(np.array([traj.t_full[0]]))
    traj.q0_clean = basis.compress(q0_full)[:, 0]

    Ls, Vs, Ns, _ = fit_gp_hyperparameters_mle(
        traj.t_samp, traj.snaps_comp, verbose=False)
    traj.Ls = np.asarray(Ls)
    traj.Vs = np.asarray(Vs)
    traj.Ns = np.asarray(Ns)


def compute_gp_means_and_derivs(traj: TrajData, t_eval_ls: np.ndarray):
    """Compute GP posterior means at t_samp and t_eval, and GP-derivative
    means at t_eval (closed-form, given MLE hypers)."""
    n_modes = traj.snaps_comp.shape[0]
    X_mle_at_samp = np.zeros((n_modes, len(traj.t_samp)))
    X_mle_at_eval = np.zeros((n_modes, len(t_eval_ls)))
    for i in range(n_modes):
        K = rbf_eval(traj.Ls[i], traj.Vs[i], traj.t_samp, traj.t_samp) \
            + (traj.Ns[i] + 1e-5) * np.eye(len(traj.t_samp))
        K_inv_y = np.linalg.solve(K, traj.snaps_comp[i])
        Ks_eval = rbf_eval(traj.Ls[i], traj.Vs[i], t_eval_ls, traj.t_samp)
        Ks_samp = rbf_eval(traj.Ls[i], traj.Vs[i], traj.t_samp, traj.t_samp)
        X_mle_at_eval[i] = Ks_eval @ K_inv_y
        X_mle_at_samp[i] = Ks_samp @ K_inv_y

    mu_z, _ = compute_gp_derivatives(
        traj.Ls, traj.Vs, traj.t_samp, t_eval_ls, traj.snaps_comp, Ns=traj.Ns)
    traj.X_mle_at_samp = X_mle_at_samp
    traj.X_mle_at_eval = X_mle_at_eval
    traj.mu_z_eval = np.asarray(mu_z)


def compute_inputs(traj: TrajData, t_samp: np.ndarray, t_eval: np.ndarray):
    """Tabulate α(t) at sample and eval times."""
    traj.inputs_at_samp = np.array(
        [float(np.asarray(traj.ifn_jax(t))[0]) for t in t_samp]
    ).reshape(1, -1)
    traj.inputs_at_eval = np.array(
        [float(np.asarray(traj.ifn_jax(t))[0]) for t in t_eval]
    ).reshape(1, -1)


def build_ls_prior(traj_list: list[TrajData], rom: opinf.ROM,
                   num_modes: int, block_reg_diag: np.ndarray):
    """Build LS prior by stacking per-trajectory D matrices and μ_z.

    Returns
    -------
    O_ls : (num_modes, n_cols)
    O_prior_scale : (num_modes, n_cols)  per-entry analytical posterior std
    """
    Ds = []
    mus = []
    for td in traj_list:
        D_j = np.array(rom.model._assemble_data_matrix(
            jnp.array(td.X_mle_at_eval), inputs=jnp.array(td.inputs_at_eval)))
        Ds.append(D_j)
        mus.append(td.mu_z_eval)  # (num_modes, n_eval)
    D_stack = np.vstack(Ds)                   # (n_traj * n_eval, n_cols)
    mu_stack = np.hstack(mus)                 # (num_modes, n_traj * n_eval)

    Reg2 = np.diag(block_reg_diag ** 2)
    DtD = D_stack.T @ D_stack
    DtD_reg_inv = np.linalg.inv(DtD + Reg2)
    O_ls = (DtD_reg_inv @ D_stack.T @ mu_stack.T).T

    # Per-row residual variance for prior scale: σ_i² ≈ noise variance of mode i
    # averaged across trajectories (or just take Ns[i] from the headline traj).
    Ns_avg = np.mean(np.stack([td.Ns for td in traj_list]), axis=0)
    sigma_per_row = np.sqrt(np.maximum(Ns_avg, 1e-8))
    col_se = np.sqrt(np.maximum(np.diag(DtD_reg_inv), 0.0))
    O_prior_scale = sigma_per_row[:, None] * col_se[None, :]
    return O_ls, O_prior_scale, D_stack, mu_stack


def gamma2_from_residuals(D: np.ndarray, mu_stack: np.ndarray, O_ls: np.ndarray,
                          floor: float = 1e-4) -> float:
    """Empirical-Bayes γ₂ from LS residual variance."""
    pred = D @ O_ls.T              # (n_total, num_modes)
    obs = mu_stack.T               # (n_total, num_modes)
    LS_resid = pred - obs
    n_resid_dof = max(LS_resid.size - O_ls.size, 1)
    gamma2_data = float(np.sum(LS_resid ** 2) / n_resid_dof)
    return max(gamma2_data, floor)


# =============================================================================
# Multi-trajectory NumPyro model
# =============================================================================
def build_multitraj_model(
    rom, num_modes: int, traj_list: list[TrajData],
    O_prior: np.ndarray, O_prior_scale: np.ndarray,
    num_eval_points: int, window_size: int,
    deriv_weight: float, integral_weight: float,
    mll_weight: float, gp_prior_scale: float,
):
    """Build NumPyro model summing GP+derivative+integral factors over trajectories.

    All factors are normalized by 1/n_traj so the prior-likelihood balance is
    comparable to the single-trajectory model.
    """
    n_traj = len(traj_list)
    # NOTE: previously norm = 1.0 / n_traj for "ablation fairness" (averaging
    # log-likelihoods across trajectories). That hobbled V3 by giving each
    # trajectory only 1/n_traj gradient signal. Use full sum (proper Bayesian
    # multi-trajectory likelihood) so V3 actually leverages its extra data.
    norm = 1.0

    # Each trajectory's t_samp and t_eval
    t_samp_jax = [jnp.array(td.t_samp) for td in traj_list]
    t_eval_np = [
        np.linspace(float(td.t_samp[0]), float(td.t_samp[-1]), num_eval_points)
        for td in traj_list
    ]
    t_eval_jax = [jnp.array(te) for te in t_eval_np]
    y_obs_list = [jnp.array(td.snaps_comp) for td in traj_list]
    inputs_eval_list = [jnp.asarray(td.inputs_at_eval) for td in traj_list]

    # Precompute kernel matrices per trajectory
    sq_diff_tt_list = [(t[:, None] - t[None, :]) ** 2 for t in t_samp_jax]
    sq_diffs_et_list = [
        (te[:, None] - ts[None, :]) ** 2
        for te, ts in zip(t_eval_jax, t_samp_jax)
    ]
    diffs_et_list = [
        te[:, None] - ts[None, :]
        for te, ts in zip(t_eval_jax, t_samp_jax)
    ]
    sq_diffs_ee_list = [(te[:, None] - te[None, :]) ** 2 for te in t_eval_jax]
    n_train_list = [len(t) for t in t_samp_jax]
    I_train_list = [jnp.eye(n) for n in n_train_list]

    # Integration windows per trajectory
    n_windows = num_eval_points // window_size
    ws_arr = np.array([i * window_size for i in range(n_windows)])
    we_arr = np.array([(i + 1) * window_size - 1 for i in range(n_windows)])
    if we_arr[-1] < num_eval_points - 1:
        we_arr[-1] = num_eval_points - 1

    trap_weights_list = []
    window_durations_list = []
    for te in t_eval_np:
        dt_eval = float(te[1] - te[0])
        traj_traps = []
        traj_durs = []
        for ws, we in zip(ws_arr, we_arr):
            n_pts = we - ws + 1
            w = jnp.ones(n_pts) * dt_eval
            w = w.at[0].set(0.5 * dt_eval)
            w = w.at[-1].set(0.5 * dt_eval)
            traj_traps.append(w)
            traj_durs.append(float(te[we] - te[ws]))
        trap_weights_list.append(traj_traps)
        window_durations_list.append(traj_durs)

    # MLE log-hypers per trajectory
    mle_log_ells = [jnp.array([jnp.log(l) for l in td.Ls]) for td in traj_list]
    mle_log_sig2s = [jnp.array([jnp.log(v) for v in td.Vs]) for td in traj_list]
    mle_log_nus = [jnp.array([jnp.log(max(n, 1e-6)) for n in td.Ns]) for td in traj_list]

    O_prior_jnp = jnp.array(O_prior)
    O_prior_scale_jnp = jnp.array(O_prior_scale)

    def _rbf_sq(ell, sig2, sq_diffs):
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def _gp_conditional(ell, sig2, nu, y_i, sq_diff_tt, sq_diffs_et,
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

    def model(gamma=0.05, gamma2=0.035, jitter=1e-4):
        # Per-trajectory hypers — separate samples per (mode, traj)
        ells_per_traj = []
        sig2s_per_traj = []
        nus_per_traj = []
        for j in range(n_traj):
            ells = jnp.stack([
                numpyro.sample(f"traj{j}_lengthscale_{i}",
                               dist.LogNormal(mle_log_ells[j][i], gp_prior_scale))
                for i in range(num_modes)])
            sig2s = jnp.stack([
                numpyro.sample(f"traj{j}_variance_{i}",
                               dist.LogNormal(mle_log_sig2s[j][i], gp_prior_scale))
                for i in range(num_modes)])
            nus = jnp.stack([
                numpyro.sample(f"traj{j}_noise_{i}",
                               dist.LogNormal(mle_log_nus[j][i], gp_prior_scale))
                for i in range(num_modes)])
            ells_per_traj.append(ells)
            sig2s_per_traj.append(sig2s)
            nus_per_traj.append(nus)

        # Shared operator
        prior_scale = gamma * O_prior_scale_jnp
        O = numpyro.sample("O", dist.Normal(O_prior_jnp, prior_scale))

        # Per-trajectory GP conditional + dynamics + constraints
        for j in range(n_traj):
            Xs_eval = []
            mu_zs = []
            deriv_vars = []
            mlls = []
            for i in range(num_modes):
                Xe, mz, dv, ml = _gp_conditional(
                    ells_per_traj[j][i], sig2s_per_traj[j][i], nus_per_traj[j][i],
                    y_obs_list[j][i],
                    sq_diff_tt_list[j], sq_diffs_et_list[j],
                    diffs_et_list[j], sq_diffs_ee_list[j],
                    I_train_list[j], n_train_list[j])
                Xs_eval.append(Xe)
                mu_zs.append(mz)
                deriv_vars.append(dv)
                mlls.append(ml)
            Xs_eval = jnp.stack(Xs_eval)
            mu_zs = jnp.stack(mu_zs)
            deriv_vars = jnp.stack(deriv_vars)
            mlls_sum = jnp.sum(jnp.stack(mlls))

            for i in range(num_modes):
                numpyro.deterministic(f"traj{j}_X_{i}", Xs_eval[i])

            # GP MLL
            if mll_weight > 0:
                numpyro.factor(f"traj{j}_gp_mll",
                               norm * mll_weight * mlls_sum)

            # Dynamics
            f_Xi = rom.model._assemble_data_matrix(
                Xs_eval, inputs=inputs_eval_list[j]) @ O.T

            # Derivative constraint
            if deriv_weight > 0:
                for i in range(num_modes):
                    total_var = deriv_vars[i] + gamma2 + jitter
                    numpyro.factor(
                        f"traj{j}_ode_constraint_{i}",
                        norm * deriv_weight * jnp.sum(
                            dist.Normal(f_Xi[:, i], jnp.sqrt(total_var))
                                .log_prob(mu_zs[i])))

            # Integral constraint
            if integral_weight > 0:
                for i in range(num_modes):
                    for w_idx, (ws, we) in enumerate(zip(ws_arr, we_arr)):
                        delta_X_obs = Xs_eval[i, we] - Xs_eval[i, ws]
                        delta_X_pred = jnp.sum(
                            trap_weights_list[j][w_idx] * f_Xi[ws:we+1, i])
                        constraint_std = (jnp.sqrt(gamma2)
                                          * window_durations_list[j][w_idx])
                        numpyro.factor(
                            f"traj{j}_integral_{i}_{w_idx}",
                            norm * integral_weight * dist.Normal(
                                delta_X_pred, constraint_std).log_prob(delta_X_obs))

    return model, t_eval_np


# =============================================================================
# ROM solve helpers
# =============================================================================
def _generate_rom_solves(operator_samples, rom, q0, time_eval,
                        input_func, max_samples=200):
    solves = []
    n = min(len(operator_samples), max_samples)
    for i in range(n):
        rom.model._extract_operators(np.array(operator_samples[i]))
        try:
            sol = rom.model.predict(state0=q0, t=time_eval, input_func=input_func)
            if sol.shape[1] == len(time_eval) and np.all(np.isfinite(sol)):
                solves.append(sol)
        except Exception:
            pass
    return np.array(solves) if solves else np.empty((0, len(q0), len(time_eval)))


# =============================================================================
# Variant orchestration
# =============================================================================
@dataclass
class VariantResult:
    name: str
    label: str
    pod_doses: tuple
    train_doses: tuple
    basis: object = None
    rom: object = None
    O_samples: np.ndarray = None
    num_modes: int = 0
    gamma2_used: float = 0.0
    runtime: float = 0.0
    losses: list = field(default_factory=list)
    eval_per_dose: dict = field(default_factory=dict)  # dose -> dict of metrics
    pod_proj_err: dict = field(default_factory=dict)   # dose -> proj err
    cond_D: float = float('nan')
    block_norms: dict = field(default_factory=dict)


def evaluate_at_doses(variant: VariantResult, traj_train: list[TrajData],
                      t_pred: np.ndarray, eval_doses: Sequence[float],
                      seed: int, schema: dict, max_samples: int = 200):
    """For each eval dose, load the FOM, integrate ROM at that dose, compute
    volume relative error and CI coverage. Always uses CLEAN projected q0
    from each eval dose's FOM (basis-projected) so initial conditions are
    physically consistent across variants."""
    rom = variant.rom
    basis = variant.basis
    O_samp = variant.O_samples
    n_total = O_samp.shape[0]

    voxel_vol = float(np.prod(traj_train[0].fom.spacing))
    V_basis = basis.entries
    ones = np.ones(V_basis.shape[0])
    vol_proj = V_basis.T @ ones
    shift_vol = ones @ basis.shift_

    # Reuse a representative chemo_meta to construct per-dose input funcs.
    chemo_meta = traj_train[0].chemo_meta
    spec = chemo_meta['chemo_spec']
    t0_chemo = chemo_meta['t0']

    for dose in eval_doses:
        print(f"    Eval dose × {dose:g}  ", end='', flush=True)
        path = _dose_path(dose)
        if not os.path.exists(path):
            print(f"⚠ missing {path}")
            continue
        fom_e = TumorTwinFOM(path)
        true_states_e = fom_e.get_states(t_pred)

        # Clean projected q0 from this dose's FOM
        q0_clean = basis.compress(true_states_e[:, :1])[:, 0]

        # Per-dose POD projection error (full trajectory, in physical space)
        proj = basis.decompress(basis.compress(true_states_e))
        proj_err = float(np.linalg.norm(true_states_e - proj)
                         / np.linalg.norm(true_states_e))
        variant.pod_proj_err[float(dose)] = proj_err

        # Build α(t) at this dose
        ifn_e = config.chemo_input_func_factory(spec, t0_chemo, dose_scale=dose)
        ifn_jax_e = make_jax_input_func(ifn_e, float(t_pred[0]),
                                        float(t_pred[-1]), n_points=4001)

        rom_solves = _generate_rom_solves(
            operator_samples=O_samp, rom=rom, q0=q0_clean,
            time_eval=t_pred, input_func=ifn_jax_e,
            max_samples=min(max_samples, n_total))
        n_stable = len(rom_solves)

        # FOM volume
        fom_vol = np.array([true_states_e[:, k].sum() * voxel_vol
                            for k in range(true_states_e.shape[1])])

        rec = dict(n_stable=n_stable, n_total=min(max_samples, n_total),
                   fom_vol=fom_vol, proj_err=proj_err)

        if n_stable > 0:
            rom_arr = np.array(rom_solves)
            rom_vols = np.array([vol_proj @ rom_arr[s] + shift_vol
                                 for s in range(rom_arr.shape[0])]) * voxel_vol
            rom_med_v = np.median(rom_vols, axis=0)
            rom_lo = np.percentile(rom_vols, 5, axis=0)
            rom_hi = np.percentile(rom_vols, 95, axis=0)

            err = float(np.linalg.norm(rom_med_v - fom_vol)
                        / np.linalg.norm(fom_vol))
            cov = float(np.mean((fom_vol >= rom_lo) & (fom_vol <= rom_hi)))
            ci_w = float(np.mean(rom_hi - rom_lo))

            # Per-mode coverage on POD coefficients
            true_comp = basis.compress(true_states_e)
            q05 = np.percentile(rom_arr, 5, axis=0)
            q95 = np.percentile(rom_arr, 95, axis=0)
            mode_cov = float(np.mean((true_comp >= q05) & (true_comp <= q95)))

            # Endpoint metrics (last 10% of time)
            n_end = max(1, len(t_pred) // 10)
            end_err = float(np.linalg.norm(rom_med_v[-n_end:] - fom_vol[-n_end:])
                            / np.linalg.norm(fom_vol[-n_end:]))
            end_cov = float(np.mean((fom_vol[-n_end:] >= rom_lo[-n_end:]) &
                                    (fom_vol[-n_end:] <= rom_hi[-n_end:])))

            rec.update(dict(
                rom_med=rom_med_v, rom_lo=rom_lo, rom_hi=rom_hi,
                vol_err=err, ci_cov=cov, ci_width=ci_w,
                mode_cov=mode_cov,
                endpoint_err=end_err, endpoint_cov=end_cov,
                rom_solves=rom_arr,
            ))
            print(f"stab {n_stable}/{rec['n_total']}  err={err:.2%}  "
                  f"cov={cov:.2%}  modecov={mode_cov:.2%}")
        else:
            rec.update(dict(vol_err=float('nan'), ci_cov=float('nan'),
                            ci_width=float('nan'), mode_cov=float('nan'),
                            endpoint_err=float('nan'), endpoint_cov=float('nan')))
            print(f"stab 0/{rec['n_total']} (all unstable)")

        variant.eval_per_dose[float(dose)] = rec


def run_variant(name: str, label: str,
                pod_doses: tuple, train_doses: tuple,
                schema: dict, num_modes: int,
                gamma2_fixed: float | None = None,
                num_steps: int = None) -> VariantResult:
    """Run a single ablation variant end-to-end."""
    p = MODEL_PARAMS
    np.random.seed(p['SEED'])
    rng_key = random.PRNGKey(p['SEED'])

    print(f"\n{'='*72}\n  Variant {name}: {label}")
    print(f"  POD doses: {pod_doses}   Train doses: {train_doses}")
    print(f"{'='*72}")

    t_pred = np.linspace(TRAINING_SPAN[0], PREDICTION_DAYS, schema['NUM_EVAL_POINTS'])

    # ── Load all needed trajectories (union of POD + train sets) ──
    all_doses = sorted(set(pod_doses) | set(train_doses))
    trajs = {d: load_trajectory(d, schema, t_pred, p['SEED']) for d in all_doses}
    pod_trajs = [trajs[d] for d in pod_doses]
    train_trajs = [trajs[d] for d in train_doses]

    # ── POD basis ──
    basis = fit_pod_basis(pod_trajs, num_modes)
    print(f"  POD basis: {num_modes} modes, energy={basis.cumulative_energy:.4%}")

    # ── Per-trajectory GP MLE on TRAIN trajectories ──
    for td in train_trajs:
        fit_per_traj_gp(td, basis, num_modes)
        print(f"  [d={td.dose}]  Ls={td.Ls.round(4)}  Vs={td.Vs.round(3)}  "
              f"Ns={td.Ns.round(6)}")

    # ── ROM scaffolding (use first train traj's input as anchor; basis shared) ──
    block_reg_diag = np.concatenate([
        np.full(1, 0.1),
        np.full(num_modes, 0.1),
        np.full(1, 0.1),
        np.full(num_modes, 0.1),
    ])
    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(train_trajs[0].t_samp),
        model=ChemoReducedOrderModel(
            operator_string="cABN",
            solver=opinf.lstsq.TikhonovSolver(regularizer=np.diag(block_reg_diag))),
    )
    # Run rom.fit on the first training trajectory just to materialize the
    # operator matrix shape; we'll override it with our own LS solution.
    for td in train_trajs:
        compute_inputs(td, td.t_samp, np.linspace(
            float(td.t_samp[0]), float(td.t_samp[-1]), schema['NUM_EVAL_POINTS']))
    rom.fit(states=train_trajs[0].snaps_noisy,
            inputs=train_trajs[0].inputs_at_samp)
    print(f"  ROM operator shape: {rom.model.operator_matrix.shape}")

    # ── GP means + derivatives at LS eval grid for each train traj ──
    for td in train_trajs:
        t_eval_ls = np.linspace(float(td.t_samp[0]), float(td.t_samp[-1]),
                                schema['NUM_EVAL_POINTS'])
        compute_gp_means_and_derivs(td, t_eval_ls)

    # ── Multi-trajectory LS prior ──
    O_ls, O_prior_scale, D_stack, mu_stack = build_ls_prior(
        train_trajs, rom, num_modes, block_reg_diag)
    cond_D = float(np.linalg.cond(D_stack))
    print(f"  LS operator norm: {np.linalg.norm(O_ls):.1f}, "
          f"shape: {O_ls.shape}, cond(D)={cond_D:.2e}")

    # Block column norms (c, A, B, N)
    block_sizes = [1, num_modes, 1, num_modes]
    block_labels = ['c', 'A', 'B', 'N']
    idx = 0
    block_norms = {}
    for sz, lab in zip(block_sizes, block_labels):
        block = D_stack[:, idx:idx+sz]
        block_norms[lab] = float(np.linalg.norm(block, axis=0).mean())
        idx += sz
    print(f"  D block col-norms: {block_norms}")

    # ── γ₂: fixed if given (V2/V3), else empirical-Bayes (V1) ──
    if gamma2_fixed is not None:
        gamma2_eff = gamma2_fixed
        print(f"  γ₂ (fixed from V1): {gamma2_eff:.4f}")
    else:
        gamma2_eff = gamma2_from_residuals(D_stack, mu_stack, O_ls)
        print(f"  γ₂ (empirical Bayes): {gamma2_eff:.4f}")

    # ── Build & run multi-trajectory SVI ──
    if num_steps is None:
        num_steps = p['NUM_STEPS']

    model, t_eval_per_traj = build_multitraj_model(
        rom=rom, num_modes=num_modes, traj_list=train_trajs,
        O_prior=O_ls, O_prior_scale=O_prior_scale,
        num_eval_points=schema['NUM_EVAL_POINTS'],
        window_size=p['WINDOW_SIZE'],
        deriv_weight=p['DERIV_WEIGHT'], integral_weight=p['INTEGRAL_WEIGHT'],
        mll_weight=p['MLL_WEIGHT'], gp_prior_scale=p['GP_PRIOR_SCALE'],
    )

    init_values = {'O': jnp.array(O_ls)}
    for j, td in enumerate(train_trajs):
        for i in range(num_modes):
            init_values[f'traj{j}_lengthscale_{i}'] = float(td.Ls[i])
            init_values[f'traj{j}_variance_{i}'] = float(td.Vs[i])
            init_values[f'traj{j}_noise_{i}'] = float(max(td.Ns[i], 1e-6))

    model_kwargs = dict(gamma=p['GAMMA'], gamma2=gamma2_eff, jitter=1e-4)
    guide_rank = 25
    guide = autoguide.AutoLowRankMultivariateNormal(
        model, rank=guide_rank,
        init_loc_fn=init_to_value(values=init_values))
    optimizer = ClippedAdam(step_size=p['LEARNING_RATE'])
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key, ik = random.split(rng_key)
    t0 = time.time()
    svi_state = svi.init(ik, **model_kwargs)

    @jax.jit
    def _step(s, _):
        s, l = svi.update(s, **model_kwargs)
        return s, l

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

    O_samp = _find_operator_samples(samples, "O")
    if O_samp.ndim == 2:
        O_samp = O_samp[np.newaxis, ...]

    variant = VariantResult(
        name=name, label=label,
        pod_doses=tuple(pod_doses), train_doses=tuple(train_doses),
        basis=basis, rom=rom, O_samples=O_samp,
        num_modes=num_modes, gamma2_used=gamma2_eff,
        runtime=runtime, losses=all_losses,
        cond_D=cond_D, block_norms=block_norms,
    )
    print(f"  SVI done in {runtime:.0f}s.  Final loss: {all_losses[-1]:.2f}")
    return variant, train_trajs, t_pred


# =============================================================================
# Plotting
# =============================================================================
def plot_variant_grid(variants: list[VariantResult], eval_doses: Sequence[float],
                      t_pred: np.ndarray, training_span: tuple,
                      save_path: str, title: str):
    """3 (variants) × N (doses) panel of tumor volume FOM vs ROM with CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_var = len(variants)
    n_dose = len(eval_doses)
    fig, axes = plt.subplots(n_var, n_dose,
                             figsize=(3.5 * n_dose, 3.5 * n_var),
                             sharey='row', squeeze=False)

    for r, var in enumerate(variants):
        for c, dose in enumerate(eval_doses):
            ax = axes[r, c]
            rec = var.eval_per_dose.get(float(dose))
            if rec is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
                continue
            ax.plot(t_pred, rec['fom_vol'], color='tab:gray', lw=2.0,
                    label='FOM')
            if rec['n_stable'] > 0:
                ax.plot(t_pred, rec['rom_med'], color='tab:purple', lw=1.7, ls='--',
                        label='ROM med')
                ax.fill_between(t_pred, rec['rom_lo'], rec['rom_hi'],
                                color='tab:purple', alpha=0.18, label='5–95%')
                txt = (f"err {rec['vol_err']:.1%}\n"
                       f"cov {rec['ci_cov']:.0%}\n"
                       f"stab {rec['n_stable']}/{rec['n_total']}\n"
                       f"proj {rec['proj_err']:.1%}")
            else:
                txt = (f"all unstable\nproj {rec['proj_err']:.1%}\n"
                       f"stab 0/{rec['n_total']}")
            ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                    fontsize=8, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            ax.axvspan(training_span[0], training_span[1],
                       color='gray', alpha=0.10)
            ax.axvline(training_span[1], color='gray', ls='--', alpha=0.5)
            if r == 0:
                ax.set_title(f'Dose × {dose:g}', fontsize=11)
            if c == 0:
                ax.set_ylabel(f'{var.name}\n{var.label}\nVolume (mm³)',
                              fontsize=9)
            if r == n_var - 1:
                ax.set_xlabel('Time (days)')
            if r == 0 and c == 0:
                ax.legend(loc='lower right', fontsize=7)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Saved variant grid: {save_path}")


def plot_dose_response(variants: list[VariantResult], eval_doses: Sequence[float],
                       save_path: str, title: str):
    """Dose-response: endpoint volume vs dose (FOM and ROM medians + bands)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    doses_arr = np.array(eval_doses)

    # FOM endpoints (same across variants — pick V1's recorded FOM volume)
    fom_endpoints = []
    for d in eval_doses:
        rec = variants[0].eval_per_dose.get(float(d))
        fom_endpoints.append(rec['fom_vol'][-1] if rec else np.nan)
    ax.plot(doses_arr, fom_endpoints, 'o-', color='black', lw=2, label='FOM')

    colors = ['tab:red', 'tab:orange', 'tab:purple']
    for var, color in zip(variants, colors):
        med = []
        lo = []
        hi = []
        for d in eval_doses:
            rec = var.eval_per_dose.get(float(d))
            if rec and rec['n_stable'] > 0:
                med.append(rec['rom_med'][-1])
                lo.append(rec['rom_lo'][-1])
                hi.append(rec['rom_hi'][-1])
            else:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        med = np.array(med); lo = np.array(lo); hi = np.array(hi)
        ax.plot(doses_arr, med, 's--', color=color, lw=1.6,
                label=f'{var.name} med')
        ax.fill_between(doses_arr, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel('Dose scale')
    ax.set_ylabel('Endpoint tumor volume (mm³)')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Saved dose-response: {save_path}")


def print_summary_table(variants: list[VariantResult], eval_doses: Sequence[float]):
    print(f"\n{'='*100}")
    print("ABLATION SUMMARY")
    print(f"{'='*100}")
    for var in variants:
        print(f"\n  {var.name}: {var.label}")
        print(f"    POD: {var.pod_doses}   Train: {var.train_doses}   "
              f"γ₂={var.gamma2_used:.4f}   cond(D)={var.cond_D:.2e}   "
              f"runtime={var.runtime:.0f}s")
        print(f"    Block norms: {var.block_norms}")
        print(f"    {'Dose':>6s} {'ProjErr':>8s} {'Stab':>9s} {'VolErr':>8s} "
              f"{'CICov':>7s} {'ModeCov':>7s} {'EndErr':>8s} {'EndCov':>7s}")
        for d in eval_doses:
            rec = var.eval_per_dose.get(float(d))
            if rec is None:
                continue
            stab_str = f"{rec['n_stable']}/{rec['n_total']}"
            verr = rec.get('vol_err', float('nan'))
            ccov = rec.get('ci_cov', float('nan'))
            mcov = rec.get('mode_cov', float('nan'))
            eerr = rec.get('endpoint_err', float('nan'))
            ecov = rec.get('endpoint_cov', float('nan'))
            print(f"    {d:>6.2g} {rec['proj_err']:>7.2%} {stab_str:>9s} "
                  f"{verr:>7.2%} {ccov:>6.2%} {mcov:>6.2%} {eerr:>7.2%} {ecov:>6.2%}")


# =============================================================================
# Main
# =============================================================================
def main(schema_name: str, variants_to_run: list[str],
         eval_doses: Sequence[float], num_steps: int):
    schema = dict(SCHEMAS[schema_name])
    print(f"=== Multi-dose ablation — schema={schema_name} ===")
    print(f"   {schema}")
    print(f"   variants: {variants_to_run}")
    print(f"   eval_doses: {eval_doses}")
    print(f"   num_steps: {num_steps}")

    # Validate all eval-dose FOMs exist before doing expensive work
    for d in eval_doses:
        path = _dose_path(d)
        if not os.path.exists(path):
            print(f"   ⚠ Missing FOM at dose×{d}: {path}")

    # ── Step 1: SNR-based mode rank from V1 (1.0x POD only) ──
    print("\n--- Selecting POD rank via SNR on 1.0x trajectory ---")
    t_pred_dummy = np.linspace(TRAINING_SPAN[0], PREDICTION_DAYS,
                               schema['NUM_EVAL_POINTS'])
    td_one = load_trajectory(1.0, schema, t_pred_dummy, MODEL_PARAMS['SEED'])
    eff_rank = select_rank_via_snr([td_one],
                                   max_modes=MODEL_PARAMS['NUM_MODES'] + 4,
                                   snr_threshold=10.0)
    num_modes = min(eff_rank, MODEL_PARAMS['NUM_MODES'])
    print(f"  → Using rank = {num_modes} for all variants")
    del td_one

    # ── Step 2: Run V1 first to compute γ₂; reuse for V2/V3 ──
    variant_specs = {
        'V1': dict(label='POD 1.0x  /  Train 1.0x',
                   pod_doses=(1.0,), train_doses=(1.0,)),
        'V2': dict(label=f'POD {DEFAULT_POD_DOSES_MULTI}  /  Train 1.0x',
                   pod_doses=DEFAULT_POD_DOSES_MULTI, train_doses=(1.0,)),
        'V3': dict(label=f'POD {DEFAULT_POD_DOSES_MULTI}  /  Train {DEFAULT_TRAIN_DOSES_MULTI}',
                   pod_doses=DEFAULT_POD_DOSES_MULTI,
                   train_doses=DEFAULT_TRAIN_DOSES_MULTI),
    }

    variants = []
    gamma2_v1 = None
    for vname in variants_to_run:
        spec = variant_specs[vname]
        gamma2_arg = None if vname == 'V1' else gamma2_v1
        var, train_trajs, t_pred = run_variant(
            name=vname, label=spec['label'],
            pod_doses=spec['pod_doses'], train_doses=spec['train_doses'],
            schema=schema, num_modes=num_modes,
            gamma2_fixed=gamma2_arg, num_steps=num_steps,
        )
        if vname == 'V1':
            gamma2_v1 = var.gamma2_used

        # ── Evaluate at all eval doses ──
        print(f"\n  --- Evaluating {vname} at doses {tuple(eval_doses)} ---")
        evaluate_at_doses(var, train_trajs, t_pred, eval_doses,
                          seed=MODEL_PARAMS['SEED'], schema=schema)
        variants.append(var)

    # ── Plot + summarize ──
    os.makedirs(FIGURE_DIR, exist_ok=True)
    grid_path = os.path.join(
        FIGURE_DIR, f'04_chemo_multidose_{schema_name}_grid.png')
    dose_resp_path = os.path.join(
        FIGURE_DIR, f'04_chemo_multidose_{schema_name}_dose_response.png')

    plot_variant_grid(variants, eval_doses, t_pred, TRAINING_SPAN, grid_path,
                      title=f'Multi-dose ablation — {schema_name}')
    plot_dose_response(variants, eval_doses, dose_resp_path,
                       title=f'Endpoint dose-response — {schema_name}')

    print_summary_table(variants, eval_doses)

    # Save numerical summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f'multidose_{schema_name}_summary.npz')
    save_dict = {}
    for var in variants:
        for d, rec in var.eval_per_dose.items():
            key = f'{var.name}_d{d:g}'.replace('.', 'p')
            save_dict[f'{key}_vol_err'] = rec.get('vol_err', float('nan'))
            save_dict[f'{key}_ci_cov'] = rec.get('ci_cov', float('nan'))
            save_dict[f'{key}_mode_cov'] = rec.get('mode_cov', float('nan'))
            save_dict[f'{key}_n_stable'] = rec['n_stable']
            save_dict[f'{key}_proj_err'] = rec['proj_err']
            save_dict[f'{key}_endpoint_err'] = rec.get('endpoint_err', float('nan'))
    np.savez(save_path, **save_dict)
    print(f"\n  💾 Saved summary: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema', default='dense_high_noise',
                        choices=list(SCHEMAS.keys()))
    parser.add_argument('--variants', nargs='+', default=['V1', 'V2', 'V3'],
                        choices=['V1', 'V2', 'V3'])
    parser.add_argument('--eval-doses', type=float, nargs='+',
                        default=list(DEFAULT_EVAL_DOSES))
    parser.add_argument('--steps', type=int, default=MODEL_PARAMS['NUM_STEPS'])
    args = parser.parse_args()
    main(args.schema, args.variants, args.eval_doses, args.steps)

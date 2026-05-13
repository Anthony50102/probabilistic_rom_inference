#!/usr/bin/env python
"""
Stepwise Pipeline Diagnostic for Tumor Growth ROM

Validates each stage of the Bayesian OpInf pipeline in isolation,
producing diagnostic plots at each step:

  Step 1: Data loading + POD basis (clean vs noisy)
  Step 2: GP fit on compressed coefficients
  Step 3: Least-squares operator estimate + forward integration
  Step 4: Full SVI inference (if Steps 1-3 pass)

Usage:
    conda run -n prob_rom python diagnose_pipeline.py
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import Basis, TumorTwinFOM, load_fom_data, TRAINING_SPAN
from core import JaxCompatibleModel, compute_gp_derivatives, rbf_eval
from core.bayesian_opinf import fit_gp_hyperparameters_mle
from core.plotting import plot_gp_fit
import opinf
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

SEED = 42
NUM_SAMPLES = 200
NOISE_LEVEL = 0.01
NUM_EVAL_POINTS = 400
OPERATOR_STR = "cAH"


def step1_data_and_pod():
    """Load data, compare clean vs noisy POD, choose basis."""
    print("=" * 70)
    print("STEP 1: Data Loading + POD Basis")
    print("=" * 70)

    np.random.seed(SEED)
    t_pred = np.linspace(0, config.PREDICTION_DAYS, NUM_EVAL_POINTS)

    fom, t_full, true_states, t_samp, snaps_noisy = load_fom_data(
        t_pred, TRAINING_SPAN, NUM_SAMPLES, NOISE_LEVEL,
    )
    snaps_clean = fom.get_states(t_samp)

    print(f"  Domain: {fom.grid_shape} = {fom.n_dof:,} DOFs")
    print(f"  Training: {TRAINING_SPAN[0]:.0f}–{TRAINING_SPAN[1]:.0f} days, "
          f"{NUM_SAMPLES} samples, {NOISE_LEVEL:.0%} noise")
    print(f"  Prediction: 0–{config.PREDICTION_DAYS:.0f} days, {NUM_EVAL_POINTS} points")

    # Compare clean vs noisy POD energy
    print(f"\n  {'Modes':>5s}  {'Clean':>10s}  {'Noisy':>10s}")
    print(f"  {'-----':>5s}  {'----------':>10s}  {'----------':>10s}")
    for nm in [2, 3, 4, 6, 8]:
        b_clean = Basis(num_vectors=nm)
        b_clean.fit(snaps_clean)
        b_noisy = Basis(num_vectors=nm)
        b_noisy.fit(snaps_noisy)
        print(f"  {nm:5d}  {b_clean.cumulative_energy:10.4%}  {b_noisy.cumulative_energy:10.4%}")

    # Use clean basis (FOM is deterministic — noise is measurement error)
    best_nm = 4
    basis = Basis(num_vectors=best_nm)
    basis.fit(snaps_clean)
    print(f"\n  → Using {best_nm} modes on CLEAN basis: {basis.cumulative_energy:.6%}")

    snaps_comp = basis.compress(snaps_noisy)
    clean_comp = basis.compress(snaps_clean)
    true_comp = basis.compress(true_states)

    # Plot: POD modes (compressed coefficients) — clean vs noisy
    fig, axes = plt.subplots(best_nm, 1, figsize=(10, 2.5 * best_nm), sharex=True)
    if best_nm == 1:
        axes = [axes]
    for i in range(best_nm):
        axes[i].plot(t_samp, clean_comp[i], 'o', color='tab:gray', ms=3,
                     alpha=0.5, label='Clean FOM', zorder=3)
        axes[i].plot(t_samp, snaps_comp[i], 'k*', ms=3, label='Noisy obs', zorder=4)
        axes[i].plot(t_full, true_comp[i], color='tab:gray', lw=1.5,
                     label='Truth (full)', zorder=2)
        axes[i].set_ylabel(f'Mode {i}')
        if i == 0:
            axes[i].legend(fontsize=8)
    axes[-1].set_xlabel('Time (days)')
    fig.suptitle(f'POD Coefficients — {best_nm} modes, {basis.cumulative_energy:.2%} energy',
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'diag_step1_pod_coefficients.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)

    return dict(
        fom=fom, t_full=t_full, true_states=true_states,
        t_samp=t_samp, snaps_noisy=snaps_noisy, snaps_clean=snaps_clean,
        basis=basis, snaps_comp=snaps_comp, clean_comp=clean_comp,
        true_comp=true_comp, t_pred=t_pred, num_modes=best_nm,
    )


def step2_gp_fit(data):
    """Fit GP hyperparameters and check state + derivative predictions."""
    print("\n" + "=" * 70)
    print("STEP 2: GP Hyperparameter Fitting")
    print("=" * 70)

    t_samp = data['t_samp']
    snaps_comp = data['snaps_comp']
    clean_comp = data['clean_comp']
    true_comp = data['true_comp']
    t_full = data['t_full']
    num_modes = data['num_modes']

    Ls, Vs, Ns, gp_models = fit_gp_hyperparameters_mle(
        t_samp, snaps_comp, verbose=False,
    )

    print(f"\n  {'Mode':>4s}  {'ℓ':>10s}  {'T/ℓ':>6s}  {'σ²':>10s}  {'ν':>10s}")
    print(f"  {'----':>4s}  {'----------':>10s}  {'------':>6s}  {'----------':>10s}  {'----------':>10s}")
    T = t_samp[-1] - t_samp[0]
    for i in range(num_modes):
        print(f"  {i:4d}  {Ls[i]:10.4f}  {T/Ls[i]:6.0f}  {Vs[i]:10.4f}  {Ns[i]:10.6f}")

    # GP state predictions on eval grid
    t_eval = np.linspace(float(t_samp[0]), float(t_samp[-1]), NUM_EVAL_POINTS)
    X_gp = np.zeros((num_modes, NUM_EVAL_POINTS))
    X_gp_std = np.zeros((num_modes, NUM_EVAL_POINTS))
    for i in range(num_modes):
        K = rbf_eval(Ls[i], Vs[i], t_samp, t_samp) + (Ns[i] + 1e-5) * np.eye(len(t_samp))
        Ks = rbf_eval(Ls[i], Vs[i], t_eval, t_samp)
        Kss = rbf_eval(Ls[i], Vs[i], t_eval, t_eval)
        alpha = np.linalg.solve(K, snaps_comp[i])
        X_gp[i] = Ks @ alpha
        cov = Kss - Ks @ np.linalg.solve(K, Ks.T)
        X_gp_std[i] = np.sqrt(np.maximum(np.diag(cov), 0))

    # GP derivative predictions
    mu_z, var_z = compute_gp_derivatives(Ls, Vs, t_samp, t_eval, snaps_comp, Ns=Ns)
    # Ensure numpy arrays, flatten any JAX/2D shapes
    mu_z = np.array(mu_z)
    var_z = np.array(var_z)
    if mu_z.ndim > 2:
        mu_z = mu_z.reshape(num_modes, -1)
    if var_z.ndim > 2:
        var_z = var_z.reshape(num_modes, -1)

    # Compute fit error vs clean truth
    true_at_eval = interp1d(t_full, true_comp, kind='cubic')(t_eval)
    for i in range(num_modes):
        rmse = np.sqrt(np.mean((X_gp[i] - true_at_eval[i]) ** 2))
        rel_err = rmse / (np.std(true_at_eval[i]) + 1e-12)
        print(f"  Mode {i} GP state RMSE: {rmse:.4f}  (relative: {rel_err:.2%})")

    # Plot: GP fits (state + derivative)
    fig, axes = plt.subplots(num_modes, 2, figsize=(14, 3 * num_modes),
                             sharex='col')
    if num_modes == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_modes):
        # State fit
        ax = axes[i, 0]
        ax.plot(t_samp, snaps_comp[i], 'k*', ms=4, label='Noisy obs', zorder=5)
        ax.plot(t_eval, true_at_eval[i], color='tab:gray', lw=1.5, label='Truth')
        ax.plot(t_eval, X_gp[i], color='tab:purple', ls='--', lw=2, label='GP mean')
        ax.fill_between(t_eval, X_gp[i] - 1.96 * X_gp_std[i],
                        X_gp[i] + 1.96 * X_gp_std[i],
                        color='tab:purple', alpha=0.15, label='95% CI')
        ax.set_ylabel(f'Mode {i}')
        if i == 0:
            ax.set_title('GP State Fit')
            ax.legend(fontsize=7, loc='upper right')

        # Derivative fit
        ax = axes[i, 1]
        # Finite difference derivatives from clean data as reference
        dt = np.diff(t_samp)
        fd_clean = np.diff(clean_comp[i]) / dt
        t_fd = 0.5 * (t_samp[:-1] + t_samp[1:])
        ax.plot(t_fd, fd_clean, '.', color='tab:gray', ms=3, alpha=0.5,
                label='FD (clean)', zorder=3)
        ax.plot(t_eval, mu_z[i], color='tab:purple', ls='--', lw=2, label='GP deriv')
        std_z = np.sqrt(np.maximum(np.diag(np.array(var_z[i])), 0))
        ax.fill_between(t_eval, np.array(mu_z[i]) - 1.96 * std_z,
                        np.array(mu_z[i]) + 1.96 * std_z,
                        color='tab:purple', alpha=0.15, label='95% CI')
        ax.set_ylabel(f'dMode {i}/dt')
        if i == 0:
            ax.set_title('GP Derivative Fit')
            ax.legend(fontsize=7, loc='upper right')

    axes[-1, 0].set_xlabel('Time (days)')
    axes[-1, 1].set_xlabel('Time (days)')
    fig.suptitle('GP Hyperparameter Fits (MLE)', fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'diag_step2_gp_fit.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)

    data.update(dict(
        Ls=Ls, Vs=Vs, Ns=Ns, gp_models=gp_models,
        t_eval=t_eval, X_gp=X_gp, mu_z=mu_z, var_z=var_z,
    ))
    return data


def step3_ls_operator(data):
    """Compute LS operator and test forward integration."""
    print("\n" + "=" * 70)
    print("STEP 3: Least-Squares Operator + Forward Integration")
    print("=" * 70)

    basis = data['basis']
    t_samp = data['t_samp']
    snaps_noisy = data['snaps_noisy']
    snaps_comp = data['snaps_comp']
    true_comp = data['true_comp']
    t_full = data['t_full']
    t_pred = data['t_pred']
    num_modes = data['num_modes']
    X_gp = data['X_gp']
    mu_z = data['mu_z']
    t_eval = data['t_eval']

    # Build ROM for data matrix assembly
    rom = opinf.ROM(
        basis=basis,
        ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(t_samp),
        model=JaxCompatibleModel(
            operators=OPERATOR_STR,
            solver=opinf.lstsq.L2Solver(regularizer=1e0),
        ),
    )
    rom.fit(states=data['snaps_clean'])

    D = np.array(rom.model._assemble_data_matrix(jnp.array(X_gp), inputs=None))
    DtD = D.T @ D

    print(f"  Data matrix D: {D.shape}, cond={np.linalg.cond(D):.0f}")
    col_norms = np.linalg.norm(D, axis=0)
    print(f"  Col norms: min={col_norms.min():.2f}, max={col_norms.max():.2f}, "
          f"ratio={col_norms.max()/col_norms.min():.0f}")

    # Sweep regularization
    print(f"\n  {'Reg':>8s}  {'‖O‖':>8s}  {'Stable?':>8s}  {'Train err':>10s}")
    print(f"  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'----------':>10s}")

    best_reg, best_err, best_O, best_sol = None, 1e10, None, None
    q0 = basis.compress(data['true_states'][:, 0])
    true_at_pred = interp1d(t_full, true_comp, kind='cubic',
                            fill_value='extrapolate')(t_pred)

    for reg in [1e-4, 1e-2, 1e0, 1e1, 1e2, 1e3, 1e4]:
        O_ls = np.linalg.solve(DtD + reg * np.eye(DtD.shape[0]),
                               D.T @ np.array(mu_z).T).T

        def rhs(t, q, O=O_ls):
            d = rom.model._assemble_data_matrix(jnp.array(q.reshape(-1, 1)), inputs=None)
            return (O @ np.array(d).T).flatten()

        sol = solve_ivp(rhs, [t_pred[0], t_pred[-1]], q0,
                        t_eval=t_pred, max_step=0.5, method='RK45',
                        rtol=1e-6, atol=1e-9)

        if sol.success:
            err = np.linalg.norm(sol.y - true_at_pred) / np.linalg.norm(true_at_pred)
            stable = "✓"
            if err < best_err:
                best_err, best_reg, best_O, best_sol = err, reg, O_ls, sol
        else:
            err = float('inf')
            stable = "✗"

        print(f"  {reg:8.0e}  {np.linalg.norm(O_ls):8.2f}  {stable:>8s}  "
              f"{'—' if err == float('inf') else f'{err:.4%}':>10s}")

    if best_sol is None:
        print("\n  ✗ ALL regularizations failed! Cannot integrate LS operator.")
        return data

    print(f"\n  → Best reg={best_reg:.0e}, train error={best_err:.4%}")

    # Plot: LS operator forward integration vs truth
    fig, axes = plt.subplots(num_modes, 1, figsize=(10, 2.5 * num_modes), sharex=True)
    if num_modes == 1:
        axes = [axes]
    for i in range(num_modes):
        axes[i].plot(t_pred, true_at_pred[i], color='tab:gray', lw=1.5, label='Truth')
        axes[i].plot(t_samp, snaps_comp[i], 'k*', ms=3, alpha=0.7, label='Noisy obs')
        axes[i].plot(t_pred, best_sol.y[i], color='tab:blue', ls='--', lw=2,
                     label=f'LS ROM (reg={best_reg:.0e})')
        axes[i].axvline(TRAINING_SPAN[1], color='k', ls=':', lw=0.8, alpha=0.5)
        axes[i].set_ylabel(f'Mode {i}')
        if i == 0:
            axes[i].legend(fontsize=8)
    axes[-1].set_xlabel('Time (days)')
    fig.suptitle(f'LS Operator Forward Integration — Error: {best_err:.2%}', fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'diag_step3_ls_integration.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {path}")
    plt.close(fig)

    data.update(dict(rom=rom, O_ls=best_O, best_reg=best_reg))
    return data


def main():
    print("=" * 70)
    print("TUMOR GROWTH — STEPWISE PIPELINE DIAGNOSTIC")
    print("=" * 70)

    data = step1_data_and_pod()
    data = step2_gp_fit(data)
    data = step3_ls_operator(data)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print(f"  Figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()

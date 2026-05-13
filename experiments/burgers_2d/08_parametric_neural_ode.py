"""
08 — Parametric MLP Ensemble (Heat-Style Stacked Trajectories)
    2D Diffusion-Reaction Equation: ∂u/∂t = κ∇²u − βu²

Black-box parametric baseline mirroring 07's setup. Uses the same training
trajectories (parametric ICs μ = (a, b)), the same global POD basis, and the
same held-out test μ as 07_parametric_ics.py — but replaces Bayesian OpInf
with an ensemble of MLP neural ODEs.

Each ensemble member: dq/dt = f_θ(q), trained by trajectory matching summed
over all M training trajectories. UQ = ensemble spread.

Usage:
    python 08_parametric_neural_ode.py
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import diffrax
import optax
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
import config_parametric as config
from config_parametric import Basis

# ── Data regime ──────────────────────────────────────────────────────────────
NUM_SAMPLES = 60
NOISE_LEVEL = 0.03

# ── Model hyperparameters ────────────────────────────────────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=4,
    HIDDEN_DIM=64,
    NUM_LAYERS=3,
    ACTIVATION='tanh',
    ENSEMBLE_SIZE=10,
    NUM_TRAIN_STEPS=2000,
    LEARNING_RATE=1e-3,
    SEED=42,
)

TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 3.0)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# Neural ODE
# =============================================================================
class NeuralODE(eqx.Module):
    layers: list

    def __init__(self, in_dim, hidden_dim, num_layers, *, key):
        keys = random.split(key, num_layers + 1)
        self.layers = []
        d_in = in_dim
        for i in range(num_layers):
            self.layers.append(eqx.nn.Linear(d_in, hidden_dim, key=keys[i]))
            d_in = hidden_dim
        self.layers.append(eqx.nn.Linear(hidden_dim, in_dim, key=keys[num_layers]))

    def __call__(self, t, y, args):
        x = y
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


# =============================================================================
# Multi-trajectory trainer
# =============================================================================
def train_single_multi_traj(model, t_train_list, y_train_list, q0_list,
                             num_steps, lr, key):
    """Train one MLP via trajectory matching summed across M trajectories."""
    M = len(t_train_list)
    t_jnp_list = [jnp.array(t) for t in t_train_list]
    y_jnp_list = [jnp.array(y) for y in y_train_list]   # each (num_modes, N)
    q0_jnp_list = [jnp.array(q) for q in q0_list]

    t0_list = [float(t[0]) for t in t_train_list]
    t1_list = [float(t[-1]) for t in t_train_list]
    dt0_list = [float(t[1] - t[0]) for t in t_train_list]

    solver = diffrax.Tsit5()
    saveat_list = [diffrax.SaveAt(ts=t_jnp) for t_jnp in t_jnp_list]
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(model):
        term = diffrax.ODETerm(model)
        total = 0.0
        for m in range(M):
            sol = diffrax.diffeqsolve(
                term, solver,
                t0=t0_list[m], t1=t1_list[m],
                dt0=dt0_list[m],
                y0=q0_jnp_list[m],
                saveat=saveat_list[m],
                adjoint=adjoint,
                max_steps=16384,
                throw=False,
            )
            y_pred = sol.ys  # (N, num_modes)
            total = total + jnp.mean((y_pred - y_jnp_list[m].T) ** 2)
        return total / M

    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state_new = opt.update(grads, opt_state,
                                            eqx.filter(model, eqx.is_array))
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss

    losses = []
    for i in range(num_steps):
        model, opt_state, loss = step(model, opt_state)
        losses.append(float(loss))

    return model, losses


# =============================================================================
# Per-trajectory data generation (same as 07)
# =============================================================================
def _generate_one(fom, mu, full_time_domain, training_span, num_samples,
                   noise_level, rng):
    ic = config.initial_conditions(*mu)
    true_states = fom.solve(ic, full_time_domain)
    t_samp = np.sort(rng.uniform(training_span[0], training_span[1], size=num_samples))
    t_samp[0] = training_span[0]
    t_samp[-1] = training_span[1]
    clean = fom.solve(ic, t_samp)
    snaps_samp = fom.noise(clean, noise_level)
    return true_states, t_samp, snaps_samp


# =============================================================================
# Run experiment
# =============================================================================
def run_experiment():
    p = MODEL_PARAMS
    np.random.seed(p['SEED'])
    num_modes = p['NUM_MODES']

    fom = config.FullOrderModel()
    full_time = config.time_domain

    print(f"\n{'='*70}")
    print(f"  Parametric MLP — {len(config.TRAINING_MUS)} training μ's, "
          f"test μ={config.TEST_MU}")
    print(f"  samples={NUM_SAMPLES} noise={NOISE_LEVEL:.0%} "
          f"NUM_MODES={num_modes} ensemble={p['ENSEMBLE_SIZE']}")
    print(f"{'='*70}")

    # ── 1. Generate training data (same seeds as 07 for fair comparison) ──
    true_states_list, t_samp_list, snaps_samp_list = [], [], []
    for m, mu in enumerate(config.TRAINING_MUS):
        local_rng = np.random.default_rng(p['SEED'] + 100 * (m + 1))
        ts, t_samp, snaps = _generate_one(
            fom, mu, full_time, TRAINING_SPAN,
            NUM_SAMPLES, NOISE_LEVEL, local_rng)
        true_states_list.append(ts)
        t_samp_list.append(t_samp)
        snaps_samp_list.append(snaps)
        print(f"  μ{m}={mu}: snaps shape={snaps.shape}")

    # ── 2. Global POD basis ───────────────────────────────────────────────
    stacked = np.concatenate(snaps_samp_list, axis=1)
    basis = Basis(num_vectors=num_modes)
    basis.fit(stacked)
    print(f"  Global POD energy ({num_modes} modes): {basis.cumulative_energy:.4%}")

    snaps_comp_list = [basis.compress(s) for s in snaps_samp_list]
    true_comp_list = [basis.compress(ts) for ts in true_states_list]
    q0_list = [snaps_comp[:, 0] for snaps_comp in snaps_comp_list]

    # ── 3. Train ensemble ─────────────────────────────────────────────────
    ensemble_size = p['ENSEMBLE_SIZE']
    num_steps = p['NUM_TRAIN_STEPS']
    lr = p['LEARNING_RATE']

    print(f"  Training {ensemble_size} ensemble members ({num_steps} steps each)...")
    t0 = time.time()
    trained_models = []
    all_losses = []

    for m in range(ensemble_size):
        key = random.PRNGKey(p['SEED'] + m)
        model = NeuralODE(in_dim=num_modes,
                          hidden_dim=p['HIDDEN_DIM'],
                          num_layers=p['NUM_LAYERS'],
                          key=key)
        model, losses = train_single_multi_traj(
            model, t_samp_list, snaps_comp_list, q0_list,
            num_steps, lr, key)
        trained_models.append(model)
        all_losses.append(losses)
        print(f"    member {m+1:2d}/{ensemble_size}  final_loss={losses[-1]:.6f}")

    runtime = time.time() - t0
    print(f"  Ensemble training took {runtime:.0f}s")

    # ── 4. Evaluate on TEST_MU ────────────────────────────────────────────
    test_ic = config.initial_conditions(*config.TEST_MU)
    test_true = fom.solve(test_ic, full_time)
    test_true_comp = basis.compress(test_true)
    test_ic_comp = basis.compress(test_ic.reshape(-1, 1))[:, 0]

    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    q0_np = np.array(test_ic_comp)

    rom_solves = []
    for model in trained_models:
        try:
            def rhs(t, y, _model=model):
                return np.array(_model(t, jnp.array(y), None))
            sol = solve_ivp(rhs, [t_pred[0], t_pred[-1]],
                            q0_np, t_eval=t_pred, method='RK45', max_step=0.01)
            if sol.success and np.all(np.isfinite(sol.y)):
                rom_solves.append(sol.y)
        except Exception:
            pass

    n_stable = len(rom_solves)
    n_total = ensemble_size
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
        rom_solves=rom_solves,
        t_pred=t_pred, full_time=full_time,
        test_true=test_true, test_true_comp=test_true_comp,
        test_mu=config.TEST_MU, training_mus=config.TRAINING_MUS,
        t_samp_list=t_samp_list,
        snaps_comp_list=snaps_comp_list,
        true_comp_list=true_comp_list,
        basis=basis, fom=fom,
        num_modes=num_modes, training_span=TRAINING_SPAN,
        runtime=runtime, losses=all_losses,
        train_error=train_err, pred_error=pred_err,
        stability_pct=stab_pct, ci_coverage=ci_cov, ci_width=ci_w,
        n_stable=n_stable, n_total=n_total,
    )


# =============================================================================
# Plotting (mirrors 07)
# =============================================================================
def plot_results(result, save_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)
    prefix = "08_parametric_neural_ode"

    rom_solves = result['rom_solves']
    num_modes = result['num_modes']
    t_pred = result['t_pred']
    t_full = result['full_time']
    true_comp = result['test_true_comp']
    training_span = result['training_span']

    # ── 1. Test-μ trajectory ───────────────────────────────────────────
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
            ax[i].axvspan(*training_span, color='gray', alpha=0.10)
            ax[i].plot(t_pred, ta[i], color='tab:gray', lw=2, label='FOM (test μ)')
            ax[i].plot(t_pred, rom_med[i], color='tab:orange', ls='--', lw=2,
                       label='MLP median')
            ax[i].fill_between(t_pred, q05[i], q95[i], color='tab:orange',
                               alpha=0.15, label='Ensemble 5–95%')
            ax[i].axvline(training_span[1], color='k', ls=':', lw=0.8, alpha=0.5)
            ax[i].set_ylabel(f'Mode {i+1}')
            if i == 0:
                ax[i].legend(loc='upper right', fontsize=9)
        ax[-1].set_xlabel('Time')
        fig.suptitle(f'Parametric MLP @ test μ={result["test_mu"]} '
                     f'({result["n_stable"]}/{result["n_total"]} stable)',
                     fontsize=13)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_test_modes.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)

    # ── 2. Training trajectories ───────────────────────────────────────
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

    # ── 3. 2D contour for TEST μ ───────────────────────────────────────
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
        axes[1, 0].set_ylabel('MLP median', fontsize=12)
        axes[2, 0].set_ylabel('Ensemble width\n(q95 − q05)', fontsize=12)
        fig.suptitle(f'2D Field — MLP — test μ={result["test_mu"]}', fontsize=14)
        fig.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_2d_contours.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {path}")
        plt.close(fig)

    # ── 4. Loss ────────────────────────────────────────────────────────
    losses = result['losses']
    mean_loss = np.mean(losses, axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(mean_loss, lw=0.8); ax[0].set_title('MSE (mean over ensemble)')
    ax[0].grid(alpha=0.3); ax[0].set_yscale('log')
    half = len(mean_loss) // 2
    ax[1].plot(range(half, len(mean_loss)), mean_loss[half:], lw=0.8)
    ax[1].set_title('MSE (last 50%)'); ax[1].grid(alpha=0.3); ax[1].set_yscale('log')
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
    path = os.path.join(save_dir, "08_parametric_neural_ode.npz")
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
    print("08 — Parametric MLP Ensemble — 2D Diffusion-Reaction")
    print("=" * 70)
    r = run_experiment()
    plot_results(r)
    save_predictions(r)


if __name__ == "__main__":
    main()

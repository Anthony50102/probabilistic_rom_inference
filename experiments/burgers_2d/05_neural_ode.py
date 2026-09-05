"""
05 — Neural ODE Baseline (2D Diffusion-Reaction)

Black-box baseline for comparison with Bayesian OpInf (04):
  - MLP: 3 → 64 → 64 → 64 → 3  (tanh activations)
  - Training: trajectory matching via MSE against noisy reduced observations
  - UQ: ensemble of 10 independently trained networks

Uses the EXACT same data pipeline as 04_unified.py:
  1. FOM solve → subsample in training span → add noise → POD compression
  2. Evaluate on t_pred = linspace(0, 2.0, 400), compare with cubic-interpolated truth

Single data regime: dense data, medium noise (3%).

Usage:
    python 05_neural_ode.py
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
import config
from config import Basis
from core import generate_trajectory
from core.plotting import plot_full_order_error

# ── Data regime ──────────────────────────────────────────────────────────────
SCHEMAS = [
    {
        "name": "dense_medium_noise",
        "label": "Dense data, medium noise",
        "NUM_SAMPLES": 60,
        "NOISE_LEVEL": 0.03,
        "NUM_EVAL_POINTS": 200,
    },
]

# ── Model hyperparameters ────────────────────────────────────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=3,
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
# Neural ODE model
# =============================================================================
class NeuralODE(eqx.Module):
    """MLP dynamics: dq/dt = f_theta(q), with tanh activations."""
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
# Training
# =============================================================================
def train_single(model, t_train, y_train, q0, num_steps, lr, key):
    """Train one neural ODE via trajectory matching. Returns (model, losses)."""
    t_train_jnp = jnp.array(t_train)
    y_train_jnp = jnp.array(y_train)  # (num_modes, num_samples)
    q0_jnp = jnp.array(q0)

    t0 = float(t_train[0])
    t1 = float(t_train[-1])
    dt0 = float(t_train[1] - t_train[0])

    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_train_jnp)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(model):
        term = diffrax.ODETerm(model)
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=t0, t1=t1,
            dt0=dt0,
            y0=q0_jnp,
            saveat=saveat,
            adjoint=adjoint,
            max_steps=16384,
            throw=False,
        )
        y_pred = sol.ys  # (len(t_train), num_modes)
        return jnp.mean((y_pred - y_train_jnp.T) ** 2)

    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state_new = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss

    losses = []
    for i in range(num_steps):
        model, opt_state, loss = step(model, opt_state)
        losses.append(float(loss))

    return model, losses


# =============================================================================
# Run experiment
# =============================================================================
def run_experiment(schema):
    """Run one data regime. Returns results dict."""
    p = MODEL_PARAMS
    np.random.seed(p['SEED'])

    noise_level = schema['NOISE_LEVEL']
    num_samples = schema['NUM_SAMPLES']
    num_modes = p['NUM_MODES']

    print(f"\n{'='*70}")
    print(f"  {schema['label']}  ({num_samples} samples, {noise_level:.0%} noise)")
    print(f"{'='*70}")

    # ── Data generation (EXACTLY as in 04) ───────────────────────────────
    (fom, t_full, true_states, t_samp, snaps_samp) = \
        generate_trajectory(config, config.time_domain, TRAINING_SPAN, num_samples, noise_level)
    basis = Basis(num_vectors=num_modes)
    basis.fit(snaps_samp)
    snaps_comp = basis.compress(snaps_samp)   # (3, num_samples)
    true_comp = basis.compress(true_states)   # (3, len(t_full))
    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    q0 = snaps_comp[:, 0]

    # ── Train ensemble ───────────────────────────────────────────────────
    ensemble_size = p['ENSEMBLE_SIZE']
    num_steps = p['NUM_TRAIN_STEPS']
    lr = p['LEARNING_RATE']

    print(f"  Training {ensemble_size} ensemble members ({num_steps} steps each)...")
    t0 = time.time()
    trained_models = []
    all_losses = []

    for m in range(ensemble_size):
        key = random.PRNGKey(p['SEED'] + m)
        model = NeuralODE(
            in_dim=num_modes,
            hidden_dim=p['HIDDEN_DIM'],
            num_layers=p['NUM_LAYERS'],
            key=key,
        )
        model, losses = train_single(model, t_samp, snaps_comp, q0, num_steps, lr, key)
        trained_models.append(model)
        all_losses.append(losses)
        final_loss = losses[-1]
        print(f"    member {m+1:2d}/{ensemble_size}  final_loss={final_loss:.6f}")

    runtime = time.time() - t0
    print(f"  Ensemble training took {runtime:.0f}s")

    # ── Evaluate using scipy solve_ivp ───────────────────────────────────
    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    q0_np = np.array(q0)

    rom_solves = []
    for model in trained_models:
        try:
            def rhs(t, y, _model=model):
                return np.array(_model(t, jnp.array(y), None))

            sol = solve_ivp(rhs, [t_pred[0], t_pred[-1]],
                          q0_np, t_eval=t_pred, method='RK45', max_step=0.01)
            if sol.success and np.all(np.isfinite(sol.y)):
                rom_solves.append(sol.y)  # (num_modes, len(t_pred))
        except Exception:
            pass

    n_stable = len(rom_solves)
    n_total = ensemble_size
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
    mean_final_loss = np.mean([l[-1] for l in all_losses])
    print(f"    Mean final loss: {mean_final_loss:.6f}")

    return {
        'schema': schema,
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'n_stable': n_stable, 'n_total': n_total,
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'runtime': runtime, 'losses': all_losses,
        'rom_solves': rom_solves,
        'snaps_comp': snaps_comp, 'true_comp': true_comp,
        't_full': t_full, 't_pred': t_pred, 't_samp': t_samp,
        'training_span': TRAINING_SPAN, 'num_modes': num_modes,
        'true_states': true_states, 'basis': basis, 'fom': fom,
        'snaps_samp': snaps_samp,
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_results(result, save_dir=None):
    """Standard diagnostic figures via the centralized plotting package."""
    from core.plotting import RunResult, figures
    if save_dir is None:
        save_dir = FIGURE_DIR
    run = RunResult.from_flat(result, "05_neural_ode")
    figures.standard(run, save_dir, f"05_{result['schema']['name']}",
                     layout="windows")


def save_predictions(result, save_dir=None):
    """Save predictions for cross-method comparison."""
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, "results", "comparison", result['schema']['name'])
    os.makedirs(save_dir, exist_ok=True)

    rom_solves = result['rom_solves']
    rom_arr = np.array(rom_solves) if len(rom_solves) > 0 else np.empty((0, result['num_modes'], len(result['t_pred'])))

    method_name = "05_neural_ode"
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
    print("05 — Neural ODE — 2D Diffusion-Reaction")
    print("=" * 70)
    print(f"Regimes: {len(schemas)}")
    for s in schemas:
        print(f"  • {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  noise={s['NOISE_LEVEL']:.0%}")
    print(f"Model:  hidden={MODEL_PARAMS['HIDDEN_DIM']}, layers={MODEL_PARAMS['NUM_LAYERS']}, "
          f"lr={MODEL_PARAMS['LEARNING_RATE']}, steps={MODEL_PARAMS['NUM_TRAIN_STEPS']}, "
          f"ensemble={MODEL_PARAMS['ENSEMBLE_SIZE']}")

    results = []
    for schema in schemas:
        r = run_experiment(schema)
        plot_results(r)
        save_predictions(r)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"SUMMARY — Neural ODE (2D Diffusion-Reaction)")
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
    if args and args[0] == "--replot":
        # Replot from a saved snapshot: --replot <path/to/plot_data/PREFIX.pkl> [save_dir]
        from core.plotting import load_plot_data
        pkl_path = args[1]
        result = load_plot_data(pkl_path)
        if len(args) > 2:
            save_dir = args[2]
        else:
            # plot_data/ lives inside save_dir
            save_dir = os.path.dirname(os.path.dirname(os.path.abspath(pkl_path)))
        plot_results(result, save_dir=save_dir)
    else:
        schema_names = args if args else None
        main(schema_names)

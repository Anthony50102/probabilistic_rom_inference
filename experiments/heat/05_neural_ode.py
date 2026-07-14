"""
05 — Neural ODE Ensemble Baseline — Heat Equation

Fair comparison against Bayesian OpInf (04_unified.py):
same data generation, basis, evaluation, and plotting conventions.

A neural ODE learns dq̂/dt = f_θ(q̂, u(t)) where q̂ ∈ R^5 is the
reduced state and u(t) ∈ R^2 is the analytical input function.
An ensemble of 20 independently-trained networks provides UQ.

Architecture:  MLP  [q̂, u] ∈ R^7 → 3×128 tanh → dq̂/dt ∈ R^5
Training:      Multi-trajectory MSE, Adam 1e-3, 2000 steps
Evaluation:    Integrate over [0, 2], compare train/pred regions

Data regimes (same as 04):
  1. Dense data, low noise    (65 samples, 1% noise)
  2. Sparse data, medium noise (20 samples, 5% noise)
  3. Dense data, high noise   (65 samples, 10% noise)

Usage:
    python 05_neural_ode.py                  # run all 3 regimes
    python 05_neural_ode.py dense_low_noise  # run one regime
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from config import (
    Basis, input_func_factory,
    input_parameters, test_parameters,
)
from step1_generate_data import TrajectorySampler
from core.plotting import plot_full_order_error

# ── Data regime definitions ──────────────────────────────────────────────────
SCHEMAS = [
    {
        "name": "sparse_low_noise",
        "label": "Sparse data, low noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.01,
        "NUM_EVAL_POINTS": 100,
    },
    {
        "name": "sparse_medium_noise",
        "label": "Sparse data, medium noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.03,
        "NUM_EVAL_POINTS": 100,
    },
    {
        "name": "sparse_high_noise",
        "label": "Sparse data, high noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.05,
        "NUM_EVAL_POINTS": 100,
    },
]

# ── Model hyperparameters ────────────────────────────────────────────────────
MODEL_PARAMS = dict(
    NUM_MODES=5,
    NUM_ICS=5,
    HIDDEN_DIM=128,
    NUM_LAYERS=3,
    ACTIVATION='tanh',
    ENSEMBLE_SIZE=20,
    NUM_TRAIN_STEPS=2000,
    LEARNING_RATE=1e-3,
    SEED=42,
)

TRAINING_SPAN = (0, 1.0)
PREDICTION_SPAN = (0, 2.0)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")


# =============================================================================
# Neural ODE model
# =============================================================================
class NeuralODE(eqx.Module):
    """MLP mapping [q̂, u] → dq̂/dt."""
    layers: list

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, *, key):
        keys = jax.random.split(key, num_layers + 1)
        dims = [in_dim] + [hidden_dim] * num_layers + [out_dim]
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


# =============================================================================
# Training
# =============================================================================
def _make_jax_input_func(params):
    """JAX-compatible scalar input function u(t) → R^2."""
    a, b = params
    def _u(t):
        return jnp.array([
            a * jnp.sin(2.0 * jnp.pi * t),
            b * jnp.sin(4.0 * jnp.pi * t),
        ])
    return _u


def _solve_trajectory(model, q0, t_obs, input_func):
    """Integrate neural ODE from q0 at times t_obs with input_func.

    Returns predicted states (num_modes, len(t_obs)).
    """
    def ode_fn(t, y, args):
        u = input_func(t)
        inp = jnp.concatenate([y, u])
        return args(inp)

    term = diffrax.ODETerm(ode_fn)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_obs)
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)
    sol = diffrax.diffeqsolve(
        term, solver, t0=t_obs[0], t1=t_obs[-1], dt0=0.01,
        y0=q0, args=model, saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=16384,
        throw=False,
    )
    return sol.ys  # (len(t_obs), num_modes)


@eqx.filter_jit
def _train_step(model, opt_state, all_q0, all_t_obs, all_y_obs,
                all_input_params, opt):
    """One gradient step over all training ICs."""

    def loss_fn(model):
        total_loss = 0.0
        n_ics = len(all_q0)
        for ic in range(n_ics):
            q0 = all_q0[ic]
            t_obs = all_t_obs[ic]
            y_obs = all_y_obs[ic]
            a, b = all_input_params[ic]
            input_func = _make_jax_input_func((a, b))
            y_pred = _solve_trajectory(model, q0, t_obs, input_func)
            # y_pred: (len(t_obs), num_modes), y_obs: (num_modes, len(t_obs))
            total_loss = total_loss + jnp.mean((y_pred - y_obs.T) ** 2)
        return total_loss / n_ics

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_single_member(key, all_q0, all_t_obs, all_y_obs,
                        all_input_params, num_steps, lr,
                        num_modes, hidden_dim, num_layers):
    """Train one ensemble member, return (model, losses)."""
    in_dim = num_modes + 2  # state + 2D input
    model = NeuralODE(in_dim, num_modes, hidden_dim, num_layers, key=key)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    losses = []
    for step in range(num_steps):
        model, opt_state, loss = _train_step(
            model, opt_state,
            all_q0, all_t_obs, all_y_obs, all_input_params, opt,
        )
        losses.append(float(loss))
        if step % 500 == 0 or step == num_steps - 1:
            print(f"      step {step:5d}/{num_steps}  loss={losses[-1]:.6f}")

    return model, np.array(losses)


def train_ensemble(all_q0, all_t_obs, all_y_obs, all_input_params, p):
    """Train full ensemble. Returns list of (model, losses)."""
    ensemble_size = p['ENSEMBLE_SIZE']
    base_key = jax.random.PRNGKey(p['SEED'])
    keys = jax.random.split(base_key, ensemble_size)

    ensemble = []
    for m in range(ensemble_size):
        print(f"    ── Ensemble member {m + 1}/{ensemble_size} ──")
        model, losses = train_single_member(
            keys[m], all_q0, all_t_obs, all_y_obs, all_input_params,
            p['NUM_TRAIN_STEPS'], p['LEARNING_RATE'],
            p['NUM_MODES'], p['HIDDEN_DIM'], p['NUM_LAYERS'],
        )
        ensemble.append((model, losses))
    return ensemble


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_ensemble(ensemble, q0, t_pred, input_params):
    """Integrate all ensemble members from q0, return stable trajectories.

    Returns array of shape (n_stable, num_modes, len(t_pred)).
    """
    input_func = _make_jax_input_func(input_params)
    t_pred_jnp = jnp.array(t_pred)
    solves = []
    for model, _ in ensemble:
        try:
            y_pred = _solve_trajectory(model, jnp.array(q0), t_pred_jnp, input_func)
            y_np = np.array(y_pred)  # (len(t_pred), num_modes)
            if np.all(np.isfinite(y_np)):
                solves.append(y_np.T)  # → (num_modes, len(t_pred))
        except Exception:
            pass
    if solves:
        return np.stack(solves, axis=0)  # (n_stable, num_modes, len(t_pred))
    return np.empty((0, q0.shape[0], len(t_pred)))


# =============================================================================
# Experiment runner
# =============================================================================
def run_experiment(schema):
    """Run one data regime. Returns results dict."""
    p = MODEL_PARAMS
    np.random.seed(p['SEED'])

    noise_level = schema['NOISE_LEVEL']
    num_samples = schema['NUM_SAMPLES']
    num_eval_points = schema['NUM_EVAL_POINTS']
    num_modes = p['NUM_MODES']
    num_ics = p['NUM_ICS']
    train_params = input_parameters[:num_ics]

    print(f"\n{'='*70}")
    print(f"  {schema['label']}  ({num_samples} samples, {noise_level:.0%} noise)")
    print(f"{'='*70}")

    # ── 1. GENERATE TRAINING DATA (same as 04) ───────────────────────
    sampler = TrajectorySampler(
        training_span=TRAINING_SPAN,
        num_samples=num_samples,
        noiselevel=noise_level,
        num_regression_points=num_eval_points,
        synced=False,
    )
    (all_true_states, all_time_sampled, all_snapshots,
     all_training_inputs) = sampler.multisample(train_params)

    # ── 2. BUILD POD BASIS ────────────────────────────────────────────
    snapshots_train = np.hstack(all_snapshots)
    basis = Basis(num_vectors=num_modes)
    basis.fit(snapshots_train)
    print(f"  POD energy: {basis.cumulative_energy:.4%}")

    # ── 3. COMPRESS ALL SNAPSHOTS AND TRUTH ──────────────────────────
    all_snapshots_comp = [basis.compress(s) for s in all_snapshots]
    all_true_comp = [basis.compress(s) for s in all_true_states]

    # ── 4. PREPARE JAX TRAINING DATA ─────────────────────────────────
    all_q0 = [jnp.array(sc[:, 0]) for sc in all_snapshots_comp]
    all_t_obs = [jnp.array(ts) for ts in all_time_sampled]
    all_y_obs = [jnp.array(sc) for sc in all_snapshots_comp]
    all_input_params_jnp = [
        jnp.array([float(p[0]), float(p[1])]) for p in train_params
    ]

    # ── 5. TRAIN ENSEMBLE ─────────────────────────────────────────────
    print(f"\n  Training {p['ENSEMBLE_SIZE']} ensemble members "
          f"({p['NUM_TRAIN_STEPS']} steps each)...")
    t0 = time.time()
    ensemble = train_ensemble(
        all_q0, all_t_obs, all_y_obs, all_input_params_jnp, p,
    )
    runtime = time.time() - t0
    print(f"  Training time: {runtime:.0f}s")

    # ── 6. EVALUATION (same structure as 04) ──────────────────────────
    t_pred = np.linspace(PREDICTION_SPAN[0], PREDICTION_SPAN[1], 400)
    max_samp = p['ENSEMBLE_SIZE']

    # Generate test IC data
    test_sampler = TrajectorySampler(
        training_span=TRAINING_SPAN,
        num_samples=num_samples,
        noiselevel=noise_level,
        num_regression_points=num_eval_points,
        synced=False,
    )
    (test_true_list, test_t_list, test_snap_list,
     test_inp_list) = test_sampler.multisample([test_parameters])
    test_true_comp = basis.compress(test_true_list[0])
    test_snaps_comp = basis.compress(test_snap_list[0])
    test_t_samp = test_t_list[0]

    # Evaluate all ICs (5 train + 1 test)
    eval_params = list(train_params) + [test_parameters]
    eval_snaps_comp = all_snapshots_comp + [test_snaps_comp]
    eval_true_comp = all_true_comp + [test_true_comp]
    eval_t_samp = all_time_sampled + [test_t_samp]
    eval_labels = [f"Train IC {i} {train_params[i]}" for i in range(num_ics)] + \
                  [f"Test IC {test_parameters}"]

    all_rom_solves = []
    all_n_stable = []
    all_train_errors = []
    all_pred_errors = []
    train_mask = t_pred <= TRAINING_SPAN[1]
    pred_mask = t_pred > TRAINING_SPAN[1]

    for ic_idx, (params, true_c) in enumerate(zip(eval_params, eval_true_comp)):
        q0 = eval_snaps_comp[ic_idx][:, 0]
        ic_solves = evaluate_ensemble(ensemble, q0, t_pred, params)
        all_rom_solves.append(ic_solves)
        all_n_stable.append(len(ic_solves))

        ti = interp1d(config.time_domain, true_c,
                      kind='cubic', fill_value='extrapolate')
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
        label = eval_labels[ic_idx]
        print(f"    {label}: {all_n_stable[-1]}/{max_samp} stable, "
              f"train={te:.4%}, pred={pe:.4%}")

    # Aggregate over training ICs
    train_ic_stable = sum(all_n_stable[:num_ics])
    train_ic_total = max_samp * num_ics
    train_errors_fin = [e for e in all_train_errors[:num_ics] if np.isfinite(e)]
    pred_errors_fin = [e for e in all_pred_errors[:num_ics] if np.isfinite(e)]
    train_error = np.mean(train_errors_fin) if train_errors_fin else float('inf')
    pred_error = np.mean(pred_errors_fin) if pred_errors_fin else float('inf')
    stability_pct = train_ic_stable / max(train_ic_total, 1) * 100

    print(f"\n    Overall: {train_ic_stable}/{train_ic_total} ({stability_pct:.0f}%)")
    print(f"    Avg train: {train_error:.4%}  |  Avg pred: {pred_error:.4%}")
    print(f"    Test IC: {all_n_stable[-1]}/{max_samp} stable, "
          f"train={all_train_errors[-1]:.4%}, pred={all_pred_errors[-1]:.4%}")

    # CI coverage
    ci_width = ci_coverage = float('nan')
    if train_ic_stable > 0:
        all_in_ci, all_widths = [], []
        for ic_idx in range(num_ics):
            if all_n_stable[ic_idx] > 0:
                ti = interp1d(config.time_domain, eval_true_comp[ic_idx],
                              kind='cubic', fill_value='extrapolate')
                ta = ti(t_pred)
                q05 = np.percentile(all_rom_solves[ic_idx], 5, axis=0)
                q95 = np.percentile(all_rom_solves[ic_idx], 95, axis=0)
                all_widths.append(np.mean(q95 - q05))
                all_in_ci.append(np.mean((ta >= q05) & (ta <= q95)))
        if all_widths:
            ci_width = np.mean(all_widths)
            ci_coverage = np.mean(all_in_ci)
            print(f"    CI coverage: {ci_coverage:.2%} (target: 90%)")

    # Collect per-member losses for plotting
    all_member_losses = np.stack([losses for _, losses in ensemble], axis=0)

    return {
        'schema': schema,
        'train_error': train_error, 'pred_error': pred_error,
        'stability_pct': stability_pct,
        'n_stable': train_ic_stable, 'n_total': train_ic_total,
        'test_train_error': all_train_errors[-1],
        'test_pred_error': all_pred_errors[-1],
        'test_n_stable': all_n_stable[-1],
        'ci_coverage': ci_coverage, 'ci_width': ci_width,
        'runtime': runtime,
        'all_member_losses': all_member_losses,
        # Per-IC data for multi-trajectory plotting
        'all_rom_solves': all_rom_solves,
        'all_snaps_comp': eval_snaps_comp,
        'all_true_comp': eval_true_comp,
        'all_t_samp': eval_t_samp,
        'all_n_stable': all_n_stable,
        'eval_labels': eval_labels,
        't_full': config.time_domain,
        't_pred': t_pred,
        'training_span': TRAINING_SPAN,
        'num_modes': p['NUM_MODES'],
        'max_samp': max_samp,
        'basis': basis,
        'all_true_states_full': all_true_states,
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_results(result, save_dir=None):
    """Standard diagnostic figures via the centralized plotting package."""
    from core.plotting import RunResult, figures
    if save_dir is None:
        save_dir = FIGURE_DIR
    run = RunResult.from_multi(result, "05_neural_ode")
    figures.standard(run, save_dir, f"05_{result['schema']['name']}",
                     layout="windows")


def save_predictions(result, save_dir=None):
    """Save predictions for cross-method comparison."""
    schema = result['schema']
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, "results", "comparison", schema['name'])
    os.makedirs(save_dir, exist_ok=True)

    all_rom_solves = result['all_rom_solves']
    method_name = "05_neural_ode"

    save_dict = {
        't_pred': result['t_pred'],
        'train_error': result['train_error'],
        'pred_error': result['pred_error'],
        'stability_pct': result['stability_pct'],
        'ci_coverage': result.get('ci_coverage', float('nan')),
        'ci_width': result.get('ci_width', float('nan')),
        'runtime': result['runtime'],
        'n_ics': len(all_rom_solves),
    }
    for ic_idx, solves in enumerate(all_rom_solves):
        if len(solves) > 0:
            save_dict[f'rom_solves_{ic_idx}'] = np.array(solves)
        else:
            save_dict[f'rom_solves_{ic_idx}'] = np.empty((0, result['num_modes'], len(result['t_pred'])))

    path = os.path.join(save_dir, f"{method_name}.npz")
    np.savez(path, **save_dict)
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
    print("05 — Neural ODE Ensemble — Cubic Heat Equation")
    print("=" * 70)
    print(f"Regimes: {len(schemas)}")
    for s in schemas:
        print(f"  • {s['label']:30s}  samples={s['NUM_SAMPLES']:3d}  noise={s['NOISE_LEVEL']:.0%}")
    print(f"Model:  {MODEL_PARAMS['NUM_LAYERS']}×{MODEL_PARAMS['HIDDEN_DIM']} "
          f"{MODEL_PARAMS['ACTIVATION']}, ensemble={MODEL_PARAMS['ENSEMBLE_SIZE']}, "
          f"lr={MODEL_PARAMS['LEARNING_RATE']}, steps={MODEL_PARAMS['NUM_TRAIN_STEPS']}")
    print(f"ICs:    {MODEL_PARAMS['NUM_ICS']} training + 1 test ({test_parameters})")

    results = []
    for schema in schemas:
        r = run_experiment(schema)
        plot_results(r)
        save_predictions(r)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"SUMMARY — Neural ODE Ensemble (Heat)")
    print(f"{'='*90}")
    print(f"{'Regime':<28s} {'Samp':>4s} {'Noise':>5s} {'Stab':>5s} "
          f"{'Train':>8s} {'Pred':>8s} {'TestPr':>8s} {'CI_cov':>7s} {'Time':>6s}")
    print(f"{'-'*28} {'-'*4} {'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
    for r in results:
        s = r['schema']
        print(f"{s['label']:<28s} {s['NUM_SAMPLES']:>4d} "
              f"{s['NOISE_LEVEL']:>4.0%} {r['stability_pct']:>4.0f}% "
              f"{r['train_error']:>7.2%} {r['pred_error']:>7.2%} "
              f"{r['test_pred_error']:>7.2%} "
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

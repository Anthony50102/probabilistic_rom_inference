"""Quick targeted test: try Euler-derived insights on Heat 04."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import experiments.heat.config  # noqa — ensure heat config loaded first
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import the production module
from importlib import import_module
mod = import_module('04_conditional_integral')

SCHEMA = {"name": "dense_low_noise", "label": "Dense data, low noise",
          "NUM_SAMPLES": 65, "NOISE_LEVEL": 0.01, "NUM_EVAL_POINTS": 150}

CONFIGS = {
    "baseline": dict(
        NUM_MODES=5, NUM_ICS=5, GAMMA=2.0, GAMMA2=0.5, DERIV_WEIGHT=1.0,
        INTEGRAL_WEIGHT=1.0, MLL_WEIGHT=0.1, GP_PRIOR_SCALE=0.1,
        WINDOW_SIZE=10, NUM_STEPS=10000, LEARNING_RATE=3e-3,
        NUM_POSTERIOR_SAMPLES=500, SEED=42),
    "euler_best": dict(
        NUM_MODES=5, NUM_ICS=5, GAMMA=8.0, GAMMA2=1.0, DERIV_WEIGHT=1.0,
        INTEGRAL_WEIGHT=5.0, MLL_WEIGHT=1.0, GP_PRIOR_SCALE=0.1,
        WINDOW_SIZE=20, NUM_STEPS=10000, LEARNING_RATE=0.01,
        NUM_POSTERIOR_SAMPLES=500, SEED=42),
    "high_lr_only": dict(
        NUM_MODES=5, NUM_ICS=5, GAMMA=2.0, GAMMA2=0.5, DERIV_WEIGHT=1.0,
        INTEGRAL_WEIGHT=1.0, MLL_WEIGHT=0.1, GP_PRIOR_SCALE=0.1,
        WINDOW_SIZE=10, NUM_STEPS=10000, LEARNING_RATE=0.01,
        NUM_POSTERIOR_SAMPLES=500, SEED=42),
    "big_window_only": dict(
        NUM_MODES=5, NUM_ICS=5, GAMMA=2.0, GAMMA2=0.5, DERIV_WEIGHT=1.0,
        INTEGRAL_WEIGHT=1.0, MLL_WEIGHT=0.1, GP_PRIOR_SCALE=0.1,
        WINDOW_SIZE=20, NUM_STEPS=10000, LEARNING_RATE=3e-3,
        NUM_POSTERIOR_SAMPLES=500, SEED=42),
}

names = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIGS.keys())
results = {}
for name in names:
    if name not in CONFIGS:
        print(f"Unknown config: {name}")
        continue
    cfg = CONFIGS[name]
    print(f"\n{'#'*70}")
    print(f"  CONFIG: {name}")
    print(f"  GAMMA={cfg['GAMMA']}, GAMMA2={cfg['GAMMA2']}, LR={cfg['LEARNING_RATE']}, "
          f"WS={cfg['WINDOW_SIZE']}, IW={cfg['INTEGRAL_WEIGHT']}, MW={cfg['MLL_WEIGHT']}")
    print(f"{'#'*70}")

    mod.MODEL_PARAMS = cfg
    t0 = time.time()
    r = mod.run_experiment(SCHEMA)
    elapsed = time.time() - t0
    results[name] = (r, elapsed)
    print(f"  → train={r['train_error']:.4%} pred={r['pred_error']:.4%} "
          f"ci={r['ci_coverage']:.1%} stable={r['stability_pct']:.0%} ({elapsed:.0f}s)")

print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"{'Config':<20} {'Train':>8} {'Pred':>8} {'CI':>6} {'Stable':>6} {'Time':>6}")
print("-" * 60)
for name, (r, t) in results.items():
    print(f"{name:<20} {r['train_error']:>7.2%} {r['pred_error']:>7.2%} "
          f"{r['ci_coverage']:>5.1%} {r['stability_pct']:>5.0%} {t:>5.0f}s")

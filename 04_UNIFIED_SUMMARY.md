# 04 Unified — Marginalised-O × Weak-Form Bayesian OpInf

`04_unified.py` is the active Bayesian OpInf method used across the PDE
experiments. It combines Gaussian-process smoothing, weak-form constraints, and
closed-form operator marginalisation.

## Current method

For each ROM mode `i`, the GP posterior gives a derivative posterior

```text
Z_i | data, θ_i ~ N(μ_{z,i}, Σ_{z,i}).
```

The operator row `O_i` is constrained by two linear-in-`O_i` blocks.

### Derivative block

```text
μ_{z,i} ≈ f(X) O_i^T,
Σ_D,i = Σ_{z,i} + γ² I.
```

### Weak-form block

Let `Ψ_w[k, j] = w_j ψ_k(t_j)` be the quadrature-weighted test-function matrix.
The weak-form data use the derivative representation

```text
w_i = Ψ_w μ_{z,i},
Ψ(X)[k, :] = ∫ ψ_k(t) d(X(t), u(t))^T dt.
```

The weak-form covariance propagates the same GP derivative uncertainty:

```text
Σ_W,i = Ψ_w Σ_{z,i} Ψ_w^T + γ² diag(∫ ψ_k(t)^2 dt).
```

Thus both likelihood blocks are "GP derivative covariance + slack" in their
respective spaces. The resulting per-mode Gaussian linear model is

```text
y_i = A(X) O_i^T + η_i,
η_i ~ N(0, blockdiag(Σ_D,i, Σ_W,i)).
```

With `O_i ~ N(0, σ_O² I)`, the conditional posterior of `O_i` and the marginal
likelihood are available in closed form. SVI therefore only explores the GP
hyperparameters (and optional hyperparameters such as hierarchical `σ_O`).

## Active experiments

| Experiment | Script | Operators | Distinguishing features |
|---|---|---|---|
| Euler | `experiments/euler/04_unified.py` | `cAH` | Single-trajectory autonomous quadratic ROM; broad GP priors. |
| Burgers 2D | `experiments/burgers_2d/04_unified.py` | `cAH` | Single-trajectory diffusion-reaction case; MLE-anchored GP priors and trace-based ridge. |
| Heat | `experiments/heat/04_unified.py` | `cAHBN` | Multi-IC shared operator; input-dependent ROM; lifted/shifted basis; deterministic OpInf prior center with stability shift. |
| Tumor | `experiments/tumor/04_unified.py` | `cA` | TumorTwin cached FOM data; autonomous growth; adaptive POD via GP SNR threshold. |

## Verified 04 results

These metrics are from the current saved `results/comparison/<schema>/04_unified.npz`
files after rerunning the full-covariance derivative/weak-form implementation.

| Experiment | Regime | Stability | Train error | Prediction error | CI coverage | Runtime |
|---|---|---:|---:|---:|---:|---:|
| Euler | dense low noise | 100.0% | 1.20% | 10.13% | 99.9% | 303s |
| Euler | sparse low noise | 96.0% | 73.32% | 70.48% | 52.0% | 61s |
| Euler | dense high noise | 100.0% | 40.28% | 72.26% | 53.0% | 323s |
| Burgers 2D | dense medium noise | 39.5% | 0.90% | 2.93% | 95.9% | 31s |
| Heat | sparse low noise | 100.0% | 3.83% | 6.01% | 68.2% | 134s |
| Heat | sparse medium noise | 100.0% | 3.90% | 5.99% | 63.8% | 129s |
| Heat | sparse high noise | 100.0% | 4.32% | 6.34% | 63.0% | 129s |
| Tumor | dense low noise | 100.0% | 2.85% | 7.35% | 93.8% | 42s |
| Tumor | dense medium noise | 100.0% | 3.42% | 4.60% | 65.0% | 33s |
| Tumor | dense high noise | 100.0% | 6.89% | 15.81% | 51.9% | 28s |

## Plot regeneration

Each `04_unified.py` run writes a `.npz` file under
`experiments/<pde>/results/comparison/<schema>/04_unified.npz`. Regenerate the
per-method 04 plots with:

```bash
conda run -n prob_rom python plot_from_npz.py \
  experiments/euler/results/comparison/dense_low_noise/04_unified.npz \
  experiments/euler/figures
```

The plotter writes schema-prefixed files such as:

```text
04_dense_low_noise_rom_trajectories.png
04_dense_low_noise_loss.png
04_dense_low_noise_operator_traces.png
04_dense_low_noise_full_order_error.png
```

Single-IC experiments also get `04_<schema>_rom_notebook.png`. Heat is multi-IC
and uses the IC-by-mode trajectory grid instead.

## Notes and limitations

- **Euler sparse/high-noise regimes** currently have high prediction error and
  under-coverage despite stable solves.
- **Burgers 2D** has low median error but low stable-solve fraction, indicating
  heavy-tailed operator samples.
- **Heat** is stable across all tested regimes but under-covers relative to the
  nominal 90% interval.
- **Tumor** performs well at low noise but under-covers at higher noise after
  adaptive SNR-based mode truncation.
- **FitzHugh-Nagumo** is not part of the active experiment set.

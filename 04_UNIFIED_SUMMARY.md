# 04 Unified — Marginalised-O × Weak-Form Bayesian OpInf

> The method lives in `core/weakform_opinf/` (see `README.md` for the
> architecture and the shared `WeakFormConfig` defaults). The results below are
> from the standardized-settings rerun (shared settings, QR operator solve,
> per-experiment closure constant). The pre-standardization outputs are kept in
> the `results_pre_standard_settings` folders and under the git tag
> `pre-standard-settings`.

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
Σ_D,i = diag(Σ_{z,i}) + γ² I,   γ² = c (S/T)².
```

`S/T` is the reference rate of the nondimensional operator prior (RMS training
POD coefficient over the training window), so `c` (`gamma2_nd`) is a
dimensionless closure constant. It is the one per-experiment setting, selected
on development data (README): `c = 1` for Euler and heat, `0.03` for Burgers 2D
(diffusion-reaction) and `0.1` for the tumor tasks.

### Weak-form block

Let `Ψ_w[k, j] = w_j ψ_k(t_j)` and `Ψ̇_w[k, j] = w_j ψ̇_k(t_j)` be the
quadrature-weighted test functions and their derivatives. By integration by
parts (the default, `weakform_mode="ibp"`) the weak-form data use the GP state

```text
w_i = -Ψ̇_w μ_{x,i},
Ψ(X)[k, :] = ∫ ψ_k(t) d(X(t), u(t))^T dt,
```

and the covariance propagates the GP state uncertainty plus the closure slack,
kept diagonal by default (`weakform_cov="diag"`):

```text
Σ_W,i = diag(Ψ̇_w Σ_{x,i} Ψ̇_w^T) + γ² diag(Σ_j (w_j ψ_k(t_j))²).
```

The slack term is the variance that the derivative block's independent
`N(0, γ²)` closure errors induce on the weak functionals through the
quadrature. (`weakform_mode="deriv"` uses `w_i = Ψ_w μ_{z,i}` and
`Ψ_w Σ_{z,i} Ψ_w^T` instead.)

Thus both likelihood blocks are "GP covariance + closure slack" in their
respective spaces. The resulting per-mode Gaussian linear model is

```text
y_i = A(X) O_i^T + η_i,
η_i ~ N(0, blockdiag(Σ_D,i, Σ_W,i)).
```

With the Gaussian prior `O_i ~ N(0, Σ_O)`, `Σ_O = diag(κ_b(j)² s_j²)` (the
nondimensional column scales `s_j` and per-block multipliers `κ_b` described in
the README), the conditional posterior of `O_i` and the marginal likelihood are
available in closed form. SVI therefore only explores the GP hyperparameters and
the block multipliers `log κ_b`.

## Active experiments

Every script runs the shared `WeakFormConfig` defaults (README); only the ROM
structure, the data and the closure constant `gamma2_nd` differ.

| Experiment | Script | Operators | Distinguishing features |
|---|---|---|---|
| Euler | `experiments/euler/04_unified.py` | `cAH` | Single-trajectory autonomous quadratic ROM, `r = 6`. |
| Burgers 2D | `experiments/burgers_2d/04_unified.py` | `cAH` | Single-trajectory diffusion-reaction case, `r = 3`; modal variances spread over orders of magnitude. |
| Heat | `experiments/heat/04_unified.py` | `cAHBN` | Multi-trajectory shared operator; input-dependent ROM; lifted/shifted basis, `r = 5`. |
| Tumor | `experiments/tumor/04_unified_benchmark.py` | `cA` / `cAN` | Three segmented TumorTwin tasks: untreated growth (acquisitions 42–44; `cA`, mean-centred `r = 3`) and single- and multi-dose chemotherapy (51–53 and 48–50; `cABN` requested; the uncentred `r = 4` basis gives `cAN` by the Galerkin rule in `benchmark_cases.galerkin_operators`). See `experiments/TUMOR_SEGMENTED_BENCHMARKS.md`. |

## Current results

Reduced-coordinate metrics recomputed from the saved
`results/comparison/<schema>/04_unified.npz` and `05_neural_ode.npz` files by
`experiments/aggregate_table.py` (`experiments/results/aggregate/`). Errors are
relative errors of the median forecast against the reduced truth, excluding the
POD residual; coverage is that of the 5–95% band over the forecast window
(nominal 90%). Each regime is one noise realisation. Heat rows average the five
training forcings; the held-out rows score the test forcing `(a, b) = (1.5, 0.5)`.

| Experiment | Regime | Stable | Train | Forecast | Coverage | NODE forecast | NODE coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| Euler | dense low noise | 100% | 1.21% | 7.89% | 98.0% | 55.85% | 41.3% |
| Euler | sparse low noise | 100% | 3.77% | 22.26% | 80.3% | 22.39% | 94.9% |
| Euler | dense high noise | 100% | 7.14% | 18.49% | 98.6% | 32.84% | 46.9% |
| Heat | sparse low noise | 100% | 1.02% | 2.13% | 99.7% | 16.94% | 34.9% |
| Heat | sparse medium noise | 100% | 1.42% | 2.69% | 100.0% | 17.10% | 42.8% |
| Heat | sparse high noise | 100% | 2.30% | 3.52% | 99.1% | 19.28% | 36.8% |
| Heat (held-out) | sparse low noise | 100% | 0.96% | 2.55% | 100.0% | 30.82% | 48.8% |
| Heat (held-out) | sparse medium noise | 100% | 1.10% | 3.28% | 100.0% | 30.64% | 54.4% |
| Heat (held-out) | sparse high noise | 100% | 1.62% | 3.88% | 100.0% | 32.25% | 58.6% |
| Burgers 2D | dense medium noise | 99.5% | 0.36% | 2.83% | 100.0% | 25.13% | 12.0% |

The tumor tasks are scored in the full-order field; their results are in
`experiments/TUMOR_SEGMENTED_BENCHMARKS.md`.

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

- **Euler sparse low noise** is the weakest PDE regime: the forecast error is
  level with the Neural ODE ensemble's and the band under-covers (80%). It is
  also where the standardized settings lost most against the earlier hand-set
  ones (16.7% before).
- **Coverage** is mostly conservative (98–100%) on the PDE regimes; on the
  tumor tasks the bands can under-cover badly (see the tumor document).
- **Burgers 2D**: one of the 200 posterior draws is non-finite; the median and
  band use the stable draws.
- **FitzHugh-Nagumo** is not part of the active experiment set.

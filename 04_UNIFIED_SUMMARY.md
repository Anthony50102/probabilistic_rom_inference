# 04 Unified — Marginalised-O × Weak-Form Bayesian OpInf

> The method lives in `core/weakform_opinf/` (see `README.md` for the
> architecture and the shared `WeakFormConfig` defaults). The results below are
> from the tempered-closure rerun (shared settings, QR operator solve,
> dynamics rows tempered by the GP's effective degrees of freedom, no
> per-experiment constant). The outputs with the earlier per-experiment closure
> constant are kept in the `results_pre_closure_tempering` folders and under the
> git tag `pre-closure-tempering`; the pre-standardization outputs in the
> `results_pre_standard_settings` folders and under `pre-standard-settings`.

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
Σ_D,i = diag(Σ_{z,i}) / α_i.
```

`α_i` is the tempering exponent below; each variance is floored at the
round-off level of the kernel nugget.

### Weak-form block

Let `Ψ_w[k, j] = w_j ψ_k(t_j)` and `Ψ̇_w[k, j] = w_j ψ̇_k(t_j)` be the
quadrature-weighted test functions and their derivatives. By integration by
parts (the default, `weakform_mode="ibp"`) the weak-form data use the GP state

```text
w_i = -Ψ̇_w μ_{x,i},
Ψ(X)[k, :] = ∫ ψ_k(t) d(X(t), u(t))^T dt,
```

and the covariance propagates the GP state uncertainty, kept diagonal by
default (`weakform_cov="diag"`):

```text
Σ_W,i = diag(Ψ̇_w Σ_{x,i} Ψ̇_w^T) / α_i.
```

(`weakform_mode="deriv"` uses `w_i = Ψ_w μ_{z,i}` and `Ψ_w Σ_{z,i} Ψ_w^T`
instead; it requires `closure="slack"`.)

### Tempering

The `n_e + K` rows of a mode are correlated functionals of one GP fit, and the
diagonal blocks treat them as independent. Each mode's dynamics likelihood is
therefore raised to the power

```text
α_i = min(1, df_i / (n_e + K)),   df_i = tr K (K + ν_i I)⁻¹,
```

the effective degrees of freedom of the mode's GP smoother at the training
times, computed once from a data-only GP fit and then fixed. Dividing the row
variances by `α_i`, as above, plus the normaliser
`−½ Σ [(α_i − 1) log 2πu + log α_i]` in the evidence, gives the power
likelihood exactly. There is no closure variance and no per-experiment
constant (README). The resulting per-mode Gaussian linear model is

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
structure and the data differ.

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
| Euler | dense low noise | 100% | 1.28% | 7.05% | 95.6% | 55.85% | 41.3% |
| Euler | sparse low noise | 100% | 7.57% | 27.92% | 99.8% | 22.39% | 94.9% |
| Euler | dense high noise | 100% | 8.25% | 12.57% | 100.0% | 32.84% | 46.9% |
| Heat | sparse low noise | 100% | 0.60% | 1.87% | 89.7% | 16.94% | 34.9% |
| Heat | sparse medium noise | 100% | 1.19% | 2.27% | 96.5% | 17.10% | 42.8% |
| Heat | sparse high noise | 100% | 2.18% | 2.83% | 98.4% | 19.28% | 36.8% |
| Heat (held-out) | sparse low noise | 100% | 0.56% | 2.30% | 75.4% | 30.82% | 48.8% |
| Heat (held-out) | sparse medium noise | 100% | 1.04% | 2.48% | 94.1% | 30.64% | 54.4% |
| Heat (held-out) | sparse high noise | 100% | 2.25% | 2.80% | 100.0% | 32.25% | 58.6% |
| Burgers 2D | dense medium noise | 100% | 0.35% | 2.36% | 100.0% | 25.13% | 12.0% |

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

- **Tempered dynamics rows.** These are the results of the dof-tempered
  likelihood (`closure="tempered"`, the default after the
  `pre-closure-tempering` tag). Against the per-experiment closure constant
  that it replaced, the forecast error is lower in nine of the ten rows and
  higher for Euler sparse low noise (22.26% before); the old rows were Euler
  7.89/22.26/18.49%, heat 2.13/2.69/3.52%, held-out heat 2.55/3.28/3.88%, and
  Burgers 2D 2.83%.
- **Euler sparse low noise** is the weakest PDE regime and the only one where
  the Neural ODE ensemble's forecast is more accurate (22.39% against
  27.92%). It is also where the standardized settings lost most against the
  earlier hand-set ones (16.7% before).
- **Coverage** is conservative (94–100%) on the PDE regimes except heat at 1%
  noise (89.7% on the training forcings, 75.4% at the held-out forcing). On
  the tumor tasks the burden bands contain the truth at every forecast time
  for multi-dose, at 74–99% of times for single-dose, and at none for
  untreated growth (see the tumor document).
- **All posterior draws are finite** in every regime.
- **FitzHugh-Nagumo** is not part of the active experiment set.

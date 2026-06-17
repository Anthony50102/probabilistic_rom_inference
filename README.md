# Probabilistic Reduced Order Model Inference

Implementation and comparison code for Bayesian operator inference and Neural
ODE reduced-order models (ROMs) learned from noisy PDE snapshot data.

The active experiment pipeline compares:

1. **Bayesian OpInf** (`04_unified.py`): Gaussian-process smoothing with
   analytically marginalised ROM operators and derivative/weak-form constraints.
2. **Neural ODE ensemble** (`05_neural_ode.py`): black-box reduced dynamics
   baseline with ensemble uncertainty bands.
3. **Comparison plots** (`06_compare_methods.py`): method-level metrics and
   full-order error comparisons from saved `.npz` outputs.

## Active PDE experiments

| Experiment | Active system | ROM operators | Notes |
|---|---|---|---|
| `euler` | Compressible Euler | `cAH` | Single trajectory, autonomous quadratic ROM. |
| `heat` | Cubic heat equation | `cAHBN` | Multi-IC, input-dependent ROM with lifted/shifted basis. |
| `burgers_2d` | 2D diffusion-reaction / Burgers-style system | `cAH` | Single trajectory plus optional parametric extension scripts. |
| `tumor` | TumorTwin tumor-growth data | `cA` | Cached FOM data, adaptive POD by GP SNR threshold. |

## Repository structure

```text
core/
  bayesian_opinf.py    # GP fitting, derivative covariance, Bayesian OpInf utilities
  bgp_jax.py           # JAX/NumPyro GP kernels and derivative kernels
  diagnostics.py       # posterior diagnostics and trace plotting helpers
  pde_models.py        # full-order PDE model implementations
  plotting.py          # shared plotting and metrics helpers
  utils.py             # data generation and utility functions

experiments/
  euler/
    04_unified.py
    05_neural_ode.py
    06_compare_methods.py
    config.py

  heat/
    04_unified.py
    05_neural_ode.py
    06_compare_methods.py
    config.py
    heat_rom.py
    step1_generate_data.py

  burgers_2d/
    04_unified.py
    05_neural_ode.py
    06_compare_methods.py
    07_parametric_ics.py
    08_parametric_neural_ode.py
    config.py
    config_parametric.py

  tumor/
    04_unified.py
    05_neural_ode.py
    05_neural_ode_chemo.py
    06_compare_methods.py
    config.py
    generate_fom_data.py
    generate_fom_data_chemo.py
    generate_fom_data_multi.py
    generate_paper.py

plot_from_npz.py       # standalone plot regeneration from saved 04_unified.npz files
```

## Bayesian OpInf method

The active `04_unified.py` method fits GP posteriors to reduced coordinates and
uses the GP derivative posterior to constrain the ROM operator. The operator is
analytically marginalised, so inference explores only GP hyperparameters and
recovers a closed-form conditional Gaussian posterior for each row of the
operator matrix.

For each ROM mode, the derivative block uses the full GP derivative covariance

```text
Σ_D = Σ_z + γ² I,
```

and the weak-form block propagates the same derivative covariance through the
test functions:

```text
Σ_W = Ψ_w Σ_z Ψ_wᵀ + γ² diag(∫ ψ_k(t)^2 dt).
```

This keeps both pointwise derivative and weak-form constraints grounded in the
same GP derivative uncertainty, with additive slack for model-form error.

## Neural ODE baseline

The `05_neural_ode.py` scripts train ensembles of reduced-state neural ODEs on
the same data regimes as the Bayesian OpInf method where implemented. The
comparison scripts treat Neural ODE outputs as method-level `.npz` files in the
same `results/comparison/<schema>/` layout.

## Running experiments

Use the `prob_rom` conda environment.

Run one Bayesian OpInf regime:

```bash
cd experiments/euler
conda run -n prob_rom python 04_unified.py dense_low_noise
```

Run the Neural ODE baseline for the same regime:

```bash
conda run -n prob_rom python 05_neural_ode.py dense_low_noise
```

Generate method-comparison plots:

```bash
conda run -n prob_rom python 06_compare_methods.py dense_low_noise
```

Regenerate standalone Bayesian OpInf plots from a saved result file:

```bash
cd ../..
conda run -n prob_rom python plot_from_npz.py \
  experiments/euler/results/comparison/dense_low_noise/04_unified.npz \
  experiments/euler/figures
```

## Generated outputs

Generated outputs are intentionally ignored by git:

- `experiments/**/figures*/`
- `experiments/**/results*/`
- `experiments/**/data/*.npz`
- `*.npz`, `*.npy`, `*.png`, `*.pkl`

The `figures_rerun_paper_v4/` directories are preserved historical paper rerun
artifacts. Current `04_unified.py` reruns write to `results/comparison/` and can
be replotted with `plot_from_npz.py`.

## Requirements

The code relies on NumPy/SciPy, Matplotlib, JAX, NumPyro, Diffrax/Equinox/Optax
for Neural ODEs, and `opinf` for ROM model scaffolding. See
`requirements.txt` and the `prob_rom` environment for the working package set.

## Citation

Citation information will be added upon publication.

## License

See [LICENSE](LICENSE) for details.

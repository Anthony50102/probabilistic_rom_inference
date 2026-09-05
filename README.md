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
| `tumor` | TumorTwin tumor-growth data | `cA` | Cached FOM data, fixed POD mode count. |
| `tumor` (chemo) | Tumor growth with chemotherapy | `cABN` | Input-driven ROM via `04_unified_chemo.py`. |

## Repository structure

```text
core/
  bayesian_opinf.py    # GP fitting, derivative covariance, Bayesian OpInf utilities
  bgp_jax.py           # JAX/NumPyro GP kernels and derivative kernels
  diagnostics.py       # posterior diagnostics and trace plotting helpers
  pde_models.py        # full-order PDE model implementations
  plotting/           # shared results, figures, comparisons, and physical plots
  utils.py             # data generation and utility functions
  weakform_opinf/      # canonical Bayesian algorithm, configuration, and pipeline

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
    04_unified_chemo.py
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

All five `04_unified*.py` experiments are thin adapters over
`core/weakform_opinf/`. `WeakFormConfig` defines the method settings;
`ExperimentSpec.prepare()` supplies the data, POD basis, ROM, and evaluation
targets. Single- and multi-trajectory cases use the same inference pipeline.

The operator is analytically marginalised. By default, SVI with an `AutoNormal`
guide infers GP hyperparameters and per-operator-block hierarchical prior
scales, then recovers a conditional Gaussian posterior for each operator row.
NUTS is also supported. GP priors are spectrum-anchored, not MLE-fitted; operator
priors are zero-mean, including heat (no least-squares prior center or stability
shift).

Pointwise derivative constraints are combined with state-based weak-form
constraints via integration by parts. The default derivative and weak-form
covariance blocks are diagonal, with additive model-error slack; full blocks
are optional configuration choices. The cross-block covariance is omitted.

Each evaluation target carries its own initial state, observations, time grid,
and training-trajectory association. IC uncertainty uses that trajectory's GP
hyperparameters; held-out targets use training-average hyperparameters on their
own observation grid, without fitting another GP. Heat headline metrics cover
training ICs only, with held-out metrics reported separately as `test_*`.

Saved results preserve truth and observation arrays for standalone plotting.
Single-IC files use `rom_solves`, `true_comp`, `snaps_comp`, and `t_samp`;
multi-IC files use indexed keys for every target plus `n_ics` and `eval_labels`.
Heat also retains `basis_shift`, and chemotherapy files retain dose/input
metadata.

## Neural ODE baseline

The `05_neural_ode.py` scripts train ensembles of reduced-state neural ODEs on
the same data regimes as the Bayesian OpInf method where implemented. The
comparison scripts treat Neural ODE outputs as method-level `.npz` files in the
same `results/comparison/<schema>/` layout.

## Shared plotting

`core/plotting/` supplies `RunResult`/`TargetResult`, per-run figures,
method-comparison charts, GP diagnostics, and physical-space paper plots.
Legacy `from core.plotting import ...` imports remain supported.

Multi-IC runs retain each target's observations and produce a trajectory
figure per target: the primary target keeps `<prefix>_rom_trajectories.png`;
subsequent targets use `<prefix>_ic_<index>_rom_trajectories.png`. Per-IC error
charts distinguish the evaluated trajectories, and heat comparison bars retain
the training/held-out split. Chemo Neural ODE figures use `05_chemo_<schema>`
to avoid overwriting autonomous tumor figures. Tumor entry points also produce
their spatial, volume, and (for chemo) uncertainty diagnostics.

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

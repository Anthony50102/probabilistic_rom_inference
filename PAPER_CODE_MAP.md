# Map from the paper to the code

Where each part of the manuscript (`GP-Bayes-Refactor/manuscript_v2`) is
implemented. Paths are relative to this repository. Unless a path is given, a
name refers to `core/weakform_opinf/`. Use the function names, not line numbers,
to find the code; the manuscript's Appendix A has the same table.

## Method (Section 2)

| Paper | Code |
|---|---|
| Reduced model and operator dictionary, Eqs. `rom`, `operator_blocks` | operator string `WeakFormConfig.operators` (`config.py`); data matrix `rom.model._assemble_data_matrix` (opinf model) |
| Graphical model and joint distribution, Eq. `joint` (Sec. 2.2) | `build_model` in `model.py` (the NumPyro `model` inside it) |
| Gaussian-process hyperpriors, Eqs. `gp_priors`, `prior_locs` (Sec. 2.3) | `spectrum_anchored_prior_locs` (`gp.py`); `_prior_locs`, `_sample_gp_hypers` (`model.py`) |
| Weak-form test functions and quadrature, Eqs. `bump`, `weak_quad` (Sec. 2.4) | `build_test_functions` (`weakform.py`) |
| GP state, derivative and weak-form conditionals, Eqs. `gp_state_posterior`, `gp_derivative_posterior`, `kernel_blocks` (Sec. 2.5) | `make_gp_conditional`, `trajectory_gp_conditional` (`gp.py`) |
| Stacked dynamics system and its weights, Eq. `stacked`; implementation note (diagonal blocks, variance floors) | `_build_blocks`, `_tempered_blocks` (inside `build_model`, `model.py`) |
| Conditional operator posterior and evidence, Eqs. `cond_post`, `evidence` | `per_mode_evidence_qr`, `per_mode_posterior_qr`, `_qr_operator_system` (`evidence.py`, QR form, see *Conditioning*); `posterior_O_fn` (`model.py`) |
| Posterior predictive sampling (Sec. 2.6) | `_draw_O`, `_predict_targets` (`pipeline.py`); initial-state uncertainty `_ic_sigma` (`pipeline.py`) |
| Posterior consistency (Sec. 2.7, struck) | none |

## Algorithm and practical considerations (Section 2.8)

| Paper | Code |
|---|---|
| Operator prior, Eq. `operator_prior` | latent `log_tau_block` in `build_model` (`op_prior_mode="block_hier"`, `hier_tau_scale`); prior precision `make_prior_prec_from_tau` (`evidence.py`) |
| Column scales, Eq. `column_scale` | `nondimensional_column_scale`, `data_scales`, `_column_degrees` (`model.py`) |
| Hand-set isotropic alternative $\Sigma_O = \sigma_O^2 I$ | `WeakFormConfig.sigma_O` |
| Effective degrees of freedom, Eq. `dof` | `effective_dof` (`gp.py`) |
| Tempering exponents, Eq. `tempering`; GP-only fit that sets them | `tempering_exponents`, `data_only_gp_fit` (`model.py`); `WeakFormConfig.closure="tempered"` |
| Power-likelihood normaliser, Eq. `power_correction` | `power_likelihood_correction` (`model.py`) |
| Hand-set alternatives to the tempering (closure variance, row weights) | `WeakFormConfig.closure`, `gamma2`, `deriv_weight`, `weakform_weight` |
| Inference target, Eq. `svi_target` | the `gp_mll` and `marg_O_evidence` factors in `build_model` (`model.py`) |
| SVI and NUTS (*Inference over the hyperparameters*) | `_run_experiment` (`pipeline.py`: `AutoNormal` guide with `ClippedAdam`, or `NUTS`); `WeakFormConfig.infer`, `num_steps`, `learning_rate` |
| Algorithm 1 | `run_experiment` (`pipeline.py`) |
| Shared settings | defaults of `WeakFormConfig` (`config.py`) |

## Experiments (Section 3)

| Paper | Code |
|---|---|
| Euler, heat and diffusion-reaction experiments | `experiments/{euler,heat,burgers_2d}/04_unified.py` (ours), `05_neural_ode.py` (Neural ODE), `06_compare_methods.py` |
| Cross-experiment table | `experiments/aggregate_table.py` |
| Tumour tasks (ours, Neural ODE, comparison) | `experiments/tumor/04_unified_benchmark.py`, `05_neural_ode_benchmark.py`, `06_compare_benchmark.py` |
| Tumour task definitions and chemotherapy ROM, Eq. `tumor_rom` | `experiments/tumor/benchmark_cases.py` (`CASES`, `galerkin_operators`); reported cases `untreated-growth`, `standard-regimen-trend-chemo` (planned course), `reduced-regimen-trend-chemo` (day-57 decision), `midcourse-reduced-trend-chemo` (day-84 decision) |
| Input-aware Gaussian-process trend, Eq. `input_kernel` (Sec. 3.4) | `WeakFormConfig.gp_input_trend` (`config.py`); `input_trend_features` (`features.py`); `trajectory_trend` (`gp.py`); switched on per case by `BenchmarkCase.input_aware_gp` (`experiments/tumor/benchmark_cases.py`) |
| Tumour scans, segmentation and noise-capped POD | `experiments/tumor/benchmark_data.py` (`_segmented_acquisition`, `observed_pod`, `noise_threshold`) |
| Tumour production fit | `fit_production` (`experiments/tumor/benchmark_models.py`) |

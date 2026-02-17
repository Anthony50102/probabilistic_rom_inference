# Copilot Context: Probabilistic ROM Inference

## Project Overview

This repository implements and compares two **Bayesian operator inference** methods for learning **reduced-order models (ROMs)** from noisy PDE snapshot data. It accompanies the paper *"Probabilistic scientific machine learning: Bayesian model reduction for nonlinear dynamical systems"* by Poole, McQuarrie, Guo, and Chaudhuri.

The central idea: given noisy observations of a high-dimensional PDE system, use **Proper Orthogonal Decomposition (POD)** to reduce dimensionality, then infer the ROM operators in a Bayesian framework that provides uncertainty quantification (UQ) on the learned dynamics.

---

## Scientific Background

### Operator Inference (OpInf)

Operator inference learns a ROM of the form:

$$\frac{d\hat{q}}{dt} = \hat{c} + \hat{A}\hat{q} + \hat{H}[\hat{q} \otimes \hat{q}] + \hat{B}u + \hat{N}[u \otimes \hat{q}]$$

where $\hat{q} \in \mathbb{R}^r$ are the POD coefficients (reduced states), $u$ is an optional input, and the operators $\hat{c}, \hat{A}, \hat{H}, \hat{B}, \hat{N}$ are inferred from data. The operator string (e.g., `"cAH"`, `"cAHBN"`) defines which operators are included. The operator matrix $\hat{O} \in \mathbb{R}^{r \times d(r,p)}$ collects all operators row-wise, and $d(\hat{q}, u)$ is the corresponding data vector.

### The Two Methods

#### 1. GP-Bayes OpInf (Baseline)
- Fits **Gaussian Processes (GPs)** to the noisy POD coefficients to get smooth state estimates and derivative estimates
- Uses GP posterior mean and covariance as weights in a **weighted least-squares** problem to infer operators
- Uncertainty comes from the GP posterior propagated through the least-squares solve
- Implemented via `scikit-learn` GP classes (`gpkernels.py`) and `WeightedLSTSQSolver` (`wlstsq.py`)
- Entry point: `01_gpbayes_opinf.ipynb` notebooks

#### 2. Full Bayesian OpInf (Our Method)

The paper describes a **three-stage hierarchical Bayesian algorithm** (Algorithm 1):

**Stage 1 — Learn GP Hyperparameters** (Bayesian):
- For each POD mode $i$, place priors on GP hyperparameters:
  - $\ell_i \sim \text{LogNormal}(\log(T/20), 1.0)$
  - $\sigma^2_i \sim \text{LogNormal}(\log(\text{std}(\hat{q}_i)^2), 0.5)$
  - $\nu_i \sim \text{LogNormal}(-8.0, 1.0)$
- GP prior: $f_i \sim \mathcal{N}(0, K_i)$ where $K_i = k(t,t; \ell_i, \sigma^2_i) + (\nu_i + \varepsilon)I$
- Observation likelihood: $\hat{q}_i \sim \mathcal{N}(f_i, \nu_i)$
- Optimize ELBO via SVI with `AutoNormal` guide

**Stage 2 — Infer Latent States** (Bayesian):
- For each mode, sample hyperparameters from Stage 1 approximate posterior
- Latent state prior: $X_i \sim \mathcal{N}(0, K_i)$
- Data likelihood: $\hat{q}_i \sim \mathcal{N}(X_i, \chi)$
- Optimize ELBO via SVI with `AutoNormal` guide

**Stage 3 — Learn Operator with Physics Constraints**:
- Operator prior: $O \sim \mathcal{N}(0, 10I)$
- For each mode $i$:
  - Sample $X_i \sim \mathcal{N}(\mu_{X_i}, \text{Stdev}_{X_i})$ from Stage 2 posterior
  - Derivative operator: $D_i = C'_\phi(C_\phi + \varepsilon I)^{-1}$
  - Derivative mean: $\mu_{z,i} = D_i X_i$
  - Derivative covariance: $A_i = C''_\phi - C'_\phi (C_\phi)^{-1} (C'_\phi)^T$
  - Constraint covariance: $\Sigma_i = A_i + \gamma I$
  - Physics constraint: $f(X)_i \hat{O}^T \sim \mathcal{N}(\mu_{z,i}, \Sigma_i)$
- Optimize ELBO via SVI with `AutoLowRankMultivariateNormal` guide

The key mathematical insight (Eq. 15–16 in the paper) is that the GP derivative $Z_i$ is **analytically marginalized out**, yielding a closed-form constraint likelihood that ties operator dynamics to the GP derivative posterior.

**Current implementation status (see Known Issues section):**
The codebase currently implements a simplified version of this algorithm. GP hyperparameters are fitted via MLE (not Bayesian Stage 1), latent states are deterministic GP means (not sampled as in Stage 2), and the operator prior is centered at a grid search result (not zero).

Key hyperparameters:
  - `GAMMA` ($\gamma$): operator prior scale
  - `GAMMA2` ($\gamma_2$): ODE constraint slack / derivative noise
- Entry point: `02_full_bayesian.ipynb` notebooks

### Pipeline (current implementation)

1. **Generate training data**: Solve the full-order PDE, subsample in time, add noise
2. **Fit POD basis**: Compress snapshots to $r$-dimensional POD coefficients
3. **Grid search for prior operator**: Find best deterministic OpInf operator via regularization sweep (used as the Bayesian prior mean)
4. **Fit GP hyperparameters**: MLE fit of RBF kernel lengthscales, variances, and noise per POD mode (via `fit_gp_hyperparameters_mle` / `SimpleGPR`)
5. **Define Bayesian model**: Prior on O centered at grid search result, deterministic latent states from GP mean, GP-derived ODE constraints as likelihood terms
6. **Run inference**: SVI with `AutoDelta` guide (MAP estimate) or MCMC (NUTS)
7. **Generate predictions**: Sample operators from posterior, solve ROMs, assess stability and accuracy
8. **Diagnostics**: Posterior correlation, ESS, R-hat, prior-posterior overlap, trace plots

---

## Repository Structure

### `core/` — Shared Library

| File | Purpose |
|------|---------|
| `bayesian_opinf.py` | **Main module for Full Bayesian method**: `JaxCompatibleModel`, GP utilities (`compute_gp_derivatives`, `fit_gp_hyperparameters_mle`), `grid_search_prior_operator`, `run_svi`, `run_mcmc`, `generate_rom_predictions`, JAX-compatible Kronecker products |
| `bayes.py` | `BayesianODE`, `BayesianROM` — posterior sampling classes for GP-Bayes baseline |
| `bgp_jax.py` | `BayesianGP`, `RBFKernel` — JAX/NumPyro Bayesian GP implementation with derivative kernel support (`get_c_phi`, `get_c_phi_dash`, `get_c_phi_double_dash`) |
| `gpkernels.py` | `GP_RBFW` — scikit-learn GP wrapper for the baseline method |
| `wlstsq.py` | `WeightedLSTSQSolver` — weighted least-squares operator inference |
| `pde_models.py` | Full-order PDE models: `Euler`, `FitzHughNagumo`, `CubicHeatBimodal` with `solve()`, `derivative()`, and visualization |
| `scaler.py` | `DataScaler` — per-mode standardization (zero mean, unit variance) for GP fitting |
| `plotting.py` | `Plotter` base class, `plot_deterministic_rom_solves`, `plot_gp_fit`, `plot_full_order_error` |
| `diagnostics.py` | `run_diagnostics`, `DiagnosticReport` — posterior correlation, ESS, R-hat, divergence detection, prior-posterior overlap, trace/rank plots |
| `utils.py` | `generate_trajectory` (data generation pipeline), `save_figure`, `summarize_experiment` |

### `experiments/` — Experiment Notebooks

Each PDE system has its own directory with:

| File | Purpose |
|------|---------|
| `config.py` | PDE-specific configuration: spatial/time domains, initial conditions, `FullOrderModel`, `Basis` (POD with lifting/scaling), `ReducedOrderModel`, GP bounds |
| `01_gpbayes_opinf.ipynb` | GP-Bayes OpInf baseline experiment |
| `02_full_bayesian.ipynb` | Full Bayesian OpInf experiment (parameterized for grid search via papermill) |
| `*_plotter.py` | Experiment-specific plotting (extends `Plotter` base class) |
| `step*.py` | Standalone scripts for individual pipeline steps |
| `run_grid_search.py` | Papermill-driven hyperparameter sweep over `GAMMA` × `GAMMA2` |
| `figures/` | Generated plots organized by date |
| `results/` | Saved MCMC samples, grid search outputs |

### Three PDE Test Cases

| System | Operator Structure | Variables | Lifting | Has Inputs |
|--------|-------------------|-----------|---------|------------|
| **Compressible Euler** | `cAH` | $(u, q, 1/\rho)$ — velocity, momentum flux, inverse density | Nondimensionalization | No |
| **FitzHugh-Nagumo** | `cAHBN` | $(q_1, q_2, q_1^2)$ — activator, inhibitor, quadratic lift | Quadratic lifting | Yes (Neumann BC) |
| **Cubic Heat** | `cAHBN` | $(q, q^2)$ — temperature, quadratic lift | Quadratic lifting + shift | Yes (source params) |

---

## Key Technical Details

### JAX/NumPyro Conventions
- All Bayesian inference uses **NumPyro** with **JAX** backend
- Platform set to CPU: `numpyro.set_platform('cpu')`
- Multi-chain MCMC uses `numpyro.set_host_device_count(4)`
- Random keys managed via `jax.random.PRNGKey`
- The `JaxCompatibleModel` wraps `opinf.models.ContinuousModel` with JAX-traceable `_assemble_data_matrix` for use inside NumPyro models

### Operator Matrix Shape
- The operator matrix `O` has shape `(num_modes, d)` where `d` depends on the operator structure
- For `"cAH"` with $r$ modes: $d = 1 + r + r(r+1)/2$ (constant + linear + quadratic Kronecker)
- For `"cAHBN"`: adds input and state-input interaction columns

### GP Derivative Computation
- RBF kernel derivatives provide analytically smooth time derivative estimates
- `compute_gp_derivatives(Ls, Vs, time_train, time_eval, y_train, Ns=None)` returns mean and covariance of the GP derivative posterior. Pass `Ns` (per-mode noise variances from MLE) to include observation noise in $K_{yy}$.
- Derivative mean: $\mu_z = K_{z,y}(K_{y,y} + \varepsilon I)^{-1} y$
- Derivative covariance: $A = K_{z,z} - K_{z,y}(K_{y,y} + \varepsilon I)^{-1} K_{z,y}^T$
- $K_{z,y}$ uses $\partial k / \partial t_1$ (derivative w.r.t. first argument — the derivative evaluation point)
- $K_{z,z}$ uses $\partial^2 k / \partial t_1 \partial t_2$ (mixed second derivative)
- These serve as the "observed" derivatives in the Bayesian ODE constraint likelihood
- GP densification (`NUM_EVAL_POINTS`) allows evaluating ODE constraints at more points than training samples
- **Note**: currently `K_yy` does NOT include observation noise — only a small jitter (1e-5). This is consistent with the paper's formulation when conditioning on latent states $X_i$, but the code conditions on noisy `y_train` instead.

### Kernel derivative functions (`bgp_jax.py`)
- `get_c_phi(ℓ, σ², t)`: $C_\phi = k(t,t) + \varepsilon I$ (kernel matrix + nugget)
- `get_c_phi_dash(ℓ, σ², t)`: $\partial k / \partial t_2 = \frac{t_i - t_j}{\ell^2} k(t_i, t_j)$ (derivative w.r.t. second argument)
- `get_dash_c_phi(ℓ, σ², t)`: $\partial k / \partial t_1 = -\frac{t_i - t_j}{\ell^2} k(t_i, t_j)$ (derivative w.r.t. first argument, = `-get_c_phi_dash`)
- `get_c_phi_double_dash(ℓ, σ², t)`: $\partial^2 k / \partial t_1 \partial t_2 = (\frac{1}{\ell^2} - \frac{(t_i-t_j)^2}{\ell^4}) k(t_i, t_j)$

### Data Scaling
- When `USE_SCALED_DATA=True`, each POD mode is standardized before GP fitting
- The Bayesian model must handle the transform/inverse-transform consistently:
  - Latent states live in scaled space
  - Operator dynamics are computed in original space (`Xs_original = Xs * std + mean`)
  - Dynamics are scaled back for GP comparison (`f_Xi_scaled = f_Xi / std`)
  - Derivatives are computed from scaled training data

### Grid Search for Prior
- `grid_search_prior_operator` sweeps regularization parameters for deterministic OpInf
- Selects the operator that produces the most stable ROM (longest integration before blowup)
- This deterministic estimate becomes the **prior mean** in the Bayesian model (differs from paper's $O \sim N(0, 10I)$)
- ROM stability is assessed by attempting `model.predict()` and checking for solver success

### Inference Guides (SVI)
- `AutoDelta`: MAP estimate (point estimate, fast) — **currently used in all notebooks**
- `AutoNormal`: Mean-field variational family (diagonal Gaussian) — paper recommends for Stages 1 & 2
- `AutoMultivariateNormal`: Full-covariance variational family (captures correlations)
- `AutoLowRankMultivariateNormal`: Low-rank + diagonal approximation — **paper recommends for Stage 3**

---

## Coding Conventions

- **Python 3.10+** features used (type hints, `match` statements)
- **NumPy** for data manipulation, **JAX/jnp** inside NumPyro models and GP kernels
- **opinf** library (version ≥ 0.5) for operator inference scaffolding
- Notebooks are parameterized via **papermill** for automated grid searches
- Configuration is centralized in `config.py` per experiment (PDE setup, basis, domain)
- The `core/` package is imported via relative path manipulation (`sys.path.insert`)
- Figures are saved with timestamps using `save_figure()` from `core/utils.py`
- Reproducibility: `np.random.seed(42)` and `jax.random.PRNGKey(42)` are standard

### Common Parameters

| Parameter | Typical Values | Description |
|-----------|---------------|-------------|
| `NUM_MODES` | 3–8 | Number of POD modes to retain |
| `NUM_SAMPLES` | 50–200 | Training snapshots (subsampled from full trajectory) |
| `NOISE_LEVEL` | 0.01–0.05 | Noise std as fraction of signal |
| `GAMMA` | 0.1–100 | Operator prior variance |
| `GAMMA2` | 0.1–100 | ODE constraint noise/stiffness |
| `NUM_EVAL_POINTS` | 100–500 | GP densification points (or None for no densification) |
| `OPERATORS` | `"cAH"`, `"cAHBN"` | OpInf operator string |

---

## Development Guidelines

### When modifying the Bayesian model (`bayesian_opinf_model`):
- The model function must be JAX-traceable (no Python control flow that depends on traced values)
- Use `jnp` (not `np`) for all array operations inside the model
- `numpyro.sample` for random variables, `numpyro.deterministic` for tracked quantities
- Keep data scaling transforms consistent between GP fitting space and operator dynamics space

### When adding a new PDE test case:
1. Add the PDE to `core/pde_models.py` (extend `_BasePDE`)
2. Create `experiments/<name>/config.py` with `FullOrderModel`, `Basis`, domain specs
3. Create a plotter extending `core.plotting.Plotter`
4. Copy and adapt `01_gpbayes_opinf.ipynb` and `02_full_bayesian.ipynb`

### When running experiments:
- Single run: open and execute the notebook interactively
- Grid search: use `run_grid_search.py` (runs papermill over GAMMA × GAMMA2 grid)
- Results are saved to `results/` subdirectories with timestamps

### Testing stability:
- ROM solves can easily blow up; always check `predict_result_.y.shape[1] == len(time_eval)` before using results
- The grid search filters for stable solves automatically
- When generating predictions from posterior samples, count and report the fraction of stable solves

---

## File Dependencies

```
config.py (per experiment)
  └── core/pde_models.py (FullOrderModel)
  └── opinf (Basis, operators)

02_full_bayesian.ipynb
  └── config.py
  └── core/bayesian_opinf.py
  │     └── core/bgp_jax.py (BayesianGP, RBFKernel)
  │     └── opinf (ContinuousModel, operators)
  │     └── numpyro (SVI, MCMC, distributions)
  └── core/plotting.py (visualization)
  └── core/diagnostics.py (post-inference checks)
  └── core/scaler.py (optional data scaling)
  └── *_plotter.py (experiment-specific plots)

01_gpbayes_opinf.ipynb
  └── config.py
  └── core/gpkernels.py (GP_RBFW)
  └── core/wlstsq.py (WeightedLSTSQSolver)
  └── core/bayes.py (BayesianODE, BayesianROM)
  └── core/plotting.py
```

---

## Known Issues and Pitfalls

See `bayesian_modeling_pitfalls.md` for a detailed guide. Key concerns:

- **Operator prior variance (GAMMA)**: Too small → posterior collapses to prior, too large → non-identifiability
- **ODE constraint stiffness (GAMMA2)**: Too small → physics not enforced, too large → overfitting to GP derivative noise
- **ROM stability**: Many posterior operator samples may produce unstable ROMs; report the stable fraction
- **Correlated posteriors**: Operator matrix elements are often highly correlated; use `AutoMultivariateNormal` if needed
- **Scaling**: If GP fits are poor, enable `USE_SCALED_DATA` to standardize POD coefficients before fitting
- **GP derivative quality**: Poor GP fits (wrong lengthscale, insufficient data) propagate directly into the Bayesian likelihood

### Code vs. Paper Discrepancies

The current implementation simplifies the paper's Algorithm 1 in several ways:

1. **Stage 1 (GP hyperparameters)**: Paper uses Bayesian inference with priors on $\ell$, $\sigma^2$, $\nu$. Code uses MLE point estimates (`fit_gp_hyperparameters_mle`/`SimpleGPR`).
2. **Stage 2 (latent states)**: Paper samples latent states $X_i \sim \mathcal{N}(0, K_i)$ with data likelihood. Code skips this entirely — uses deterministic GP mean predictions.
3. **Stage 3 (operator prior)**: Paper uses $O \sim \mathcal{N}(0, 10I)$. Code uses $O \sim \mathcal{N}(\hat{O}_{\text{grid}}, \gamma I)$ (centered at grid search result).
4. **Stage 3 (latent states not sampled)**: Paper samples $X_i$ from Stage 2 posterior. Code uses `numpyro.deterministic` (fixed GP mean). The `build_bayesian_opinf_model` has a `sample_X` flag but it defaults to `False` and notebooks don't use it.
5. **Derivative mean uses noisy data**: Paper computes $\mu_{z,i} = D_i X_i$ using the (sampled) latent states. Code calls `compute_gp_derivatives(..., y_train)` with the raw noisy observations. Even when `sample_X=True`, the derivative doesn't use the sampled X.
6. **~~Missing noise in K_yy~~** (FIXED): `compute_gp_derivatives` now accepts an optional `Ns` parameter with per-mode observation noise variances. When provided, $K_{yy} = k(t,t) + (\nu_i + \varepsilon)I$.
7. **SVI guide**: Paper recommends `AutoLowRankMultivariateNormal` for Stage 3. All notebooks use `AutoDelta` (MAP estimate).
8. **FitzHugh-Nagumo notebook cruft**: Contains legacy model definitions below the active code (artifact of earlier development).

### Implementation Notes

- **`compute_gp_derivatives` called inside numpyro model**: Since it depends only on fixed data (not on any `numpyro.sample` site), it returns identical values every call. When GP hyperparameters become sampled (full Bayesian Stage 1), this will need to depend on the sampled lengthscales/variances.
- **`SimpleGPR.predict()` uses NumPy**: The GP prediction is NumPy-based, not JAX. Currently safe because X is `numpyro.deterministic`, but would break if X becomes a sampled variable requiring JAX tracing. Precomputing Xs_means outside the model (current approach) avoids this issue.

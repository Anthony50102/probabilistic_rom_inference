# Probabilistic Reduced Order Model Inference

Implementation and comparison of Bayesian reduced-order modeling methods for operator inference.

## Overview

This repository accompanies a scientific paper comparing two Bayesian approaches for learning reduced-order models (ROMs) from noisy data:

1. **GP-Bayes OpInf** (Baseline): Gaussian Process-based Bayesian operator inference
2. **Full Bayesian OpInf** (Our Method): Fully Bayesian approach using Stochastic Variational Inference (SVI)

Both methods are tested on three PDE systems:
- **FitzHugh-Nagumo** equations (reaction-diffusion)
- **Compressible Euler** equations (fluid dynamics)
- **Cubic Heat** equation (nonlinear diffusion)

## Repository Structure

```
probabilistic_rom_inference/
├── core/                           # Shared library code
│   ├── bayes.py                    # BayesianODE, BayesianROM classes
│   ├── bgp_jax.py                  # Bayesian GP with JAX/NumPyro (Full Bayesian)
│   ├── gpkernels.py                # GP kernel implementations (GP-Bayes)
│   ├── pde_models.py               # PDE full-order model implementations
│   ├── scaler.py                   # Data normalization utilities
│   └── wlstsq.py                   # Weighted least squares solver
│
├── experiments/                    # Main experiment notebooks
│   ├── fitz_nagumo/
│   │   ├── config.py               # FitzHugh-Nagumo configuration
│   │   ├── 01_gpbayes_opinf.ipynb  # GP-Bayes OpInf method
│   │   ├── 02_full_bayesian.ipynb  # Full Bayesian method
│   │   ├── figures/                # Generated figures
│   │   └── results/                # Saved results (MCMC samples, etc.)
│   │
│   ├── euler/
│   │   ├── config.py               # Euler equations configuration
│   │   ├── 01_gpbayes_opinf.ipynb
│   │   ├── 02_full_bayesian.ipynb
│   │   ├── figures/
│   │   └── results/
│   │
│   └── heat/
│       ├── config.py               # Heat equation configuration
│       ├── 01_gpbayes_opinf.ipynb
│       ├── 02_full_bayesian.ipynb
│       ├── figures/
│       └── results/
│
├── archive/                        # Deprecated notebooks (kept for reference)
└── helpers/                        # Legacy helper code
```

## Methods

### GP-Bayes OpInf (Baseline)
- Uses Gaussian Process regression to smooth noisy state data
- Derives time derivatives from GP posterior
- Performs weighted least squares for operator inference
- Uncertainty quantification via GP posterior

### Full Bayesian OpInf (Our Method)
- Fully Bayesian treatment using Stochastic Variational Inference (SVI)
- Joint posterior over GP hyperparameters and ROM operators
- Implemented in JAX/NumPyro for efficient inference
- More principled uncertainty quantification

## Data Scaling

When `USE_SCALED_DATA=True` in a notebook, each POD mode is standardized to zero mean and unit variance before GP fitting via `DataScaler` (`core/scaler.py`). This improves GP hyperparameter learning and numerical conditioning, especially when POD modes have very different magnitudes.

### How scaling flows through the Bayesian model

1. **GP fitting**: GPs are trained on scaled data $\tilde{q}_i = (\hat{q}_i - \mu_i) / \sigma_i$, so lengthscales, variances, and noise are all in scaled space.
2. **Latent states**: `Xs_means` (GP predictions) live in scaled space.
3. **Data matrix assembly**: States are unscaled back to original space ($\hat{q} = \sigma\tilde{q} + \mu$) before assembling the operator data matrix $d(\hat{q})$.
4. **Operator dynamics**: $O \cdot d(\hat{q})$ is computed in original space, then divided by $\sigma_i$ to get the scaled-space derivative $d\tilde{q}_i/dt$.
5. **ODE constraint**: The scaled operator dynamics are compared against GP derivative estimates (also in scaled space) via a multivariate normal likelihood.

The operator $O$ is always learned and used for predictions in **original (unscaled) space**.

### Implicit mode-dependent constraint scaling

The ODE constraint slack `gamma2` is applied uniformly in scaled space:

$$\frac{d\tilde{q}_i}{dt} \sim \mathcal{N}\!\left(\frac{[O \cdot d(\hat{q})]_i}{\sigma_i},\; \text{cov}_z^{(\text{scaled})}_i + \gamma_2 I\right)$$

This means the effective constraint tolerance in original space is mode-dependent: $\gamma_{2,i}^{(\text{original})} = \sigma_i^2 \cdot \gamma_2$. Modes with larger amplitude get proportionally more slack, which acts as a natural regularization — all modes contribute comparably to the likelihood regardless of their raw scale.

## Requirements

```
numpy
scipy
matplotlib
jax
jaxlib
numpyro
opinf
scikit-learn
```

## Usage

Each experiment follows the same structure:

1. **Configuration** (`config.py`): Defines the PDE, domain, and ROM structure
2. **Notebook 01**: Runs GP-Bayes OpInf (baseline)
3. **Notebook 02**: Runs Full Bayesian OpInf (our method)

To run an experiment:
```bash
cd experiments/fitz_nagumo
jupyter notebook 01_gpbayes_opinf.ipynb
```

## Citation

If you use this code, please cite:
```
[Citation to be added upon publication]
```

## License

See [LICENSE](LICENSE) for details.

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

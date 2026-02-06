# core/__init__.py
"""
Core modules for Probabilistic Reduced Order Model Inference.

This package contains shared code for two Bayesian operator inference methods:
1. GP-Bayes OpInf (baseline) - Gaussian Process based Bayesian operator inference
2. Full Bayesian OpInf - Fully Bayesian approach using SVI/MCMC

Methods are tested on three PDE systems:
- FitzHugh-Nagumo equations
- Compressible Euler equations  
- Heat equation with cubic nonlinearity
"""

from .bayes import BayesianODE, BayesianROM
from .gpkernels import GP_RBFW, fit_gaussian_processes
from .wlstsq import WeightedLSTSQSolver, WeightedLSTSQSolverMulti
from .bgp_jax import BayesianGP, RBFKernel, get_c_phi
from .scaler import DataScaler
from .utils import summarize_experiment, save_figure, generate_trajectory
from . import pde_models

# Full Bayesian OpInf utilities
from .bayesian_opinf import (
    JaxCompatibleModel,
    SimpleGPR,
    GridSearchResult,
    SVIResult,
    MCMCResult,
    # Core functions
    grid_search_prior_operator,
    fit_gp_hyperparameters_mle,
    compute_gp_derivatives,
    generate_gp_samples,
    build_bayesian_opinf_model,
    run_svi,
    run_mcmc,
    generate_rom_predictions,
    # Kronecker products
    Quadraticckron,
    Cubicckron,
    khatri_rao,
    binom,
    rbf_eval,
    flatten_time,
)

__all__ = [
    # GP-Bayes OpInf (baseline method)
    "BayesianODE",
    "BayesianROM", 
    "GP_RBFW",
    "fit_gaussian_processes",
    "WeightedLSTSQSolver",
    "WeightedLSTSQSolverMulti",
    # Full Bayesian OpInf (our method)
    "BayesianGP",
    "RBFKernel",
    "get_c_phi",
    "DataScaler",
    # Full Bayesian OpInf shared utilities
    "JaxCompatibleModel",
    "SimpleGPR",
    "GridSearchResult",
    "SVIResult",
    "MCMCResult",
    "grid_search_prior_operator",
    "fit_gp_hyperparameters_mle",
    "compute_gp_derivatives",
    "generate_gp_samples",
    "build_bayesian_opinf_model",
    "run_svi",
    "run_mcmc",
    "generate_rom_predictions",
    "Quadraticckron",
    "Cubicckron",
    "khatri_rao",
    "binom",
    "rbf_eval",
    "flatten_time",
    # Utilities
    "summarize_experiment",
    "save_figure",
    "generate_trajectory",
    "pde_models",
]

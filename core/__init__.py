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
from .plotting import (
    Plotter,
    plot_deterministic_rom_solves,
    plot_gp_fit,
    plot_full_order_error,
    compute_derivatives_fourth_order,
    rbf_eval as plotting_rbf_eval,
    flatten_time as plotting_flatten_time,
)

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

# Bayesian diagnostics
from .diagnostics import (
    DiagnosticReport,
    run_diagnostics,
    compute_posterior_correlation,
    compute_ess,
    compute_rhat,
    detect_divergences,
    compute_prior_posterior_overlap,
    plot_correlation_matrix,
    plot_ess,
    plot_trace,
    plot_rank,
    plot_prior_posterior,
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
    # Plotting
    "Plotter",
    "plot_deterministic_rom_solves",
    "plot_gp_fit",
    "plot_full_order_error",
    "compute_derivatives_fourth_order",
    # Diagnostics
    "DiagnosticReport",
    "run_diagnostics",
    "compute_posterior_correlation",
    "compute_ess",
    "compute_rhat",
    "detect_divergences",
    "compute_prior_posterior_overlap",
    "plot_correlation_matrix",
    "plot_ess",
    "plot_trace",
    "plot_rank",
    "plot_prior_posterior",
]

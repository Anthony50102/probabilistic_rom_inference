# core/__init__.py
"""
Core modules for Probabilistic Reduced Order Model Inference.

This package contains shared code for the active Bayesian operator inference
experiments.

Active experiments currently cover:
- Compressible Euler equations
- Cubic heat equation
- 2D diffusion-reaction / Burgers-style system
- TumorTwin tumor-growth data
"""

from .bgp_jax import BayesianGP, RBFKernel, get_c_phi
from .utils import summarize_experiment, save_figure, generate_trajectory
from . import pde_models
from .plotting import (
    Plotter,
    plot_deterministic_rom_solves,
    plot_gp_fit,
    plot_operator_derivative_fit,
    plot_full_order_error,
    save_paper_figure,
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
    find_latent_state_key,
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
    StabilityReport,
    run_diagnostics,
    diagnose_stability,
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
    # Full Bayesian OpInf (our method)
    "BayesianGP",
    "RBFKernel",
    "get_c_phi",
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
    "find_latent_state_key",
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
    "plot_operator_derivative_fit",
    "plot_full_order_error",
    "save_paper_figure",
    "compute_derivatives_fourth_order",
    # Diagnostics
    "DiagnosticReport",
    "StabilityReport",
    "run_diagnostics",
    "diagnose_stability",
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

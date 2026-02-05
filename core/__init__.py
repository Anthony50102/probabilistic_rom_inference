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
    # Utilities
    "summarize_experiment",
    "save_figure",
    "generate_trajectory",
    "pde_models",
]

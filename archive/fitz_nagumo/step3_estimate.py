# step3_estimate.py
"""Estimate the system parameters with GP-powered OpInf."""

__all__ = [
    "estimate_posterior",
    "compute_deterministic_operator",
]

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import logging
import warnings
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

import config
from core import bayes, wlstsq

import opinf


__MAXOPTVAL = 1e12  # Ceiling for optimization.
__DEFAULT_SEARCH_GRID = np.logspace(-16, 4, 81)  # Search grid.


def compute_deterministic_operator(
    gps: list,
    inputs: np.ndarray,
    regularizer: float = 1e-6,
) -> np.ndarray:
    """Compute a deterministic OpInf operator (for use as prior mean).
    
    This fits a standard (non-Bayesian) OpInf model using the GP mean estimates.
    
    Parameters
    ----------
    gps : list of trained gpkernel.GP_RBFW objects.
        Gaussian processes for each state variable, already fit to data.
    inputs : (k,) ndarray or None
        Inputs corresponding to the GP state estimates (if present).
    regularizer : float
        L2 regularization for the deterministic solve.
        
    Returns
    -------
    operator : (r, d) ndarray
        Deterministic operator matrix (each row is one mode's operator).
    """
    rom = config.ReducedOrderModel()
    rom.state_dimension = len(gps)
    
    # Get GP mean estimates (no uncertainty)
    state_estimates = np.array([gp.state_estimate for gp in gps])
    ddt_estimates = np.array([gp.ddt_estimate for gp in gps])
    
    # Assemble data matrix
    D = rom._assemble_data_matrix(state_estimates, inputs)
    
    # Solve standard least squares: min ||D @ O - ddt||^2 + reg * ||O||^2
    # Solution: O = (D'D + reg*I)^{-1} D' ddt
    DtD = D.T @ D
    reg_matrix = regularizer * np.eye(D.shape[1])
    
    operators = []
    for i in range(len(gps)):
        Dt_ddt = D.T @ ddt_estimates[i]
        operator_row = la.solve(DtD + reg_matrix, Dt_ddt, assume_a='pos')
        operators.append(operator_row)
    
    return np.array(operators)


def _posterior_autoregularized_multisample(
    regularizer_grid: np.ndarray,
    time_domain: np.ndarray,
    time_domain_estimated: np.ndarray,
    snapshots_estimated: np.ndarray,
    num_samples: int,
    lstsq_solver: wlstsq.WeightedLSTSQSolver,
    rom: opinf.models.ContinuousModel,
    prior_mean: np.ndarray = None,
    weighted_rhs: np.ndarray = None,
) -> bayes.BayesianROM:
    r"""Use an error-based optimization to select an appropriate regularization
    hyperparamter for the operator inference regression.

    .. math::
       \ohat_i = (D^T W_i D + \lambda_i I)^{-1} D^T W_i ddts_i

    Use ``num_samples`` posterior draws to check that the posterior gives
    stable solutions.

    Parameters
    ----------
    regularizer_grid : (num_regs,) ndarray
        Grid of regularization values to try (followed by an optimization).
    time_domain : (k,) ndarray
        Time domain over which to solve the model for a stability check.
    time_domain_estimated : (m',) ndarray
        Time domain corresponding to the GP estimates of the snapshots.
    snapshots_estimated : (r, m') ndarray
        GP state estimates of the available training snapshots.
    num_samples : int
        Number of posterior draws to do for the stability check.
    lstsq_solvers : list of wlstsq.WeightedLSTSQSolver objects
        Solvers for the least-squares problem(s) (already 'fit' to the data).
    rom : opinf.models.ContinuousModel
        Reduced-order model object (**not** fit to the data)

    Returns
    -------
    Bayesian model.
    """
    shift = np.mean(snapshots_estimated, axis=1).reshape((-1, 1))
    limits = 5 * np.abs(snapshots_estimated - shift).max(axis=1)
    snapshotnorm = la.norm(snapshots_estimated)
    initial_conditions = snapshots_estimated[:, 0]

    def unstable(_solution, size):
        """Return True if the solution is unstable."""
        if _solution.shape[-1] != size:
            return True
        return np.any(np.abs(_solution - shift).max(axis=1) > limits)

    def get_bayesian_model(reg):
        """Form and solve the regression for the given regularization value.
        
        If prior_mean is provided, uses Bayesian linear regression with:
            Prior: O ~ N(prior_mean, (1/reg^2) * I)
            Posterior mean: (D'WD + reg^2*I)^{-1} (D'W*ddt + reg^2 * prior_mean)
        Otherwise uses zero prior mean (standard regularization).
        """
        lstsq_solver.regularizer = reg
        reg2 = lstsq_solver.regularizer**2
        
        # If no prior mean, just use the standard solve
        if prior_mean is None:
            means = np.atleast_2d(lstsq_solver.solve())
            rom._extract_operators(means)
            
            # Posterior covariance
            precisions = []
            for subsolver, mean in zip(lstsq_solver.solvers, means):
                RsqrtD = subsolver.data_matrix  # = Rsqrt @ D
                invSigma = (RsqrtD.T @ RsqrtD) + (reg2 * np.eye(mean.size))
                precisions.append(invSigma)
        else:
            # With prior mean, we need to modify the posterior mean calculation
            precisions = []
            posterior_means = []
            
            for i, subsolver in enumerate(lstsq_solver.solvers):
                # Precision matrix: D'WD + reg^2*I
                RsqrtD = subsolver.data_matrix  # = Rsqrt @ D
                invSigma = (RsqrtD.T @ RsqrtD) + (reg2 * np.eye(RsqrtD.shape[1]))
                precisions.append(invSigma)
                
                # Posterior mean with prior
                # Standard: mean = (D'WD + reg^2*I)^{-1} D'W*ddt
                # With prior: mean = (D'WD + reg^2*I)^{-1} (D'W*ddt + reg^2 * prior_mean)
                # weighted_rhs[i] = Rsqrt @ ddt[i], so RsqrtD.T @ weighted_rhs[i] = D'W*ddt
                rhs_term = RsqrtD.T @ weighted_rhs[i]
                
                # Add prior contribution: reg^2 * prior_mean[i]
                rhs_term = rhs_term + reg2 * prior_mean[i]
                
                mean_i = la.solve(invSigma, rhs_term, assume_a='pos')
                posterior_means.append(mean_i)
            
            means = np.atleast_2d(np.array(posterior_means))
            rom._extract_operators(means)

        try:
            return bayes.BayesianROM(means, np.array(precisions), rom)
        except np.linalg.LinAlgError as ex:
            if ex.args[0] == "Matrix is not positive definite":
                return None
            raise

    def _training_error(logreg):
        """Get the solution for a single regularization candidate."""
        opinf_regularizer = 10**logreg
        print(
            f"Testing regularizer {opinf_regularizer:.4e}...",
            end="",
            flush=True,
        )
        bayesian_model = get_bayesian_model(opinf_regularizer)
        if bayesian_model is None:
            print("Covariance not SPD")
            return __MAXOPTVAL

        # Sample the posterior distribution and check for stability.
        draws = []
        for _ in range(num_samples):
            for tmdmn in (time_domain, time_domain_estimated):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    draw = bayesian_model.predict(
                        initial_conditions=initial_conditions,
                        timepoints=tmdmn,
                        input_func=rom.input_func,
                    )
                if unstable(draw, tmdmn.size):
                    print("UNSTABLE")
                    return __MAXOPTVAL
            draws.append(draw)

        rom_solution = np.mean(draws, axis=0)
        error = la.norm(rom_solution - snapshots_estimated) / snapshotnorm
        print(f"{error:.2%} error")
        return error

    # Test each regularization hyperparameter.
    regularizer_grid = np.atleast_1d(regularizer_grid)
    if (num_tests := len(regularizer_grid)) == 1:
        search_bounds = [regularizer_grid[0] / 10, 10 * regularizer_grid[0]]
    else:
        # GRID SEARCH.
        _smallest_error, _best_reg_index = __MAXOPTVAL, None
        regularizer_grid = np.sort(regularizer_grid)
        print("\nGRIDSEARCH")
        for i, reg in enumerate(regularizer_grid):
            print(f"({i+1:d}/{num_tests:d}) ", end="")
            if (error := _training_error(np.log10(reg))) < _smallest_error:
                _smallest_error = error
                _best_reg_index = i
        if _best_reg_index is None:
            raise ValueError("grid search failed!")
        best_reg = regularizer_grid[_best_reg_index]

        if _best_reg_index == 0:
            print("\nWARNING: extend regularizer_grid to the left!")
            search_bounds = [best_reg / 100, regularizer_grid[1]]
        elif _best_reg_index == num_tests - 1:
            print("\nWARNING: extend regularizer_grid to the right!")
            search_bounds = [regularizer_grid[-2], 100 * best_reg]
        else:
            search_bounds = [
                regularizer_grid[_best_reg_index - 1],
                regularizer_grid[_best_reg_index + 1],
            ]

        message = f"Best regularization via gridsearch: {best_reg:.4e}"
        print(message + "\n")
        logging.info(message)

    # Follow up grid search with minimization-based search.
    print("1D OPTIMIZATION")
    opt_result = opt.minimize_scalar(
        _training_error, method="bounded", bounds=np.log10(search_bounds)
    )

    if opt_result.success and opt_result.fun != __MAXOPTVAL:
        regularizer = 10**opt_result.x
        message = f"Best regularization via optimization: {regularizer:.4e}"
        print(message)
        logging.info(message)
    else:
        regularizer = best_reg
        print("Optimization failed, falling back on gridsearch")

    return get_bayesian_model(regularizer)


def estimate_posterior(
    time_domain: np.ndarray,
    gps: list,
    inputs: np.ndarray,
    use_prior: bool = False,
    prior_regularizer: float = 1e-6,
) -> bayes.BayesianROM:
    """Construct the posterior parameter distribution.

    Parameters
    ----------
    time_domain : (k,) ndarray
        Time domain over which to solve the model.
    gps : list of trained gpkernel.GP_RBFW objects.
        Gaussian processes for each state variable, already fit to data.
    inputs : (k,) ndarray or None
        Inputs corresponding to the GP state estimates (if present).
    use_prior : bool, optional
        If True, use a deterministic OpInf solve as the prior mean.
        This makes the method more comparable to Full Bayesian OpInf.
        Default is False (zero prior mean).
    prior_regularizer : float, optional
        Regularization parameter for the deterministic solve used to 
        compute the prior mean. Only used if use_prior=True.
        Default is 1e-6.

    Returns
    -------
    bayes.BayesianROM
        Initialized Bayesian model.
    """
    with opinf.utils.TimedBlock("constructing posterior hyperparameters\n"):
        rom = config.ReducedOrderModel()
        rom.state_dimension = len(gps)

        # Construct the data matrix RHS ddts vector, and weight matrix.
        state_estimates = np.array([gp.state_estimate for gp in gps])
        D = rom._assemble_data_matrix(state_estimates, inputs)
        rhs = np.array([gp.ddt_estimate for gp in gps])
        Rsqrts = np.array([gp.sqrtW for gp in gps])

        # Compute weighted rhs: Rsqrt @ rhs for each mode
        # This is needed when using a prior mean
        weighted_rhs = np.array([Rsqrts[i] @ rhs[i] for i in range(len(gps))])

        # Pepare a solver and reduced-order model.
        lstsq_solver = wlstsq.WeightedLSTSQSolver(Rsqrts, 1)
        lstsq_solver.fit(D, rhs)

        # Optionally compute prior mean from deterministic solve.
        prior_mean = None
        if use_prior:
            logging.info("Computing deterministic operator for prior mean...")
            prior_mean = compute_deterministic_operator(
                gps=gps,
                inputs=inputs,
                regularizer=prior_regularizer,
            )
            logging.info(f"Prior mean shape: {prior_mean.shape}")

        # Choose the regularizer automatically through an optimization.
        return _posterior_autoregularized_multisample(
            regularizer_grid=__DEFAULT_SEARCH_GRID,
            time_domain=time_domain,
            time_domain_estimated=gps[0].t_estimation,
            snapshots_estimated=state_estimates,
            num_samples=20,
            lstsq_solver=lstsq_solver,
            rom=rom,
            prior_mean=prior_mean,
            weighted_rhs=weighted_rhs if use_prior else None,
        )
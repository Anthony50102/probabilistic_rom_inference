# core/bayesian_opinf.py
"""
Shared utilities for Full Bayesian Operator Inference.

This module provides:
- JAX-compatible OpInf model class
- GP regression and derivative computation
- Bayesian inference model builders
- Grid search for prior operator
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from typing import Callable, Optional, Tuple, List, Union
from dataclasses import dataclass

import opinf
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide, Predictive, MCMC, NUTS, init_to_value
from numpyro.optim import Adam

from .bgp_jax import BayesianGP, RBFKernel, get_c_phi


# =============================================================================
# JAX-Compatible Kronecker Products
# =============================================================================

def binom(x, y):
    """Binomial coefficient using gamma functions."""
    return jnp.exp(gammaln(x + 1) - gammaln(y + 1) - gammaln(x - y + 1))


def Quadraticckron(state):
    """Quadratic Kronecker product for state."""
    return jnp.concatenate([state[i] * state[:i + 1] for i in range(state.shape[0])], axis=0)


def Cubicckron(state):
    """Cubic Kronecker product for state."""
    state2 = Quadraticckron(state)
    lens = binom(jnp.arange(2, len(state) + 2), 2).astype(int)
    return jnp.concatenate([state[i] * state2[:lens[i]] for i in range(state.shape[0])], axis=0)


def khatri_rao(a, b):
    """Khatri-Rao product (column-wise Kronecker)."""
    return jnp.vstack([jnp.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T


# =============================================================================
# JAX-Compatible OpInf Model
# =============================================================================

class JaxCompatibleModel(opinf.models.ContinuousModel):
    """OpInf ContinuousModel with JAX-compatible data matrix assembly."""
    
    def __init__(self, operators, solver=None, *args, **kwargs):
        super().__init__(operators, solver, *args, **kwargs)
    
    def _assemble_data_matrix(self, states, inputs):
        """Assemble data matrix using JAX operations."""
        blocks = []
        for i in self._indices_of_operators_to_infer:
            op = self.operators[i]
            if isinstance(op, opinf.operators.ConstantOperator):
                block = jnp.ones((1, jnp.atleast_1d(states).shape[-1]))
            elif isinstance(op, opinf.operators.LinearOperator):
                block = jnp.atleast_2d(states)
            elif isinstance(op, opinf.operators.QuadraticOperator):
                block = Quadraticckron(jnp.atleast_2d(states))
            elif isinstance(op, opinf.operators.CubicOperator):
                block = Cubicckron(jnp.atleast_2d(states))
            elif isinstance(op, opinf.operators.InputOperator):
                block = jnp.atleast_2d(inputs)
            elif isinstance(op, opinf.operators.StateInputOperator):
                block = khatri_rao(jnp.atleast_2d(inputs), jnp.atleast_2d(states))
            else:
                raise ValueError(f"Unknown operator type: {type(op)}")
            blocks.append(block.T)
        return jnp.hstack(blocks)


# =============================================================================
# GP Utilities
# =============================================================================

def flatten_time(t: jnp.ndarray) -> jnp.ndarray:
    """Flatten time array to 1D."""
    return jnp.ravel(t)


def rbf_eval(lengthscale: float, variance: float, t1: jnp.ndarray, t2: jnp.ndarray) -> jnp.ndarray:
    """Evaluate RBF kernel between two time arrays."""
    t1, t2 = flatten_time(t1), flatten_time(t2)
    diff = t1[:, None] - t2[None, :]
    return variance * jnp.exp(-diff**2 / (2 * lengthscale**2))


def compute_gp_derivatives(
    Ls: np.ndarray,
    Vs: np.ndarray, 
    time_train: np.ndarray,
    time_eval: np.ndarray,
    y_train: np.ndarray,
    Ns: Optional[np.ndarray] = None,
    jitter: float = 1e-5
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute GP derivative mean and covariance.
    
    Parameters
    ----------
    Ls : array (num_modes,)
        Lengthscales per mode
    Vs : array (num_modes,)
        Variances per mode
    time_train : array (n_train,)
        Training time points
    time_eval : array (n_eval,)
        Evaluation time points
    y_train : array (num_modes, n_train)
        Training observations
    Ns : array (num_modes,), optional
        Observation noise variances per mode. When provided, K_yy includes
        the noise term so that derivatives are computed by conditioning on
        noisy observations rather than interpolating through them.
    jitter : float
        Numerical stability term
        
    Returns
    -------
    mu_z : array (num_modes, n_eval)
        Derivative means
    cov_z : array (num_modes, n_eval, n_eval)
        Derivative covariances
    """
    num_modes = len(Ls)
    mu_z, cov_z = [], []
    
    for i in range(num_modes):
        ell2 = Ls[i]**2
        
        # Kernel matrices — include observation noise if provided
        noise_i = Ns[i] if Ns is not None else 0.0
        K_yy = rbf_eval(Ls[i], Vs[i], time_train, time_train) + (noise_i + jitter) * jnp.eye(len(time_train))
        
        # Derivative kernel K_zy
        diff_zy = time_eval[:, None] - time_train[None, :]
        K_zy = -(diff_zy / ell2) * rbf_eval(Ls[i], Vs[i], time_eval, time_train)
        
        # Second derivative kernel K_zz
        diff_zz = time_eval[:, None] - time_eval[None, :]
        K_zz = ((1 - (diff_zz**2 / ell2)) / ell2) * rbf_eval(Ls[i], Vs[i], time_eval, time_eval)
        
        # Conditional mean and covariance
        w = jnp.linalg.solve(K_yy, y_train[i])
        mu_z.append(K_zy @ w)
        cov_z.append(K_zz - K_zy @ jnp.linalg.solve(K_yy, K_zy.T))
    
    return jnp.array(mu_z), jnp.array(cov_z)


class SimpleGPR:
    """Simple Gaussian Process Regression with RBF kernel and MLE hyperparameter estimation."""
    
    def __init__(self, length_scale_init=1.0, variance_init=1.0, noise_init=0.01):
        self.length_scale = length_scale_init
        self.variance = variance_init
        self.noise = noise_init
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def rbf_kernel(self, X1, X2):
        """Compute RBF kernel matrix."""
        dists = cdist(X1, X2, 'sqeuclidean')
        return self.variance * np.exp(-dists / (2 * self.length_scale**2))
    
    def neg_log_marginal_likelihood(self, params):
        """Negative log marginal likelihood for optimization."""
        ls, var, noise = np.exp(params)
        self.length_scale, self.variance, self.noise = ls, var, noise
        
        K = self.rbf_kernel(self.X_train, self.X_train)
        K_noise = K + (noise + 1e-8) * np.eye(len(self.X_train))
        
        try:
            L = np.linalg.cholesky(K_noise)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
            nll = -(-0.5 * self.y_train @ alpha - np.sum(np.log(np.diag(L))) 
                    - 0.5 * len(self.X_train) * np.log(2 * np.pi))
            return nll
        except np.linalg.LinAlgError:
            return 1e10
    
    def fit(self, X, y, verbose=False):
        """Fit GP hyperparameters via MLE."""
        self.X_train, self.y_train = X, y
        init = np.log([self.length_scale, self.variance, self.noise])
        result = minimize(self.neg_log_marginal_likelihood, init, method='L-BFGS-B')
        self.length_scale, self.variance, self.noise = np.exp(result.x)
        
        if verbose:
            print(f"  L={self.length_scale:.4f}, V={self.variance:.4f}, N={self.noise:.6f}")
        
        K = self.rbf_kernel(self.X_train, self.X_train)
        self.K_inv = np.linalg.inv(K + (self.noise + 1e-8) * np.eye(len(self.X_train)))
        return self
    
    def predict(self, X_test, return_std=True):
        """Predict at test points."""
        K_star = self.rbf_kernel(self.X_train, X_test)
        mean = K_star.T @ self.K_inv @ self.y_train
        if return_std:
            K_ss = self.rbf_kernel(X_test, X_test)
            cov = K_ss - K_star.T @ self.K_inv @ K_star
            return mean, np.sqrt(np.diag(cov) + self.noise)
        return mean


# =============================================================================
# Grid Search for Prior Operator
# =============================================================================

@dataclass
class GridSearchResult:
    """Result from grid search for prior operator."""
    best_reg: float
    best_error: float
    operator: np.ndarray
    rom: opinf.ROM
    stable_results: List[Tuple]


def grid_search_prior_operator(
    basis,
    time_domain_sampled: np.ndarray,
    snapshots_sampled: np.ndarray,
    snapshots_compressed: np.ndarray,
    operators: str = "cAH",
    inputs: Optional[np.ndarray] = None,
    input_func: Optional[Callable] = None,
    reg_values: Optional[List[float]] = None,
    verbose: bool = True
) -> GridSearchResult:
    """
    Find optimal prior operator via regularization grid search.
    
    Parameters
    ----------
    basis : opinf.basis.PODBasis
        Fitted POD basis
    time_domain_sampled : array
        Sampled time points
    snapshots_sampled : array
        Full-order snapshots
    snapshots_compressed : array
        Reduced snapshots
    operators : str
        Operator string (e.g., "cAH", "cAHBN")
    inputs : array, optional
        Input values at sampled times
    input_func : callable, optional
        Input function for prediction
    reg_values : list, optional
        Regularization values to test
    verbose : bool
        Print progress
        
    Returns
    -------
    GridSearchResult
        Best operator and associated metadata
    """
    if reg_values is None:
        reg_values = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 1e2, 1e3, 1e4]
    
    best_reg, best_error = None, float('inf')
    best_operator, best_rom = None, None
    stable_results = []
    
    if verbose:
        print(f"Testing {len(reg_values)} regularization values...")
    
    for reg in reg_values:
        try:
            rom = opinf.ROM(
                basis=basis,
                ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(time_domain_sampled),
                model=JaxCompatibleModel(operators=operators, solver=opinf.lstsq.L2Solver(regularizer=reg))
            )
            
            if inputs is not None:
                rom.fit(states=snapshots_sampled, inputs=inputs)
            else:
                rom.fit(states=snapshots_sampled)
            
            operator = rom.model.operator_matrix
            rom.model._extract_operators(np.array(operator))
            
            # Test stability
            if input_func is not None:
                pred = rom.model.predict(state0=snapshots_compressed[:, 0], t=time_domain_sampled, input_func=input_func)
            else:
                pred = rom.model.predict(state0=snapshots_compressed[:, 0], t=time_domain_sampled)
            
            sol = rom.model.predict_result_
            
            if sol.t.shape[0] == time_domain_sampled.shape[0]:
                error = np.linalg.norm(pred - snapshots_compressed) / np.linalg.norm(snapshots_compressed)
                stable_results.append((reg, error, operator, rom))
                
                if verbose:
                    print(f"  reg={reg:.1e}: STABLE, error={error:.6f}")
                
                if error < best_error:
                    best_error, best_reg = error, reg
                    best_operator, best_rom = operator, rom
            else:
                if verbose:
                    print(f"  reg={reg:.1e}: UNSTABLE")
                    
        except Exception as e:
            if verbose:
                print(f"  reg={reg:.1e}: FAILED ({str(e)[:40]})")
    
    if best_operator is None:
        raise RuntimeError("No stable operator found!")
    
    if verbose:
        print(f"\n✅ Best reg: {best_reg:.1e}, error: {best_error:.6f}")
    
    return GridSearchResult(
        best_reg=best_reg,
        best_error=best_error,
        operator=best_operator,
        rom=best_rom,
        stable_results=stable_results
    )


# =============================================================================
# GP Hyperparameter Fitting
# =============================================================================

def fit_gp_hyperparameters_mle(
    time_domain: np.ndarray,
    snapshots: np.ndarray,
    time_range_factor: float = 10.0,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[SimpleGPR]]:
    """
    Fit GP hyperparameters via MLE for each mode.
    
    Parameters
    ----------
    time_domain : array (n_time,)
        Time points
    snapshots : array (num_modes, n_time)
        Observations per mode
    time_range_factor : float
        Initial lengthscale = time_range / factor
    verbose : bool
        Print progress
        
    Returns
    -------
    Ls : array (num_modes,)
        Lengthscales
    Vs : array (num_modes,)
        Variances  
    Ns : array (num_modes,)
        Noise levels
    gp_models : list
        Fitted GP models
    """
    num_modes = snapshots.shape[0]
    time_range = time_domain.max() - time_domain.min()
    
    Ls, Vs, Ns = [], [], []
    gp_models = []
    
    if verbose:
        print("Fitting GP hyperparameters via MLE...")
    
    for i in range(num_modes):
        gp = SimpleGPR(
            length_scale_init=time_range / time_range_factor,
            variance_init=np.var(snapshots[i]),
            noise_init=0.01
        )
        gp.fit(time_domain[:, None], snapshots[i], verbose=verbose)
        gp_models.append(gp)
        
        Ls.append(gp.length_scale)
        Vs.append(gp.variance)
        Ns.append(gp.noise)
        
        if verbose:
            print(f"  Mode {i}: L={gp.length_scale:.4f}, V={gp.variance:.4f}, N={gp.noise:.6f}")
    
    return np.array(Ls), np.array(Vs), np.array(Ns), gp_models


def generate_gp_samples(
    gp_models: List[SimpleGPR],
    time_train: np.ndarray,
    time_eval: np.ndarray,
    snapshots: np.ndarray,
    num_samples: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate GP predictions at evaluation points.
    
    Returns
    -------
    Xs_means : array (num_modes, n_eval)
    Xs_covs : array (num_modes, n_eval, n_eval)
    """
    from .bgp_jax import BayesianGP
    
    num_modes = len(gp_models)
    Xi_samples = [[] for _ in range(num_modes)]
    
    gp = BayesianGP()
    gp.X_train = time_train[:, None]
    
    for j in range(num_modes):
        gp.y_train = snapshots[j]
        for _ in range(num_samples):
            mean, _, _ = gp.predict_with_hypers(
                X_test=time_eval[:, None],
                lengthscale=gp_models[j].length_scale,
                variance=gp_models[j].variance,
                noise=gp_models[j].noise
            )
            Xi_samples[j].append(mean)
    
    Xi_samples = [np.array(x) for x in Xi_samples]
    
    Xs_means = np.stack([x.mean(axis=0) for x in Xi_samples], axis=0)
    Xs_covs = np.stack([np.cov(x.T) for x in Xi_samples], axis=0)
    
    return Xs_means, Xs_covs


# =============================================================================
# Bayesian Inference Model Builders
# =============================================================================

# TODO: Review this function as it uses the prior operator as the mean of a Gaussian likelihood.
# Make sure that this function is only used when this is fixed
def build_bayesian_opinf_model(
    prior_operator: np.ndarray,
    rom,
    Ls_means: np.ndarray,
    Vs_means: np.ndarray,
    time_domain_sampled: np.ndarray,
    snapshots: np.ndarray,
    Xs_means: np.ndarray,
    Ns_means: Optional[np.ndarray] = None,
    inputs_eval: Optional[np.ndarray] = None,
    data_scaler = None,
    sample_X: bool = False,
    Xs_covs: Optional[np.ndarray] = None
):
    """
    Build a numpyro model for Bayesian operator inference.
    
    Parameters
    ----------
    prior_operator : array
        Prior mean for operator
    rom : OpInf model
        ROM for data matrix assembly
    Ls_means, Vs_means : arrays
        GP hyperparameters
    time_domain_sampled : array
        Training time points
    snapshots : array
        Training data
    Xs_means : array
        GP mean for latent states
    Ns_means : array, optional
        GP observation noise variances per mode (from MLE). Passed to
        compute_gp_derivatives so that K_yy properly accounts for noise.
    inputs_eval : array, optional
        Inputs at eval times
    data_scaler : DataScaler, optional
        For scaled data
    sample_X : bool
        Whether to sample X (True) or use deterministic (False)
    Xs_covs : array, optional
        GP covariance for latent states (required if sample_X=True)
        
    Returns
    -------
    model : callable
        Numpyro model function
    """
    num_modes = len(Ls_means)
    use_scaled = data_scaler is not None
    
    def model(time, gamma=1e-1, gamma2=1e2, normalization=1e-6):
        num_time_steps = time.shape[0]
        
        # Sample operator
        O = numpyro.sample(
            "O",
            dist.Normal(loc=prior_operator, scale=gamma * jnp.ones_like(prior_operator))
        )
        
        # Latent states
        Xs = []
        for i in range(num_modes):
            if sample_X and Xs_covs is not None:
                X = numpyro.sample(
                    f"X{i}",
                    dist.MultivariateNormal(
                        loc=Xs_means[i],
                        covariance_matrix=Xs_covs[i] + normalization * jnp.eye(Xs_covs[i].shape[0])
                    )
                )
            else:
                X = numpyro.deterministic(f"X{i}", Xs_means[i])
            Xs.append(X)
        Xs = jnp.array(Xs)
        
        # Transform to original space if scaled
        if use_scaled:
            Xs_original = jnp.array([
                Xs[i] * data_scaler.stds_[i, 0] + data_scaler.means_[i, 0]
                for i in range(num_modes)
            ])
        else:
            Xs_original = Xs
        
        # Compute operator dynamics
        f_Xi = rom.model._assemble_data_matrix(Xs_original, inputs=inputs_eval) @ O.T
        
        # Scale derivatives if needed
        if use_scaled:
            f_Xi_scaled = jnp.array([f_Xi.T[i] / data_scaler.stds_[i, 0] for i in range(num_modes)])
        else:
            f_Xi_scaled = f_Xi.T
        
        # GP derivatives
        y_train = data_scaler.transform(snapshots) if use_scaled else snapshots
        mu_z, cov_z = compute_gp_derivatives(Ls_means, Vs_means, time_domain_sampled, time, y_train, Ns=Ns_means)
        
        # ODE constraints
        for i in range(num_modes):
            constraint_cov = cov_z[i] + gamma2 * jnp.eye(num_time_steps)
            numpyro.sample(
                f'ode_constraint{i}',
                dist.MultivariateNormal(f_Xi_scaled[i], constraint_cov),
                obs=mu_z[i]
            )
    
    return model


# =============================================================================
# Inference Runners
# =============================================================================

@dataclass
class SVIResult:
    """Result from SVI inference."""
    samples: dict
    params: dict
    losses: List[float]


def run_svi(
    model: Callable,
    rng_key: jax.random.PRNGKey,
    time_eval: np.ndarray,
    gamma: float = 1e-1,
    gamma2: float = 1e2,
    normalization: float = 1e-6,
    num_steps: int = 50000,
    learning_rate: float = 0.0001,
    num_samples: int = 1000,
    verbose: bool = True,
    guide: Optional[autoguide.AutoGuide] = None
) -> SVIResult:
    """
    Run SVI inference. Defaults to AutoDelta guide.
    
    Returns
    -------
    SVIResult
        Samples, parameters, and loss history
    """
    if guide is None:
        guide = autoguide.AutoDelta(model)
    else:
        guide = guide(model)

    optimizer = Adam(step_size=learning_rate)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    if verbose:
        print(f"Running SVI (gamma={gamma}, gamma2={gamma2})...")
    
    results = svi.run(
        rng_key=rng_key,
        num_steps=num_steps,
        time=time_eval,
        gamma=gamma,
        gamma2=gamma2,
        normalization=normalization,
        progress_bar=verbose
    )
    
    params = results.params

    # Use guide.sample_posterior() to get latent samples with original site names.
    # This works uniformly across all guide types (AutoDelta, AutoNormal,
    # AutoDiagonalNormal, AutoMultivariateNormal, etc.), unlike Predictive
    # which filters out latent sites already provided by the guide.
    # We must pass the model's kwargs so that guides which internally call
    # Predictive (e.g. AutoDelta for deterministic sites) can run the model.
    model_kwargs = dict(
        time=time_eval, gamma=gamma, gamma2=gamma2, normalization=normalization
    )
    rng_key, sample_key, pred_key = jax.random.split(rng_key, 3)
    posterior_samples = guide.sample_posterior(
        sample_key, params, sample_shape=(num_samples,), **model_kwargs
    )
    
    # For guides that don't return deterministic sites (e.g. AutoNormal),
    # run the model with posterior samples to collect them.
    has_deterministic = any(k.startswith('X') for k in posterior_samples)
    if not has_deterministic:
        predictive = Predictive(
            model, posterior_samples=posterior_samples, num_samples=num_samples
        )
        model_output = predictive(pred_key, **model_kwargs)
        # Merge: latent samples from guide + deterministic/observed sites from model
        samples = {**posterior_samples, **model_output}
    else:
        samples = posterior_samples
    
    if verbose:
        print(f"✅ SVI complete! Final loss: {results.losses[-1]:.4f}")
        print(f"   Sample keys: {sorted(samples.keys())}")
    
    return SVIResult(samples=samples, params=params, losses=list(results.losses))


@dataclass
class MCMCResult:
    """Result from MCMC inference."""
    samples: dict
    mcmc: MCMC


def run_mcmc(
    model: Callable,
    rng_key: jax.random.PRNGKey,
    time_eval: np.ndarray,
    init_values: Optional[dict] = None,
    gamma: float = 1e0,
    gamma2: float = 1e0,
    normalization: float = 1e-6,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    target_accept: float = 0.9,
    verbose: bool = True
) -> MCMCResult:
    """
    Run MCMC inference.
    
    Returns
    -------
    MCMCResult
        Samples and MCMC object
    """
    if init_values is None:
        init_values = {}
    
    # Only use init_strategy if we have init values
    if init_values:
        nuts_kernel = NUTS(
            model,
            target_accept_prob=target_accept,
            init_strategy=init_to_value(values=init_values),
            max_tree_depth=12
        )
    else:
        nuts_kernel = NUTS(
            model,
            target_accept_prob=target_accept,
            max_tree_depth=12
        )
    
    mcmc = MCMC(
        nuts_kernel,
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=verbose
    )
    
    if verbose:
        print(f"Running MCMC ({num_chains} chains, {num_warmup} warmup, {num_samples} samples)...")
    
    mcmc.run(
        rng_key,
        time=time_eval,
        gamma=gamma,
        gamma2=gamma2,
        normalization=normalization
    )
    
    samples = mcmc.get_samples()
    
    if verbose:
        print(f"✅ MCMC complete! {len(samples['O'])} samples collected.")
    
    return MCMCResult(samples=samples, mcmc=mcmc)


# =============================================================================
# Prediction Utilities
# =============================================================================

def _find_operator_samples(samples: dict, site_name: str = "O") -> np.ndarray:
    """
    Robustly extract operator samples from a samples dict, regardless of
    whether samples came from MCMC, SVI with any guide type, or raw params.
    
    Search order:
      1. Exact match: "O"
      2. AutoDelta/AutoNormal loc: "O_auto_loc"
      3. Prefixed: "auto_O_auto_loc" or similar
      4. Fuzzy: any key containing 'O' but not 'ode', 'X', 'constraint'
    
    Returns the array and raises KeyError with helpful diagnostics if not found.
    """
    # 1. Exact match (MCMC or properly merged SVI samples)
    if site_name in samples:
        return np.asarray(samples[site_name])
    
    # 2. AutoDelta / AutoNormal suffix pattern
    auto_loc_key = f"{site_name}_auto_loc"
    if auto_loc_key in samples:
        return np.asarray(samples[auto_loc_key])
    
    # 3. Full prefixed pattern (raw SVI params)
    prefixed_patterns = [
        f"auto_{site_name}_auto_loc",
        f"auto_{site_name}",
    ]
    for pattern in prefixed_patterns:
        if pattern in samples:
            return np.asarray(samples[pattern])
    
    # 4. Fuzzy match: find keys containing the site name
    exclude = {'ode', 'constraint', 'latent'}
    candidates = [
        k for k in samples.keys()
        if site_name in k and not any(ex in k.lower() for ex in exclude)
    ]
    if candidates:
        # Prefer shortest key (most likely the right one)
        best = min(candidates, key=len)
        return np.asarray(samples[best])
    
    raise KeyError(
        f"Cannot find operator '{site_name}' in samples.\n"
        f"  Available keys: {sorted(samples.keys())}\n"
        f"  Hint: If using SVI, ensure run_svi() merges guide.sample_posterior() "
        f"output with model predictions."
    )


def generate_rom_predictions(
    samples: dict,
    rom,
    snapshots_compressed: np.ndarray,
    time_eval: np.ndarray,
    num_modes: int,
    num_pulls: int = 200,
    input_func: Optional[Callable] = None,
    data_scaler = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate ROM predictions from posterior samples.
    
    Handles sample dicts from any inference method:
      - MCMC: keys are original site names ("O", "X0", ...)
      - SVI + guide.sample_posterior(): same as MCMC
      - SVI raw params: keys like "auto_O_auto_loc"
    
    Returns
    -------
    Os : array
        Operator samples
    Xs : array
        Latent state samples
    rom_solves : array
        Successful ROM solutions
    """
    Os, Xs, rom_solves = [], [], []
    
    # Robustly find operator samples
    O_samples = _find_operator_samples(samples, "O")
    
    # If single sample (point estimate), expand to allow iteration
    if O_samples.ndim == 2:
        O_samples = O_samples[np.newaxis, ...]  # Add batch dimension
    
    for i in range(min(num_pulls, len(O_samples))):
        O = O_samples[i]
        Os.append(O)
        
        # Extract X samples - they're deterministic so should be in samples
        try:
            X_sampled = []
            for j in range(num_modes):
                if f'X{j}' in samples:
                    X_j = samples[f'X{j}']
                    # Handle batch dimension
                    X_j = X_j[i] if X_j.ndim > 1 else X_j
                else:
                    X_j = snapshots_compressed[j]  # Fallback to data
                X_sampled.append(X_j)
            X_sampled = np.array(X_sampled)
        except Exception:
            X_sampled = snapshots_compressed  # Fallback to data
            
        if data_scaler is not None:
            X_orig = data_scaler.inverse_transform(X_sampled)
        else:
            X_orig = X_sampled
        Xs.append(X_orig)
        
        rom.model._extract_operators(np.array(O))
        
        try:
            if input_func is not None:
                rom.model.predict(state0=snapshots_compressed[:, 0], t=time_eval, input_func=input_func)
            else:
                rom.model.predict(state0=snapshots_compressed[:, 0], t=time_eval)
            
            if rom.model.predict_result_.y.shape[1] >= time_eval.size:
                rom_solves.append(rom.model.predict_result_.y)
        except:
            pass
    
    return np.array(Os), np.array(Xs), np.array(rom_solves) if rom_solves else np.array([])

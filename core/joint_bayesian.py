# core/joint_bayesian.py
"""
Joint Bayesian Operator Inference — single-SVI full posterior.

This module implements a single-step SVI approach where GP hyperparameters
(lengthscale, variance, noise per mode), latent states, and the ROM operator
are all jointly inferred in one NumPyro model.

Key differences from the staged approach in ``bayesian_opinf.py``:
- GP hyperparameters have priors and are sampled (not fixed via MLE)
- Latent states X are drawn from the GP prior conditioned on hyperparameters
- Observations q_obs ~ N(X, noise) link latent states to data
- ODE constraints use derivative kernels computed from sampled hyperparameters
- Operator O ~ N(0, γI) is zero-centered (no grid search prior)
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass

import opinf
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide, Predictive
from numpyro.optim import Adam

from .bayesian_opinf import (
    rbf_eval,
    compute_gp_derivatives,
    generate_rom_predictions,
    SVIResult,
    GridSearchResult,
    grid_search_prior_operator,
)


# =============================================================================
# Joint Bayesian Model Builder
# =============================================================================

def build_joint_bayesian_model(
    rom,
    num_modes: int,
    time_domain_sampled: np.ndarray,
    snapshots: np.ndarray,
    inputs_eval: Optional[np.ndarray] = None,
    data_scaler=None,
    # GP hyperparameter prior config
    gp_lengthscale_prior_loc: Optional[np.ndarray] = None,
    gp_lengthscale_prior_scale: float = 1.0,
    gp_variance_prior_loc: Optional[np.ndarray] = None,
    gp_variance_prior_scale: float = 0.5,
    gp_noise_prior_loc: Optional[np.ndarray] = None,
    gp_noise_prior_scale: float = 1.0,
    # ODE constraint slack prior config
    gamma2_prior_loc: Optional[float] = None,
    gamma2_prior_scale: float = 1.0,
    learn_gamma2: bool = True,
    # Densification
    num_eval_points: Optional[int] = None,
):
    """
    Build a NumPyro model for joint Bayesian inference of GP hyperparameters,
    latent states, and the ROM operator in a single SVI step.

    Parameters
    ----------
    rom : opinf.ROM
        Fitted ROM with ``rom.model._assemble_data_matrix`` available.
    num_modes : int
        Number of POD modes.
    time_domain_sampled : array (n_train,)
        Training time points.
    snapshots : array (num_modes, n_train)
        Training observations (POD coefficients).
    inputs_eval : array (p, n_eval), optional
        Inputs at evaluation time points.
    data_scaler : DataScaler, optional
        For scaled data transforms.
    gp_lengthscale_prior_loc : array (num_modes,), optional
        LogNormal location for lengthscale priors. Defaults to log(T/20).
    gp_lengthscale_prior_scale : float
        LogNormal scale for lengthscale priors.
    gp_variance_prior_loc : array (num_modes,), optional
        LogNormal location for variance priors. Defaults to log(var(data)).
    gp_variance_prior_scale : float
        LogNormal scale for variance priors.
    gp_noise_prior_loc : array (num_modes,), optional
        LogNormal location for noise priors. Defaults to -8.0.
    gp_noise_prior_scale : float
        LogNormal scale for noise priors.
    gamma2_prior_loc : float, optional
        LogNormal location for per-mode ODE constraint slack γ₂.
        Defaults to ``log(gamma2)`` where ``gamma2`` is the model kwarg.
        Only used when ``learn_gamma2=True``.
    gamma2_prior_scale : float
        LogNormal scale for γ₂ prior. Larger values allow the model
        more freedom to find the right constraint tightness.
    learn_gamma2 : bool
        If True (default), sample a per-mode γ₂ from a LogNormal prior
        and let the model learn how tight the ODE constraints should be.
        If False, use the fixed ``gamma2`` kwarg (original behavior).
    num_eval_points : int, optional
        Number of densification points for ODE constraints.
        If None, uses training time points.

    Returns
    -------
    model : callable
        NumPyro model function with signature
        ``model(gamma=1.0, gamma2=1.0, normalization=1e-6)``
    time_eval : array
        The evaluation time points used for ODE constraints.
    """
    t_train = jnp.array(time_domain_sampled)
    n_train = len(t_train)
    T = float(t_train[-1] - t_train[0])
    y_obs = jnp.array(snapshots)
    use_scaled = data_scaler is not None

    # Evaluation time points (densified or same as training)
    if num_eval_points is not None:
        time_eval = jnp.linspace(float(t_train[0]), float(t_train[-1]), num_eval_points)
    else:
        time_eval = t_train
    n_eval = len(time_eval)

    inputs_eval_jnp = jnp.array(inputs_eval) if inputs_eval is not None else None

    # Default GP hyperparameter prior locations
    if gp_lengthscale_prior_loc is None:
        gp_lengthscale_prior_loc = jnp.full(num_modes, jnp.log(T / 20.0))
    else:
        gp_lengthscale_prior_loc = jnp.broadcast_to(
            jnp.asarray(gp_lengthscale_prior_loc, dtype=float), (num_modes,)
        )

    if gp_variance_prior_loc is None:
        gp_variance_prior_loc = jnp.array([
            jnp.log(jnp.var(y_obs[i]) + 1e-8) for i in range(num_modes)
        ])
    else:
        gp_variance_prior_loc = jnp.broadcast_to(
            jnp.asarray(gp_variance_prior_loc, dtype=float), (num_modes,)
        )

    if gp_noise_prior_loc is None:
        gp_noise_prior_loc = jnp.full(num_modes, -8.0)
    else:
        gp_noise_prior_loc = jnp.broadcast_to(
            jnp.asarray(gp_noise_prior_loc, dtype=float), (num_modes,)
        )

    # Store gamma2 prior config
    _learn_gamma2 = learn_gamma2
    _gamma2_prior_scale = gamma2_prior_scale
    _gamma2_prior_loc = gamma2_prior_loc  # resolved at model call time if None

    # Operator shape from ROM
    op_shape = rom.model.operator_matrix.shape  # (num_modes, d)

    # ---- T2: Precompute constant time-difference matrices ----
    # These are the same for every SVI step; computing them once avoids
    # redundant work inside the hot model function.
    sq_diff_tt = (t_train[:, None] - t_train[None, :]) ** 2  # (n_train, n_train)
    I_train = jnp.eye(n_train)
    I_eval = jnp.eye(n_eval)

    if num_eval_points is not None:
        diffs_et = time_eval[:, None] - t_train[None, :]       # (n_eval, n_train)
        sq_diffs_et = diffs_et ** 2
        sq_diffs_ee = (time_eval[:, None] - time_eval[None, :]) ** 2  # (n_eval, n_eval)
    else:
        diffs_et = t_train[:, None] - t_train[None, :]         # eval == train
        sq_diffs_et = sq_diff_tt
        sq_diffs_ee = sq_diff_tt

    # Precompute scaler arrays for vectorized un/re-scaling
    if use_scaled:
        scale_stds = jnp.array([data_scaler.stds_[i, 0] for i in range(num_modes)])
        scale_means = jnp.array([data_scaler.means_[i, 0] for i in range(num_modes)])

    # ---- T1: Define vmapped helper functions ----
    # These close over the precomputed diff matrices and are called inside the
    # model via jax.vmap to batch kernel/Cholesky/solve across modes.

    def _rbf_sq(ell, sig2, sq_diffs):
        """RBF kernel from precomputed squared differences."""
        return sig2 * jnp.exp(-sq_diffs / (2.0 * ell ** 2))

    def _single_gp_forward(ell, sig2, X_raw, base_jitter):
        """Kernel → Cholesky → non-centered X for one mode (T3: single factorisation)."""
        # Adaptive jitter: scale with kernel variance so the condition number
        # stays bounded regardless of σ² magnitude (prevents Cholesky NaN).
        eff_jitter = jnp.maximum(base_jitter, sig2 * 1e-4)
        K = _rbf_sq(ell, sig2, sq_diff_tt) + eff_jitter * I_train
        L = jnp.linalg.cholesky(K)
        X = L @ X_raw
        return L, X

    def _single_interp_and_deriv(ell, sig2, L, X):
        """GP interpolation + derivative conditioning for one mode.

        Reuses the Cholesky factor L from the GP prior (T3) so there is only
        ONE factorisation per mode per SVI step.  K_inv is applied via
        cho_solve (two triangular solves) instead of a fresh LU.
        """
        ell2 = ell ** 2
        # Cross-kernel K(t_eval, t_train) — shared by interpolation & derivative
        K_et = _rbf_sq(ell, sig2, sq_diffs_et)

        # K_i^{-1} @ X via Cholesky — reused for interp mean AND derivative mean
        K_inv_X = jax.scipy.linalg.cho_solve((L, True), X)
        X_eval = K_et @ K_inv_X

        # Derivative cross-covariance K'(t_eval, t_train)
        K_zy = -(diffs_et / ell2) * K_et
        mu_z = K_zy @ K_inv_X                          # reuses K_inv_X

        # Derivative auto-covariance K''(t_eval, t_eval)
        K_ee = _rbf_sq(ell, sig2, sq_diffs_ee)
        K_zz = ((1.0 - sq_diffs_ee / ell2) / ell2) * K_ee

        # Conditional covariance A = K'' − K' @ K_i^{-1} @ K'^T
        K_inv_Kzy_T = jax.scipy.linalg.cho_solve((L, True), K_zy.T)
        A = K_zz - K_zy @ K_inv_Kzy_T
        A = 0.5 * (A + A.T)                            # enforce symmetry

        return X_eval, mu_z, A

    # Batch the pure-math helpers across modes
    _batch_gp_forward = jax.vmap(_single_gp_forward, in_axes=(0, 0, 0, None))
    _batch_interp_deriv = jax.vmap(_single_interp_and_deriv)

    def model(gamma=1.0, gamma2=1.0, jitter=1e-4):

        # --- Sample or fix γ₂ per mode ---
        if _learn_gamma2:
            g2_loc = _gamma2_prior_loc if _gamma2_prior_loc is not None else jnp.log(gamma2)
            gamma2_arr = jnp.stack([
                numpyro.sample(f"gamma2_{i}", dist.LogNormal(g2_loc, _gamma2_prior_scale))
                for i in range(num_modes)
            ])
        else:
            gamma2_arr = jnp.full(num_modes, gamma2)

        # --- Sample GP hyperparameters per mode ---
        ells = jnp.stack([
            numpyro.sample(f"lengthscale_{i}",
                           dist.LogNormal(gp_lengthscale_prior_loc[i], gp_lengthscale_prior_scale))
            for i in range(num_modes)
        ])
        sig2s = jnp.stack([
            numpyro.sample(f"variance_{i}",
                           dist.LogNormal(gp_variance_prior_loc[i], gp_variance_prior_scale))
            for i in range(num_modes)
        ])
        nus = jnp.stack([
            numpyro.sample(f"noise_{i}",
                           dist.LogNormal(gp_noise_prior_loc[i], gp_noise_prior_scale))
            for i in range(num_modes)
        ])

        # --- Sample X_raw per mode (non-centered parameterisation) ---
        X_raws = jnp.stack([
            numpyro.sample(f"X_raw_{i}", dist.Normal(jnp.zeros(n_train), jnp.ones(n_train)))
            for i in range(num_modes)
        ])

        # --- T1+T3: Batched GP forward (kernel, Cholesky, X = L @ X_raw) ---
        Ls, Xs = _batch_gp_forward(ells, sig2s, X_raws, jitter)

        # Register deterministic latent states and observation likelihoods
        for i in range(num_modes):
            numpyro.deterministic(f"X_{i}", Xs[i])
            numpyro.sample(f"obs_{i}", dist.Normal(Xs[i], jnp.sqrt(nus[i])), obs=y_obs[i])

        # --- T1+T3: Batched interpolation + derivative conditioning ---
        # Single vmapped call does: GP interp to eval pts, derivative mean μ_z,
        # and derivative conditional covariance A — reusing L from above.
        Xs_eval_interp, mu_zs, As = _batch_interp_deriv(ells, sig2s, Ls, Xs)

        if num_eval_points is not None:
            Xs_eval = Xs_eval_interp
            for i in range(num_modes):
                numpyro.deterministic(f"X_eval_{i}", Xs_eval[i])
        else:
            Xs_eval = Xs  # no densification — use exact latent states

        # --- Sample operator O ~ N(0, γ·I) ---
        O = numpyro.sample(
            "O",
            dist.Normal(jnp.zeros(op_shape), gamma * jnp.ones(op_shape)),
        )

        # --- Transform to original space if scaled (vectorised) ---
        if use_scaled:
            Xs_eval_original = Xs_eval * scale_stds[:, None] + scale_means[:, None]
        else:
            Xs_eval_original = Xs_eval

        # --- Compute operator dynamics: f(X) @ O^T ---
        f_Xi = rom.model._assemble_data_matrix(
            Xs_eval_original, inputs=inputs_eval_jnp
        ) @ O.T

        if use_scaled:
            f_Xi_scaled = f_Xi.T / scale_stds[:, None]
        else:
            f_Xi_scaled = f_Xi.T

        # --- ODE constraints (loop only for numpyro site naming) ---
        for i in range(num_modes):
            numpyro.deterministic(f"mu_z_{i}", mu_zs[i])
            # Ensure constraint covariance is well-conditioned: γ₂ must
            # dominate numerical errors in A (which can have small negative
            # eigenvalues from the K' @ K^{-1} @ K'^T subtraction).
            g2_eff = jnp.maximum(gamma2_arr[i], 1e-2) + jitter
            constraint_cov = As[i] + g2_eff * I_eval
            numpyro.sample(
                f"ode_constraint_{i}",
                dist.MultivariateNormal(
                    loc=f_Xi_scaled[i],
                    covariance_matrix=constraint_cov,
                ),
                obs=mu_zs[i],
            )

    return model, np.array(time_eval)


# =============================================================================
# Joint SVI Runner
# =============================================================================

def run_joint_svi(
    model: Callable,
    rng_key: jax.random.PRNGKey,
    gamma: float = 1.0,
    gamma2: float = 1.0,
    jitter: float = 1e-4,
    num_steps: int = 50000,
    learning_rate: float = 1e-3,
    num_samples: int = 1000,
    verbose: bool = True,
    guide_class=None,
) -> SVIResult:
    """
    Run SVI for the joint Bayesian model.

    Parameters
    ----------
    model : callable
        NumPyro model from ``build_joint_bayesian_model``.
    rng_key : PRNGKey
    gamma : float
        Operator prior scale.
    gamma2 : float
        ODE constraint slack.
    jitter : float
        Nugget for numerical stability.
    num_steps : int
        Number of SVI iterations.
    learning_rate : float
        Adam learning rate.
    num_samples : int
        Number of posterior samples to draw.
    verbose : bool
        Show progress bar.
    guide_class : autoguide class, optional
        Defaults to ``AutoNormal``.

    Returns
    -------
    SVIResult
        Samples, parameters, and loss history.
    """
    if guide_class is None:
        guide_class = autoguide.AutoNormal

    guide = guide_class(model)
    optimizer = Adam(step_size=learning_rate)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    if verbose:
        print(f"Running joint SVI (gamma={gamma}, gamma2={gamma2}, "
              f"guide={guide_class.__name__}, steps={num_steps})...")

    results = svi.run(
        rng_key=rng_key,
        num_steps=num_steps,
        gamma=gamma,
        gamma2=gamma2,
        jitter=jitter,
        progress_bar=verbose,
    )

    params = results.params

    model_kwargs = dict(gamma=gamma, gamma2=gamma2, jitter=jitter)
    rng_key, sample_key, pred_key = jax.random.split(rng_key, 3)

    posterior_samples = guide.sample_posterior(
        sample_key, params, sample_shape=(num_samples,), **model_kwargs,
    )

    predictive = Predictive(
        model, posterior_samples=posterior_samples, num_samples=num_samples,
        return_sites=None,
    )
    model_output = predictive(pred_key, **model_kwargs)
    samples = {**model_output, **posterior_samples}

    if verbose:
        print(f"✅ Joint SVI complete! Final loss: {results.losses[-1]:.4f}")
        print(f"   Sample keys: {sorted(samples.keys())}")

    return SVIResult(samples=samples, params=params, losses=list(results.losses))


# =============================================================================
# Post-inference Utilities
# =============================================================================

def extract_gp_posterior(
    samples: dict,
    num_modes: int,
    time_train: np.ndarray,
    time_eval: np.ndarray,
    y_train: np.ndarray,
    jitter: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the GP fit from the joint posterior.

    Computes the GP predictive mean and std at ``time_eval`` using the
    posterior median hyperparameters.

    Parameters
    ----------
    samples : dict
        Posterior samples from ``run_joint_svi``.
    num_modes : int
    time_train : array (n_train,)
    time_eval : array (n_eval,)
    y_train : array (num_modes, n_train)
    jitter : float

    Returns
    -------
    gp_means : array (num_modes, n_eval)
    gp_stds : array (num_modes, n_eval)
    Ls : array (num_modes,)  — posterior median lengthscales
    Vs : array (num_modes,)  — posterior median variances
    Ns : array (num_modes,)  — posterior median noise levels
    """
    Ls = np.array([np.median(samples[f"lengthscale_{i}"]) for i in range(num_modes)])
    Vs = np.array([np.median(samples[f"variance_{i}"]) for i in range(num_modes)])
    Ns = np.array([np.median(samples[f"noise_{i}"]) for i in range(num_modes)])

    t_tr = jnp.array(time_train)
    t_ev = jnp.array(time_eval)
    gp_means, gp_stds = [], []

    for i in range(num_modes):
        K_train = rbf_eval(Ls[i], Vs[i], t_tr, t_tr) + (Ns[i] + jitter) * jnp.eye(len(t_tr))
        K_eval_train = rbf_eval(Ls[i], Vs[i], t_ev, t_tr)
        K_eval_eval = rbf_eval(Ls[i], Vs[i], t_ev, t_ev)

        alpha = jnp.linalg.solve(K_train, y_train[i])
        mu = K_eval_train @ alpha
        cov = K_eval_eval - K_eval_train @ jnp.linalg.solve(K_train, K_eval_train.T)
        std = jnp.sqrt(jnp.maximum(jnp.diag(cov), 0.0))

        gp_means.append(np.array(mu))
        gp_stds.append(np.array(std))

    return np.array(gp_means), np.array(gp_stds), Ls, Vs, Ns


def extract_derivative_posterior(
    Ls: np.ndarray,
    Vs: np.ndarray,
    Ns: np.ndarray,
    time_train: np.ndarray,
    time_eval: np.ndarray,
    y_train: np.ndarray,
    jitter: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GP derivative posterior at ``time_eval`` given hyperparameters.

    Parameters
    ----------
    Ls, Vs, Ns : array (num_modes,)
        GP hyperparameters.
    time_train, time_eval : arrays
    y_train : array (num_modes, n_train)

    Returns
    -------
    mu_z : array (num_modes, n_eval)
        Derivative posterior means.
    std_z : array (num_modes, n_eval)
        Derivative posterior standard deviations.
    """
    mu_z, cov_z = compute_gp_derivatives(Ls, Vs, time_train, time_eval, y_train, Ns=Ns)
    std_z = np.array([np.sqrt(np.maximum(np.diag(cov_z[i]), 0.0)) for i in range(len(Ls))])
    return np.array(mu_z), std_z


def gp_based_opinf_baseline(
    basis,
    gp_means: np.ndarray,
    time_eval: np.ndarray,
    snapshots_compressed: np.ndarray,
    operators: str,
    inputs: Optional[np.ndarray] = None,
    input_func: Optional[Callable] = None,
    ivp_method: Optional[str] = None,
    reg_values: Optional[List[float]] = None,
) -> GridSearchResult:
    """
    Fit a deterministic OpInf operator on GP posterior means as a baseline.

    This represents the "best case" ROM from the GP fit alone (no Bayesian
    operator inference).

    Parameters
    ----------
    basis : opinf.basis.PODBasis
    gp_means : array (num_modes, n_eval)
        GP posterior mean states at ``time_eval``.
    time_eval : array (n_eval,)
    snapshots_compressed : array (num_modes, n_train)
        Original compressed snapshots (used for initial condition).
    operators : str
        Operator string (e.g. "cAH", "cAHBN").
    inputs : array, optional
    input_func : callable, optional
    ivp_method : str, optional
    reg_values : list, optional

    Returns
    -------
    GridSearchResult
    """
    # Reconstruct full-order from GP means to use with grid search
    gp_full = basis.decompress(gp_means)

    return grid_search_prior_operator(
        basis=basis,
        time_domain_sampled=time_eval,
        snapshots_sampled=gp_full,
        snapshots_compressed=gp_means,
        operators=operators,
        inputs=inputs,
        input_func=input_func,
        reg_values=reg_values if reg_values is not None else np.logspace(-16, 4, 41).tolist(),
        verbose=True,
        ivp_method=ivp_method,
    )

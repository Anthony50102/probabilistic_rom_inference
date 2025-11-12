#!/usr/bin/env python3
"""
Simplified FitzHugh-Nagumo Operator Inference with Bayesian GPs and MCMC.

This script performs:
1. Generate training data from the FitzHugh-Nagumo PDE
2. Dimensionality reduction via POD
3. Gaussian Process Regression (GPR) for each POD mode
4. MCMC-based operator inference with stability constraints
"""

import argparse
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append("../")

import opinf
import config
import step1_generate_data as step1
from scaler import DataScaler


# ============================================================================
# GP Regression Implementation
# ============================================================================

class SimpleGPR:
    """Simple Gaussian Process Regression with RBF kernel."""
    
    def __init__(self, length_scale_init=1.0, variance_init=1.0, noise_init=0.01):
        self.length_scale = length_scale_init
        self.variance = variance_init
        self.noise = noise_init
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        
    def rbf_kernel(self, X1, X2, length_scale, variance):
        """RBF kernel function."""
        dists = cdist(X1, X2, 'sqeuclidean')
        return variance * np.exp(-dists / (2 * length_scale**2))
    
    def neg_log_marginal_likelihood(self, params):
        """Negative log marginal likelihood for optimization."""
        length_scale, variance, noise = np.exp(params)
        
        K = self.rbf_kernel(self.X_train, self.X_train, length_scale, variance)
        K_noise = K + (noise + 1e-8) * np.eye(len(self.X_train))
        
        try:
            L = np.linalg.cholesky(K_noise)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
            log_likelihood = (-0.5 * self.y_train.T @ alpha 
                            - np.sum(np.log(np.diag(L))) 
                            - 0.5 * len(self.X_train) * np.log(2 * np.pi))
            return -log_likelihood
        except np.linalg.LinAlgError:
            return 1e10
    
    def fit(self, X, y, verbose=True):
        """Fit GP by optimizing hyperparameters."""
        self.X_train = X
        self.y_train = y
        
        init_params = np.log([self.length_scale, self.variance, self.noise])
        
        result = minimize(
            self.neg_log_marginal_likelihood,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        
        self.length_scale, self.variance, self.noise = np.exp(result.x)
        
        if verbose:
            print(f"  Length scale: {self.length_scale:.6f}")
            print(f"  Variance: {self.variance:.6f}")
            print(f"  Noise: {self.noise:.6f}")
        
        K = self.rbf_kernel(self.X_train, self.X_train, self.length_scale, self.variance)
        K_noise = K + (self.noise + 1e-8) * np.eye(len(self.X_train))
        self.K_inv = np.linalg.inv(K_noise)
        
        return self
    
    def predict(self, X_test, return_std=True):
        """Predict at test points."""
        K_star = self.rbf_kernel(self.X_train, X_test, self.length_scale, self.variance)
        mean = K_star.T @ self.K_inv @ self.y_train
        
        if return_std:
            K_star_star = self.rbf_kernel(X_test, X_test, self.length_scale, self.variance)
            cov = K_star_star - K_star.T @ self.K_inv @ K_star
            std = np.sqrt(np.diag(cov) + self.noise)
            return mean, std
        
        return mean


# ============================================================================
# JAX-Compatible Operator Model
# ============================================================================

def binom(x, y):
    """Binomial coefficient using JAX."""
    return jnp.exp(jax.scipy.special.gammaln(x + 1) - 
                   jax.scipy.special.gammaln(y + 1) - 
                   jax.scipy.special.gammaln(x - y + 1))

def Quadraticckron(state):
    """Compute Kronecker product for quadratic terms."""
    return jnp.concatenate(
        [state[i] * state[:i + 1] for i in range(state.shape[0])],
        axis=0,
    )

def khatri_rao(a, b):
    """Khatri-Rao product."""
    return jnp.vstack([jnp.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T

class JaxCompatibleModel(opinf.models.ContinuousModel):
    """OpInf model with JAX-compatible data matrix assembly."""
    
    def _assemble_data_matrix(self, states, inputs):
        """Assemble the data matrix for operator inference."""
        blocks = []
        for i in self._indices_of_operators_to_infer:
            op = self.operators[i]
            if isinstance(op, opinf.operators.ConstantOperator):
                block = jnp.ones((1, jnp.atleast_1d(states).shape[-1]))
            elif isinstance(op, opinf.operators.LinearOperator):
                block = jnp.atleast_2d(states)
            elif isinstance(op, opinf.operators.QuadraticOperator):
                block = Quadraticckron(jnp.atleast_2d(states))
            elif isinstance(op, opinf.operators.InputOperator):
                block = jnp.atleast_2d(inputs)
            elif isinstance(op, opinf.operators.StateInputOperator):
                block = khatri_rao(jnp.atleast_2d(inputs), jnp.atleast_2d(states))
            else:
                raise ValueError(f"Unknown operator type: {type(op)}")
            blocks.append(block.T)

        return jnp.hstack(blocks)


# ============================================================================
# GP Derivative Computation
# ============================================================================

def flatten_time(t):
    """Return t with shape (n,) no matter if (n,), (n,1) or (1,n) was given."""
    return jnp.ravel(t)

def rbf_eval(lengthscale, variance, t, t2):
    """Full n×n RBF kernel matrix."""
    t = flatten_time(t)
    t2 = flatten_time(t2)
    diff = t[:, None] - t2[None, :]
    ell2 = lengthscale ** 2
    return variance * jnp.exp(-diff**2 / (2.0 * ell2))

def joint_gp_derivatives(Ls, Vs, time1, time2, snapshots_compressed, num_modes, use_scaled, data_scaler):
    """Compute GP derivative mean and covariance."""
    K_yys = []
    K_zys = []
    K_zzs = []
    
    for i in range(num_modes):
        ell2 = Ls[i]**2
        
        rbf_yy = rbf_eval(Ls[i], Vs[i], time1, time1) 
        rbf_zy = rbf_eval(Ls[i], Vs[i], time2, time1)
        rbf_zz = rbf_eval(Ls[i], Vs[i], time2, time2)
        
        K_yy = rbf_yy + 1e-5 * np.eye(len(time1))
        
        diff_zy = time2[:, None] - time1[None, :]
        K_zy = -(diff_zy / ell2) * rbf_zy
        
        diff_zz = time2[:, None] - time2[None, :]
        K_zz = ((1 - (diff_zz**2 / ell2)) / ell2) * rbf_zz
        
        K_yys.append(K_yy)
        K_zys.append(K_zy)
        K_zzs.append(K_zz)

    snapshots_for_gp = data_scaler.transform(snapshots_compressed) if use_scaled else snapshots_compressed
    
    mu_z = []
    cov_z = []
    for i in range(num_modes):
        w = jnp.linalg.solve(K_yys[i], snapshots_for_gp[i])
        mu_zi = K_zys[i] @ w
        mu_z.append(mu_zi)

        cov_zi = K_zzs[i] - K_zys[i] @ jnp.linalg.solve(K_yys[i], K_zys[i].T)
        cov_z.append(cov_zi)

    return jnp.array(mu_z), jnp.array(cov_z)


# ============================================================================
# MCMC Model
# ============================================================================

def mcmc_model(time, rom, loaded_operator, Ls, Vs, time_domain_sampled, 
               snapshots_compressed, num_modes, inputs_eval_time, 
               Xs_means, Xs_covs, data_scaler, use_scaled,
               gamma, gamma2, normalization):
    """MCMC model for operator inference with GP constraints."""
    
    num_time_steps = time.shape[0]
    
    # Sample operator perturbation
    O_standardized = numpyro.sample(
        "O_standardized", dist.Normal(0.0, 1.0).expand(loaded_operator.shape)
    )
    O_uncertainty = numpyro.sample(
        "O_uncertainty", dist.Normal(jnp.zeros(loaded_operator.shape), 
                                    gamma * jnp.ones(loaded_operator.shape))
    )
    O = numpyro.deterministic("O", O_standardized * gamma + O_uncertainty + loaded_operator)

    # Sample latent states
    Xs = []
    for i in range(num_modes):
        Xs.append(
            numpyro.sample(
                f"X{i}",
                dist.MultivariateNormal(
                    loc=Xs_means[i], 
                    covariance_matrix=Xs_covs[i] + normalization * jnp.eye(Xs_covs[i].shape[0])
                ),
            )
        )
    
    Xs = jnp.array(Xs)
    
    # Transform to original space if using scaled data
    if use_scaled:
        Xs_original = jnp.array([
            Xs[i] * data_scaler.stds_[i, 0] + data_scaler.means_[i, 0] 
            for i in range(num_modes)
        ])
    else:
        Xs_original = Xs
    
    # Compute dynamics in original space
    f_Xi_ohat = rom.model._assemble_data_matrix(Xs_original, inputs=inputs_eval_time) @ O.T
    
    # Transform dynamics derivatives if using scaled space
    if use_scaled:
        f_Xi_scaled = jnp.array([
            f_Xi_ohat.T[i] / data_scaler.stds_[i, 0] 
            for i in range(num_modes)
        ])
    else:
        f_Xi_scaled = f_Xi_ohat.T

    # Get GP derivatives
    mu_z, cov_z = joint_gp_derivatives(Ls, Vs, time_domain_sampled, time, 
                                       snapshots_compressed, num_modes, 
                                       use_scaled, data_scaler)
    
    # ODE constraints
    for i in range(num_modes):
        mu_zi = mu_z[i]
        cov_zi = cov_z[i]
        constraint_cov = cov_zi + gamma2 * jnp.eye(num_time_steps)

        numpyro.sample(
            f'ode_constraint{i}',
            dist.MultivariateNormal(mu_zi, constraint_cov),
            obs=f_Xi_scaled[i]
        )


# ============================================================================
# Derivative Matching Visualization
# ============================================================================

def plot_derivative_matching(rom, operator, Ls, Vs, time_domain_sampled, time_domain_eval,
                            snapshots_compressed, num_modes, input_func, data_scaler, 
                            use_scaled, title="Operator Derivative Matching", 
                            save_path=None):
    """
    Plot comparison between GP-predicted derivatives and operator-predicted derivatives.
    
    This helps verify that the operator is producing derivatives consistent with the GP.
    """
    # Compute GP derivatives
    K_yys, K_zys, K_zzs = [], [], []
    for i in range(num_modes):
        ell2 = Ls[i]**2
        
        rbf_yy = rbf_eval(Ls[i], Vs[i], time_domain_sampled, time_domain_sampled)
        rbf_zy = rbf_eval(Ls[i], Vs[i], time_domain_eval, time_domain_sampled)
        
        K_yy = rbf_yy + 1e-5 * np.eye(len(time_domain_sampled))
        
        diff_zy = time_domain_eval[:, None] - time_domain_sampled[None, :]
        K_zy = -(diff_zy / ell2) * rbf_zy
        
        K_yys.append(K_yy)
        K_zys.append(K_zy)
    
    # Select appropriate data
    snapshots_for_gp = data_scaler.transform(snapshots_compressed) if use_scaled else snapshots_compressed
    
    # Compute GP derivative predictions
    mu_z_gp = []
    for i in range(num_modes):
        w = np.linalg.solve(K_yys[i], snapshots_for_gp[i])
        mu_zi = K_zys[i] @ w
        mu_z_gp.append(mu_zi)
    
    # Compute operator derivatives
    # Use the mean of the latent states (snapshots) at eval points
    # For simplicity, interpolate snapshots to eval points
    from scipy.interpolate import interp1d
    
    X_eval = np.zeros((num_modes, len(time_domain_eval)))
    for i in range(num_modes):
        if use_scaled:
            interp_func = interp1d(time_domain_sampled, snapshots_for_gp[i], 
                                  kind='cubic', fill_value='extrapolate')
        else:
            interp_func = interp1d(time_domain_sampled, snapshots_compressed[i], 
                                  kind='cubic', fill_value='extrapolate')
        X_eval[i] = interp_func(time_domain_eval)
    
    # Transform to original space for operator application if needed
    if use_scaled:
        X_eval_orig = data_scaler.inverse_transform(X_eval)
    else:
        X_eval_orig = X_eval
    
    # Get inputs at eval points
    inputs_eval = input_func(time_domain_eval)
    
    # Compute operator derivatives in original space
    # Ensure operator is a NumPy array
    operator_np = np.array(operator) if not isinstance(operator, np.ndarray) else operator
    rom.model._extract_operators(operator_np)
    f_X_operator = rom.model._assemble_data_matrix(X_eval_orig, inputs=inputs_eval) @ operator_np.T
    
    # Scale back if needed
    if use_scaled:
        f_X_scaled = np.array([
            f_X_operator.T[i] / data_scaler.stds_[i, 0] 
            for i in range(num_modes)
        ])
    else:
        f_X_scaled = f_X_operator.T
    
    # Create plot
    fig, axes = plt.subplots(num_modes, 1, figsize=(12, 3*num_modes))
    if num_modes == 1:
        axes = [axes]
    
    for i in range(num_modes):
        axes[i].plot(time_domain_eval, mu_z_gp[i], 'b-', linewidth=2, 
                    label='GP Derivative', alpha=0.7)
        axes[i].plot(time_domain_eval, f_X_scaled[i], 'r--', linewidth=2, 
                    label='Operator Derivative', alpha=0.7)
        
        # Plot difference
        diff = mu_z_gp[i] - f_X_scaled[i]
        rel_error = np.linalg.norm(diff) / (np.linalg.norm(mu_z_gp[i]) + 1e-10)
        
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel(f'Mode {i} Derivative')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'Mode {i} - Relative Error: {rel_error:.4f}')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Derivative matching plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


# ============================================================================
# Main Function
# ============================================================================

def main(gamma, gamma2, use_scaled_data=True, 
         training_span=(0, 6), num_samples=400, num_pod_modes=3,
         num_warmup=200, num_mcmc_samples=200, num_chains=1,
         normalization=1e-4, seed=42, output_dir="mcmc_results"):
    """
    Main execution function.
    
    Parameters
    ----------
    gamma : float
        Operator uncertainty scale
    gamma2 : float
        ODE constraint covariance scale
    use_scaled_data : bool
        Whether to use scaled data for GP fitting
    training_span : tuple
        Time span for training data
    num_samples : int
        Number of snapshots to sample
    num_pod_modes : int
        Number of POD modes to use
    num_warmup : int
        MCMC warmup iterations
    num_mcmc_samples : int
        MCMC sampling iterations
    num_chains : int
        Number of MCMC chains
    normalization : float
        Regularization for numerical stability
    seed : int
        Random seed
    output_dir : str
        Directory to save results
    """
    
    print("="*70)
    print("FitzHugh-Nagumo Bayesian Operator Inference")
    print("="*70)
    print(f"Configuration:")
    print(f"  gamma (operator uncertainty): {gamma}")
    print(f"  gamma2 (ODE constraint): {gamma2}")
    print(f"  use_scaled_data: {use_scaled_data}")
    print(f"  training_span: {training_span}")
    print(f"  num_samples: {num_samples}")
    print(f"  num_pod_modes: {num_pod_modes}")
    print(f"  MCMC: {num_warmup} warmup + {num_mcmc_samples} samples × {num_chains} chains")
    print("="*70)
    
    # Set random seeds
    np.random.seed(seed)
    rng_key = jax.random.PRNGKey(seed)
    
    # Update config time domain
    config.time_domain = np.linspace(0, 8, 801)
    time_domain = config.time_domain  # Store for later use
    
    # ========================================================================
    # Step 1: Generate Data
    # ========================================================================
    print("\n[1/8] Generating training data...")
    
    (model, time_domain, true_states, time_domain_sampled, snapshots_sampled) = \
        step1.trajectory(training_span, num_samples, config, noiselevel=0.01)
    
    time_domain_eval_training = np.linspace(0, training_span[-1], num_samples)
    input_func = config.ReducedOrderModel.input_func
    inputs = input_func(time_domain_sampled)
    inputs_eval_time = input_func(time_domain_eval_training)
    
    print(f"  Generated {snapshots_sampled.shape[1]} snapshots over [{training_span[0]}, {training_span[1]}]")
    
    # ========================================================================
    # Step 2: Dimensionality Reduction (POD)
    # ========================================================================
    print(f"\n[2/8] Performing POD with {num_pod_modes} modes...")
    
    basis = config.Basis(num_vectors=num_pod_modes)
    basis.fit(snapshots_sampled)
    snapshots_compressed = basis.compress(snapshots_sampled)
    full_states_compressed = basis.compress(true_states)
    
    print(f"  Reduced from {snapshots_sampled.shape[0]} to {num_pod_modes} dimensions")
    
    # ========================================================================
    # Step 3: Data Scaling
    # ========================================================================
    print(f"\n[3/8] Setting up data scaler (use_scaled={use_scaled_data})...")
    
    data_scaler = DataScaler(num_modes=num_pod_modes)
    data_scaler.fit(snapshots_compressed)
    
    if use_scaled_data:
        snapshots_compressed_scaled = data_scaler.transform(snapshots_compressed)
        training_data = snapshots_compressed_scaled
        print(f"  Using SCALED data (mean≈0, std≈1)")
    else:
        training_data = snapshots_compressed
        print(f"  Using UNSCALED data")
    
    # ========================================================================
    # Step 4: Fit Gaussian Process for Each Mode
    # ========================================================================
    print(f"\n[4/8] Fitting GPR models for each POD mode...")
    
    gp_models = []
    gp_hyperparams = {'lengthscales': [], 'variances': [], 'noises': []}
    
    for i in range(num_pod_modes):
        print(f"\n  Mode {i}:")
        gp = SimpleGPR(
            length_scale_init=training_span[-1]/10,
            variance_init=1.0 if use_scaled_data else np.var(training_data[i]),
            noise_init=0.01
        )
        gp.fit(time_domain_sampled[:, None], training_data[i])
        gp_models.append(gp)
        
        gp_hyperparams['lengthscales'].append(gp.length_scale)
        gp_hyperparams['variances'].append(gp.variance)
        gp_hyperparams['noises'].append(gp.noise)
    
    Ls = np.array(gp_hyperparams['lengthscales'])
    Vs = np.array(gp_hyperparams['variances'])
    
    # Compute GP predictions for latent state statistics
    print(f"\n  Computing latent state statistics...")
    time_domain_test = time_domain_eval_training
    gp_predictions = []
    
    for i in range(num_pod_modes):
        mean, std = gp_models[i].predict(time_domain_test[:, None])
        gp_predictions.append(mean)
    
    Xs_means = np.array(gp_predictions)
    Xs_covs = np.array([np.cov(gp_predictions[i][None, :]) * np.eye(len(time_domain_test)) 
                        for i in range(num_pod_modes)])
    
    # ========================================================================
    # Step 5: Generate Prior Operator via Line Search
    # ========================================================================
    print(f"\n[5/8] Finding optimal prior operator via line search...")
    
    # Test different regularization values
    reg_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    best_reg = None
    best_error = float('inf')
    best_operator = None
    best_rom = None
    stable_results = []
    
    print(f"  Testing {len(reg_values)} regularization values...")
    
    for reg in reg_values:
        try:
            # Create ROM with current regularization
            rom_candidate = opinf.ROM(
                basis=basis,
                ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(time_domain_sampled),
                model=JaxCompatibleModel(
                    operators="cAHBN",
                    solver=opinf.lstsq.L2Solver(regularizer=reg),
                )
            )
            
            # Fit the ROM
            rom_candidate.fit(states=snapshots_sampled, inputs=inputs)
            candidate_operator = rom_candidate.model.operator_matrix
            
            # Extract and set operator
            rom_candidate.model._extract_operators(np.array(candidate_operator))
            
            # Test stability on training domain
            deter_pred = rom_candidate.model.predict(
                state0=snapshots_compressed[:, 0], 
                t=time_domain_sampled, 
                input_func=input_func
            )
            deter_sol = rom_candidate.model.predict_result_
            
            # Check if stable (completed all timesteps)
            if deter_sol.t.shape[0] == snapshots_sampled.shape[1]:
                # Compute fit error
                error = np.linalg.norm(deter_pred - snapshots_compressed) / np.linalg.norm(snapshots_compressed)
                stable_results.append((reg, error, candidate_operator, rom_candidate))
                
                print(f"    reg={reg:.1e}: STABLE, error={error:.6f}")
                
                if error < best_error:
                    best_error = error
                    best_reg = reg
                    best_operator = candidate_operator
                    best_rom = rom_candidate
            else:
                print(f"    reg={reg:.1e}: UNSTABLE (terminated at {deter_sol.t.shape[0]}/{snapshots_sampled.shape[1]} steps)")
                
        except Exception as e:
            print(f"    reg={reg:.1e}: FAILED ({str(e)[:50]})")
            continue
    
    if best_operator is None:
        raise RuntimeError("No stable operator found! Try different regularization values.")
    
    print(f"\n  Best regularization: {best_reg:.1e}")
    print(f"  Best fit error: {best_error:.6f}")
    print(f"  Operator shape: {best_operator.shape}")
    
    # Use best ROM
    rom = best_rom
    loaded_operator = best_operator
    
    # Generate and save plot of best fit
    print(f"\n  Generating best fit plot...")
    fig, axes = plt.subplots(num_pod_modes, 1, figsize=(12, 3*num_pod_modes))
    if num_pod_modes == 1:
        axes = [axes]
    
    # Get best prediction
    best_pred = rom.model.predict(
        state0=snapshots_compressed[:, 0],
        t=time_domain_sampled,
        input_func=input_func
    )
    
    for i in range(num_pod_modes):
        axes[i].plot(time_domain_sampled, snapshots_compressed[i], 'k-', linewidth=2, label='True', alpha=0.7)
        axes[i].plot(time_domain_sampled, best_pred[i], 'r--', linewidth=2, label=f'ROM (reg={best_reg:.1e})')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel(f'Mode {i}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    fig.suptitle(f'Best Prior Operator Fit (error={best_error:.6f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"prior_operator_fit_g{gamma}_g2{gamma2}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Best fit plot saved to: {plot_path}")
    
    # Generate derivative matching plot for prior operator
    print(f"\n  Generating prior operator derivative matching plot...")
    plot_derivative_matching(
        rom=rom,
        operator=loaded_operator,
        Ls=Ls,
        Vs=Vs,
        time_domain_sampled=time_domain_sampled,
        time_domain_eval=time_domain_eval_training,
        snapshots_compressed=snapshots_compressed,
        num_modes=num_pod_modes,
        input_func=input_func,
        data_scaler=data_scaler,
        use_scaled=use_scaled_data,
        title=f"Prior Operator Derivative Matching (reg={best_reg:.1e})",
        save_path=os.path.join(output_dir, f"prior_derivative_matching_g{gamma}_g2{gamma2}.png")
    )
    
    # ========================================================================
    # Step 6: MCMC Sampling
    # ========================================================================
    print(f"\n[6/8] Running MCMC...")
    
    numpyro.set_host_device_count(num_chains)
    
    nuts_kernel = NUTS(
        lambda time: mcmc_model(
            time, rom, loaded_operator, Ls, Vs, time_domain_sampled,
            snapshots_compressed, num_pod_modes, inputs_eval_time,
            Xs_means, Xs_covs, data_scaler, use_scaled_data,
            gamma, gamma2, normalization
        ),
        target_accept_prob=0.9,
    )
    
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_mcmc_samples,
        num_chains=num_chains,
        chain_method='parallel',
        progress_bar=True,
    )
    
    mcmc.run(rng_key, time=time_domain_eval_training)
    
    # ========================================================================
    # Step 7: Extract Results
    # ========================================================================
    print(f"\n[7/8] Extracting results...")
    
    samples = mcmc.get_samples()
    mcmc.print_summary()
    
    # Compute statistics
    O_samples = samples['O']
    O_mean = np.array(np.mean(O_samples, axis=0))  # Convert to NumPy
    O_std = np.array(np.std(O_samples, axis=0))    # Convert to NumPy
    
    print(f"\n" + "="*70)
    print("Results Summary:")
    print(f"  Number of samples: {len(O_samples)}")
    print(f"  Operator mean norm: {np.linalg.norm(O_mean):.4f}")
    print(f"  Operator std norm: {np.linalg.norm(O_std):.4f}")
    print(f"  Prior operator norm: {np.linalg.norm(loaded_operator):.4f}")
    print("="*70)
    
    # Generate derivative matching plot for posterior mean operator
    print(f"\nGenerating posterior operator derivative matching plot...")
    plot_derivative_matching(
        rom=rom,
        operator=O_mean,
        Ls=Ls,
        Vs=Vs,
        time_domain_sampled=time_domain_sampled,
        time_domain_eval=time_domain_eval_training,
        snapshots_compressed=snapshots_compressed,
        num_modes=num_pod_modes,
        input_func=input_func,
        data_scaler=data_scaler,
        use_scaled=use_scaled_data,
        title="Posterior Mean Operator Derivative Matching",
        save_path=os.path.join(output_dir, f"posterior_derivative_matching_g{gamma}_g2{gamma2}.png")
    )
    
    # ========================================================================
    # Step 8: Generate Operator Plots
    # ========================================================================
    print(f"\n[8/8] Generating operator plots...")
    
    try:
        # Import plotting utilities
        sys.path.append("../")
        import fitz_plotter
        
        # Create plotter instance
        plotter = fitz_plotter.FitzPlotter(
            numPODmodes=num_pod_modes,
            time_domain_training=time_domain_sampled,
            time_domain_prediction=time_domain,
            time_domain_eval_training=time_domain_eval_training,
            time_domain_eval_prediction=time_domain_eval_training,  # Use training for now
            snapshots_training=snapshots_compressed,
            snapshots_prediction=full_states_compressed,
            scaler=data_scaler if use_scaled_data else None
        )
        
        # Generate operator trajectory plots
        os.makedirs(output_dir, exist_ok=True)
        
        plot_file = os.path.join(output_dir, f"operator_trajectories_g{gamma}_g2{gamma2}.png")
        
        plotter.operator_plot(
            q0=snapshots_compressed[:, 0],
            operator_samples=O_samples,
            latent_state_samples=[samples[f'X{i}'] for i in range(num_pod_modes)],
            rom=rom,
            input_func=input_func,
            figsize=(21, 12),
            max_num_samples=min(len(O_samples), 100),  # Limit to 100 samples for speed
            plot_samples=False,  # Only plot mean/median/percentiles
            save=True,
            save_path=plot_file
        )
        
        print(f"  Operator plots saved to: {plot_file}")
        
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")
        print(f"  Continuing without plots...")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    print(f"\nSaving results...")
    
    results = {
        'O_samples': O_samples,
        'O_mean': O_mean,
        'O_std': O_std,
        'prior_operator': loaded_operator,
        'hyperparameters': {
            'gamma': gamma,
            'gamma2': gamma2,
            'use_scaled_data': use_scaled_data,
        },
        'gp_hyperparams': gp_hyperparams,
        'latent_states': {
            f'X{i}': samples[f'X{i}'] for i in range(num_pod_modes)
        }
    }
    
    output_file = os.path.join(output_dir, f"results_g{gamma}_g2{gamma2}.npz")
    np.savez(output_file, **results)
    print(f"Results saved to: {output_file}")
    
    print(f"\n" + "="*70)
    print("All outputs saved:")
    print(f"  1. Prior operator fit: prior_operator_fit_g{gamma}_g2{gamma2}.png")
    print(f"  2. Prior derivative matching: prior_derivative_matching_g{gamma}_g2{gamma2}.png")
    print(f"  3. Posterior derivative matching: posterior_derivative_matching_g{gamma}_g2{gamma2}.png")
    print(f"  4. Operator trajectories: operator_trajectories_g{gamma}_g2{gamma2}.png")
    print(f"  5. Numerical results: results_g{gamma}_g2{gamma2}.npz")
    print("="*70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FitzHugh-Nagumo Bayesian Operator Inference"
    )
    parser.add_argument("--gamma", type=float, required=True,
                       help="Operator uncertainty scale (e.g., 5e1)")
    parser.add_argument("--gamma2", type=float, required=True,
                       help="ODE constraint covariance scale (e.g., 5e-1)")
    parser.add_argument("--use_scaled", type=bool, default=True,
                       help="Use scaled data for GP fitting")
    parser.add_argument("--num_warmup", type=int, default=200,
                       help="MCMC warmup iterations")
    parser.add_argument("--num_samples", type=int, default=200,
                       help="MCMC sampling iterations")
    parser.add_argument("--num_chains", type=int, default=1,
                       help="Number of MCMC chains")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_dir", type=str, default="mcmc_results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    main(
        gamma=args.gamma,
        gamma2=args.gamma2,
        use_scaled_data=args.use_scaled,
        num_warmup=args.num_warmup,
        num_mcmc_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        output_dir=args.output_dir,
    )
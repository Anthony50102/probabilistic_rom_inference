import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO, autoguide
import numpy as np
from typing import Optional, Tuple, Dict
import warnings


class BayesianPPCA:
    """
    Bayesian Probabilistic Principal Component Analysis using NUTS sampling.
    
    Based on Tipping & Bishop (1999) "Probabilistic principal component analysis"
    
    Model:
    t = W @ x + μ + ε
    where:
    - q: n-dimensional observation
    - qhat ~ N(0, I): q-dimensional latent variable  
    - V: n*r loading matrix
    - qbar: n-dimensional mean
    - ε ~ N(0, sig^2I): isotropic Gaussian noise
    
    This gives: t ~ N(qbar, V @ V^T + sig^2I)
    """
    
    def __init__(self, latent_dim: int, obs_dim: int):
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.mcmc = None
        self.samples = None

    def penalty(self, V, strength = 5000):
        '''
        Add a soft penalty to punish the loss between V^T V - I
        '''
        loss = jnp.sum(((V.T @ V) - jnp.eye(V.shape[1]))**2)
        numpyro.factor("orth_penalty", -strength * loss)

    # def prior_V(self, feature_dim: int = None, heteroskedastic:bool = False):
    #     if feature_dim == None and heteroskedastic == True:
    #         raise ValueError "Feature dim must be passed if using heteroskedastic"
    #     # Give different noises across feature dims
    #     if heteroskedastic:
        
    def model(self, data: jnp.ndarray, penalty: bool = False, heteroskedastic:bool = True) -> None:
        """
        NumPyro model for PPCA.
        
        Args:
            data: Array of shape (n_samples, obs_dim)
        """
        n_samples, obs_dim = data.shape
        
        # Priors
        # Loading matrix W with appropriate scale
        V = numpyro.sample("V", dist.Normal(
            jnp.zeros((obs_dim, self.latent_dim)),
            jnp.ones((obs_dim, self.latent_dim))
        ).to_event(2))

        if penalty:
            self.penalty(V)
        
        # Mean parameter
        qbar = numpyro.sample("qbar", dist.Normal(
            jnp.zeros(obs_dim),
            jnp.ones(obs_dim) * 2.0
        ).to_event(1))
        
        # Noise variance (use log-normal to ensure positivity)
        log_sigma2 = numpyro.sample("log_sigma2", dist.Normal(0.0, 1.0))
        sigma2 = jnp.exp(log_sigma2)
        
        # Latent variables for each observation
        with numpyro.plate("data", n_samples):
            qhat = numpyro.sample("qhat", dist.Normal(
                jnp.zeros(self.latent_dim),
                jnp.ones(self.latent_dim)
            ).to_event(1))
            
            # Observation model: t = W @ x + μ + ε
            mean = jnp.matmul(qhat, V.T) + qbar
            
            # Likelihood with isotropic Gaussian noise
            numpyro.sample("obs", dist.Normal(
                mean,
                jnp.sqrt(sigma2) * jnp.ones_like(mean)
            ).to_event(1), obs=data)
    
    def fit(self, 
            data: jnp.ndarray, 
            num_samples: int = 1000,
            warmup_steps: int = 500,
            num_chains: int = 1,
            step_size: float = 0.01,
            model: str = 'normal',
            penalty: bool = False,
            rng_key: Optional[jax.random.PRNGKey] = None) -> None:
        """
        Fit the PPCA model using NUTS sampling.
        
        Args:
            data: Training data of shape (n_samples, obs_dim)
            num_samples: Number of MCMC samples
            warmup_steps: Number of warmup steps
            num_chains: Number of MCMC chains
            step_size: Initial step size for NUTS
            model: Which model to use, 'normal' or 'marginal'
            rng_key: JAX random key. If None, will create one.
        """
        # Convert to JAX array if needed
        data = jnp.asarray(data)
        
        # Create RNG key if not provided
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        
        # Set up NUTS sampler
        nuts_kernel = NUTS(self.marginalized_qhat_model if model == 'marginal' else self.model, 
                           step_size=step_size, 
                           adapt_step_size=True)
        
        # Run MCMC
        self.mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            num_warmup=warmup_steps,
            num_chains=num_chains
        )
        
        print("Running NUTS sampling...")
        self.mcmc.run(rng_key, data, penalty = penalty)
        
        # Get samples
        self.samples = self.mcmc.get_samples()
        
        print(f"Sampling completed. Collected {num_samples} samples.")
    
    def fit_svi(self,
                data: jnp.ndarray,
                num_iterations: int = 5000,
                learning_rate: float = 0.01,
                penalty: bool = False,
                rng_key: Optional[jax.random.PRNGKey] = None) -> None:
        """
        Fit using Stochastic Variational Inference - even faster!
        """
        data = jnp.asarray(data)
        
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        
        # Use automatic guide
        guide = autoguide.AutoNormal(self.model)
        
        # Set up SVI
        optimizer = numpyro.optim.Adam(learning_rate)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())
        
        print("Running SVI optimization...")
        svi_result = svi.run(rng_key, num_iterations, data, penalty)
        
        # Extract posterior samples
        params = svi_result.params
        predictive = numpyro.infer.Predictive(
            guide, params=params, num_samples=1000
        )
        self.samples = predictive(rng_key, data)
        
        print(f"SVI completed. Final loss: {svi_result.losses[-1]:.4f}")
        
    def get_posterior_mean(self) -> Dict[str, jnp.ndarray]:
        """Get posterior means of parameters."""
        if self.samples is None:
            raise ValueError("Model must be fitted first.")
            
        posterior_means = {}
        for param_name, samples in self.samples.items():
            if param_name != "qhat":  # qhat is latent variables, not model parameters
                posterior_means[param_name] = jnp.mean(samples, axis=0)
                
        return posterior_means
    
    def get_principal_components(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extract principal components from posterior samples.
        
        Returns:
            Tuple of (eigenvalues, eigenvectors) averaged over posterior samples
        """
        if self.samples is None:
            raise ValueError("Model must be fitted first.")
            
        W_samples = self.samples["V"]  # Shape: (n_samples, obs_dim, latent_dim)
        sigma2_samples = jnp.exp(self.samples["log_sigma2"])  # Shape: (n_samples,)
        
        eigenvalues_list = []
        eigenvectors_list = []
        
        for i in range(W_samples.shape[0]):
            W = W_samples[i]  # (obs_dim, latent_dim)
            sigma2 = sigma2_samples[i]
            
            # Compute covariance matrix C = W @ W^T + σ²I
            C = W @ W.T + sigma2 * jnp.eye(self.obs_dim)
            
            # Eigendecomposition
            eigenvals, eigenvecs = jnp.linalg.eigh(C)
            
            # Sort in descending order
            idx = jnp.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            eigenvalues_list.append(eigenvals)
            eigenvectors_list.append(eigenvecs)
        
        # Average over samples
        mean_eigenvalues = jnp.stack(eigenvalues_list).mean(axis=0)
        mean_eigenvectors = jnp.stack(eigenvectors_list).mean(axis=0)
        
        return mean_eigenvalues, mean_eigenvectors
    
    def compress(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Project data into latent space using posterior mean parameters.
        
        Args:
            data: Data to transform, shape (n_samples, obs_dim)
            
        Returns:
            Latent representations, shape (n_samples, latent_dim)
        """
        if self.samples is None:
            raise ValueError("Model must be fitted first.")
            
        # Convert to JAX array if needed
        data = jnp.asarray(data)
        
        posterior_means = self.get_posterior_mean()
        W_mean = posterior_means["V"]  # (obs_dim, latent_dim)
        mu_mean = posterior_means["qbar"]  # (obs_dim,)
        sigma2_mean = jnp.exp(posterior_means["log_sigma2"])
        
        # Center the data
        centered_data = data - mu_mean
        
        # Compute M = W^T @ W + σ²I
        M = W_mean.T @ W_mean + sigma2_mean * jnp.eye(self.latent_dim)
        
        # Project: x = M^(-1) @ W^T @ (t - μ)
        x = jnp.linalg.solve(M, W_mean.T @ centered_data.T).T
        
        return x
    
    def decompress(self, latent_data: jnp.ndarray) -> jnp.ndarray:
        """
        Reconstruct observations from latent representations.
        
        Args:
            latent_data: Latent representations, shape (n_samples, latent_dim)
            
        Returns:
            Reconstructed observations, shape (n_samples, obs_dim)
        """
        if self.samples is None:
            raise ValueError("Model must be fitted first.")
            
        # Convert to JAX array if needed
        latent_data = jnp.asarray(latent_data)
        
        posterior_means = self.get_posterior_mean()
        W_mean = posterior_means["V"]  # (obs_dim, latent_dim)
        mu_mean = posterior_means["qbar"]  # (obs_dim,)
        
        # Reconstruct: t = W @ x + μ
        reconstructed = latent_data @ W_mean.T + mu_mean
        
        return reconstructed
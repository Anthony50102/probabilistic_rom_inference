import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import NUTS, MCMC
import numpy as np
import matplotlib.pyplot as plt
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
    
    def __init__(self, latent_dim: int, obs_dim: int, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.device = device
        self.mcmc = None
        self.samples = None
        
    def model(self, data: torch.Tensor) -> None:
        """
        Pyro model for PPCA.
        
        Args:
            data: Tensor of shape (n_samples, obs_dim)
        """
        n_samples, obs_dim = data.shape
        
        # Priors
        # Loading matrix W with appropriate scale
        V = pyro.sample("V", dist.Normal(
            torch.zeros(obs_dim, self.latent_dim, device=self.device),
            torch.ones(obs_dim, self.latent_dim, device=self.device)
        ).to_event(2))
        
        # Mean parameter
        qbar = pyro.sample("qbar", dist.Normal(
            torch.zeros(obs_dim, device=self.device),
            torch.ones(obs_dim, device=self.device) * 2.0
        ).to_event(1))
        
        # Noise variance (use log-normal to ensure positivity)
        log_sigma2 = pyro.sample("log_sigma2", dist.Normal(
            torch.tensor(0.0, device=self.device),
            torch.tensor(1.0, device=self.device)
        ))
        sigma2 = torch.exp(log_sigma2)
        
        # Latent variables for each observation
        with pyro.plate("data", n_samples):
            qhat = pyro.sample("qhat", dist.Normal(
                torch.zeros(self.latent_dim, device=self.device),
                torch.ones(self.latent_dim, device=self.device)
            ).to_event(1))
            
            # Observation model: t = W @ x + μ + ε
            mean = torch.matmul(qhat, V.T) + qbar
            
            # Likelihood with isotropic Gaussian noise
            pyro.sample("obs", dist.Normal(
                mean,
                torch.sqrt(sigma2) * torch.ones_like(mean)
            ).to_event(1), obs=data)
    
    def fit(self, 
            data: torch.Tensor, 
            num_samples: int = 1000,
            warmup_steps: int = 500,
            num_chains: int = 1,
            step_size: float = 0.01) -> None:
        """
        Fit the PPCA model using NUTS sampling.
        
        Args:
            data: Training data of shape (n_samples, obs_dim)
            num_samples: Number of MCMC samples
            warmup_steps: Number of warmup steps
            num_chains: Number of MCMC chains
            step_size: Initial step size for NUTS
        """
        data = data.to(self.device)
        
        # Set up NUTS sampler
        nuts_kernel = NUTS(self.model, step_size=step_size, adapt_step_size=True)
        
        # Run MCMC
        self.mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains
        )
        
        print("Running NUTS sampling...")
        self.mcmc.run(data)
        
        # Get samples
        self.samples = self.mcmc.get_samples()
        
        print(f"Sampling completed. Collected {num_samples} samples.")
        
    def get_posterior_mean(self) -> Dict[str, torch.Tensor]:
        """Get posterior means of parameters."""
        if self.samples is None:
            raise ValueError("Model must be fitted first.")
            
        posterior_means = {}
        for param_name, samples in self.samples.items():
            if param_name != "x":  # x is latent variables, not model parameters
                posterior_means[param_name] = samples.mean(dim=0)
                
        return posterior_means
    
    def get_principal_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract principal components from posterior samples.
        
        Returns:
            Tuple of (eigenvalues, eigenvectors) averaged over posterior samples
        """
        if self.samples is None:
            raise ValueError("Model must be fitted first.")
            
        W_samples = self.samples["V"]  # Shape: (n_samples, obs_dim, latent_dim)
        sigma2_samples = torch.exp(self.samples["log_sigma2"])  # Shape: (n_samples,)
        
        eigenvalues_list = []
        eigenvectors_list = []
        
        for i in range(W_samples.shape[0]):
            W = W_samples[i]  # (obs_dim, latent_dim)
            sigma2 = sigma2_samples[i]
            
            # Compute covariance matrix C = W @ W^T + σ²I
            C = W @ W.T + sigma2 * torch.eye(self.obs_dim, device=self.device)
            
            # Eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(C)
            
            # Sort in descending order
            idx = torch.argsort(eigenvals, descending=True)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            eigenvalues_list.append(eigenvals)
            eigenvectors_list.append(eigenvecs)
        
        # Average over samples
        mean_eigenvalues = torch.stack(eigenvalues_list).mean(dim=0)
        mean_eigenvectors = torch.stack(eigenvectors_list).mean(dim=0)
        
        return mean_eigenvalues, mean_eigenvectors
    
    def compress(self, data: torch.Tensor) -> torch.Tensor:
        """
        Project data into latent space using posterior mean parameters.
        
        Args:
            data: Data to transform, shape (n_samples, obs_dim)
            
        Returns:
            Latent representations, shape (n_samples, latent_dim)
        """
        if self.samples is None:
            raise ValueError("Model must be fitted first.")
            
        posterior_means = self.get_posterior_mean()
        W_mean = posterior_means["V"]  # (obs_dim, latent_dim)
        mu_mean = posterior_means["qbar"]  # (obs_dim,)
        sigma2_mean = torch.exp(posterior_means["log_sigma2"])
        
        # Center the data
        centered_data = data - mu_mean
        
        # Compute M = W^T @ W + σ²I
        M = W_mean.T @ W_mean + sigma2_mean * torch.eye(self.latent_dim, device=self.device)
        
        # Project: x = M^(-1) @ W^T @ (t - μ)
        x = torch.linalg.solve(M, W_mean.T @ centered_data.T).T
        
        return x
    
    def decompress(self, latent_data: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct observations from latent representations.
        
        Args:
            latent_data: Latent representations, shape (n_samples, latent_dim)
            
        Returns:
            Reconstructed observations, shape (n_samples, obs_dim)
        """
        if self.samples is None:
            raise ValueError("Model must be fitted first.")
            
        posterior_means = self.get_posterior_mean()
        W_mean = posterior_means["V"]  # (obs_dim, latent_dim)
        mu_mean = posterior_means["qbar"]  # (obs_dim,)
        
        # Reconstruct: t = W @ x + μ
        reconstructed = latent_data @ W_mean.T + mu_mean
        
        return reconstructed
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import NUTS, MCMC
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import transforms
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
pyro.set_rng_seed(42)
np.random.seed(42)

class RBFKernel:
    """Radial Basis Function (Squared Exponential) Kernel"""
    
    def __init__(self):
        pass
    
    def __call__(self, X1, X2, lengthscale, variance):
        """
        Compute RBF kernel matrix
        
        Args:
            X1: Input tensor of shape (n1, d)
            X2: Input tensor of shape (n2, d) 
            lengthscale: Kernel lengthscale parameter
            variance: Kernel variance parameter
            
        Returns:
            Kernel matrix of shape (n1, n2)
        """
        # Compute squared distances
        X1_expanded = X1.unsqueeze(1)  # (n1, 1, d)
        X2_expanded = X2.unsqueeze(0)  # (1, n2, d)
        
        squared_dist = torch.sum((X1_expanded - X2_expanded) ** 2, dim=-1)
        
        # RBF kernel: k(x,x') = σ² * exp(-||x-x'||²/(2ℓ²))
        return variance * torch.exp(-0.5 * squared_dist / lengthscale**2)

class BayesianGP:
    """Fully Bayesian Gaussian Process"""
    
    def __init__(self, kernel='rbf'):
        if kernel == 'rbf':
            self.kernel = RBFKernel()
        else:
            raise ValueError("Kernel must be 'rbf'")
    
    def model(self, X, y=None):
        """
        Pyro model for Bayesian GP
        
        Args:
            X: Input locations (n, d)
            y: Observed outputs (n,) - None for prediction
        """
        n = X.shape[0]
        
        # Priors for hyperparameters
        lengthscale = pyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
        variance = pyro.sample("variance", dist.LogNormal(0.0, 1.0))
        noise = pyro.sample("noise", dist.LogNormal(-2.0, 1.0))
        
        # Compute kernel matrix
        K = self.kernel(X, X, lengthscale, variance)
        
        # Add noise to diagonal for numerical stability and observation noise
        K_noise = K + (noise + 1e-6) * torch.eye(n)
        
        # GP prior over function values
        f = pyro.sample("f", dist.MultivariateNormal(torch.zeros(n), K_noise))
        
        # Likelihood
        if y is not None:
            with pyro.plate("data", n):
                pyro.sample("y", dist.Normal(f, torch.sqrt(noise)), obs=y)
        
        return f
    
    def fit(self, X_train, y_train, num_samples=1000, warmup_steps=500):
        """
        Fit the GP using NUTS sampling
        
        Args:
            X_train: Training inputs (n, d)
            y_train: Training outputs (n,)
            num_samples: Number of MCMC samples
            warmup_steps: Number of warmup steps
        """
        self.X_train = X_train
        self.y_train = y_train
        
        # Initialize NUTS sampler
        nuts_kernel = NUTS(self.model, jit_compile=True, ignore_jit_warnings=True)
        
        # Run MCMC
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
        mcmc.run(X_train, y_train)
        
        # Store samples
        self.samples = mcmc.get_samples()
        
        print(f"MCMC completed with {num_samples} samples")
        self._print_summary()
    
    def _print_summary(self):
        """Print summary statistics of posterior samples"""
        print("\nPosterior Summary:")
        print("-" * 40)
        for param in ['lengthscale', 'variance', 'noise']:
            samples = self.samples[param]
            print(f"{param:12s}: mean={samples.mean():.3f}, std={samples.std():.3f}")
    
    def predict(self, X_test, num_samples=None):
        """
        Make predictions at test points
        
        Args:
            X_test: Test input locations (m, d)
            num_samples: Number of posterior samples to use (None = all)
            
        Returns:
            mean: Predictive mean (m,)
            std: Predictive standard deviation (m,)
            samples: Function samples at test points (num_samples, m)
        """
        if not hasattr(self, 'samples'):
            raise RuntimeError("Model must be fitted before making predictions")
        
        X_test = torch.as_tensor(X_test, dtype=torch.float32)
        n_train = self.X_train.shape[0]
        n_test = X_test.shape[0]
        
        if num_samples is None:
            param_samples = self.samples
        else:
            # Randomly select subset of samples
            indices = torch.randperm(len(self.samples['lengthscale']))[:num_samples]
            param_samples = {k: v[indices] for k, v in self.samples.items()}
        
        predictions = []
        
        for i in range(len(param_samples['lengthscale'])):
            # Get hyperparameters for this sample
            ls = param_samples['lengthscale'][i]
            var = param_samples['variance'][i]
            noise = param_samples['noise'][i]
            
            # Kernel matrices
            K_train = self.kernel(self.X_train, self.X_train, ls, var)
            K_train_noise = K_train + (noise + 1e-6) * torch.eye(n_train)
            K_test_train = self.kernel(X_test, self.X_train, ls, var)
            K_test = self.kernel(X_test, X_test, ls, var)
            
            # GP posterior predictive
            try:
                L = torch.linalg.cholesky(K_train_noise)
                alpha = torch.cholesky_solve(self.y_train.unsqueeze(-1), L).squeeze()
                
                # Predictive mean
                mean = K_test_train @ alpha
                
                # Predictive covariance
                v = torch.triangular_solve(K_test_train.T, L, upper=False)[0]
                cov = K_test - v.T @ v
                
                # Sample from predictive distribution
                cov_noise = cov + 1e-6 * torch.eye(n_test)
                L_pred = torch.linalg.cholesky(cov_noise)
                
                # Generate sample
                eps = torch.randn(n_test)
                sample = mean + L_pred @ eps
                
                predictions.append(sample)
                
            except torch.linalg.LinAlgError:
                # Fallback for numerical issues
                print(f"Numerical issue in sample {i}, skipping...")
                continue
        
        if not predictions:
            raise RuntimeError("All predictions failed due to numerical issues")
        
        # Stack predictions
        prediction_samples = torch.stack(predictions)
        
        # Compute statistics
        pred_mean = prediction_samples.mean(dim=0)
        pred_std = prediction_samples.std(dim=0)
        
        return pred_mean, pred_std, prediction_samples
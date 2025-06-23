import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
import numpy as np
import matplotlib.pyplot as plt
from jax import random
import warnings
from functools import partial
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
key = random.PRNGKey(42)
np.random.seed(42)

# ------------------------
# Kernel derivative functions
# ------------------------
@jax.jit
def flatten_time(t: jnp.ndarray) -> jnp.ndarray:
    """Return t with shape (n,) no matter if (n,), (n,1) or (1,n) was given."""
    return jnp.ravel(t)

@jax.jit
def rbf_kernel_no_nugget(lengthscale: float, variance: float, t: jnp.ndarray) -> jnp.ndarray:
    """Full n×n RBF kernel matrix K_ij = variance * exp(-(t_i-t_j)^2 / (2*ell^2))."""
    t = flatten_time(t)
    diff = t[:, None] - t[None, :]
    ell2 = lengthscale ** 2
    return variance * jnp.exp(-diff**2 / (2.0 * ell2))

@jax.jit
def get_c_phi(lengthscale: float, variance: float, t: jnp.ndarray, nugget: float = 1e-4) -> jnp.ndarray:
    """Kernel matrix plus nugget on the diagonal."""
    kmat = rbf_kernel_no_nugget(lengthscale, variance, t)
    return kmat + nugget * jnp.eye(kmat.shape[0])

@jax.jit
def get_c_phi_dash(lengthscale: float, variance: float, t: jnp.ndarray) -> jnp.ndarray:
    """Derivative with respect to the second time argument (dt2)."""
    t = flatten_time(t)
    diff = t[:, None] - t[None, :]
    ell2 = lengthscale ** 2
    return (diff / ell2) * rbf_kernel_no_nugget(lengthscale, variance, t)

@jax.jit
def get_dash_c_phi(lengthscale: float, variance: float, t: jnp.ndarray) -> jnp.ndarray:
    """Derivative with respect to the first time argument (dt1)."""
    return -get_c_phi_dash(lengthscale, variance, t)

@jax.jit
def get_c_phi_double_dash(lengthscale: float, variance: float, t: jnp.ndarray) -> jnp.ndarray:
    """Second mixed derivative with respect to both time arguments."""
    t = flatten_time(t)
    diff = t[:, None] - t[None, :]
    ell2 = lengthscale ** 2
    return (1.0 / ell2 - diff**2 / ell2**2) * rbf_kernel_no_nugget(lengthscale, variance, t)

class RBFKernel:
    """Radial Basis Function (Squared Exponential) Kernel"""
    
    def __init__(self):
        pass
    
    @partial(jax.jit,
             static_argnums=(0,),
             static_argnames=['lengthscale','variance'])
    def __call__(self, X1, X2, lengthscale, variance):
        """
        Compute RBF kernel matrix
        
        Args:
            X1: Input array of shape (n1, d)
            X2: Input array of shape (n2, d) 
            lengthscale: Kernel lengthscale parameter
            variance: Kernel variance parameter
            
        Returns:
            Kernel matrix of shape (n1, n2)
        """
        # Compute squared distances
        X1_expanded = X1[:, jnp.newaxis, :]  # (n1, 1, d)
        X2_expanded = X2[jnp.newaxis, :, :]  # (1, n2, d)
        
        squared_dist = jnp.sum((X1_expanded - X2_expanded) ** 2, axis=-1)
        
        # RBF kernel: k(x,x') = σ² * exp(-||x-x'||²/(2ℓ²))
        return variance * jnp.exp(-0.5 * squared_dist / lengthscale**2)

class BayesianGP:
    """Fully Bayesian Gaussian Process with derivative support"""
    
    def __init__(self, kernel:str='rbf', normalization=1e-6):
        if kernel == 'rbf':
            self.kernel = RBFKernel()
        else:
            raise ValueError("Kernel must be 'rbf'")
        
        self.normalization = normalization
    
    def model(self, X, y=None):
        """
        NumPyro model for Bayesian GP
        
        Args:
            X: Input locations (n, d)
            y: Observed outputs (n,) - None for prediction
        """
        n = X.shape[0]
        
        # Priors for hyperparameters
        lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
        variance = numpyro.sample("variance", dist.LogNormal(0.0, 1.0))
        noise = numpyro.sample("noise", dist.LogNormal(-2.0, 1.0))
        
        # Compute kernel matrix
        K = self.kernel(X, X, lengthscale, variance)
        
        # Add noise to diagonal for numerical stability and observation noise
        K_noise = K + (noise + self.normalization) * jnp.eye(n)
        
        # GP prior over function values
        f = numpyro.sample("f", dist.MultivariateNormal(jnp.zeros(n), K_noise))
        
        # Likelihood
        if y is not None:
            with numpyro.plate("data", n):
                numpyro.sample("y", dist.Normal(f, jnp.sqrt(noise)), obs=y)
        
        return f
    
    def fit(self, X_train, y_train, num_samples=1000, warmup_steps=500, rng_key=None):
        """
        Fit the GP using NUTS sampling
        
        Args:
            X_train: Training inputs (n, d)
            y_train: Training outputs (n,)
            num_samples: Number of MCMC samples
            warmup_steps: Number of warmup steps
            rng_key: JAX random key
        """
        if rng_key is None:
            rng_key = random.PRNGKey(0)
            
        self.X_train = jnp.asarray(X_train)
        self.y_train = jnp.asarray(y_train)
        
        # Initialize NUTS sampler
        nuts_kernel = NUTS(self.model)
        
        # Run MCMC
        mcmc = MCMC(
            nuts_kernel, 
            num_samples=num_samples, 
            num_warmup=warmup_steps,
            progress_bar=True
        )
        mcmc.run(rng_key, self.X_train, self.y_train)
        
        # Store samples
        self.samples = mcmc.get_samples()
        
        print(f"\nMCMC completed with {num_samples} samples")
        self._print_summary()
    
    def _print_summary(self):
        """Print summary statistics of posterior samples"""
        print("\nPosterior Summary:")
        print("-" * 40)
        for param in ['lengthscale', 'variance', 'noise']:
            samples = self.samples[param]
            print(f"{param:12s}: mean={samples.mean():.3f}, std={samples.std():.3f}")
    
    def predict(self, X_test, num_samples=None, rng_key=None):
        """
        Make predictions at test points
        
        Args:
            X_test: Test input locations (m, d)
            num_samples: Number of posterior samples to use (None = all)
            rng_key: JAX random key
            
        Returns:
            mean: Predictive mean (m,)
            std: Predictive standard deviation (m,)
            samples: Function samples at test points (num_samples, m)
        """
        if not hasattr(self, 'samples'):
            raise RuntimeError("Model must be fitted before making predictions")
        
        if rng_key is None:
            rng_key = random.PRNGKey(1)
            
        X_test = jnp.asarray(X_test)
        n_train = self.X_train.shape[0]
        n_test = X_test.shape[0]
        
        if num_samples is None:
            param_samples = self.samples
        else:
            # Randomly select subset of samples
            total_samples = len(self.samples['lengthscale'])
            indices = random.choice(rng_key, total_samples, shape=(num_samples,), replace=False)
            param_samples = {k: v[indices] for k, v in self.samples.items()}
        
        predictions = []
        
        # Split key for multiple predictions
        num_pred = len(param_samples['lengthscale'])
        keys = random.split(rng_key, num_pred)
        
        for i in range(num_pred):
            # Get hyperparameters for this sample
            ls = param_samples['lengthscale'][i]
            var = param_samples['variance'][i]
            noise = param_samples['noise'][i]
            
            # Kernel matrices
            K_train = self.kernel(self.X_train, self.X_train, ls, var)
            K_train_noise = K_train + (noise + self.normalization) * jnp.eye(n_train)
            K_test_train = self.kernel(X_test, self.X_train, ls, var)
            K_test = self.kernel(X_test, X_test, ls, var)
            
            # GP posterior predictive
            try:
                # Cholesky decomposition
                L = jnp.linalg.cholesky(K_train_noise)
                
                # Solve for alpha = K_train_noise^{-1} @ y_train
                alpha = jax.scipy.linalg.cho_solve((L, True), self.y_train)
                
                # Predictive mean
                mean = K_test_train @ alpha
                
                # Predictive covariance
                v = jax.scipy.linalg.solve_triangular(L, K_test_train.T, lower=True)
                cov = K_test - v.T @ v
                
                # Sample from predictive distribution
                cov_noise = cov + self.normalization * jnp.eye(n_test)
                L_pred = jnp.linalg.cholesky(cov_noise)
                
                # Generate sample
                eps = random.normal(keys[i], shape=(n_test,))
                sample = mean + L_pred @ eps
                
                predictions.append(sample)
                
            except (jnp.linalg.LinAlgError, ValueError):
                # Handle numerical issues
                print(f"Numerical issue in sample {i}, skipping...")
                continue
        
        if not predictions:
            raise RuntimeError("All predictions failed due to numerical issues")
        
        # Stack predictions
        prediction_samples = jnp.stack(predictions)
        
        # Compute statistics
        pred_mean = prediction_samples.mean(axis=0)
        pred_std = prediction_samples.std(axis=0)
        
        return pred_mean, pred_std, prediction_samples
    
    # ------------------------
    # Kernel matrix utility methods
    # ------------------------
    def get_kernel_matrix(self, t, nugget=1e-4, sample_idx=None):
        """
        Get the kernel matrix for given time points.
        
        Args:
            t: Time points
            nugget: Small value added to diagonal for numerical stability
            sample_idx: Index of posterior sample to use (None = use mean)
            
        Returns:
            Kernel matrix of shape [n, n]
        """
        if not hasattr(self, 'samples'):
            raise RuntimeError("Model must be fitted before computing kernel matrices")
        
        if sample_idx is None:
            # Use posterior mean
            lengthscale = self.samples['lengthscale'].mean()
            variance = self.samples['variance'].mean()
        else:
            lengthscale = self.samples['lengthscale'][sample_idx]
            variance = self.samples['variance'][sample_idx]
        
        t = jnp.asarray(t)
        if t.ndim == 2:
            t = flatten_time(t)
        return get_c_phi(lengthscale, variance, t, nugget)
    
    def get_kernel_derivative_dt1(self, t, sample_idx=None):
        """
        Get the kernel derivative matrix with respect to first time argument.
        
        Args:
            t: Time points
            sample_idx: Index of posterior sample to use (None = use mean)
            
        Returns:
            Derivative kernel matrix of shape [n, n]
        """
        if not hasattr(self, 'samples'):
            raise RuntimeError("Model must be fitted before computing kernel derivatives")
        
        if sample_idx is None:
            lengthscale = self.samples['lengthscale'].mean()
            variance = self.samples['variance'].mean()
        else:
            lengthscale = self.samples['lengthscale'][sample_idx]
            variance = self.samples['variance'][sample_idx]
        
        t = jnp.asarray(t)
        if t.ndim == 2:
            t = flatten_time(t)
        return get_dash_c_phi(lengthscale, variance, t)
    
    def get_kernel_derivative_dt2(self, t, sample_idx=None):
        """
        Get the kernel derivative matrix with respect to second time argument.
        
        Args:
            t: Time points
            sample_idx: Index of posterior sample to use (None = use mean)
            
        Returns:
            Derivative kernel matrix of shape [n, n]
        """
        if not hasattr(self, 'samples'):
            raise RuntimeError("Model must be fitted before computing kernel derivatives")
        
        if sample_idx is None:
            lengthscale = self.samples['lengthscale'].mean()
            variance = self.samples['variance'].mean()
        else:
            lengthscale = self.samples['lengthscale'][sample_idx]
            variance = self.samples['variance'][sample_idx]
        
        t = jnp.asarray(t)
        if t.ndim == 2:
            t = flatten_time(t)
        return get_c_phi_dash(lengthscale, variance, t)
    
    def get_kernel_double_derivative(self, t, sample_idx=None):
        """
        Get the kernel second mixed derivative matrix.
        
        Args:
            t: Time points
            sample_idx: Index of posterior sample to use (None = use mean)
            
        Returns:
            Second derivative kernel matrix of shape [n, n]
        """
        if not hasattr(self, 'samples'):
            raise RuntimeError("Model must be fitted before computing kernel derivatives")
        
        if sample_idx is None:
            lengthscale = self.samples['lengthscale'].mean()
            variance = self.samples['variance'].mean()
        else:
            lengthscale = self.samples['lengthscale'][sample_idx]
            variance = self.samples['variance'][sample_idx]
        
        t = jnp.asarray(t)
        if t.ndim == 2:
            t = flatten_time(t)
        return get_c_phi_double_dash(lengthscale, variance, t)
    
    def get_As(self, t, sample_idx=None):
        """
        Compute A matrices for derivative processes.
        
        Args:
            t: Time points
            sample_idx: Index of posterior sample to use (None = use mean)
            
        Returns:
            List containing A matrix
        """
        if not hasattr(self, 'samples'):
            raise RuntimeError("Model must be fitted before computing A matrices")
        
        if sample_idx is None:
            lengthscale = self.samples['lengthscale'].mean()
            variance = self.samples['variance'].mean()
        else:
            lengthscale = self.samples['lengthscale'][sample_idx]
            variance = self.samples['variance'][sample_idx]
        
        t = jnp.asarray(t)
        if t.ndim == 2:
            t = flatten_time(t)
            
        CDashs = get_c_phi_dash(lengthscale, variance, t)
        DashCs = get_dash_c_phi(lengthscale, variance, t)
        CPhis = get_c_phi(lengthscale, variance, t, nugget=1e-6)
        CDoubleDashs = get_c_phi_double_dash(lengthscale, variance, t)
        
        A = CDoubleDashs - jnp.dot(DashCs, jnp.linalg.solve(CPhis, CDashs))
        return [A]
    
    def get_Ds(self, t, sample_idx=None):
        """
        Get D function for derivative processes.
        
        Args:
            t: Time points
            sample_idx: Index of posterior sample to use (None = use mean)
            
        Returns:
            List containing D function
        """
        if not hasattr(self, 'samples'):
            raise RuntimeError("Model must be fitted before computing D functions")
        
        if sample_idx is None:
            lengthscale = self.samples['lengthscale'].mean()
            variance = self.samples['variance'].mean()
        else:
            lengthscale = self.samples['lengthscale'][sample_idx]
            variance = self.samples['variance'][sample_idx]
        
        t = jnp.asarray(t)
        if t.ndim == 2:
            t = flatten_time(t)
            
        DashCs = get_dash_c_phi(lengthscale, variance, t)
        CPhis = get_c_phi(lengthscale, variance, t, nugget=1e-6)
        
        def getProdWithD(x):
            """Defines a function to get a product of a vector with the matrix D"""
            return jnp.dot(DashCs, jnp.linalg.solve(CPhis, x))
        
        return [getProdWithD]
    
    @property
    def hyperparameters(self):
        """
        Get hyperparameters as a dictionary (using posterior means).
        
        Returns:
            Dictionary with 'lengthscale', 'variance', and 'noise'
        """
        if not hasattr(self, 'samples'):
            raise RuntimeError("Model must be fitted before accessing hyperparameters")
        
        return {
            'lengthscale': float(self.samples['lengthscale'].mean()),
            'variance': float(self.samples['variance'].mean()),
            'noise': float(self.samples['noise'].mean())
        }
    
    def get_hyperparameters_samples(self):
        """
        Get all posterior samples of hyperparameters.
        
        Returns:
            Dictionary with arrays of samples for each hyperparameter
        """
        if not hasattr(self, 'samples'):
            raise RuntimeError("Model must be fitted before accessing hyperparameters")
        
        return {
            'lengthscale': self.samples['lengthscale'],
            'variance': self.samples['variance'],
            'noise': self.samples['noise']
        }
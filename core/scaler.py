"""
Data scaling utilities for GP fitting and operator inference.

This module provides scaling/normalization functionality to improve
GP hyperparameter learning and numerical stability.
"""
import numpy as np
import jax.numpy as jnp


class DataScaler:
    """
    Scales data for GP fitting and maintains transformations for inverse operations.
    
    Supports standardization (zero mean, unit variance) for each POD mode independently.
    """
    
    def __init__(self, num_modes):
        """
        Initialize the scaler.
        
        Parameters
        ----------
        num_modes : int
            Number of POD modes to scale
        """
        self.num_modes = num_modes
        self.means_ = None
        self.stds_ = None
        self.fitted_ = False
        
    def fit(self, data):
        """
        Compute scaling parameters from training data.
        
        Parameters
        ----------
        data : np.ndarray or jnp.ndarray
            Shape (num_modes, num_time_points)
            Training snapshots to compute scaling from
        """
        data = np.array(data)
        assert data.shape[0] == self.num_modes, \
            f"Expected {self.num_modes} modes, got {data.shape[0]}"
        
        # Compute mean and std for each mode
        self.means_ = np.mean(data, axis=1, keepdims=True)  # Shape: (num_modes, 1)
        self.stds_ = np.std(data, axis=1, keepdims=True)    # Shape: (num_modes, 1)
        
        # Prevent division by zero for constant modes
        self.stds_ = np.where(self.stds_ < 1e-10, 1.0, self.stds_)
        
        self.fitted_ = True
        return self
    
    def transform(self, data):
        """
        Scale data to zero mean and unit variance.
        
        Parameters
        ----------
        data : np.ndarray or jnp.ndarray
            Shape (num_modes, num_time_points) or (num_modes,)
            Data to scale
            
        Returns
        -------
        scaled_data : same type as input
            Scaled data
        """
        if not self.fitted_:
            raise RuntimeError("Scaler must be fitted before transforming")
        
        is_jax = isinstance(data, jnp.ndarray)
        data_np = np.array(data)
        
        # Handle both (num_modes, n_points) and (num_modes,) shapes
        if data_np.ndim == 1:
            scaled = (data_np - self.means_.ravel()) / self.stds_.ravel()
        else:
            scaled = (data_np - self.means_) / self.stds_
        
        return jnp.array(scaled) if is_jax else scaled
    
    def inverse_transform(self, scaled_data):
        """
        Transform scaled data back to original space.
        
        Parameters
        ----------
        scaled_data : np.ndarray or jnp.ndarray
            Shape (num_modes, num_time_points) or (num_modes,)
            Scaled data
            
        Returns
        -------
        original_data : same type as input
            Data in original scale
        """
        if not self.fitted_:
            raise RuntimeError("Scaler must be fitted before inverse transforming")
        
        is_jax = isinstance(scaled_data, jnp.ndarray)
        scaled_np = jnp.array(scaled_data)
        
        # Handle both (num_modes, n_points) and (num_modes,) shapes
        if scaled_np.ndim == 1:
            original = scaled_np * self.stds_.ravel() + self.means_.ravel()
        else:
            original = scaled_np * self.stds_ + self.means_
        
        return jnp.array(original) if is_jax else original
    
    def scale_variance(self, variance, mode_idx):
        """
        Scale a GP variance hyperparameter for a given mode.
        
        When data is standardized, the GP variance should be scaled accordingly.
        
        Parameters
        ----------
        variance : float
            Variance in original data space
        mode_idx : int
            Index of the POD mode
            
        Returns
        -------
        scaled_variance : float
            Variance in scaled space
        """
        if not self.fitted_:
            raise RuntimeError("Scaler must be fitted before scaling variance")
        
        return variance / (self.stds_[mode_idx, 0] ** 2)
    
    def unscale_variance(self, scaled_variance, mode_idx):
        """
        Unscale a GP variance hyperparameter back to original space.
        
        Parameters
        ----------
        scaled_variance : float
            Variance in scaled space
        mode_idx : int
            Index of the POD mode
            
        Returns
        -------
        variance : float
            Variance in original data space
        """
        if not self.fitted_:
            raise RuntimeError("Scaler must be fitted before unscaling variance")
        
        return scaled_variance * (self.stds_[mode_idx, 0] ** 2)
    
    def scale_derivatives(self, derivatives):
        """
        Scale derivative data (only divides by std, no mean shift).
        
        Parameters
        ----------
        derivatives : np.ndarray or jnp.ndarray
            Shape (num_modes, num_time_points)
            Derivative data to scale
            
        Returns
        -------
        scaled_derivatives : same type as input
            Scaled derivatives
        """
        if not self.fitted_:
            raise RuntimeError("Scaler must be fitted before scaling derivatives")
        
        is_jax = isinstance(derivatives, jnp.ndarray)
        deriv_np = np.array(derivatives)
        
        # Derivatives only scale by std, not shifted by mean
        scaled = deriv_np / self.stds_
        
        return jnp.array(scaled) if is_jax else scaled
    
    def unscale_derivatives(self, scaled_derivatives):
        """
        Unscale derivative data back to original space.
        
        Parameters
        ----------
        scaled_derivatives : np.ndarray or jnp.ndarray
            Shape (num_modes, num_time_points)
            Scaled derivative data
            
        Returns
        -------
        derivatives : same type as input
            Derivatives in original scale
        """
        if not self.fitted_:
            raise RuntimeError("Scaler must be fitted before unscaling derivatives")
        
        is_jax = isinstance(scaled_derivatives, jnp.ndarray)
        scaled_np = np.array(scaled_derivatives)
        
        # Derivatives only scale by std
        original = scaled_np * self.stds_
        
        return jnp.array(original) if is_jax else original
    
    def get_params(self):
        """
        Get the scaling parameters.
        
        Returns
        -------
        params : dict
            Dictionary with 'means' and 'stds' keys
        """
        if not self.fitted_:
            raise RuntimeError("Scaler must be fitted before getting parameters")
        
        return {
            'means': self.means_.copy(),
            'stds': self.stds_.copy(),
        }
    
    def __repr__(self):
        if self.fitted_:
            return (f"DataScaler(num_modes={self.num_modes}, fitted=True, "
                   f"mean_range=[{self.means_.min():.3e}, {self.means_.max():.3e}], "
                   f"std_range=[{self.stds_.min():.3e}, {self.stds_.max():.3e}])")
        else:
            return f"DataScaler(num_modes={self.num_modes}, fitted=False)"

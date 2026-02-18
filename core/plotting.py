"""
Plotting utilities for Probabilistic ROM Inference.

Provides reusable visualization functions for ROM analysis and debugging.
This module contains:
- Base Plotter class for all experiment-specific plotters
- Utility functions for GP kernels and finite differences
- Functions for plotting deterministic and Bayesian ROM results
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable
import jax.numpy as jnp


# =============================================================================
# Utility Functions
# =============================================================================

def flatten_time(t: jnp.ndarray) -> jnp.ndarray:
    """Return t with shape (n,) no matter if (n,), (n,1) or (1,n) was given."""
    return jnp.ravel(t)


def rbf_eval(lengthscale: float, variance: float, t: jnp.ndarray, t2: jnp.ndarray) -> jnp.ndarray:
    """Full n×n RBF kernel matrix K_ij = variance * exp(-(t_i-t_j)^2 / (2*ell^2))."""
    t = flatten_time(t)
    t2 = flatten_time(t2)
    diff = t[:, None] - t2[None, :]
    ell2 = lengthscale ** 2
    return variance * jnp.exp(-diff**2 / (2.0 * ell2))


def compute_derivatives_fourth_order(snapshots: np.ndarray, time_points: np.ndarray) -> np.ndarray:
    """
    Compute derivatives using 4th order finite differences.
    Works best for uniformly spaced time points.
    
    Parameters
    ----------
    snapshots : np.ndarray
        Snapshot data, shape (n_modes, n_time)
    time_points : np.ndarray
        Time points, shape (n_time,)
        
    Returns
    -------
    derivatives : np.ndarray
        Computed derivatives, shape (n_modes, n_time)
    """
    n_modes, n_time = snapshots.shape
    derivatives = np.zeros_like(snapshots)
    dt = time_points[1] - time_points[0]  # Assumes uniform spacing
    
    # 4th order central differences for interior points
    for i in range(2, n_time - 2):
        derivatives[:, i] = (-snapshots[:, i+2] + 8*snapshots[:, i+1] - 
                            8*snapshots[:, i-1] + snapshots[:, i-2]) / (12 * dt)
    
    # Use 2nd order for near-boundary points
    for i in [1, n_time-2]:
        derivatives[:, i] = (snapshots[:, i+1] - snapshots[:, i-1]) / (2 * dt)
    
    # First and last points
    derivatives[:, 0] = (snapshots[:, 1] - snapshots[:, 0]) / dt
    derivatives[:, -1] = (snapshots[:, -1] - snapshots[:, -2]) / dt
    
    return derivatives


# =============================================================================
# Base Plotter Class
# =============================================================================

class Plotter:
    """
    Base plotter class for ROM visualization.
    
    Provides common initialization and attributes for all experiment-specific plotters.
    
    Parameters
    ----------
    numPODmodes : int
        Number of POD modes
    time_domain_training : np.ndarray
        Time domain of the training snapshots, shape (t_train,)
    time_domain_prediction : np.ndarray
        Time domain of the full snapshots, shape (t_pred,)
    time_domain_eval_training : np.ndarray
        Time domain for evaluating GP in training domain, shape (t_eval_train,)
    time_domain_eval_prediction : np.ndarray
        Time domain for evaluating ROM in prediction domain, shape (t_eval_pred,)
    snapshots_training : np.ndarray
        Training snapshots, shape (numPODmodes, t_train)
    snapshots_prediction : np.ndarray
        Full snapshots for comparison, shape (numPODmodes, t_pred)
    scaler : optional
        Data scaler for inverse transforms (used by some experiments)
    """
    
    def __init__(self, 
                 numPODmodes: int,
                 time_domain_training: np.ndarray, 
                 time_domain_prediction: np.ndarray, 
                 time_domain_eval_training: np.ndarray,
                 time_domain_eval_prediction: np.ndarray,
                 snapshots_training: np.ndarray, 
                 snapshots_prediction: np.ndarray,
                 scaler=None,
                 ) -> None:
        self.numPODmodes = numPODmodes
        self.time_domain_training = time_domain_training
        self.time_domain_prediction = time_domain_prediction
        self.time_domain_eval_training = time_domain_eval_training
        self.time_domain_eval_prediction = time_domain_eval_prediction
        self.snapshots_training = snapshots_training
        self.snapshots_prediction = snapshots_prediction
        self.scaler = scaler

        # GP hyperparameters (set by gp_plot_state)
        self.gp_lengthscales = None
        self.gp_variances = None
        self.gp_noises = None
        
        # Computed derivatives
        self.snapshots_training_derivatives = None
    
    def _to_original_space(self, data_scaled):
        """Convert scaled data back to original space."""
        if self.scaler is not None:
            return self.scaler.inverse_transform(data_scaled)
        return data_scaled
    
    def _to_scaled_space(self, data_original):
        """Convert original data to scaled space."""
        if self.scaler is not None:
            return self.scaler.transform(data_original)
        return data_original

    def gp_plot_state(self,
                      lengthscales: np.ndarray | List,
                      variances: np.ndarray | List,
                      noises: np.ndarray | List,
                      gp_class,
                      double: bool = True,
                      figsize: Tuple[int, int] = (12, 8),
                      max_num_samples: int = 1000
                      ):
        """
        Plot GP state estimates in the training domain.
        
        Parameters
        ----------
        lengthscales : np.ndarray or List
            GP lengthscales, shape (numPODmodes, num_samples)
        variances : np.ndarray or List 
            GP variances, shape (numPODmodes, num_samples)
        noises : np.ndarray or List
            GP noise levels, shape (numPODmodes, num_samples)
        gp_class : class
            BayesianGP class to use for predictions
        double : bool
            If True, show both training and eval grids
        figsize : tuple
            Figure size
        max_num_samples : int
            Maximum number of samples to plot
        """
        plt.clf()

        # Store hyperparameters
        self.gp_lengthscales = lengthscales if isinstance(lengthscales, np.ndarray) else np.array(lengthscales)
        self.gp_variances = variances if isinstance(variances, np.ndarray) else np.array(variances)
        self.gp_noises = noises if isinstance(noises, np.ndarray) else np.array(noises)

        num_samples = min(self.gp_lengthscales.shape[1], self.gp_variances.shape[1], 
                         self.gp_noises.shape[1], max_num_samples)
        print(f"Number of samples: {num_samples}")

        if double:
            fig, ax = plt.subplots(self.numPODmodes, 2, figsize=figsize, sharey='row', sharex='col')
        else:
            fig, ax = plt.subplots(self.numPODmodes, 1, figsize=figsize)
            ax = np.array([[ax[i], None] for i in range(self.numPODmodes)])
        
        gp = gp_class()
        gp.X_train = self.time_domain_training[:, None]

        # Get scaled training data if scaler available
        if self.scaler is not None:
            snapshots_scaled = self.scaler.transform(self.snapshots_training)
        else:
            snapshots_scaled = self.snapshots_training

        for i in range(self.numPODmodes):
            gp.y_train = snapshots_scaled[i]

            # Plot original training data
            ax[i, 0].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
            if double:
                ax[i, 1].plot(self.time_domain_training, self.snapshots_training[i], 'k*')

            means, stds, eval_means, eval_stds = [], [], [], []
            for j in range(num_samples):
                mean, std, _ = gp.predict_with_hypers(
                    X_test=self.time_domain_training[:, None],
                    lengthscale=self.gp_lengthscales[i][j],
                    variance=self.gp_variances[i][j],
                    noise=self.gp_noises[i][j]
                )
                means.append(mean)
                stds.append(std)
                
                if double:
                    eval_mean, eval_std, _ = gp.predict_with_hypers(
                        X_test=self.time_domain_eval_training[:, None],
                        lengthscale=self.gp_lengthscales[i][j],
                        variance=self.gp_variances[i][j],
                        noise=self.gp_noises[i][j]
                    )
                    eval_means.append(eval_mean)
                    eval_stds.append(eval_std)
            
            means, stds = np.array(means), np.array(stds)
            
            # Inverse transform if needed
            if self.scaler is not None:
                means_orig = means * self.scaler.stds_[i, 0] + self.scaler.means_[i, 0]
                stds_orig = stds * self.scaler.stds_[i, 0]
            else:
                means_orig, stds_orig = means, stds

            ax[i, 0].plot(self.time_domain_training, means_orig.T, alpha=0.3)
            ax[i, 0].fill_between(
                self.time_domain_training,
                np.mean(means_orig, axis=0) - 2*np.mean(stds_orig, axis=0),
                np.mean(means_orig, axis=0) + 2*np.mean(stds_orig, axis=0),
                alpha=0.3, color="gray", label="Mean ± 2 std"
            )

            if double:
                eval_means, eval_stds = np.array(eval_means), np.array(eval_stds)
                if self.scaler is not None:
                    eval_means_orig = eval_means * self.scaler.stds_[i, 0] + self.scaler.means_[i, 0]
                    eval_stds_orig = eval_stds * self.scaler.stds_[i, 0]
                else:
                    eval_means_orig, eval_stds_orig = eval_means, eval_stds
                    
                ax[i, 1].plot(self.time_domain_eval_training, eval_means_orig.T, alpha=0.3)
                ax[i, 1].fill_between(
                    self.time_domain_eval_training,
                    np.mean(eval_means_orig, axis=0) - 2*np.mean(eval_stds_orig, axis=0),
                    np.mean(eval_means_orig, axis=0) + 2*np.mean(eval_stds_orig, axis=0),
                    alpha=0.3, color="gray", label="Mean ± 2 std"
                )

            ax[i, 0].set_title(f"Mode {i+1} Training Domain")
            ax[i, 0].legend()
            ax[i, 0].grid()
            
            if double:
                ax[i, 1].set_title(f"Mode {i+1} Eval Grid")
                ax[i, 1].legend()
                ax[i, 1].grid()

        fig.suptitle("GP Hyperparameter Samples", fontsize=16)
        fig.tight_layout()
        return fig, ax

    def gp_plot_derivatives(self,
                           figsize: Tuple[int, int] = (12, 8),
                           eval: bool = True
                           ):
        """
        Plot GP derivative estimates in the training domain.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        eval : bool
            If True, show predictions on eval grid
        """
        if self.gp_lengthscales is None:
            raise ValueError("Must call gp_plot_state first to set GP hyperparameters")
        
        plt.clf()

        # Get scaled training data if scaler available
        if self.scaler is not None:
            snapshots_scaled = self.scaler.transform(self.snapshots_training)
        else:
            snapshots_scaled = self.snapshots_training

        K_yys, K_zys, K_zzs = [], [], []
        for i in range(self.numPODmodes):
            ell2 = self.gp_lengthscales[i].mean(axis=0)**2
            
            # Standard RBF kernels
            rbf_yy = rbf_eval(self.gp_lengthscales[i].mean(axis=0), 
                             self.gp_variances[i].mean(axis=0),
                             self.time_domain_training, self.time_domain_training)
            rbf_zy = rbf_eval(self.gp_lengthscales[i].mean(axis=0),
                             self.gp_variances[i].mean(axis=0),
                             self.time_domain_eval_training, self.time_domain_training)
            rbf_zz = rbf_eval(self.gp_lengthscales[i].mean(axis=0),
                             self.gp_variances[i].mean(axis=0),
                             self.time_domain_eval_training, self.time_domain_eval_training)

            # K_yy with noise term
            K_yy = rbf_yy + 1e-5 * np.eye(len(self.time_domain_training))

            # K_zy: derivative kernel
            diff_zy = self.time_domain_eval_training[:, None] - self.time_domain_training[None, :]
            K_zy = -(diff_zy / ell2) * rbf_zy
            
            # K_zz: second derivative kernel
            diff_zz = self.time_domain_eval_training[:, None] - self.time_domain_eval_training[None, :]
            K_zz = ((1 - (diff_zz**2 / ell2)) / ell2) * rbf_zz
            
            K_yys.append(K_yy)
            K_zys.append(K_zy)
            K_zzs.append(K_zz)

        # Compute GP derivative predictions
        mu_z_scaled, cov_z_scaled = [], []
        for i in range(self.numPODmodes):
            w = jnp.linalg.solve(K_yys[i], snapshots_scaled[i])
            mu_zi = K_zys[i] @ w
            mu_z_scaled.append(mu_zi)
            cov_zi = K_zzs[i] - K_zys[i] @ jnp.linalg.solve(K_yys[i], K_zys[i].T)
            cov_z_scaled.append(cov_zi)

        # Compute finite difference derivatives in original space
        self.snapshots_training_derivatives = compute_derivatives_fourth_order(
            self.snapshots_training, self.time_domain_training
        )

        # Convert GP predictions back to original space
        if self.scaler is not None:
            mu_z = [mu_z_scaled[i] * self.scaler.stds_[i, 0] for i in range(self.numPODmodes)]
            std_z = [jnp.sqrt(jnp.diag(cov_z_scaled[i])) * self.scaler.stds_[i, 0] 
                    for i in range(self.numPODmodes)]
        else:
            mu_z = mu_z_scaled
            std_z = [jnp.sqrt(jnp.diag(cov_z_scaled[i])) for i in range(self.numPODmodes)]

        fig, ax = plt.subplots(self.numPODmodes, 1, figsize=figsize, sharex=True)
        if self.numPODmodes == 1:
            ax = [ax]
            
        for i in range(self.numPODmodes):
            ax[i].plot(self.time_domain_training, self.snapshots_training_derivatives[i], 
                      'k*', label='Finite Diff')
            if eval:
                ax[i].plot(self.time_domain_eval_training, mu_z[i], label='GP Mean')
                ax[i].fill_between(
                    self.time_domain_eval_training,
                    mu_z[i] - 2 * std_z[i],
                    mu_z[i] + 2 * std_z[i],
                    color='gray', alpha=0.3, label='± 2 Std'
                )
            ax[i].set_title(f"Mode {i+1} Derivative")
            ax[i].legend()
            ax[i].grid()
        
        fig.tight_layout()
        return fig, ax

    def operator_plot(self,
                     q0: np.ndarray,
                     operator_samples: np.ndarray | List,
                     latent_state_samples: np.ndarray | List,
                     rom,
                     input_func: Optional[Callable] = None,
                     figsize: Tuple[int, int] = (12, 8),
                     max_num_samples: int = 1000,
                     plot_samples: bool = False,
                     plot_single: bool = False,
                     training_span: Optional[Tuple[float, float]] = None,
                     save: bool = False,
                     save_path: str = "operator_inference_trajectories.png"
                     ):
        """
        Plot operator inference trajectories from posterior samples.
        
        Parameters
        ----------
        q0 : np.ndarray
            Initial condition for ROM
        operator_samples : np.ndarray or List
            Operator samples from posterior
        latent_state_samples : np.ndarray or List
            Latent state samples from posterior
        rom : opinf.ROM
            ROM object for predictions
        input_func : callable, optional
            Input function for ROM
        figsize : tuple
            Figure size
        max_num_samples : int
            Maximum samples to use
        plot_samples : bool
            If True, plot individual samples
        plot_single : bool
            If True, single column layout
        save : bool
            If True, save figure
        save_path : str
            Path to save figure
        """
        plt.clf()

        operator_samples = operator_samples if isinstance(operator_samples, np.ndarray) else np.array(operator_samples)
        latent_state_samples = np.transpose(latent_state_samples, (1, 0, 2)) if isinstance(latent_state_samples, np.ndarray) else np.transpose(np.array(latent_state_samples), (1, 0, 2))

        print(f"Operator samples: {operator_samples.shape}, Latent states: {latent_state_samples.shape}")
        samples = min(operator_samples.shape[0], latent_state_samples.shape[0], max_num_samples)

        # Generate ROM solves
        rom_solves_training, rom_solves_prediction = [], []
        for i in range(samples):
            operator = operator_samples[i]
            rom.model._extract_operators(operator)
            
            # Training domain
            if input_func is not None:
                rom.model.predict(state0=q0, t=self.time_domain_eval_training, input_func=input_func)
            else:
                rom.model.predict(state0=q0, t=self.time_domain_eval_training)
            if rom.model.predict_result_.y.shape[1] < self.time_domain_eval_training.size:
                continue
            rom_solves_training.append(rom.model.predict_result_.y)

            # Prediction domain
            if input_func is not None:
                rom.model.predict(state0=q0, t=self.time_domain_eval_prediction, input_func=input_func)
            else:
                rom.model.predict(state0=q0, t=self.time_domain_eval_prediction)
            if rom.model.predict_result_.y.shape[1] < self.time_domain_eval_prediction.size:
                continue
            rom_solves_prediction.append(rom.model.predict_result_.y)

        rom_solves_training = np.array(rom_solves_training)
        rom_solves_prediction = np.array(rom_solves_prediction)
        print(f"Stable solves - Training: {len(rom_solves_training)}, Prediction: {len(rom_solves_prediction)}")

        if plot_single:
            fig, ax = plt.subplots(self.numPODmodes, 1, figsize=figsize, sharey=False, sharex=True)
            if self.numPODmodes == 1:
                ax = [ax]
            
            for i in range(self.numPODmodes):
                # Training span shading
                if training_span is not None:
                    ax[i].axvspan(training_span[0], training_span[1],
                                  color='tab:blue', alpha=0.06, zorder=0)

                # True solution
                ax[i].plot(self.time_domain_prediction, self.snapshots_prediction[i],
                          color='tab:gray', lw=2, label='True solution')

                # Training snapshots
                ax[i].plot(self.time_domain_training, self.snapshots_training[i],
                          'k*', ms=5, label='Training data', zorder=5)

                # ROM median
                ax[i].plot(self.time_domain_eval_prediction, 
                          np.median(rom_solves_prediction[:, i, :], axis=0),
                          color='tab:blue', alpha=0.9, lw=2, label='ROM median')

                # ROM 5-95% band
                ax[i].fill_between(
                    self.time_domain_eval_prediction,
                    np.percentile(rom_solves_prediction[:, i, :], 5, axis=0),
                    np.percentile(rom_solves_prediction[:, i, :], 95, axis=0),
                    color='tab:blue', alpha=0.15, label='ROM 5\u201395%'
                )

                ax[i].set_ylabel(f'Mode {i+1}')
                ax[i].grid(True, alpha=0.3)
                if i == 0:
                    ax[i].legend(loc='upper right', fontsize=9)

            ax[-1].set_xlabel('Time')
        else:
            fig, ax = plt.subplots(self.numPODmodes, 3, figsize=figsize, sharey='row', sharex='col')
            if self.numPODmodes == 1:
                ax = ax.reshape(1, -1)

            for i in range(self.numPODmodes):
                # Training span shading on all columns
                if training_span is not None:
                    for j in range(3):
                        ax[i, j].axvspan(training_span[0], training_span[1],
                                         color='tab:blue', alpha=0.06, zorder=0)

                ax[i, 0].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                ax[i, 1].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                ax[i, 2].plot(self.time_domain_prediction, self.snapshots_prediction[i], 
                             color='tab:gray', lw=2)

                if plot_samples:
                    ax[i, 0].plot(self.time_domain_eval_training, rom_solves_training[:, i, :].T, alpha=0.3, lw=2)
                    ax[i, 1].plot(self.time_domain_eval_prediction, rom_solves_prediction[:, i, :].T, alpha=0.3, lw=2)
                    ax[i, 2].plot(self.time_domain_eval_prediction, rom_solves_prediction[:, i, :].T, alpha=0.3, lw=2)

                # Mean
                ax[i, 0].plot(self.time_domain_eval_training, rom_solves_training[:, i, :].T.mean(axis=1), alpha=0.8, lw=2)
                ax[i, 1].plot(self.time_domain_eval_prediction, rom_solves_prediction[:, i, :].T.mean(axis=1), alpha=0.8, lw=2)
                ax[i, 2].plot(self.time_domain_eval_prediction, rom_solves_prediction[:, i, :].T.mean(axis=1), alpha=0.8, lw=2)

                # Median
                ax[i, 0].plot(self.time_domain_eval_training, np.median(rom_solves_training[:, i, :], axis=0), alpha=0.8, linestyle='--', lw=2)
                ax[i, 1].plot(self.time_domain_eval_prediction, np.median(rom_solves_prediction[:, i, :], axis=0), alpha=0.8, linestyle='--', lw=2)
                ax[i, 2].plot(self.time_domain_eval_prediction, np.median(rom_solves_prediction[:, i, :], axis=0), alpha=0.8, linestyle='--', lw=2)

                # 5th and 95th percentiles
                ax[i, 0].fill_between(self.time_domain_eval_training, np.percentile(rom_solves_training[:, i, :], 5, axis=0), np.percentile(rom_solves_training[:, i, :], 95, axis=0), alpha=0.2)
                ax[i, 1].fill_between(self.time_domain_eval_prediction, np.percentile(rom_solves_prediction[:, i, :], 5, axis=0), np.percentile(rom_solves_prediction[:, i, :], 95, axis=0), alpha=0.2)
                ax[i, 2].fill_between(self.time_domain_eval_prediction, np.percentile(rom_solves_prediction[:, i, :], 5, axis=0), np.percentile(rom_solves_prediction[:, i, :], 95, axis=0), alpha=0.2)

                # Set y-limits
                yvals = np.asarray(self.snapshots_prediction[i])
                ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
                if np.isclose(ymin, ymax):
                    pad = 1.0 if np.isclose(ymax, 0.0) else abs(ymax) * 0.75
                    ymin -= pad
                    ymax += pad
                else:
                    ymin = ymin - abs(ymin) * 0.75
                    ymax = ymax * 1.75
                ax[i, 0].set_ylim(float(ymin), float(ymax))
                ax[i, 1].set_ylim(float(ymin), float(ymax))
                ax[i, 2].set_ylim(float(ymin), float(ymax))

                for j in range(3):
                    ax[i, j].grid()

        fig.suptitle("Operator Inference Trajectories", fontsize=16)
        fig.tight_layout()
        
        if save:
            fig.savefig(save_path, dpi=300)
        
        return fig, ax, rom_solves_training, rom_solves_prediction

    def operator_plot_trajectories(self,
                                  draws_training: np.ndarray | List,
                                  draws_prediction: np.ndarray | List,
                                  time_domain_training: Optional[np.ndarray] = None,
                                  time_domain_prediction: Optional[np.ndarray] = None,
                                  figsize: Tuple[int, int] = (12, 8),
                                  plot_single: bool = False
                                  ):
        """
        Plot ROM trajectories from pre-computed draws.
        
        Parameters
        ----------
        draws_training : np.ndarray or List
            Pre-computed draws on training domain
        draws_prediction : np.ndarray or List
            Pre-computed draws on prediction domain
        time_domain_training : np.ndarray, optional
            Time domain for training (defaults to eval_training)
        time_domain_prediction : np.ndarray, optional
            Time domain for prediction (defaults to eval_prediction)
        figsize : tuple
            Figure size
        plot_single : bool
            If True, single column layout
        """
        plt.clf()

        if time_domain_training is None:
            time_domain_training = self.time_domain_eval_training
        if time_domain_prediction is None:
            time_domain_prediction = self.time_domain_eval_prediction

        if isinstance(draws_training, list):
            draws_training = np.array(draws_training)
        if isinstance(draws_prediction, list):
            draws_prediction = np.array(draws_prediction)

        # Inverse transform if scaler available
        if self.scaler is not None:
            draws_training = np.array([self.scaler.inverse_transform(draws_training[j]) 
                                       for j in range(draws_training.shape[0])])
            draws_prediction = np.array([self.scaler.inverse_transform(draws_prediction[j]) 
                                         for j in range(draws_prediction.shape[0])])

        if plot_single:
            fig, ax = plt.subplots(self.numPODmodes, 1, figsize=figsize, sharey=False)
            if self.numPODmodes == 1:
                ax = [ax]
            
            for i in range(self.numPODmodes):
                ax[i].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                ax[i].plot(self.time_domain_prediction, self.snapshots_prediction[i],
                          color='tab:gray', lw=2, label='Ground Truth')
                ax[i].plot(time_domain_prediction, np.median(draws_prediction, axis=0)[i],
                          alpha=0.8, linestyle='--', lw=2, label='Median')
                ax[i].fill_between(
                    time_domain_prediction,
                    np.percentile(draws_prediction, 5, axis=0)[i],
                    np.percentile(draws_prediction, 95, axis=0)[i],
                    alpha=0.2
                )
                ax[i].set_ylabel(f'Mode {i+1}')
                ax[i].legend()
        else:
            fig, ax = plt.subplots(self.numPODmodes, 3, figsize=figsize, sharey='row', sharex='col')
            if self.numPODmodes == 1:
                ax = ax.reshape(1, -1)

            for i in range(self.numPODmodes):
                ax[i, 0].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                ax[i, 1].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                ax[i, 2].plot(self.time_domain_prediction, self.snapshots_prediction[i], 
                             color='tab:gray', lw=2)

                # Mean
                ax[i, 0].plot(time_domain_training, draws_training.mean(axis=0)[i], alpha=0.8, lw=2)
                ax[i, 1].plot(time_domain_prediction, draws_prediction.mean(axis=0)[i], alpha=0.8, lw=2)
                ax[i, 2].plot(time_domain_prediction, draws_prediction.mean(axis=0)[i], alpha=0.8, lw=2)

                # Median
                ax[i, 0].plot(time_domain_training, np.median(draws_training, axis=0)[i], alpha=0.8, linestyle='--', lw=2)
                ax[i, 1].plot(time_domain_prediction, np.median(draws_prediction, axis=0)[i], alpha=0.8, linestyle='--', lw=2)
                ax[i, 2].plot(time_domain_prediction, np.median(draws_prediction, axis=0)[i], alpha=0.8, linestyle='--', lw=2)

                # Percentiles
                ax[i, 0].fill_between(time_domain_training, np.percentile(draws_training, 5, axis=0)[i], np.percentile(draws_training, 95, axis=0)[i], alpha=0.2)
                ax[i, 1].fill_between(time_domain_prediction, np.percentile(draws_prediction, 5, axis=0)[i], np.percentile(draws_prediction, 95, axis=0)[i], alpha=0.2)
                ax[i, 2].fill_between(time_domain_prediction, np.percentile(draws_prediction, 5, axis=0)[i], np.percentile(draws_prediction, 95, axis=0)[i], alpha=0.2)

                # Y-limits
                yvals = np.asarray(self.snapshots_prediction[i])
                ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
                if np.isclose(ymin, ymax):
                    pad = 1.0 if np.isclose(ymax, 0.0) else abs(ymax) * 0.5
                    ymin -= pad
                    ymax += pad
                else:
                    ymin = ymin - abs(ymin) * 0.5
                    ymax = ymax * 1.5
                for j in range(3):
                    ax[i, j].set_ylim(float(ymin), float(ymax))
                    ax[i, j].grid()

        fig.suptitle("Operator Inference Trajectories", fontsize=16)
        fig.tight_layout()
        return fig, ax


def plot_deterministic_rom_solves(
    grid_search_result,
    snapshots_compressed: np.ndarray,
    time_sampled: np.ndarray,
    time_eval_training: np.ndarray,
    time_eval_prediction: np.ndarray,
    time_full: Optional[np.ndarray] = None,
    true_states_compressed: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[float, float]] = None,
    alpha_all: float = 0.3,
    alpha_best: float = 0.9,
    lw_all: float = 1.5,
    lw_best: float = 2.5,
):
    """
    Plot all stable deterministic ROM solves from grid search.
    
    Displays all stable solves in orange with transparency and the chosen
    best operator in blue with less transparency for visual debugging.
    
    Parameters
    ----------
    grid_search_result : GridSearchResult
        Result from grid_search_prior_operator containing stable_results
    snapshots_compressed : np.ndarray
        Compressed training snapshots, shape (num_modes, num_samples)
    time_sampled : np.ndarray
        Training time points
    time_eval_training : np.ndarray
        Dense time points for training domain evaluation
    time_eval_prediction : np.ndarray
        Dense time points for prediction domain evaluation
    time_full : np.ndarray, optional
        Full time domain for true trajectory (if available)
    true_states_compressed : np.ndarray, optional
        True compressed states over full time domain
    figsize : tuple, optional
        Figure size. Default computed from num_modes
    alpha_all : float
        Transparency for all stable solves (default: 0.3)
    alpha_best : float
        Transparency for best solve (default: 0.9)
    lw_all : float
        Line width for all stable solves (default: 1.5)
    lw_best : float
        Line width for best solve (default: 2.5)
        
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    num_modes = snapshots_compressed.shape[0]
    q0 = snapshots_compressed[:, 0]
    
    if figsize is None:
        figsize = (14, 3 * num_modes)
    
    fig, axes = plt.subplots(num_modes, 2, figsize=figsize, sharex='col')
    if num_modes == 1:
        axes = axes.reshape(1, -1)
    
    # Collect all stable solves
    training_solves = []
    prediction_solves = []
    best_training_solve = None
    best_prediction_solve = None
    
    stable_results = grid_search_result.stable_results
    best_operator = grid_search_result.operator
    
    for reg, error, operator, rom in stable_results:
        rom.model._extract_operators(operator)
        
        # Training domain solve
        rom.model.predict(state0=q0, t=time_eval_training)
        train_sol = rom.model.predict_result_.y
        train_stable = train_sol.shape[1] == len(time_eval_training)
        
        # Prediction domain solve
        rom.model.predict(state0=q0, t=time_eval_prediction)
        pred_sol = rom.model.predict_result_.y
        pred_stable = pred_sol.shape[1] == len(time_eval_prediction)
        
        if train_stable:
            training_solves.append((train_sol, operator))
        if pred_stable:
            prediction_solves.append((pred_sol, operator))
        
        # Check if this is the best operator
        if np.allclose(operator, best_operator):
            if train_stable:
                best_training_solve = train_sol
            if pred_stable:
                best_prediction_solve = pred_sol
    
    print(f"Stable training solves: {len(training_solves)}/{len(stable_results)}")
    print(f"Stable prediction solves: {len(prediction_solves)}/{len(stable_results)}")
    
    # Plot each mode
    for i in range(num_modes):
        # Left: Training domain
        # Plot all stable solves in orange
        for idx, (sol, _) in enumerate(training_solves):
            label = 'Other stable solves' if idx == 0 else None
            axes[i, 0].plot(time_eval_training, sol[i], color='tab:orange', 
                           alpha=alpha_all, lw=lw_all, label=label)
        
        # Plot best solve in blue
        if best_training_solve is not None:
            axes[i, 0].plot(time_eval_training, best_training_solve[i], 
                           color='tab:blue', alpha=alpha_best, lw=lw_best, 
                           label='Best (chosen)')
        
        # Plot training data points
        axes[i, 0].plot(time_sampled, snapshots_compressed[i], 'k*', ms=4, 
                       label='Training data', zorder=5)
        
        axes[i, 0].set_ylabel(f'Mode {i+1}')
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].legend(loc='upper right', fontsize=8)
            axes[i, 0].set_title('Training Domain')
        
        # Right: Prediction domain
        # Plot true trajectory if available
        if time_full is not None and true_states_compressed is not None:
            axes[i, 1].plot(time_full, true_states_compressed[i], 
                           color='gray', lw=1.5, alpha=0.7, label='True trajectory')
        
        # Plot all stable solves in orange
        for idx, (sol, _) in enumerate(prediction_solves):
            label = 'Other stable solves' if idx == 0 else None
            axes[i, 1].plot(time_eval_prediction, sol[i], color='tab:orange', 
                           alpha=alpha_all, lw=lw_all, label=label)
        
        # Plot best solve in blue
        if best_prediction_solve is not None:
            axes[i, 1].plot(time_eval_prediction, best_prediction_solve[i], 
                           color='tab:blue', alpha=alpha_best, lw=lw_best, 
                           label='Best (chosen)')
        
        # Plot training data points
        axes[i, 1].plot(time_sampled, snapshots_compressed[i], 'ko', ms=4, 
                       label='Training data', zorder=5)
        
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].legend(loc='upper right', fontsize=8)
            axes[i, 1].set_title('Prediction Domain')
    
    axes[-1, 0].set_xlabel('Time')
    axes[-1, 1].set_xlabel('Time')
    fig.suptitle('Deterministic ROM Solves: Grid Search Results', fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig, axes


def plot_gp_fit(
    gp_models: List,
    snapshots_compressed: np.ndarray,
    time_sampled: np.ndarray,
    time_eval: np.ndarray,
    lengthscales: np.ndarray,
    variances: np.ndarray,
    figsize: Optional[Tuple[float, float]] = None,
    plot_derivatives: bool = True,
):
    """
    Plot GP fit quality for states and optionally derivatives.
    
    Works with both MLE-fitted GPs (single hyperparameter values) and
    Bayesian GPs (samples of hyperparameters).
    
    Parameters
    ----------
    gp_models : List
        List of fitted GP models (one per mode), with predict method
    snapshots_compressed : np.ndarray
        Compressed training snapshots, shape (num_modes, num_samples)
    time_sampled : np.ndarray
        Training time points
    time_eval : np.ndarray
        Dense time points for GP evaluation
    lengthscales : np.ndarray
        GP lengthscales, shape (num_modes,) for MLE or (num_modes, num_samples) for Bayesian
    variances : np.ndarray
        GP variances, shape (num_modes,) for MLE or (num_modes, num_samples) for Bayesian
    figsize : tuple, optional
        Figure size. Default computed from num_modes
    plot_derivatives : bool
        If True, also plot derivative predictions
        
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    num_modes = snapshots_compressed.shape[0]
    
    if figsize is None:
        figsize = (14, 3 * num_modes) if not plot_derivatives else (14, 3 * num_modes)
    
    ncols = 2 if plot_derivatives else 1
    fig, axes = plt.subplots(num_modes, ncols, figsize=figsize, squeeze=False, sharex="col")
    
    # Compute finite difference derivatives for comparison
    if plot_derivatives:
        fd_derivatives = compute_derivatives_fourth_order(snapshots_compressed, time_sampled)
    
    for i in range(num_modes):
        gp = gp_models[i]
        
        # Get GP predictions on dense grid
        mean_pred, std_pred = gp.predict(time_eval[:, None], return_std=True)
        
        # Get GP predictions on training points (for residual check)
        mean_train, std_train = gp.predict(time_sampled[:, None], return_std=True)
        
        # === Left: State fit ===
        ax_state = axes[i, 0]
        
        # Training data
        ax_state.plot(time_sampled, snapshots_compressed[i], 'k*', ms=5, 
                     label='Training data', zorder=5)
        
        # GP mean prediction
        ax_state.plot(time_eval, mean_pred, 'tab:blue', lw=2, label='GP mean')
        
        # GP uncertainty (±2σ)
        ax_state.fill_between(time_eval, 
                             mean_pred - 2*std_pred, 
                             mean_pred + 2*std_pred,
                             color='tab:blue', alpha=0.2, label='±2σ')
        
        ax_state.set_ylabel(f'Mode {i+1}')
        ax_state.grid(True, alpha=0.3)
        if i == 0:
            ax_state.legend(loc='upper right', fontsize=8)
            ax_state.set_title('GP State Fit')
        
        # === Right: Derivative fit ===
        if plot_derivatives:
            ax_deriv = axes[i, 1]
            
            # Compute GP derivative prediction using kernel derivatives
            ell = lengthscales[i] if lengthscales.ndim == 1 else lengthscales[i].mean()
            var = variances[i] if variances.ndim == 1 else variances[i].mean()
            ell2 = ell ** 2
            
            # Build kernel matrices for derivative prediction
            # K_yy: state-state kernel at training points
            rbf_yy = rbf_eval(ell, var, time_sampled, time_sampled)
            K_yy = rbf_yy + 1e-6 * np.eye(len(time_sampled))
            
            # K_zy: derivative-state cross-kernel (eval points vs training points)
            rbf_zy = rbf_eval(ell, var, time_eval, time_sampled)
            diff_zy = time_eval[:, None] - time_sampled[None, :]
            K_zy = -(diff_zy / ell2) * rbf_zy
            
            # K_zz: derivative-derivative kernel at eval points
            rbf_zz = rbf_eval(ell, var, time_eval, time_eval)
            diff_zz = time_eval[:, None] - time_eval[None, :]
            K_zz = ((1 - (diff_zz**2 / ell2)) / ell2) * rbf_zz
            
            # Derivative mean and variance
            alpha = jnp.linalg.solve(K_yy, snapshots_compressed[i])
            mu_deriv = K_zy @ alpha
            cov_deriv = K_zz - K_zy @ jnp.linalg.solve(K_yy, K_zy.T)
            std_deriv = jnp.sqrt(jnp.maximum(jnp.diag(cov_deriv), 1e-10))
            
            # Finite difference derivatives (ground truth proxy)
            ax_deriv.plot(time_sampled, fd_derivatives[i], 'ko', ms=5, 
                         label='Finite diff', zorder=5)
            
            # GP derivative prediction
            ax_deriv.plot(time_eval, mu_deriv, 'tab:orange', lw=2, label='GP derivative')
            ax_deriv.fill_between(time_eval,
                                 mu_deriv - 2*std_deriv,
                                 mu_deriv + 2*std_deriv,
                                 color='tab:orange', alpha=0.2, label='±2σ')
            
            ax_deriv.grid(True, alpha=0.3)
            if i == 0:
                ax_deriv.legend(loc='upper right', fontsize=8)
                ax_deriv.set_title('GP Derivative Fit')
    
    axes[-1, 0].set_xlabel('Time')
    if plot_derivatives:
        axes[-1, 1].set_xlabel('Time')
    
    fig.suptitle('GP Fit Quality', fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig, axes


def plot_full_order_error(
    rom_solves: np.ndarray,
    basis,
    true_states: np.ndarray,
    time_domain_full: np.ndarray,
    time_domain_eval: np.ndarray,
    training_span: Tuple[float, float],
    figsize: Optional[Tuple[float, float]] = None,
    error_type: str = 'relative',
):
    """
    Plot full order prediction error over time, comparing ROM predictions to projection error.
    
    Parameters
    ----------
    rom_solves : np.ndarray
        ROM solutions, shape (num_samples, num_modes, num_time_eval)
    basis : opinf.basis
        POD basis with decompress method
    true_states : np.ndarray
        True full order states, shape (n_dof, num_time_full)
    time_domain_full : np.ndarray
        Full time domain for true states
    time_domain_eval : np.ndarray
        Time domain for ROM evaluation (should match rom_solves)
    training_span : tuple
        (t_start, t_end) for training region
    figsize : tuple, optional
        Figure size
    error_type : str
        'relative' for relative error, 'absolute' for absolute error
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    from scipy.interpolate import interp1d
    
    if figsize is None:
        figsize = (12, 5)
    
    num_samples = rom_solves.shape[0]
    
    # Interpolate true states onto evaluation time grid
    interp_truth = interp1d(time_domain_full, true_states, axis=1, 
                           kind='linear', fill_value='extrapolate')
    true_states_interp = interp_truth(time_domain_eval)
    
    # Compute projection error (best possible with this basis)
    # Project true states onto basis and reconstruct
    true_compressed = basis.compress(true_states_interp)
    true_projected = basis.decompress(true_compressed)
    
    if error_type == 'relative':
        norm_truth = np.linalg.norm(true_states_interp, axis=0)
        norm_truth = np.maximum(norm_truth, 1e-10)  # Avoid division by zero
        projection_error = np.linalg.norm(true_states_interp - true_projected, axis=0) / norm_truth
    else:
        projection_error = np.linalg.norm(true_states_interp - true_projected, axis=0)
    
    # Compute ROM prediction errors for each sample
    rom_errors = []
    for i in range(num_samples):
        rom_full_order = basis.decompress(rom_solves[i])  # (n_dof, num_time_eval)
        if error_type == 'relative':
            error = np.linalg.norm(true_states_interp - rom_full_order, axis=0) / norm_truth
        else:
            error = np.linalg.norm(true_states_interp - rom_full_order, axis=0)
        rom_errors.append(error)
    
    rom_errors = np.array(rom_errors)  # (num_samples, num_time_eval)
    
    # Statistics
    rom_error_mean = rom_errors.mean(axis=0)
    rom_error_median = np.median(rom_errors, axis=0)
    rom_error_5 = np.percentile(rom_errors, 5, axis=0)
    rom_error_95 = np.percentile(rom_errors, 95, axis=0)
    
    # Create plot with 3 subplots sharing x-axis
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1]})
    ylabel = 'Relative Error' if error_type == 'relative' else 'Absolute Error'

    # --- Subplot 1: ROM prediction error ---
    ax_rom = axes[0]
    ax_rom.axvspan(training_span[0], training_span[1], color='tab:blue', alpha=0.1)
    ax_rom.plot(time_domain_eval, rom_error_median, 'tab:blue', lw=2,
                label='ROM error (median)')
    ax_rom.fill_between(time_domain_eval, rom_error_5, rom_error_95,
                        color='tab:blue', alpha=0.3, label='ROM error (5\u201395%)')
    ax_rom.plot(time_domain_eval, rom_error_mean, 'tab:orange', lw=1.5,
                linestyle=':', label='ROM error (mean)')
    ax_rom.set_ylabel(ylabel)
    ax_rom.set_title('ROM Prediction Error')
    ax_rom.legend(loc='upper left', fontsize=9)
    ax_rom.grid(True, alpha=0.3)
    ax_rom.set_yscale('log')

    # --- Subplot 2: Projection error (basis limit) ---
    ax_proj = axes[1]
    ax_proj.axvspan(training_span[0], training_span[1], color='tab:blue', alpha=0.1)
    ax_proj.plot(time_domain_eval, projection_error, 'k--', lw=2,
                 label='Projection error (basis limit)')
    ax_proj.set_ylabel(ylabel)
    ax_proj.set_title('Projection Error (Basis Limit)')
    ax_proj.legend(loc='upper left', fontsize=9)
    ax_proj.grid(True, alpha=0.3)
    ax_proj.set_yscale('log')

    # --- Subplot 3: ROM error minus projection error ---
    ax_diff = axes[2]
    ax_diff.axvspan(training_span[0], training_span[1], color='tab:blue', alpha=0.1)
    rom_minus_proj = np.maximum(rom_error_median - projection_error, 1e-16)
    ax_diff.plot(time_domain_eval, rom_minus_proj, 'tab:purple', lw=2,
                 label='ROM error \u2212 projection error')
    ax_diff.set_xlabel('Time')
    ax_diff.set_ylabel(ylabel)
    ax_diff.set_title('Excess ROM Error (Above Basis Limit)')
    ax_diff.legend(loc='upper left', fontsize=9)
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_yscale('log')

    plt.tight_layout()

    return fig, axes

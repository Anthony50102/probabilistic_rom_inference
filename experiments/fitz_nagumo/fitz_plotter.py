import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Callable
import jax.numpy as jnp

from core import BayesianGP
from core.plotting import Plotter, rbf_eval, flatten_time, compute_derivatives_fourth_order, _ylim_from_truth


# =============================================================================
# Standalone utility functions
# =============================================================================

def plot_fitz_grid_search(
    grid_search_result,
    snapshots_compressed: np.ndarray,
    time_sampled: np.ndarray,
    time_eval_training: np.ndarray,
    time_eval_prediction: np.ndarray,
    num_modes: int,
    input_func: Callable,
    time_full: Optional[np.ndarray] = None,
    true_states_compressed: Optional[np.ndarray] = None,
    training_span: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Plot all stable deterministic ROM solves from grid search (with input support).

    Uses operator_plot style: single column, modes as rows, purple
    median + 5-95% band from all stable solves, gray true trajectory,
    training span shading, best solve in blue.
    """
    from core.plotting import plot_deterministic_rom_solves
    return plot_deterministic_rom_solves(
        grid_search_result=grid_search_result,
        snapshots_compressed=snapshots_compressed,
        time_sampled=time_sampled,
        time_eval_training=time_eval_training,
        time_eval_prediction=time_eval_prediction,
        time_full=time_full,
        true_states_compressed=true_states_compressed,
        input_func=input_func,
        training_span=training_span,
        figsize=figsize,
    )


class FitzPlotter(Plotter):
    """Plotter for FitzHugh-Nagumo equation experiments."""
    
    def __init__(self, *args, scaler=None, **kwargs) -> None:
        super().__init__(*args, scaler=scaler, **kwargs)

    def gp_plot_state(self,
                lengthscales: np.ndarray | List,
                variances: np.ndarray | List,
                noises: np.ndarray | List,
                double: bool = True,
                figsize=(12,8),
                max_num_samples: int = 1000
                ):
        plt.clf()

        # Put them in shapes (numPODmodes, num_samples)
        self.gp_lengthscales = lengthscales if not isinstance(lengthscales, list) else np.array(lengthscales)
        self.gp_variances = variances if not isinstance(variances, list) else np.array(variances)
        self.gp_noises = noises if not isinstance(noises, list) else np.array(noises)

        num_samples = min(self.gp_lengthscales.shape[1], self.gp_variances.shape[1], self.gp_noises.shape[1], max_num_samples)
        print(f"Number of samples: {num_samples}")

        if double:
            fig, ax = plt.subplots(self.numPODmodes, 2, figsize = figsize, sharey='row', sharex='col')
        
        gp = BayesianGP()
        gp.X_train = self.time_domain_training[:, None]

        for i in range(self.numPODmodes):
            # Use scaled data for GP if scaler is available
            if self.scaler is not None:
                y_train_scaled = self.scaler.transform(self.snapshots_training)[i]
            else:
                y_train_scaled = self.snapshots_training[i]
            
            gp.y_train = y_train_scaled

            # Plot original (unscaled) data
            ax[i,0].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
            ax[i,1].plot(self.time_domain_training, self.snapshots_training[i], 'k*')

            means, stds, eval_means, eval_stds = [], [], [], []
            for j in range(num_samples):
                mean, std, _ = gp.predict_with_hypers(X_test=self.time_domain_training[:, None], lengthscale=self.gp_lengthscales[i][j], variance=self.gp_variances[i][j], noise=self.gp_noises[i][j])
                means.append(mean)
                stds.append(std)
                eval_mean, eval_std, _ = gp.predict_with_hypers(X_test=self.time_domain_eval_training[:, None], lengthscale=self.gp_lengthscales[i][j], variance=self.gp_variances[i][j], noise=self.gp_noises[i][j])
                eval_means.append(eval_mean)
                eval_stds.append(eval_std)
            
            means, stds, eval_means, eval_stds = np.array(means), np.array(stds), np.array(eval_means), np.array(eval_stds)
            
            # Inverse transform predictions back to original scale for plotting
            if self.scaler is not None:
                means_orig = np.array([self.scaler.inverse_transform(np.vstack([means[j] for _ in range(self.numPODmodes)]))[i] for j in range(num_samples)])
                eval_means_orig = np.array([self.scaler.inverse_transform(np.vstack([eval_means[j] for _ in range(self.numPODmodes)]))[i] for j in range(num_samples)])
                stds_orig = stds * self.scaler.stds_[i, 0]
                eval_stds_orig = eval_stds * self.scaler.stds_[i, 0]
            else:
                means_orig, eval_means_orig = means, eval_means
                stds_orig, eval_stds_orig = stds, eval_stds
            
            ax[i,0].plot(self.time_domain_training, means_orig.T, alpha = .3)
            ax[i,1].plot(self.time_domain_eval_training, eval_means_orig.T, alpha = .3)
        
            ax[i,0].fill_between(self.time_domain_training, np.mean(means_orig, axis=0)-2*np.mean(stds_orig, axis=0), np.mean(means_orig, axis=0)+2*np.mean(stds_orig, axis=0), alpha = .3, color = "gray", label = "Mean $\\pm$ 2 std")
            ax[i,1].fill_between(self.time_domain_eval_training, np.mean(eval_means_orig, axis=0)-2*np.mean(eval_stds_orig, axis=0), np.mean(eval_means_orig, axis=0)+2*np.mean(eval_stds_orig, axis=0), alpha = .3, color = "gray", label = "Mean $\\pm$ 2 std")

            ax[i,0].set_title(f"Mode {i+1} Training Domain")
            ax[i,1].set_title(f"Mode {i+1} Training Domain Increase Density")
            ax[i,0].legend()
            ax[i,1].legend()

        fig.suptitle("GP Hyperparameter Samples", fontsize=16)
        fig.tight_layout()
        fig.show()

    def gp_plot_derivatives(
                self,
                eval: bool = True,
                lengthscales: np.ndarray | List = None,
                variances: np.ndarray | List = None,
                noises: np.ndarray | List = None,
                figsize=(12,8),
                max_num_samples: int = 1000
                ):
        plt.clf()

        # Determine if we're working with scaled data
        if self.scaler is not None:
            snapshots_scaled = self.scaler.transform(self.snapshots_training)
        else:
            snapshots_scaled = self.snapshots_training

        K_yys, K_zys, K_zzs = [], [], []
        for i in range(self.numPODmodes):
            ell2 = self.gp_lengthscales[i].mean(axis=0)**2
            
            # Standard RBF kernels
            rbf_yy = rbf_eval(self.gp_lengthscales[i].mean(axis=0), self.gp_variances[i].mean(axis=0), self.time_domain_training, self.time_domain_training)
            rbf_zy = rbf_eval(self.gp_lengthscales[i].mean(axis=0), self.gp_variances[i].mean(axis=0), self.time_domain_eval_training, self.time_domain_training)
            rbf_zz = rbf_eval(self.gp_lengthscales[i].mean(axis=0), self.gp_variances[i].mean(axis=0), self.time_domain_eval_training, self.time_domain_eval_training)

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

        # Compute GP derivative predictions in scaled space
        mu_z_scaled = []
        cov_z_scaled = []
        for i in range(self.numPODmodes):
            w = jnp.linalg.solve(K_yys[i], snapshots_scaled[i])
            mu_zi = K_zys[i] @ w
            mu_z_scaled.append(mu_zi)

            cov_zi = K_zzs[i] - K_zys[i] @ jnp.linalg.solve(K_yys[i], K_zys[i].T)
            cov_z_scaled.append(cov_zi)

        # Compute finite difference derivatives in original space
        self.snapshots_training_derivatives = compute_derivatives_fourth_order(self.snapshots_training, self.time_domain_training)

        # Convert GP predictions back to original space for plotting
        if self.scaler is not None:
            mu_z = [mu_z_scaled[i] * self.scaler.stds_[i, 0] for i in range(self.numPODmodes)]
            std_z = [jnp.sqrt(jnp.diag(cov_z_scaled[i])) * self.scaler.stds_[i, 0] for i in range(self.numPODmodes)]
        else:
            mu_z = mu_z_scaled
            std_z = [jnp.sqrt(jnp.diag(cov_z_scaled[i])) for i in range(self.numPODmodes)]

        fig, ax = plt.subplots(self.numPODmodes, 1, figsize = figsize, sharex=True)
        for i in range(self.numPODmodes):
            ax[i].plot(self.time_domain_training, self.snapshots_training_derivatives[i], 'k*', label='Training Data')
            if eval:
                ax[i].plot(self.time_domain_eval_training, mu_z[i], label='Predicted Mean')
                ax[i].fill_between(self.time_domain_eval_training, 
                                mu_z[i] - 2 * std_z[i], 
                                mu_z[i] + 2 * std_z[i], 
                                color='gray', alpha=0.3, label='Predicted Mean ± 2 Std Dev')
                ax[i].set_title(f"Mode {i+1} Derivative Prediction on Eval Grid")
            ax[i].legend()
        
        fig.tight_layout()
        fig.show()

    def operator_plot(
                    self,
                    # TODO: Add support for List q0
                    q0: np.ndarray | List,
                    operator_samples: np.ndarray | List,
                    latent_state_samples: np.ndarray | List,
                    rom,
                    input_func = None,
                    figsize: tuple = (12, 8),
                    max_num_samples = 1000,
                    plot_samples: bool = False,
                    plot_single: bool = False,
                    training_span: tuple = None,
                    save=False,
                    save_path: str = "operator_inference_trajectories.png"
                    ):
        plt.clf()

        self.operator_samples =  operator_samples if isinstance(operator_samples, np.ndarray) else np.array(operator_samples)
        self.latent_state_samples = np.transpose(latent_state_samples, (1,0,2)) if isinstance(latent_state_samples, np.ndarray) else np.transpose(np.array(latent_state_samples), (1,0,2))

        print(self.operator_samples.shape, self.latent_state_samples.shape)
        samples = min(self.operator_samples.shape[0], self.latent_state_samples.shape[0], max_num_samples)

        # Generate ROM solves
        rom_solves_training, rom_solves_prediction = [], []
        for i in range(samples):
            operator = self.operator_samples[i]
            rom.model._extract_operators(operator)
            # TODO: Can't cheat like this with starting value
            rom.model.predict(state0=q0, t=self.time_domain_eval_training, input_func=input_func)
            if rom.model.predict_result_.y.shape[1] < self.time_domain_eval_training.size:
                print("Bad solve within training domain, skipping", rom.model.predict_result_.y.shape)
                continue
            rom_solves_training.append(rom.model.predict_result_.y)

            rom.model.predict(state0=q0, t=self.time_domain_eval_prediction, input_func=input_func)
            if rom.model.predict_result_.y.shape[1] < self.time_domain_eval_prediction.size:
                print("Bad solve within prediction domain, skipping", rom.model.predict_result_.y.shape)
                continue
            rom_solves_prediction.append(rom.model.predict_result_.y)

        rom_solves_training, rom_solves_prediction = np.array(rom_solves_training), np.array(rom_solves_prediction)
        print(rom_solves_training.shape, rom_solves_prediction.shape)

        if plot_single:
            # Single row with one column per POD mode
            fig, ax = plt.subplots(self.numPODmodes, 1, figsize=figsize, sharey=False)
            
            # Handle case where numPODmodes = 1
            if self.numPODmodes == 1:
                ax = [ax]
            
            for i in range(self.numPODmodes):
                # Training span shading
                if training_span is not None:
                    ax[i].axvspan(training_span[0], training_span[1],
                                  color='gray', alpha=0.10, zorder=0)

                # Plot training data
                ax[i].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                
                # Plot ground truth
                ax[i].plot(self.time_domain_prediction, self.snapshots_prediction[i], 
                          color='tab:gray', lw=2, label='Ground Truth')
                
                # Plot the median (dashed purple)
                ax[i].plot(self.time_domain_eval_prediction, np.median(rom_solves_prediction[:,i,:], axis=0), 
                          color='tab:purple', linestyle='--', alpha=0.9, lw=2, label='Median')
                
                # Plot the 5th and 95th percentiles
                ax[i].fill_between(self.time_domain_eval_prediction, 
                                  np.percentile(rom_solves_prediction[:,i,:], 5, axis=0), 
                                  np.percentile(rom_solves_prediction[:,i,:], 95, axis=0), 
                                  color='tab:purple', alpha=0.15)
                
                ax[i].set_ylim(*_ylim_from_truth(self.snapshots_prediction[i]))
                ax[i].set_xlabel('Time')
                ax[i].set_ylabel(f'Mode {i+1}')
                ax[i].legend()
            
            fig.suptitle("Operator Inference Trajectories", fontsize=16)
        else:
            # Original three-column layout
            fig, ax = plt.subplots(self.numPODmodes, 3, figsize=figsize, sharey='row', sharex='col')

            for i in range(self.numPODmodes):
                # Training span shading on all columns
                if training_span is not None:
                    for j in range(3):
                        ax[i, j].axvspan(training_span[0], training_span[1],
                                         color='gray', alpha=0.10, zorder=0)

                ax[i, 0].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                ax[i, 1].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                ax[i, 2].plot(self.time_domain_prediction, self.snapshots_prediction[i], color='tab:gray', lw=2)

                if plot_samples:
                    ax[i, 0].plot(self.time_domain_eval_training, rom_solves_training[:,i,:].T, alpha = .3, lw=2)
                    ax[i, 1].plot(self.time_domain_eval_prediction, rom_solves_prediction[:,i,:].T, alpha = .3, lw=2)
                    ax[i, 2].plot(self.time_domain_eval_prediction, rom_solves_prediction[:,i,:].T, alpha = .3, lw=2)

                # Plot the median (dashed purple)
                ax[i, 0].plot(self.time_domain_eval_training, np.median(rom_solves_training[:,i,:], axis=0), color='tab:purple', linestyle='--', alpha=0.9, lw=2)
                ax[i, 1].plot(self.time_domain_eval_prediction, np.median(rom_solves_prediction[:,i,:], axis=0), color='tab:purple', linestyle='--', alpha=0.9, lw=2)
                ax[i, 2].plot(self.time_domain_eval_prediction, np.median(rom_solves_prediction[:,i,:], axis=0), color='tab:purple', linestyle='--', alpha=0.9, lw=2)

                # Plot the 5th and 95th percentiles
                ax[i, 0].fill_between(self.time_domain_eval_training, np.percentile(rom_solves_training[:,i,:], 5, axis=0), np.percentile(rom_solves_training[:,i,:], 95, axis=0), color='tab:purple', alpha=0.15)
                ax[i, 1].fill_between(self.time_domain_eval_prediction, np.percentile(rom_solves_prediction[:,i,:], 5, axis=0), np.percentile(rom_solves_prediction[:,i,:], 95, axis=0), color='tab:purple', alpha=0.15)
                ax[i, 2].fill_between(self.time_domain_eval_prediction, np.percentile(rom_solves_prediction[:,i,:], 5, axis=0), np.percentile(rom_solves_prediction[:,i,:], 95, axis=0), color='tab:purple', alpha=0.15)

                ymin, ymax = _ylim_from_truth(self.snapshots_prediction[i])

                ax[i, 0].set_ylim(ymin, ymax)
                ax[i, 1].set_ylim(ymin, ymax)
                ax[i, 2].set_ylim(ymin, ymax)

            fig.suptitle("Operator Inference Trajectories", fontsize=16)
        
        fig.tight_layout()
        if save:
            fig.savefig(save_path, dpi=300)
        fig.show()
    
    def operator_plot_trajectories(
                    self,
                    draws_training: List | np.ndarray,
                    draws_prediction: List | np.ndarray,
                    time_domain_training = None,
                    time_domain_prediction = None,
                    figsize: tuple = (12, 8),
                    plot_single: bool = False,
    ):
        plt.clf()

        if time_domain_training is None:
            time_domain_training = self.time_domain_eval_training

        if time_domain_prediction is None:
            time_domain_prediction = self.time_domain_eval_prediction

        if isinstance(draws_training, list):
            draws_training = np.array(draws_training)
        if isinstance(draws_prediction, list):
            draws_prediction = np.array(draws_prediction)

        # Inverse transform draws if scaler is available
        if self.scaler is not None:
            draws_training_orig = np.array([self.scaler.inverse_transform(draws_training[j]) for j in range(draws_training.shape[0])])
            draws_prediction_orig = np.array([self.scaler.inverse_transform(draws_prediction[j]) for j in range(draws_prediction.shape[0])])
        else:
            draws_training_orig = draws_training
            draws_prediction_orig = draws_prediction

        if plot_single:
            # Single row with one column per POD mode
            fig, ax = plt.subplots(self.numPODmodes, 1, figsize=figsize, sharey=False)
            
            # Handle case where numPODmodes = 1
            if self.numPODmodes == 1:
                ax = [ax]
            
            for i in range(self.numPODmodes):
                # Plot training data
                ax[i].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                
                # Plot ground truth
                ax[i].plot(self.time_domain_prediction, self.snapshots_prediction[i], 
                          color='tab:gray', lw=2, label='Ground Truth')
                
                # Plot the median (dashed purple)
                ax[i].plot(time_domain_prediction, np.median(draws_prediction_orig, axis=0)[i], 
                          color='tab:purple', linestyle='--', alpha=0.9, lw=2, label='Median')
                
                # Plot the 5th and 95th percentiles
                ax[i].fill_between(time_domain_prediction, 
                                  np.percentile(draws_prediction_orig, 5, axis=0)[i], 
                                  np.percentile(draws_prediction_orig, 95, axis=0)[i], 
                                  color='tab:purple', alpha=0.15)
                
                ax[i].set_ylim(*_ylim_from_truth(self.snapshots_prediction[i]))
                ax[i].set_xlabel('Time')
                ax[i].set_ylabel(f'Mode {i+1}')
                ax[i].legend()
            
            fig.suptitle("Operator Inference Trajectories", fontsize=16)
        else:
            # Original three-column layout
            fig, ax = plt.subplots(self.numPODmodes, 3, figsize=figsize, sharey='row', sharex='col')

            for i in range(self.numPODmodes):
                ax[i, 0].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                ax[i, 1].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                ax[i, 2].plot(self.time_domain_prediction, self.snapshots_prediction[i], color='tab:gray', lw=2)

                # Plot the median (dashed purple)
                ax[i, 0].plot(time_domain_training, np.median(draws_training_orig, axis=0)[i], color='tab:purple', linestyle='--', alpha=0.9, lw=2)
                ax[i, 1].plot(time_domain_prediction, np.median(draws_prediction_orig, axis=0)[i], color='tab:purple', linestyle='--', alpha=0.9, lw=2) 
                ax[i, 2].plot(time_domain_prediction, np.median(draws_prediction_orig, axis=0)[i], color='tab:purple', linestyle='--', alpha=0.9, lw=2)

                # Plot the 5th and 95th percentiles
                ax[i, 0].fill_between(time_domain_training, np.percentile(draws_training_orig, 5, axis=0)[i], np.percentile(draws_training_orig, 95, axis=0)[i], color='tab:purple', alpha=0.15)
                ax[i, 1].fill_between(time_domain_prediction, np.percentile(draws_prediction_orig, 5, axis=0)[i], np.percentile(draws_prediction_orig, 95, axis=0)[i], color='tab:purple', alpha=0.15)
                ax[i, 2].fill_between(time_domain_prediction, np.percentile(draws_prediction_orig, 5, axis=0)[i], np.percentile(draws_prediction_orig, 95, axis=0)[i], color='tab:purple', alpha=0.15)

                ymin, ymax = _ylim_from_truth(self.snapshots_prediction[i])

                for j in range(3):
                    ax[i, j].set_ylim(ymin, ymax)
   
            fig.suptitle("Operator Inference Trajectories", fontsize=16)
        
        fig.tight_layout()
        fig.show()
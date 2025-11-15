import importlib
import sys
sys.path.append("../")
import plotter
importlib.reload(plotter)
from plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from helpers.bgp_jax import BayesianGP
import jax.numpy as jnp

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

def compute_derivatives_fourth_order(snapshots, time_points):
    """
    Compute derivatives using 4th order finite differences.
    Works best for uniformly spaced time points.
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



class FitzPlotter(Plotter):
    def __init__(self, *args, scaler=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaler = scaler
    
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
            ax[i,0].grid()
            ax[i,1].grid()

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
            ax[i].grid()
        
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
                # Plot training data
                ax[i].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                
                # Plot ground truth
                ax[i].plot(self.time_domain_prediction, self.snapshots_prediction[i], 
                          color='tab:gray', lw=2, label='Ground Truth')
                
                # Plot the mean
                # ax[i].plot(self.time_domain_eval_prediction, rom_solves_prediction[:,i,:].T.mean(axis=1), 
                #           alpha=0.8, lw=2, label='Mean')
                
                # Plot the median
                ax[i].plot(self.time_domain_eval_prediction, np.median(rom_solves_prediction[:,i,:], axis=0), 
                          alpha=0.8, linestyle='--', lw=2, label='Median')
                
                # Plot the 5th and 95th percentiles
                ax[i].fill_between(self.time_domain_eval_prediction, 
                                  np.percentile(rom_solves_prediction[:,i,:], 5, axis=0), 
                                  np.percentile(rom_solves_prediction[:,i,:], 95, axis=0), 
                                  alpha=0.2)
                
                # Set y-limits based on ground truth
                yvals = np.asarray(self.snapshots_prediction[i])
                ymin = np.nanmin(yvals)
                ymax = np.nanmax(yvals)
                
                if np.isclose(ymin, ymax):
                    if np.isclose(ymax, 0.0):
                        pad = 1.0
                    else:
                        pad = abs(ymax) * 0.75
                        ymin -= pad
                        ymax += pad
                else:
                    ymin = ymin - abs(ymin) * 0.75
                    ymax = ymax * 1.75
                
                ax[i].set_ylim(float(ymin), float(ymax))
                ax[i].set_xlabel('Time')
                ax[i].set_ylabel(f'Mode {i+1}')
                # ax[i].grid()
                ax[i].legend()
            
            fig.suptitle("Operator Inference Trajectories", fontsize=16)
        else:
            # Original three-column layout
            fig, ax = plt.subplots(self.numPODmodes, 3, figsize=figsize, sharey='row', sharex='col')

            for i in range(self.numPODmodes):
                ax[i, 0].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                ax[i, 1].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
                ax[i, 2].plot(self.time_domain_prediction, self.snapshots_prediction[i], color='tab:gray', lw=2)

                if plot_samples:
                    ax[i, 0].plot(self.time_domain_eval_training, rom_solves_training[:,i,:].T, alpha = .3, lw=2)
                    ax[i, 1].plot(self.time_domain_eval_prediction, rom_solves_prediction[:,i,:].T, alpha = .3, lw=2)
                    ax[i, 2].plot(self.time_domain_eval_prediction, rom_solves_prediction[:,i,:].T, alpha = .3, lw=2)

                # Plot the mean
                ax[i, 0].plot(self.time_domain_eval_training, rom_solves_training[:,i,:].T.mean(axis=1), alpha = .8, lw=2)
                ax[i, 1].plot(self.time_domain_eval_prediction, rom_solves_prediction[:,i,:].T.mean(axis=1), alpha = .8, lw=2)
                ax[i, 2].plot(self.time_domain_eval_prediction, rom_solves_prediction[:,i,:].T.mean(axis=1), alpha = .8, lw=2)

                # Plot the median
                ax[i, 0].plot(self.time_domain_eval_training, np.median(rom_solves_training[:,i,:], axis=0), alpha = .8, linestyle='--', lw=2)
                ax[i, 1].plot(self.time_domain_eval_prediction, np.median(rom_solves_prediction[:,i,:], axis=0), alpha = .8, linestyle='--', lw=2)
                ax[i, 2].plot(self.time_domain_eval_prediction, np.median(rom_solves_prediction[:,i,:], axis=0), alpha = .8, linestyle='--', lw=2)

                # Plot the 5th and 95th percentiles
                ax[i, 0].fill_between(self.time_domain_eval_training, np.percentile(rom_solves_training[:,i,:], 5, axis=0), np.percentile(rom_solves_training[:,i,:], 95, axis=0), alpha=.2)
                ax[i, 1].fill_between(self.time_domain_eval_prediction, np.percentile(rom_solves_prediction[:,i,:], 5, axis=0), np.percentile(rom_solves_prediction[:,i,:], 95, axis=0), alpha=.2)
                ax[i, 2].fill_between(self.time_domain_eval_prediction, np.percentile(rom_solves_prediction[:,i,:], 5, axis=0), np.percentile(rom_solves_prediction[:,i,:], 95, axis=0), alpha=.2)

                yvals = np.asarray(self.snapshots_prediction[i])
                ymin = np.nanmin(yvals)
                ymax = np.nanmax(yvals)

                if np.isclose(ymin, ymax):
                    if np.isclose(ymax, 0.0):
                        pad = 1.0  # arbitrary small window around zero
                    else:
                        pad = abs(ymax) * 0.75
                        ymin -= pad
                        ymax += pad
                else:
                    ymin = ymin - abs(ymin) * 0.75
                    ymax = ymax * 1.75

                ax[i, 0].set_ylim(float(ymin), float(ymax))
                ax[i, 1].set_ylim(float(ymin), float(ymax))
                ax[i, 2].set_ylim(float(ymin), float(ymax))

                ax[i,0].grid()
                ax[i,1].grid()
                ax[i,2].grid()

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
                
                # Plot the median
                ax[i].plot(time_domain_prediction, np.median(draws_prediction_orig, axis=0)[i], 
                          alpha=0.8, linestyle='--', lw=2, label='Median')
                
                # Plot the 5th and 95th percentiles
                ax[i].fill_between(time_domain_prediction, 
                                  np.percentile(draws_prediction_orig, 5, axis=0)[i], 
                                  np.percentile(draws_prediction_orig, 95, axis=0)[i], 
                                  alpha=0.2)
                
                # Set y-limits based on ground truth
                yvals = np.asarray(self.snapshots_prediction[i])
                ymin = np.nanmin(yvals)
                ymax = np.nanmax(yvals)
                
                if np.isclose(ymin, ymax):
                    if np.isclose(ymax, 0.0):
                        pad = 1.0
                    else:
                        pad = abs(ymax) * 0.5
                        ymin -= pad
                        ymax += pad
                else:
                    ymin = ymin - abs(ymin) * 0.5
                    ymax = ymax * 1.5
                
                ax[i].set_ylim(float(ymin), float(ymax))
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

                # Plot the mean
                ax[i, 0].plot(time_domain_training, draws_training_orig.mean(axis=0)[i], alpha=0.8, lw=2)
                ax[i, 1].plot(time_domain_prediction, draws_prediction_orig.mean(axis=0)[i], alpha=0.8, lw=2)
                ax[i, 2].plot(time_domain_prediction, draws_prediction_orig.mean(axis=0)[i], alpha=0.8, lw=2)

                # Plot the median
                ax[i, 0].plot(time_domain_training, np.median(draws_training_orig, axis=0)[i], alpha=0.8, linestyle='--', lw=2)
                ax[i, 1].plot(time_domain_prediction, np.median(draws_prediction_orig, axis=0)[i], alpha=0.8, linestyle='--', lw=2) 
                ax[i, 2].plot(time_domain_prediction, np.median(draws_prediction_orig, axis=0)[i], alpha=0.8, linestyle='--', lw=2)

                # Plot the 5th and 95th percentiles
                ax[i, 0].fill_between(time_domain_training, np.percentile(draws_training_orig, 5, axis=0)[i], np.percentile(draws_training_orig, 95, axis=0)[i], alpha=.2)
                ax[i, 1].fill_between(time_domain_prediction, np.percentile(draws_prediction_orig, 5, axis=0)[i], np.percentile(draws_prediction_orig, 95, axis=0)[i], alpha=.2)
                ax[i, 2].fill_between(time_domain_prediction, np.percentile(draws_prediction_orig, 5, axis=0)[i], np.percentile(draws_prediction_orig, 95, axis=0)[i], alpha=.2)

                yvals = np.asarray(self.snapshots_prediction[i])
                ymin = np.nanmin(yvals)
                ymax = np.nanmax(yvals)

                if np.isclose(ymin, ymax):
                    if np.isclose(ymax, 0.0):
                        pad = 1.0  # arbitrary small window around zero
                    else:
                        pad = abs(ymax) * 0.5
                        ymin -= pad
                        ymax += pad
                else:
                    ymin = ymin - abs(ymin) * 0.5
                    ymax = ymax * 1.5

                ax[i, 0].set_ylim(float(ymin), float(ymax))
                ax[i, 1].set_ylim(float(ymin), float(ymax))
                ax[i, 2].set_ylim(float(ymin), float(ymax))

                ax[i,0].grid()
                ax[i,1].grid()
                ax[i,2].grid()
   
            fig.suptitle("Operator Inference Trajectories", fontsize=16)
        
        fig.tight_layout()
        fig.show()
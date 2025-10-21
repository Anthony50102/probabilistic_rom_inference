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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
            gp.y_train = self.snapshots_training[i]

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
            ax[i,0].plot(self.time_domain_training, means.T, alpha = .3)
            ax[i,1].plot(self.time_domain_eval_training, eval_means.T, alpha = .3)
        
            ax[i,0].fill_between(self.time_domain_training, np.mean(means, axis=0)-2*np.mean(stds, axis=0), np.mean(means, axis=0)+2*np.mean(stds, axis=0), alpha = .3, color = "gray", label = "Mean $\pm$ 2 std")
            ax[i,1].fill_between(self.time_domain_eval_training, np.mean(eval_means, axis=0)-2*np.mean(eval_stds, axis=0), np.mean(eval_means, axis=0)+2*np.mean(eval_stds, axis=0), alpha = .3, color = "gray", label = "Mean $\pm$ 2 std")

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

        K_yys, K_zys, K_zzs = [], [], []
        for i in range(self.numPODmodes):
            ell2 = self.gp_lengthscales[i].mean(axis=0)**2
            
            # Standard RBF kernels
            rbf_yy = rbf_eval(self.gp_lengthscales[i].mean(axis=0), self.gp_variances[i].mean(axis=0), self.time_domain_training, self.time_domain_training)
            rbf_zy = rbf_eval(self.gp_lengthscales[i].mean(axis=0), self.gp_variances[i].mean(axis=0), self.time_domain_eval_training, self.time_domain_training)
            rbf_zz = rbf_eval(self.gp_lengthscales[i].mean(axis=0), self.gp_variances[i].mean(axis=0), self.time_domain_eval_training, self.time_domain_eval_training)

            # K_yy with noise term
            K_yy = rbf_yy + 1e-5 * np.eye(len(self.time_domain_training))  # Fixed: use eye instead of diag

            # K_zy: derivative kernel - note the correct difference computation
            diff_zy = self.time_domain_eval_training[:, None] - self.time_domain_training[None, :]  # (250, 150)
            K_zy = -(diff_zy / ell2) * rbf_zy  # (250, 150)
            
            # K_zz: second derivative kernel
            diff_zz = self.time_domain_eval_training[:, None] - self.time_domain_eval_training[None, :]  # (250, 250)
            K_zz = ((1 - (diff_zz**2 / ell2)) / ell2) * rbf_zz  # (250, 250)
            
            K_yys.append(K_yy)
            K_zys.append(K_zy)
            K_zzs.append(K_zz)

        # Now the prediction should work
        mu_z = []
        cov_z = []
        for i in range(self.numPODmodes):
            w = jnp.linalg.solve(K_yys[i], self.snapshots_training[i])  # w shape: (150,)
            mu_zi = K_zys[i] @ w  # (250, 150) @ (150,) = (250,)
            mu_z.append(mu_zi)

            cov_zi = K_zzs[i] - K_zys[i] @ jnp.linalg.solve(K_yys[i], K_zys[i].T)
            cov_z.append(cov_zi)


        self.snapshots_training_derivatives = compute_derivatives_fourth_order(self.snapshots_training, self.time_domain_training)


        fig, ax = plt.subplots(self.numPODmodes, 1, figsize = figsize, sharex=True)
        for i in range(self.numPODmodes):
            ax[i].plot(self.time_domain_training, self.snapshots_training_derivatives[i], 'k*', label='Training Data')
            if eval:
                ax[i].plot(self.time_domain_eval_training, mu_z[i], label='Predicted Mean')
                ax[i].fill_between(self.time_domain_eval_training, 
                                mu_z[i] - 2 * jnp.sqrt(jnp.diag(cov_z[i])), 
                                mu_z[i] + 2 * jnp.sqrt(jnp.diag(cov_z[i])), 
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
                    figsize: tuple = (12, 8),
                    max_num_samples = 1000,
                    plot_samples: bool = False
                    ):
        plt.clf()

        self.operator_samples =  operator_samples if isinstance(operator_samples, np.ndarray) else np.array(operator_samples)
        self.latent_state_samples = np.transpose(latent_state_samples, (1,0,2)) if isinstance(latent_state_samples, np.ndarray) else np.transpose(np.array(latent_state_samples), (1,0,2))

        print(self.operator_samples.shape, self.latent_state_samples.shape)
        samples = min(self.operator_samples.shape[0], self.latent_state_samples.shape[0], max_num_samples)

        fig, ax = plt.subplots(self.numPODmodes, 3, figsize=figsize, sharey='row', sharex='col')

        rom_solves_training, rom_solves_prediction = [], []
        for i in range(samples):
            operator = self.operator_samples[i]
            rom.model._extract_operators(operator)
            # TODO: Can't cheat like this with starting value
            rom.model.predict(state0=q0, t=self.time_domain_eval_training)
            if rom.model.predict_result_.y.shape[1] < self.time_domain_eval_training.size:
                print("Bad solve within training domain, skipping", rom.model.predict_result_.y.shape)
                continue
            rom_solves_training.append(rom.model.predict_result_.y)


            rom.model.predict(state0=q0, t=self.time_domain_eval_prediction)
            if rom.model.predict_result_.y.shape[1] < self.time_domain_eval_prediction.size:
                print("Bad solve within prediction domain, skipping", rom.model.predict_result_.y.shape)
                continue
            rom_solves_prediction.append(rom.model.predict_result_.y)

        rom_solves_training, rom_solves_prediction = np.array(rom_solves_training), np.array(rom_solves_prediction)
        print(rom_solves_training.shape, rom_solves_prediction.shape)

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
        fig.show()
    
    def operator_plot_trajectories(
                    self,
                    draws_training: List | np.ndarray,
                    draws_prediction: List | np.ndarray,
                    time_domain_training = None,
                    time_domain_prediction = None,
                    figsize: tuple = (12, 8),
    ):
        plt.clf()

        if time_domain_training is None:
            time_domain_training = self.time_domain_eval_training

        if time_domain_prediction is None:
            time_domain_prediction = self.time_domain_eval_prediction

        fig, ax = plt.subplots(self.numPODmodes, 3, figsize=figsize, sharey='row', sharex='col')

        if isinstance(draws_training, list):
            draws_training = np.array(draws_training)
        if isinstance(draws_prediction, list):
            draws_prediction = np.array(draws_prediction)

        for i in range(self.numPODmodes):
            ax[i, 0].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
            ax[i, 1].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
            ax[i, 2].plot(self.time_domain_prediction, self.snapshots_prediction[i], color='tab:gray', lw=2)

            # Plot the mean
            ax[i, 0].plot(time_domain_training, draws_training.mean(axis=0)[i], alpha=0.8, lw=2)
            ax[i, 1].plot(time_domain_prediction, draws_prediction.mean(axis=0)[i], alpha=0.8, lw=2)
            ax[i, 2].plot(time_domain_prediction, draws_prediction.mean(axis=0)[i], alpha=0.8, lw=2)

            # Plot the median
            ax[i, 0].plot(time_domain_training, np.median(draws_training, axis=0)[i], alpha=0.8, linestyle='--', lw=2)
            ax[i, 1].plot(time_domain_prediction, np.median(draws_prediction, axis=0)[i], alpha=0.8, linestyle='--', lw=2) 
            ax[i, 2].plot(time_domain_prediction, np.median(draws_prediction, axis=0)[i], alpha=0.8, linestyle='--', lw=2)

            # Plot the 5th and 95th percentiles
            ax[i, 0].fill_between(time_domain_training, np.percentile(draws_training, 5, axis=0)[i], np.percentile(draws_training, 95, axis=0)[i], alpha=.2)
            ax[i, 1].fill_between(time_domain_prediction, np.percentile(draws_prediction, 5, axis=0)[i], np.percentile(draws_prediction, 95, axis=0)[i], alpha=.2)
            ax[i, 2].fill_between(time_domain_prediction, np.percentile(draws_prediction, 5, axis=0)[i], np.percentile(draws_prediction, 95, axis=0)[i], alpha=.2)

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
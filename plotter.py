import matplotlib.pyplot as plt
from helpers.bgp_jax import BayesianGP
import numpy as np
from typing import List
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


class Plotter:
    def __init__(self, 
                 numPODmodes: int,
                 time_domain_training: np.ndarray, 
                 time_domain_prediction: np.ndarray, 
                 time_domain_eval_training: np.ndarray,
                 time_domain_eval_prediction: np.ndarray,
                 snapshots_training: np.ndarray, 
                 snapshots_prediction: np.ndarray,
                 ) -> None:
        '''
        Initializes the Plotter class with the given parameters.
        numPODmodes: number of POD modes
        time_domain_training: time domain of the training snapshots, shape = (t_train,)
        time_domain_prediction: time domain of the full snapshots, shape = (t_pred,)
        time_domain_eval_training: time domain for evaluating GP in training domain, shape = (t_eval_train,)
        time_domain_eval_prediction: time domain for evaluating ROM in prediction domain, shape = (t_eval_pred,)
        snapshots_training: training snapshots, shape = (numPODmodes, t_train)
        snapshots_prediction: full snapshots, shape = (numPODmodes, t_pred)
        '''

        self.numPODmodes = numPODmodes
        self.time_domain_training = time_domain_training
        self.time_domain_prediction = time_domain_prediction
        self.time_domain_eval_training = time_domain_eval_training
        self.time_domain_eval_prediction = time_domain_eval_prediction
        self.snapshots_training = snapshots_training
        self.snapshots_prediction = snapshots_prediction

        self.lengthscales = None
        self.variances = None
        self.noises = None

    def gp_plot_state(self,
                lengthscales: np.ndarray | List,
                variances: np.ndarray | List,
                noises: np.ndarray | List,
                double: bool = True,
                figsize=(12,8),
                max_num_samples: int = 1000
                ):

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
        plt.tight_layout()
        plt.show()
    
    def gp_plot_derivatives(
                self,
                eval: bool = True,
                lengthscales: np.ndarray | List = None,
                variances: np.ndarray | List = None,
                noises: np.ndarray | List = None,
                figsize=(12,8),
                max_num_samples: int = 1000
                ):
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
            print(K_yys[i].shape, self.snapshots_training[i].shape)
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
        
        plt.tight_layout()
        plt.show()

    def operator_plot(
                    self,
                    operator_samples: np.ndarray | List,
                    latent_state_samples: np.ndarray | List,
                    rom,
                    figsize: tuple = (12, 8),
                    max_num_samples = 1000,
                    plot_samples: bool = False
                    ):

        self.operator_samples =  operator_samples if isinstance(operator_samples, np.ndarray) else np.array(operator_samples)
        self.latent_state_samples = np.transpose(latent_state_samples, (1,0,2)) if isinstance(latent_state_samples, np.ndarray) else np.transpose(np.array(latent_state_samples), (1,0,2))

        print(self.operator_samples.shape, self.latent_state_samples.shape)
        samples = min(self.operator_samples.shape[0], self.latent_state_samples.shape[0], max_num_samples)

        fig, ax = plt.subplots(self.numPODmodes, 3, figsize=figsize)

        # for i in range(self.numPODmodes):
        #     ax[i, 0].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
        #     ax[i, 1].plot(self.time_domain_training, self.snapshots_training[i], 'k*')
        #     ax[i, 2].plot(self.time_domain_prediction, self.snapshots_prediction[i], 'k*')

        #     rom_solves_training, rom_solves_prediction = [], []
        #     for j in range(samples):
        #         operator = self.operator_samples[j]
        #         rom.model._extract_operators(operator)
        #         # TODO: Can't cheat like this with starting value
        #         rom.model.predict(state0=self.snapshots_training[:, 0], t=self.time_domain_eval_training)
        #         if rom.model.predict_result_.y.shape[1] < self.time_domain_eval_training.size:
        #             print("Bad solve within training domain, skipping", rom.model.predict_result_.y.shape)
        #             continue
        #         rom_solves_training.append(rom.model.predict_result_.y)


        #         rom.model.predict(state0=self.snapshots_prediction[:, 0], t=self.time_domain_eval_prediction)
        #         if rom.model.predict_result_.y.shape[1] < self.time_domain_eval_prediction.size:
        #             print("Bad solve within prediction domain, skipping", rom.model.predict_result_.y.shape)
        #             continue
        #         rom_solves_prediction.append(rom.model.predict_result_.y)

        rom_solves_training, rom_solves_prediction = [], []
        for i in range(samples):
            operator = self.operator_samples[i]
            rom.model._extract_operators(operator)
            # TODO: Can't cheat like this with starting value
            rom.model.predict(state0=self.snapshots_training[:, 0], t=self.time_domain_eval_training)
            if rom.model.predict_result_.y.shape[1] < self.time_domain_eval_training.size:
                print("Bad solve within training domain, skipping", rom.model.predict_result_.y.shape)
                continue
            rom_solves_training.append(rom.model.predict_result_.y)


            rom.model.predict(state0=self.snapshots_prediction[:, 0], t=self.time_domain_eval_prediction)
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

            ax[i,0].grid()
            ax[i,1].grid()
            ax[i,2].grid()

        plt.tight_layout()
        plt.show()

    # def trajectory_plot(self, 
    #                     time_snaps: np.ndarray, 
    #                     time: np.ndarray, 
    #                     snapshots: np.ndarray, 
    #                     samples: np.ndarray, 
    #                     figsize: tuple = (12, 8), 
    #                     title: str = "Trajectory",
    #                     xtitle: str = "t",
    #                     ytitle: str | None = None,
    #                     title_y: float = 0.95,  # Reduced default value
    #                     title_x: float = 0.5,   
    #                     shade: float = 0,
    #                     grid: bool = False,
    #                     plot_all_samples: bool = True,  # New parameter
    #                     confidence_level: float = 0.95,  # New parameter for CI
    #                     truth_time: np.ndarray | None = None,  # Ground truth time axis
    #                     truth_data: np.ndarray | None = None,  # Ground truth data
    #                     plot_training_data: bool = True,  # Whether to plot observed data
    #                     ):
    #     '''
    #     time_snaps: the time axis of the observed data shape = (r, t)
    #     time: the time axis of the results shape = (r, t1)
    #     snapshots: the observed data compressed or not, shape = (r, t)
    #     samples: the results of many draws from a model, shape = (n, r, t1)
    #     title_x: horizontal position of title (0=left, 0.5=center, 1=right)
    #     title_y: vertical position of title (0=bottom, 1=top)
    #     plot_all_samples: if True, plot all samples; if False, plot mean with confidence intervals
    #     confidence_level: confidence level for intervals (e.g., 0.95 for 95% CI)
    #     truth_time: time axis for ground truth data, shape = (r, t2) or (t2,)
    #     truth_data: ground truth data, shape = (r, t2)
    #     plot_training_data: if True, plot the training/observed data
        
    #     Plotting combinations:
    #     - Training domain: training data + predictions (truth_time=None, truth_data=None)
    #     - Training domain with truth: training data + predictions + truncated truth
    #     - Test domain: training data + predictions + truth (full truth domain)
    #     - Test domain without training: predictions + truth (plot_training_data=False)
    #     '''
    #     modes = samples.shape[1]
    #     ndraws = samples.shape[0]
    #     print(f"Modes: {modes}, Number of draws: {ndraws}")
        
    #     # Determine if we have ground truth data
    #     has_truth = truth_time is not None and truth_data is not None
        
    #     fig, ax = plt.subplots(modes, 1, figsize=figsize, sharex=True)
        
    #     if modes == 1:
    #         ax = [ax]
        
    #     # Set x-axis label on the bottom subplot only (due to sharex=True)
    #     ax[-1].set_xlabel(xtitle)
        
    #     # Set y-axis label if provided
    #     if ytitle is not None:
    #         # Set ylabel on the middle subplot for better positioning
    #         middle_idx = modes // 2
    #         ax[middle_idx].set_ylabel(ytitle)
        
    #     # Plot ground truth data first (so it appears behind other plots)
    #     if has_truth:
    #         for i in range(modes):
    #             if len(truth_time.shape) == 1:  # 1d case
    #                 ax[i].plot(truth_time, truth_data[i], color='gray', linewidth=2,
    #                           label='Ground truth' if i == 0 else "", alpha=0.8)
    #             else:  # Handle multi-dimensional time if needed
    #                 ax[i].plot(truth_time[i], truth_data[i], color='gray', linewidth=2,
    #                           label='Ground truth' if i == 0 else "", alpha=0.8)
        
    #     # Plot samples based on the chosen method
    #     if plot_all_samples:
    #         # Plot all sample trajectories (semi-transparent lines)
    #         for i in range(ndraws):
    #             for j in range(modes):
    #                 if len(time.shape) == 1:  # 1d case
    #                     ax[j].plot(time, samples[i, j], alpha=0.2, color='tab:blue',
    #                               label='Model samples' if i == 0 and j == 0 else "")
    #                 else:  # Handle multi-dimensional time if needed
    #                     ax[j].plot(time[j], samples[i, j], alpha=0.2, color='tab:blue',
    #                               label='Model samples' if i == 0 and j == 0 else "")
    #     else:
    #         # Plot mean with confidence intervals
    #         alpha = 1 - confidence_level
    #         lower_percentile = (alpha / 2) * 100
    #         upper_percentile = (1 - alpha / 2) * 100
            
    #         for j in range(modes):
    #             # Calculate mean and percentiles across samples
    #             mean_trajectory = np.mean(samples[:, j, :], axis=0)
    #             lower_bound = np.percentile(samples[:, j, :], lower_percentile, axis=0)
    #             upper_bound = np.percentile(samples[:, j, :], upper_percentile, axis=0)
                
    #             if len(time.shape) == 1:  # 1d case
    #                 time_axis = time
    #             else:  # Handle multi-dimensional time if needed
    #                 time_axis = time[j]
                
    #             # Plot mean line
    #             ax[j].plot(time_axis, mean_trajectory, color='tab:blue', linewidth=2,
    #                       label='Mean trajectory' if j == 0 else "")
                
    #             # Plot confidence interval as filled area
    #             ax[j].fill_between(time_axis, lower_bound, upper_bound, 
    #                               color='tab:blue', alpha=0.3,
    #                               label=f'{confidence_level*100:.0f}% CI' if j == 0 else "")
        
    #     # Plot the observed/training data (black stars)
    #     if plot_training_data:
    #         for i in range(modes):
    #             if len(time_snaps.shape) == 1:  # 1d case
    #                 ax[i].plot(time_snaps, snapshots[i], 'k*', markersize=8,
    #                           label='Training data' if i == 0 else "")
    #             else:  # Handle multi-dimensional time_snaps if needed
    #                 ax[i].plot(time_snaps[i], snapshots[i], 'k*', markersize=8,
    #                           label='Training data' if i == 0 else "")
        
    #     # Add legend to the first subplot
    #     ax[0].legend(loc='upper right')
        
    #     # Add grid and styling
    #     for i in range(modes):
    #         if grid:
    #             ax[i].grid(True, alpha=0.3)
    #         if shade != 0:
    #             ax[i].axvspan(0, shade, alpha=0.1, color="gray")
            
    #         # Set x-axis limits based on available data
    #         all_times = []
            
    #         # Always include prediction time domain
    #         if len(time.shape) == 1:
    #             all_times.extend([min(time), max(time)])
    #         else:
    #             all_times.extend([min(time[i]), max(time[i])])
            
    #         # Include training data time domain if being plotted
    #         if plot_training_data:
    #             if len(time_snaps.shape) == 1:
    #                 all_times.extend([min(time_snaps), max(time_snaps)])
    #             else:
    #                 all_times.extend([min(time_snaps[i]), max(time_snaps[i])])
            
    #         # Include ground truth time domain if available (this extends to test domain)
    #         if has_truth:
    #             if len(truth_time.shape) == 1:
    #                 all_times.extend([min(truth_time), max(truth_time)])
    #             else:
    #                 all_times.extend([min(truth_time[i]), max(truth_time[i])])
            
    #         ax[i].set_xlim(min(all_times), max(all_times))
    #         # ax[i].set_title(f'Mode {i+1}', fontsize=10)
        
    #     # First do tight_layout to get proper spacing
    #     plt.tight_layout()
        
    #     # Then add the title with proper positioning
    #     fig.suptitle(title, x=title_x, y=title_y, fontsize=14)
        
    #     # Adjust the top margin to accommodate the main title
    #     # Use a more conservative adjustment based on title_y
    #     top_margin = title_y - 0.03
    #     plt.subplots_adjust(top=top_margin)
        
    #     plt.show()
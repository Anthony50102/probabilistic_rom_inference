import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib.pyplot as plt
import numpy as np
from typing import List

from core import BayesianGP
from core.plotting import Plotter


class HeatPlotter(Plotter):
    """Plotter for Heat equation experiments."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_initial_conditions = self.snapshots_training.shape[1]
        self.snapshots_prediction_parameters = None

    def gp_plot_state(self,
                      samples,
                      figsize=(20,12),
                      ):
        '''
        Plot the GP state estimates in the training domain.
        Rows per intial condition
        '''
        print(self.num_initial_conditions)
        fig, ax = plt.subplots(self.num_initial_conditions, self.numPODmodes, figsize=figsize, sharex='col')

        gp = BayesianGP()
        for j in range(self.num_initial_conditions):
            gp.X_train = self.time_domain_training[j,][:,None]

            Ls = np.array([samples[f'lengthscale{i}{j}'].mean() for i in range(self.numPODmodes)]) # i is POD mode j is intitial condition
            Vs = np.array([samples[f'variance{i}{j}'].mean() for i in range(self.numPODmodes)])
            Ns = np.array([samples[f'noise{i}{j}'].mean() for i in range(self.numPODmodes)])
            # TODO: fix this to take actual samples
            for k in range(self.numPODmodes):
                gp.y_train = self.snapshots_training[j][k]
                mean, std, _ = gp.predict_with_hypers(X_test=self.time_domain_eval_training[:,None], lengthscale=Ls[k], variance=Vs[k], noise=Ns[k])
                ax[j][k].plot(self.time_domain_eval_training, mean)
                ax[j][k].fill_between(self.time_domain_eval_training, mean-2*std, mean+2*std, alpha=0.3)
                ax[j][k].plot(self.time_domain_training[j], self.snapshots_training[j][k], 'k*')
    
    def gp_plot_states_hyperparams(self,
                                   lengthscales, # Shape (samples, num_initial_conditions, numPODmodes)
                                   variances,    # Shape (samples, num_initial_conditions, numPODmodes)
                                   noises,       # Shape (samples, num_initial_conditions, numPODmodes)
                                   figsize=(20,12),
                                ):
        
        samples = min(lengthscales.shape[0], variances.shape[0], noises.shape[0])

        fig, ax = plt.subplots(self.num_initial_conditions, self.numPODmodes, figsize=figsize, sharex='col')

        lengthscale_mean = lengthscales.mean(axis=0) # i is POD mode j is intitial condition
        variance_mean = variances.mean(axis=0)
        noise_mean = noises.mean(axis=0)

        means = np.zeros((samples, self.num_initial_conditions, self.numPODmodes, self.time_domain_eval_training.shape[0]))

        gp = BayesianGP()
        for i in range(samples):
            for j in range(self.num_initial_conditions):
                gp.X_train = self.time_domain_training[j,][:,None]
                # TODO: fix this to take actual samples
                for k in range(self.numPODmodes):
                    Ls = lengthscales[i][j][k]
                    Vs = variances[i][j][k]
                    Ns = noises[i][j][k]
                    gp.y_train = self.snapshots_training[j][k]
                    mean, std, _ = gp.predict_with_hypers(X_test=self.time_domain_eval_training[:,None], lengthscale=Ls, variance=Vs, noise=Ns)
                    means[i][j][k] = mean
                    ax[j][k].plot(self.time_domain_eval_training, mean)
        
        # compute the mean and std over the samples
        means_mean = means.mean(axis=0)
        means_std = means.std(axis=0)

        for i in range(self.num_initial_conditions):
            for j in range(self.numPODmodes):
                ax[i][j].plot(self.time_domain_eval_training, means_mean[i][j], color='tab:orange', lw=2)
                ax[i][j].fill_between(self.time_domain_eval_training, 
                                     means_mean[i][j]-2*means_std[i][j], 
                                     means_mean[i][j]+2*means_std[i][j], 
                                     color='tab:orange', alpha=0.3)
                ax[i][j].plot(self.time_domain_training[i], self.snapshots_training[i][j], 'k*')
        
        fig.show()

    
    def gp_plot_derivatives(self,
                            figsize=(20,12),
                            ):
          '''
          Plot the GP derivative estimates in the training domain.
          Rows per intial condition
          '''
          fig, ax = plt.subplots(self.num_initial_conditions, self.numPODmodes, figsize=figsize, sharex='col')
    
          gp = BayesianGP()
          for j in range(self.num_initial_conditions):
                gp.X_train = self.time_domain_training[j,][:,None]
    
                Ls = np.array([self.lengthscales[i][j] for i in range(self.numPODmodes)]) # i is POD mode j is intitial condition
                Vs = np.array([self.variances[i][j] for i in range(self.numPODmodes)])
                Ns = np.array([self.noises[i][j] for i in range(self.numPODmodes)])
    
    def operator_plot(self,
                  q0: np.ndarray | List,
                  operator_samples: np.ndarray | List,
                  latent_state_samples: np.ndarray | List,
                  snapshots_training_prediction_parameters: np.ndarray | List,
                  time_domain_training_prediction_parameters: np.ndarray | List,
                  snapshots_prediction_new_initial: np.ndarray | List,
                  rom,
                  input_func,
                  input_parameters_training: np.ndarray | List,
                  input_parameters_prediction: np.ndarray | List,
                  figsize=(20,12),
                  max_num_samples = 100,
                  plot_samples = False,
                  plot_single = True,
                  ):
        plt.clf()

        fig, ax = plt.subplots(self.num_initial_conditions + 1, self.numPODmodes, figsize=figsize, sharex='col', sharey='col') 

        rom_solves_training, rom_solves_prediction = [[] for _ in range(self.num_initial_conditions + 1)], [[] for _ in range(self.num_initial_conditions + 1)]

        for i in range(self.num_initial_conditions + 1):
            for j in range(min(max_num_samples, operator_samples.shape[0])):

                    O = operator_samples[j]
                    rom._extract_operators(np.array(O))

                    if i == self.num_initial_conditions:
                        if self.snapshots_prediction_parameters is None:
                            self.snapshots_prediction_parameters = snapshots_training_prediction_parameters
                        rom.predict(state0=self.snapshots_prediction_parameters[:, 0], t=self.time_domain_eval_training, input_func=input_func(input_parameters_prediction))

                    else:
                        rom.predict(state0=self.snapshots_training[i, :, 0], t=self.time_domain_eval_training, input_func=input_func(input_parameters_training[i]))
                    if rom.predict_result_.y.shape[1] < self.time_domain_eval_training.size:
                        print("Bad solve, skipping", rom.predict_result_.y.shape)
                        continue
                    rom_solves_training[i].append(rom.predict_result_.y)

                    if i == self.num_initial_conditions:
                        rom.predict(state0=self.snapshots_prediction_parameters[:, 0], t=self.time_domain_eval_prediction, input_func=input_func(input_parameters_prediction))
                    else:
                        rom.predict(state0=self.snapshots_training[i, :, 0], t=self.time_domain_eval_prediction, input_func=input_func(input_parameters_training[i]))
                    
                    if rom.predict_result_.y.shape[1] < self.time_domain_eval_prediction.size:
                        print("Bad solve, skipping", rom.predict_result_.y.shape)
                        continue
                    
                    rom_solves_prediction[i].append(rom.predict_result_.y)
            
            rom_solves_training[i] = np.array(rom_solves_training[i])
            print(rom_solves_training[i].shape)
            rom_solves_prediction[i] = np.array(rom_solves_prediction[i]) 
            print(rom_solves_prediction[i].shape)
        
        # Find minimum number of samples across all initial conditions
        min_samples_training = min(arr.shape[0] for arr in rom_solves_training)
        min_samples_prediction = min(arr.shape[0] for arr in rom_solves_prediction)
        
        # Truncate all arrays to the minimum sample count
        rom_solves_training = [arr[:min_samples_training] for arr in rom_solves_training]
        rom_solves_prediction = [arr[:min_samples_prediction] for arr in rom_solves_prediction]
        
        # Convert to numpy arrays and permute dimensions correctly
        rom_solves_training = np.permute_dims(np.array(rom_solves_training), (1,0,2,3)) # (samples, initial conditions, POD modes, time)
        rom_solves_prediction = np.permute_dims(np.array(rom_solves_prediction), (1,0,2,3)) # (samples, initial conditions, POD modes, time)
        print(np.array(rom_solves_training).shape, np.array(rom_solves_prediction).shape)

        # Calculate statistics over the sample dimension (axis=0)
        rom_solves_training_mean = rom_solves_training.mean(axis=0)  # (initial conditions, POD modes, time)
        rom_solves_training_median = np.median(rom_solves_training, axis=0)
        rom_solves_training_95 = np.percentile(rom_solves_training, 95, axis=0)
        rom_solves_training_5 = np.percentile(rom_solves_training, 5, axis=0)

        # Plot the within training domain
        for i in range(self.num_initial_conditions + 1):
            for j in range(self.numPODmodes):
                # Plot the snapshots (truth data)
                if i < self.num_initial_conditions:
                    ax[i,j].plot(self.time_domain_training[i], self.snapshots_training[i][j], 'k*', label='Truth', alpha=0.5)
                else:
                    ax[i,j].plot(time_domain_training_prediction_parameters, self.snapshots_prediction_parameters[j], 'k*', label='Truth', alpha=0.5)

                # Plot the predictions means and stds
                ax[i,j].plot(self.time_domain_eval_training, rom_solves_training_mean[i, j], '--', color='tab:orange', alpha=0.8, lw=2, label='Mean')
                ax[i,j].plot(self.time_domain_eval_training, rom_solves_training_median[i, j], '-', color='tab:blue', alpha=0.8, lw=2, label='Median')
                ax[i,j].fill_between(self.time_domain_eval_training, 
                                    rom_solves_training_5[i, j], 
                                    rom_solves_training_95[i, j], 
                                    color='tab:blue', alpha=0.3, label='95% CI')

                ax[i,j].grid()
        
        if not plot_single:
            fig.show()

        # Create a new plot for the out-of-sample predictions
        fig, ax = plt.subplots(self.num_initial_conditions + 1, self.numPODmodes, figsize=figsize, sharex='col', sharey='col')

        # Calculate statistics over the sample dimension (axis=0)
        rom_solves_prediction_mean = rom_solves_prediction.mean(axis=0)  # (initial conditions, POD modes, time)
        rom_solves_prediction_median = np.median(rom_solves_prediction, axis=0)
        rom_solves_prediction_95 = np.percentile(rom_solves_prediction, 95, axis=0)
        rom_solves_prediction_5 = np.percentile(rom_solves_prediction, 5, axis=0)

        for i in range(self.num_initial_conditions + 1):
            for j in range(self.numPODmodes):
                # Plot the snapshots (truth data) and the truth data
                if i < self.num_initial_conditions:
                    ax[i,j].plot(self.time_domain_training[i], self.snapshots_training[i][j], 'k*', label='Truth', alpha=0.5)
                else:
                    ax[i,j].plot(time_domain_training_prediction_parameters, self.snapshots_prediction_parameters[j], 'k*', label='Truth', alpha=0.5)
                    # ax[i,j].plot(self.time_domain_prediction, snapshots_prediction_new_initial[j], color="tab:gray", label='Truth', alpha=0.5)

                # Plot the predictions means and stds
                ax[i,j].plot(self.time_domain_eval_prediction, rom_solves_prediction_mean[i, j], '--', color='tab:orange', alpha=0.8, lw=2, label='Mean')
                ax[i,j].plot(self.time_domain_eval_prediction, rom_solves_prediction_median[i, j], '-', color='tab:blue', alpha=0.8, lw=2, label='Median')
                ax[i,j].fill_between(self.time_domain_eval_prediction, 
                                    rom_solves_prediction_5[i, j], 
                                    rom_solves_prediction_95[i, j], 
                                    color='tab:blue', alpha=0.3, label='95% CI')
                
                ax[i,j].grid()
                ax[i,j].axvspan(self.time_domain_eval_training[0], self.time_domain_eval_training[-1], color='tab:blue', alpha=0.15)

        fig.show()

        # Create a new plot for the out-of-sample predictions
        fig, ax = plt.subplots(self.num_initial_conditions + 1, self.numPODmodes, figsize=figsize, sharex='col', sharey='col')

        for i in range(self.num_initial_conditions + 1):
            for j in range(self.numPODmodes):
                # Plot the snapshots (truth data) and the truth data
                if i < self.num_initial_conditions:
                    ax[i,j].plot(self.time_domain_prediction, self.snapshots_prediction[i][j], color="tab:gray", label='Truth', alpha=0.5)
                else:
                    ax[i,j].plot(self.time_domain_prediction, snapshots_prediction_new_initial[j], color="tab:gray", label='Truth', alpha=0.5)

                # Plot the predictions means and stds
                ax[i,j].plot(self.time_domain_eval_prediction, rom_solves_prediction_mean[i, j], '--', color='tab:orange', alpha=0.8, lw=2, label='Mean')
                ax[i,j].plot(self.time_domain_eval_prediction, rom_solves_prediction_median[i, j], '-', color='tab:blue', alpha=0.8, lw=2, label='Median')
                ax[i,j].fill_between(self.time_domain_eval_prediction, 
                                    rom_solves_prediction_5[i, j], 
                                    rom_solves_prediction_95[i, j], 
                                    color='tab:blue', alpha=0.3, label='95% CI')

                ax[i,j].grid()
                ax[i,j].axvspan(self.time_domain_eval_training[0], self.time_domain_eval_training[-1], color='tab:blue', alpha=0.15)

        if not plot_single:
            fig.show()
        
        plt.clf()
    
    def operator_plot_trajectories(
            self,
            snapshots_training_new_initial,
            time_domain_training_new_initial,
            draws_training,
            draws_prediction,
            time_domain_prediction,
            time_domain_training_prediction_parameters,
            true_states_compressed,
            figsize=(20,12),
            max_num_samples = 100,
            plot_samples = False,
            plot_single = True
            ):
        
        plt.clf()

        fig, ax = plt.subplots(self.num_initial_conditions+1, self.numPODmodes, figsize=figsize, sharex='col', sharey='col')

        for i in range(self.num_initial_conditions + 1):
            for j in range(self.numPODmodes):
                # Plot the snapshots (truth data)
                if i < self.num_initial_conditions:
                    ax[i,j].plot(self.time_domain_training[i], self.snapshots_training[i][j], 'k*', label='Truth', alpha=0.5)
                else:
                    ax[i,j].plot(time_domain_training_new_initial, snapshots_training_new_initial[j], 'k*', label='Truth', alpha=0.5)

                # Plot the predictions means and stds
                if plot_samples:
                    for k in range(min(max_num_samples, draws_training.shape[0])):
                        ax[i,j].plot(self.time_domain_eval_training, draws_training[k,i,j,:], color='tab:blue', alpha=0.1)

                ax[i,j].plot(self.time_domain_eval_training, draws_training.mean(axis=0)[i,j,:], '--', color='tab:orange', alpha=0.8, lw=2, label='Mean')
                ax[i,j].plot(self.time_domain_eval_training, np.median(draws_training, axis=0)[i,j,:], '-', color='tab:blue', alpha=0.8, lw=2, label='Median')
                ax[i,j].fill_between(self.time_domain_eval_training, 
                                    np.percentile(draws_training, 5, axis=0)[i,j,:], 
                                    np.percentile(draws_training, 95, axis=0)[i,j,:], 
                                    color='tab:blue', alpha=0.3, label='95% CI')

                ax[i,j].grid()

        if not plot_single:
            fig.show()

        # Create a new plot for the out-of-sample predictions
        fig, ax = plt.subplots(self.num_initial_conditions + 1, self.numPODmodes, figsize=figsize, sharex='col', sharey='col')

        for i in range(self.num_initial_conditions + 1):
            for j in range(self.numPODmodes):
                # Plot the snapshots (truth data)
                if i < self.num_initial_conditions:
                    ax[i,j].plot(self.time_domain_training[i], self.snapshots_training[i][j], 'k*', label='Truth', alpha=0.5)
                else:
                    ax[i,j].plot(time_domain_training_new_initial, snapshots_training_new_initial[j], 'k*', label='Truth', alpha=0.5)

                # Plot the predictions means and stds
                if plot_samples:
                    for k in range(min(max_num_samples, draws_prediction.shape[0])):
                        ax[i,j].plot(self.time_domain_eval_prediction, draws_prediction[k,i,j,:], color='tab:blue', alpha=0.1)

                ax[i,j].plot(self.time_domain_eval_prediction, draws_prediction.mean(axis=0)[i,j,:], '--', color='tab:orange', alpha=0.8, lw=2, label='Mean')
                ax[i,j].plot(self.time_domain_eval_prediction, np.median(draws_prediction, axis=0)[i,j,:], '-', color='tab:blue', alpha=0.8, lw=2, label='Median')
                ax[i,j].fill_between(self.time_domain_eval_prediction, 
                                    np.percentile(draws_prediction, 5, axis=0)[i,j,:], 
                                    np.percentile(draws_prediction, 95, axis=0)[i,j,:], 
                                    color='tab:blue', alpha=0.3, label='95% CI')

                ax[i,j].grid()
                ax[i,j].axvspan(self.time_domain_eval_training[0], self.time_domain_eval_training[-1], color='tab:blue', alpha=0.15)

        fig.show()

        fig, ax = plt.subplots(self.num_initial_conditions + 1, self.numPODmodes, figsize=figsize, sharex='col', sharey='col')

        for i in range(self.num_initial_conditions + 1):
            for j in range(self.numPODmodes):
                # Plot the snapshots (truth data)
                ax[i,j].plot(self.time_domain_prediction, true_states_compressed[i,j,:], color="tab:gray", label='Truth', alpha=0.5)

                # Plot the predictions means and stds
                if plot_samples:
                    for k in range(min(max_num_samples, draws_prediction.shape[0])):
                        ax[i,j].plot(self.time_domain_eval_prediction, draws_prediction[k,i,j,:], color='tab:blue', alpha=0.1)

                ax[i,j].plot(self.time_domain_eval_prediction, draws_prediction.mean(axis=0)[i,j,:], '--', color='tab:orange', alpha=0.8, lw=2, label='Mean')
                ax[i,j].plot(self.time_domain_eval_prediction, np.median(draws_prediction, axis=0)[i,j,:], '-', color='tab:blue', alpha=0.8, lw=2, label='Median')
                ax[i,j].fill_between(self.time_domain_eval_prediction, 
                                    np.percentile(draws_prediction, 5, axis=0)[i,j,:], 
                                    np.percentile(draws_prediction, 95, axis=0)[i,j,:], 
                                    color='tab:blue', alpha=0.3, label='95% CI')

                ax[i,j].grid()
                ax[i,j].axvspan(self.time_domain_eval_training[0], self.time_domain_eval_training[-1], color='tab:blue', alpha=0.15)
        
        if not plot_single:
            fig.show()
        
        plt.clf()
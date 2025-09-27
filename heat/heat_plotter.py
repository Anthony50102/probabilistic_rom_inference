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

class HeatPlotter(Plotter):
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
                  plot_samples = False
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

                # Plot the predictions means and stds - now using pre-calculated statistics
                ax[i,j].plot(self.time_domain_eval_training, rom_solves_training_mean[i, j], label='GP Mean', alpha=0.5)
                ax[i,j].plot(self.time_domain_eval_training, rom_solves_training_median[i, j], label='GP Median', alpha=0.5)
                ax[i,j].fill_between(self.time_domain_eval_training, 
                                    rom_solves_training_5[i, j], 
                                    rom_solves_training_95[i, j], 
                                    alpha=0.3, label='95% CI')
        
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
                    ax[i,j].plot(self.time_domain_prediction, self.snapshots_prediction[i][j], color="tab:gray", label='Truth', alpha=0.5)
                else:
                    ax[i,j].plot(time_domain_training_prediction_parameters, self.snapshots_prediction_parameters[j], 'k*', label='Truth', alpha=0.5)
                    ax[i,j].plot(self.time_domain_prediction, snapshots_prediction_new_initial[j], color="tab:gray", label='Truth', alpha=0.5)

                # Plot the predictions means and stds - now using pre-calculated statistics
                ax[i,j].plot(self.time_domain_eval_prediction, rom_solves_prediction_mean[i, j], label='GP Mean', alpha=0.5)
                ax[i,j].plot(self.time_domain_eval_prediction, rom_solves_prediction_median[i, j], label='GP Median', alpha=0.5)
                ax[i,j].fill_between(self.time_domain_eval_prediction, 
                                    rom_solves_prediction_5[i, j], 
                                    rom_solves_prediction_95[i, j], 
                                    alpha=0.3, label='95% CI')
                
        
        fig.show()
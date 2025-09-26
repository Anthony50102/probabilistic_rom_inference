import importlib
import sys
sys.path.append("../")
import plotter
importlib.reload(plotter)
from plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np

from helpers.bgp_jax import BayesianGP

class HeatPlotter(Plotter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_initial_conditions = self.snapshots_training.shape[1]

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

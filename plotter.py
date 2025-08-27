import matplotlib.pyplot as plt
import numpy as np

class plotter:

    def __init__(self) -> None:
        pass
    
    def trajectory_plot(self, time_snaps: np.ndarray, time: np.ndarray, snapshots: np.ndarray, samples: np.ndarray, figsize: tuple = (12,8)):
        '''
        time_snaps: the time axis of the observed data shape = (r, t)
        time: the time axis of the results shape = (r, t1)
        snapshots: the observed data compressed or not, shape = (r, t)
        samples: the results of many draws from a model, shape = (n, r, t1)
        Need to adapt this to use the multiparameter
        '''
        modes = samples.shape[1]
        ndraws = samples.shape[0]
        print(modes, ndraws)
        fig, ax = plt.subplots(modes, figsize=figsize)

        # Plot the observed data first
        for i in range(modes):
            if len(time_snaps.shape) == 1: # 1d
                ax[i].plot(time_snaps, snapshots[i], 'k*')

        for i in range(ndraws):
            for j in range(modes):
                if len(time_snaps.shape) == 1: # 1d
                    ax[j].plot(time, samples[i,j], alpha=.2)
        
        plt.show()

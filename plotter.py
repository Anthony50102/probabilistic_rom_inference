import matplotlib.pyplot as plt
from helpers.bgp_jax import BayesianGP
import numpy as np
from typing import List
import jax.numpy as jnp

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



Best Results

RESULT 1 HYPERS

```
# Data generation settings
TRAINING_SPAN = (0, 0.08)  # Subset of time domain for training
NUM_SAMPLES = 250           # Number of training samples
NOISE_LEVEL = 0.03        # Noise level for training data

# Data scaling
USE_SCALED_DATA = False    # Standardize POD coefficients for GP fitting

# Inference settings
RUN_SVI = True
RUN_MCMC = False           # More expensive, optional
USE_SVI_FOR_MCMC_INIT = False  # Initialize MCMC from SVI result (requires RUN_SVI=True)
GUIDE = numpyro.infer.autoguide.AutoDelta  # Guide for SVI; ignored if not using SVI

# GP densification: number of points at which to evaluate ODE constraints
# Set to None to use the original training times (no densification)
NUM_EVAL_POINTS = 400 

# Hyperparameters
GAMMA = 1e1     # Operator prior variance
GAMMA2 = 1e1     # ODE constraint stiffness
```

RESULT 2 HYPERS:

```
# Data generation settings
TRAINING_SPAN = (0, 0.08)  # Subset of time domain for training
NUM_SAMPLES = 55           # Number of training samples
NOISE_LEVEL = 0.05        # Noise level for training data

# Data scaling
USE_SCALED_DATA = False    # Standardize POD coefficients for GP fitting

# Inference settings
RUN_SVI = True
RUN_MCMC = False           # More expensive, optional
USE_SVI_FOR_MCMC_INIT = False  # Initialize MCMC from SVI result (requires RUN_SVI=True)
GUIDE = numpyro.infer.autoguide.AutoDelta  # Guide for SVI; ignored if not using SVI

# GP densification: number of points at which to evaluate ODE constraints
# Set to None to use the original training times (no densification)
NUM_EVAL_POINTS = 150 

# Hyperparameters
GAMMA = 1e2     # Operator prior variance
GAMMA2 = 1e2     # ODE constraint stiffness
```
import numpy as np
from mlfinlab.ymws.KCA_optimal_v2 import fitKCA_optimized as fitKCA

import matplotlib.pyplot as plt

def generate_inter_kcapva(bar_series, q, forecast_steps=0):
    """
    Generate interbar features using KCA with optional forecasting.
    
    Inputs:
      bar_series  : 1D numpy array of inter-bar series
      q             : Scalar to seed the process noise covariance in KCA.
      forecast_steps: Number of steps to forecast ahead (default=0; no forecasting)
    
    Returns:
      features: Dictionary containing:
          - 'position', 'velocity', 'acceleration': the state estimates.
          - 'position_std', 'velocity_std', 'acceleration_std': the corresponding standard deviations.
          - 'position_t', 'velocity_t', 'acceleration_t': the t-values (state estimate divided by std).
    """
    # Create a time vector corresponding to the interbar data.
    t = np.arange(len(bar_series))
    
    # Run KCA; if forecast_steps > 0, the last forecasted state(s) will be appended.
    x_mean, x_std, _ = fitKCA(t, bar_series, q, fwd=forecast_steps, truncate_past=False)    
    
    # Compute t-values as the ratio of the state estimate to its standard deviation.
    # A small epsilon is added to the denominator to avoid division by zero.
    epsilon = 1e-8
    position_t     = x_mean[:, 0] / (x_std[:, 0] + epsilon)
    velocity_t     = x_mean[:, 1] / (x_std[:, 1] + epsilon)
    acceleration_t = x_mean[:, 2] / (x_std[:, 2] + epsilon)
    
    features = {
        'position': x_mean[:, 0],
        'velocity': x_mean[:, 1],
        'acceleration': x_mean[:, 2],
        'position_std': x_std[:, 0],
        'velocity_std': x_std[:, 1],
        'acceleration_std': x_std[:, 2],
        'position_t': position_t,
        'velocity_t': velocity_t,
        'acceleration_t': acceleration_t,
    }
    return features


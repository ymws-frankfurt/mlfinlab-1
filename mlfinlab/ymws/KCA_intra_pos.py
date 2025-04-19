import numpy as np
from mlfinlab.ymws.KCA_optimal_v2 import fitKCA_optimized as fitKCA

import matplotlib.pyplot as plt

def generate_intra_kcapos(signed_tick_array, q, forecast_steps=0, em_iter=5):    
    """
    Generate intrabar features using KCA with optional forecasting.
    
    Inputs:
      signed_tick_array  : 1D numpy array of intrabar prices (e.g., minute bars)
      q             : Scalar to seed the process noise covariance in KCA.
      forecast_steps: Number of steps to forecast ahead (default=0; no forecasting)
    
    Returns:
      features: Dictionary containing:
          - 'position', 'velocity', 'acceleration': the state estimates.
          - 'position_std', 'velocity_std', 'acceleration_std': the corresponding standard deviations.
          - 'position_t', 'velocity_t', 'acceleration_t': the t-values (state estimate divided by std).
    """
    # Create a time vector corresponding to the intrabar data.
    t = np.arange(len(signed_tick_array))
    
    # Run KCA; if forecast_steps > 0, the last forecasted state(s) will be appended.
    x_mean, x_std, _ = fitKCA(
        t, signed_tick_array, q,
        fwd=forecast_steps,
        truncate_past=False,
        em_iter=em_iter
    )    

    # Compute t-values as the ratio of the state estimate to its standard deviation.
    # A small epsilon is added to the denominator to avoid division by zero.
    epsilon = 1e-8
    # position_t     = x_mean[:, 0] / (x_std[:, 0] + epsilon)
    # velocity_t     = x_mean[:, 1] / (x_std[:, 1] + epsilon)
    # acceleration_t = x_mean[:, 2] / (x_std[:, 2] + epsilon)
    
    features = {
        'position': x_mean[:, 0],
        # 'velocity': x_mean[:, 1],
        # 'acceleration': x_mean[:, 2],
        # 'position_std': x_std[:, 0],
        # 'velocity_std': x_std[:, 1],
        # 'acceleration_std': x_std[:, 2],
        # 'position_t': position_t,
        # 'velocity_t': velocity_t,
        # 'acceleration_t': acceleration_t,
    }
    return features

# ---------------------------
# Example usage:
if __name__ == "__main__":
    # Simulate an intrabar price series: e.g., minute bars for one trading day (e.g., 390 minutes)
    n_minutes = 390
    np.random.seed(42)
    t_sim = np.arange(n_minutes)
    # Create a synthetic price series: a base price with a slight sine fluctuation and noise.
    price_sim = 100 + np.sin(t_sim / 50) + 0.5 * np.random.randn(n_minutes)
    
    # Process noise seed parameter (this may require calibration for your intraday data)
    q_value = 0.001
    # Number of forecast steps (set to zero if you don't need forecasts)
    forecast_steps = 30  
    
    features = generate_intra_kcapos(price_sim, q_value, forecast_steps)
    
    # For demonstration, plot the position estimates and the corresponding t-values.
    plt.figure(figsize=(12, 6))
    total_points = len(features['position'])
    t_total = np.arange(total_points)
    plt.plot(t_total, features['position'], 'b-', label='KCA Position')
    plt.fill_between(t_total,
                     features['position'] - 2 * features['position_std'],
                     features['position'] + 2 * features['position_std'],
                     color='b', alpha=0.2, label='Confidence Interval')
    plt.xlabel('Time (bar index)')
    plt.ylabel('Price / Position')
    plt.title('Intrabar KCA Position with Forecasting')
    plt.legend()
    plt.show()

    # Plot t-values as an additional feature.
    plt.figure(figsize=(12, 4))
    plt.plot(t_total, features['position_t'], 'r-', label='Position t-value')
    plt.xlabel('Time (bar index)')
    plt.ylabel('t-value')
    plt.title('Intrabar KCA Position t-values')
    plt.legend()
    plt.show()

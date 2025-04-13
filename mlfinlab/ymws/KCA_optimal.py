import numpy as np
from pykalman import KalmanFilter
from numba import njit
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# JIT-compiled forecast loop using the state-space prediction formulas:
#
#   x[t+1] = A @ x[t]
#   P[t+1] = A @ P[t] @ A.T + Q
#
# This avoids repeatedly calling kf.filter_update and speeds up forecasting.
@njit
def forecast_loop(x0, P0, A, Q, fwd):
    n_state = A.shape[0]
    x_forecast = np.empty((fwd, n_state))
    P_forecast = np.empty((fwd, n_state, n_state))
    x_prev = x0.copy()
    P_prev = P0.copy()
    for i in range(fwd):
        x_new = A.dot(x_prev)
        P_new = A.dot(P_prev).dot(A.T) + Q
        x_forecast[i, :] = x_new
        P_forecast[i, :, :] = P_new
        x_prev = x_new
        P_prev = P_new
    return x_forecast, P_forecast

# -----------------------------------------------------------------------------
# Optimized KCA implementation with pre-allocation and manual forecast loop.
def fitKCA_optimized(t, z, q, fwd=0):
    """
    Optimized implementation of Kinetic Component Analysis (KCA).

    Inputs:
      t   : 1D numpy array of time indices.
      z   : 1D numpy array of measurements.
      q   : Scalar to seed the process noise covariance Q.
      fwd : Number of forecast steps to compute (default=0).

    Returns:
      x_mean_total: Smoothed state means (and forecasted means if fwd > 0) 
                    for [position, velocity, acceleration].
      x_std_total : Standard deviations (from state covariances) for each state.
      x_cov_total : Full state covariance matrices.
    """
    # Compute time step h from the time vector
    h = (t[-1] - t[0]) / t.shape[0]
    
    # Define state transition matrix A (from Taylor expansion)
    A = np.array([[1, h, 0.5 * h**2],
                  [0, 1, h],
                  [0, 0, 1]])
    # Process noise covariance matrix (seed)
    Q = q * np.eye(A.shape[0])
    
    # Observation matrix H: we only observe the position
    H = np.array([[1, 0, 0]])
    
    # Initialize the Kalman filter with the given matrices.
    kf = KalmanFilter(transition_matrices=A, 
                      transition_covariance=Q,
                      observation_matrices=H)
    
    # Optionally run a few EM iterations to estimate noise covariances.
    kf = kf.em(z, n_iter=5)
    
    # Run smoothing to get estimates of the latent states from the noisy data.
    x_mean, x_cov = kf.smooth(z)
    
    n_smoothed = x_mean.shape[0]
    
    if fwd > 0:
        # Preallocate arrays for the total number of time steps (smoothed + forecast)
        total_steps = n_smoothed + fwd
        x_mean_total = np.empty((total_steps, x_mean.shape[1]))
        x_cov_total = np.empty((total_steps, x_cov.shape[1], x_cov.shape[2]))
        
        # Copy the smoothed estimates into the first part of the arrays.
        x_mean_total[:n_smoothed, :] = x_mean
        x_cov_total[:n_smoothed, :, :] = x_cov
        
        # Use the last smoothed state as the starting point for forecasting.
        # Forecast fwd steps ahead using the JIT-compiled loop.
        x_forecast, P_forecast = forecast_loop(x_mean_total[n_smoothed - 1, :],
                                               x_cov_total[n_smoothed - 1, :, :],
                                               A, Q, fwd)
        x_mean_total[n_smoothed:, :] = x_forecast
        x_cov_total[n_smoothed:, :, :] = P_forecast
    else:
        x_mean_total = x_mean
        x_cov_total = x_cov

    # Compute the standard deviations for each state from the diagonal of the covariances.
    n_steps, n_state = x_cov_total.shape[0], x_cov_total.shape[1]
    x_std_total = np.empty((n_steps, n_state))
    for i in range(n_steps):
        x_std_total[i, :] = np.sqrt(np.diag(x_cov_total[i, :, :]))
    
    return x_mean_total, x_std_total, x_cov_total

# -----------------------------------------------------------------------------
# Function to process multiple time series in parallel using Joblib.
def fitKCA_parallel(t_list, z_list, q, fwd=0, n_jobs=-1):
    """
    Run fitKCA_optimized on multiple time series in parallel.
    
    Inputs:
      t_list: List of numpy arrays (time indices for each series).
      z_list: List of numpy arrays (observations for each series).
      q     : Scalar for the process noise covariance.
      fwd   : Forecast steps.
      n_jobs: Number of parallel jobs (-1 uses all available cores).
    
    Returns:
      List of tuples (x_mean_total, x_std_total, x_cov_total) for each series.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(fitKCA_optimized)(t, z, q, fwd) for t, z in zip(t_list, z_list)
    )
    return results

# -----------------------------------------------------------------------------
# Example usage:
if __name__ == '__main__':
    # Create a sample time series: a noisy sine wave.
    n_obs = 300
    t = np.linspace(0, 10, n_obs)
    signal = np.sin(t)
    np.random.seed(0)
    noise = 0.5 * np.random.randn(n_obs)
    z = signal + noise

    q = 0.001   # Process noise seed
    fwd_steps = 40  # Forecast 20 steps ahead

    # Run the optimized KCA on the sample series.
    x_mean_total, x_std_total, x_cov_total = fitKCA_optimized(t, z, q, fwd=fwd_steps)
    
    # Construct a time vector for the forecasted series.
    dt = (t[-1] - t[0]) / n_obs
    t_total = np.linspace(t[0], t[-1] + fwd_steps * dt, n_obs + fwd_steps)
    
    # Plot the results: observations, estimated position, and a 2-standard-deviation band.
    plt.figure(figsize=(10, 6))
    plt.plot(t, z, 'kx', label='Observations')
    plt.plot(t_total, x_mean_total[:, 0], 'b-', label='Estimated Position')
    plt.fill_between(t_total,
                     x_mean_total[:, 0] - 2 * x_std_total[:, 0],
                     x_mean_total[:, 0] + 2 * x_std_total[:, 0],
                     color='b', alpha=0.2, label='Confidence Interval')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Optimized KCA with Forecasting')
    plt.legend()
    plt.show()
    
    # Example of parallel processing with multiple time series:
    # (For demonstration, we use two copies of the same series.)
    t_list = [t, t]
    z_list = [z, z]
    results = fitKCA_parallel(t_list, z_list, q, fwd=fwd_steps, n_jobs=-1)
    # results is a list of tuples with the KCA outputs for each series.

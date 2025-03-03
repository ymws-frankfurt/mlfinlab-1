import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from numba import njit
# https://chatgpt.com/share/67bc54da-1974-8000-a69a-2ac82cb210a0
# ---------------------------------------------------------------------------
# JIT-compiled helper to compute weights (returns weights in chronological order)
@njit
def get_weights_jit(diff_amt, size):
    weights = np.empty(size)
    weights[0] = 1.0
    for k in range(1, size):
        weights[k] = -weights[k - 1] * (diff_amt - k + 1) / k
    # Reverse so that the first element corresponds to the oldest observation
    return weights[::-1]

# ---------------------------------------------------------------------------
# Fixed window fractional differencing using vectorized convolution.
def frac_diff_fixed_vectorized(process, diff_amt, thresh=0.01):
    """
    Fractionally difference each column of a DataFrame using a fixed-width
    (constant window) approach. Under the hood, this function computes the
    weights using a JIT-compiled helper and then applies np.convolve.
    """
    out_dict = {}
    for col in process.columns:
        # Fill missing values and drop any remaining NAs
        series = process[col].fillna(method='ffill').dropna()
        arr = series.values
        n = len(arr)
        # Compute full weights using the JIT helper
        weights = get_weights_jit(diff_amt, n)
        # Determine the effective window length based on the cutoff threshold:
        cum_weights = np.cumsum(np.abs(weights))
        cum_weights /= cum_weights[-1]
        window = int(np.argmax(cum_weights >= (1 - thresh))) + 1
        # Select only the significant portion of weights
        weights_window = weights[-window:]
        # Apply convolution in "valid" mode so that only complete windows are computed
        fd_arr = np.convolve(arr, weights_window, mode='valid')
        # Prepend NaNs to maintain alignment with the original series
        pad = np.full(window - 1, np.nan)
        full_arr = np.concatenate([pad, fd_arr])
        out_dict[col] = pd.Series(full_arr, index=series.index)
    return pd.DataFrame(out_dict)

# ---------------------------------------------------------------------------
# JIT-compiled core function for expanding window fractional differencing.
@njit
def frac_diff_expanding_jit_core(arr, weights, skip):
    n = len(arr)
    out = np.empty(n)
    # Mark initial entries as NaN (insufficient data)
    for i in range(skip):
        out[i] = np.nan
    for i in range(skip, n):
        s = 0.0
        # For each time index i, use the last (i+1) weights and observations
        for j in range(i + 1):
            s += weights[n - (i + 1) + j] * arr[j]
        out[i] = s
    return out

# Wrapper for expanding window approach using JIT.
def frac_diff_expanding_jit(process, diff_amt, thresh=0.01):
    """
    Fractionally difference each column of a DataFrame using an expanding window.
    The weights are computed via a JIT-compiled routine, and the computation
    over the growing window is also JIT-compiled.
    """
    out_dict = {}
    for col in process.columns:
        series = process[col].fillna(method='ffill').dropna()
        arr = series.values
        n = len(arr)
        weights = get_weights_jit(diff_amt, n)
        # Determine the number of observations to skip based on the threshold.
        # Here we define skip as the first index where the cumulative sum of the
        # absolute weights exceeds the threshold.
        cum_weights = np.cumsum(np.abs(weights))
        cum_weights /= cum_weights[-1]
        skip = int(np.argmax(cum_weights >= thresh))
        if skip < 1:
            skip = 1
        fd_arr = frac_diff_expanding_jit_core(arr, weights, skip)
        out_dict[col] = pd.Series(fd_arr, index=series.index)
    return pd.DataFrame(out_dict)

# ---------------------------------------------------------------------------
# New function that endogenously determines the optimal d* via a grid search
# using the ADF test, then returns the fractionally differentiated series.
def optimal_frac_diff(process, thresh=0.01, method='fixed',
                      adf_pval_threshold=0.05, d_min=0, d_max=2, grid_steps=21):
    """
    Returns the fractionally differentiated version of the input series along with the
    optimal differencing amount d* determined via a grid search using the ADF test.
    
        Parameters:
      process : pd.Series, pd.DataFrame, or np.ndarray
          Input time series data.
      thresh : float, default 0.01
          Threshold for cutting off weights.
      method : str, {'fixed', 'expanding'}, default 'fixed'
          Whether to use the fixed window (vectorized convolution) or expanding
          window (JIT-compiled) method.
      adf_pval_threshold : float, default 0.05
          P-value threshold for the Augmented Dickey-Fuller test.
      d_min : float, default 0
          Minimum candidate fractional differencing value.
      d_max : float, default 2
          Maximum candidate fractional differencing value.
      grid_steps : int, default 21
          Number of candidate d values to test.
    
    Returns:
      tuple:
        - pd.DataFrame: The fractionally differentiated series computed using the optimal d*.
        - float: The optimal d value found during the grid search.
    """
    # Ensure input is a DataFrame.
    if isinstance(process, (pd.Series, np.ndarray)):
        process = pd.DataFrame(process)
    elif not isinstance(process, pd.DataFrame):
        raise ValueError("Input process must be a pandas Series, DataFrame, or numpy ndarray.")
    
    optimal_d = None
    d_values = np.linspace(d_min, d_max, grid_steps)
    
    for d in d_values:
        if method == 'fixed':
            diff_series = frac_diff_fixed_vectorized(process, d, thresh)
        elif method == 'expanding':
            diff_series = frac_diff_expanding_jit(process, d, thresh)
        else:
            raise ValueError("Method must be either 'fixed' or 'expanding'.")
        
        # Use the first column for the ADF test (assumes univariate test).
        diff_col = diff_series.iloc[:, 0].dropna()
        if len(diff_col) < 10:
            continue  # Skip if too few data points for a reliable test.
        try:
            adf_result = adfuller(diff_col, maxlag=1, regression='c', autolag=None)
        except Exception:
            continue
        p_value = adf_result[1]
        if p_value <= adf_pval_threshold:
            optimal_d = d
            break  # Accept the first (i.e. minimum) d that passes.
    
    # If no candidate passes, default to d_max.
    if optimal_d is None:
        optimal_d = d_max
    
    # Recompute the final fractionally differenced series using the optimal d.
    if method == 'fixed':
        final_series = frac_diff_fixed_vectorized(process, optimal_d, thresh)
    else:
        final_series = frac_diff_expanding_jit(process, optimal_d, thresh)
    
    return final_series, optimal_d

# ---------------------------------------------------------------------------
# Example usage:
if __name__ == '__main__':
    # Generate a synthetic non-stationary series (e.g., a random walk)
    np.random.seed(42)
    n_points = 500
    data = np.cumsum(np.random.randn(n_points))
    series = pd.Series(data, name='price')

    # Compute optimally fractionally differentiated series using both approaches.
    diff_series_fixed = optimal_frac_diff(series, thresh=0.99, method='fixed')
    diff_series_expanding = optimal_frac_diff(series, thresh=0.01, method='expanding')

    # print("Fixed Window Optimal Fractionally Differenced Series (first 10 rows):")
    # print(diff_series_fixed.head(10))
    # print("\nExpanding Window Optimal Fractionally Differenced Series (first 10 rows):")
    # print(diff_series_expanding.head(10))

    # Combine the original and differenced series into one DataFrame.
    combined_df = pd.concat([series, diff_series_fixed, diff_series_expanding], axis=1)
    combined_df.columns = ['Original Series', 'Fixed Window Differenced', 'Expanding Window Differenced']

    # Use df.plot() to visualize the series.
    # The style list sets a solid line for the first two series and a dashed line for the expanding window.
    ax = combined_df.plot(title='Optimal Fractional Differentiation', figsize=(12, 6),
                            style=['-', '-', '--'], alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

"""
This module contains an implementation of an exponentially weighted moving average based on sample size.
The inspiration and context for this code was from a blog post by writen by Maksim Ivanov:
https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
"""
# https://chatgpt.com/share/67bf885c-09c0-8000-8640-ca32f1cdaffe
import numpy as np
from numba import jit, float64, int64

@jit((float64[:], int64), nopython=True, nogil=True)
def ewma(arr_in, window):
    """
    Compute the bias-corrected exponentially weighted moving average (EWMA)
    for a 1D numpy array using a decay window (span).
    
    The EWMA is computed recursively as:
        s[0] = arr_in[0]
        s[t] = (1 - α)*s[t-1] + arr_in[t]
    with α = 2 / (window + 1), and the bias is corrected by dividing by the 
    cumulative weight W[t] which is updated recursively:
        W[0] = 1
        W[t] = (1 - α)*W[t-1] + 1
    The final output is given by:
        ewma_arr[t] = s[t] / W[t]
    
    :param arr_in: 1D numpy array (float64) of tick-level values.
    :param window: Decay window (span) as an integer.
    :return: A numpy array of the same length as arr_in containing the EWMA.
    """
    arr_length = arr_in.shape[0]
    ewma_arr = np.empty(arr_length, dtype=np.float64)
    alpha = 2 / (window + 1)
    
    # Initialize the first element and weight
    s_old = arr_in[0]
    weight = 1.0
    ewma_arr[0] = s_old
    
    # Recursive update for the EWMA and cumulative weight
    for i in range(1, arr_length):
        weight = weight * (1 - alpha) + 1.0
        s_old = s_old * (1 - alpha) + arr_in[i]
        ewma_arr[i] = s_old / weight

    return ewma_arr

'''
# Imports
import numpy as np
from numba import jit
from numba import float64
from numba import int64


@jit((float64[:], int64), nopython=False, nogil=True)
def ewma(arr_in, window):  # pragma: no cover
    """
    Exponentially weighted moving average specified by a decay ``window`` to provide better adjustments for
    small windows via:
        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    :param arr_in: (np.ndarray), (float64) A single dimensional numpy array
    :param window: (int64) The decay window, or 'span'
    :return: (np.ndarray) The EWMA vector, same length / shape as ``arr_in``
    """
    arr_length = arr_in.shape[0]
    ewma_arr = np.empty(arr_length, dtype=float64)
    alpha = 2 / (window + 1)
    weight = 1
    ewma_old = arr_in[0]
    ewma_arr[0] = ewma_old
    for i in range(1, arr_length):
        weight += (1 - alpha)**i
        ewma_old = ewma_old * (1 - alpha) + arr_in[i]
        ewma_arr[i] = ewma_old / weight

    return ewma_arr
'''
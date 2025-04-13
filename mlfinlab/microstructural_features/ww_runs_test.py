import math
import itertools


def runs_z_score(signed_ticks):
    """
    Given a list of signed ticks (e.g., 1 for buy-initiated trades, -1 for sell-initiated trades),
    compute the Wald-Wolfowitz runs test z-score.
    
    A run is defined as a sequence of consecutive identical tick values. This function computes
    the number of runs in the provided tick series and then calculates a z-score based on the 
    expected number of runs under a random hypothesis.
    
    Parameters:
        signed_ticks (list of int): A list of signed ticks, where each entry is typically 1 or -1.
    
    Returns:
        float: The standardized test statistic (z-score) representing the deviation of the observed 
               number of runs from its expected value under randomness.
    """
    if not signed_ticks:
        return 0.0

    # Count the number of runs using itertools.groupby
    R = sum(1 for _ in itertools.groupby(signed_ticks))
    
    # Count the occurrences of 1's (buy ticks) and -1's (sell ticks)
    n1 = signed_ticks.count(1)
    n2 = signed_ticks.count(-1)
    n = n1 + n2

    # If there are fewer than two ticks, not enough data to perform the test.
    if n < 2:
        return 0.0

    # Expected number of runs and standard error under the randomness assumption.
    muR = (2 * n1 * n2) / n + 1
    seR = math.sqrt(((2 * n1 * n2) * (2 * n1 * n2 - n)) / (n**2 * (n - 1)))
    
    # Avoid division by zero in case of a degenerate sequence.
    if seR == 0:
        return 0.0

    # Return the computed z-score.
    return (R - muR) / seR


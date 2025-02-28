import math
import itertools


def runs_z_score(prices):
    """
    Given a list of prices, compute the Wald-Wolfowitz runs test z-score.
    This function converts prices into signed ticks, counts the number
    of runs (consecutive groups), and then calculates the standardized
    z-score based on the observed number of runs.
    
    Returns:
        z: The test statistic, which can be used as a feature.
    """
    def _compute_signed_ticks(prices):
        """
        Convert a list of prices into signed ticks:
        - 1 if price increases (buy-initiated)
        - -1 if price decreases (sell-initiated)
        - If the price is unchanged, inherit the previous tick sign.
        The first tick is assumed to be 1.
        """
        if not prices:
            return []
        ticks = [1]  # default for the first tick
        for prev, curr in zip(prices, prices[1:]):
            diff = curr - prev
            if diff > 0:
                ticks.append(1)
            elif diff < 0:
                ticks.append(-1)
            else:
                ticks.append(ticks[-1])
        return ticks

    ticks = _compute_signed_ticks(prices)
    
    # Count the number of runs using itertools.groupby
    R = sum(1 for _ in itertools.groupby(ticks))
    
    # Count of positive ticks (assumed "buy") and negative ticks ("sell")
    n1 = ticks.count(1)
    n2 = ticks.count(-1)
    n = n1 + n2
    
    # Guard against division by zero or insufficient data
    if n < 2:
        return 0.0
    
    # Compute standard error and expected runs under randomness.
    seR = math.sqrt(((2 * n1 * n2) * (2 * n1 * n2 - n)) / (n**2 * (n - 1)))
    muR = (2 * n1 * n2) / n + 1
    
    # If standard error is zero (or nearly zero), return 0 to avoid division by zero.
    if seR == 0:
        return 0.0
    
    # Calculate and return the z-score.
    return (R - muR) / seR


# THIS SCRIPT IS NOT BE USED SINCE MFGPB IS ABANDONED!!!
import sys
sys.path.append("/app/scripts/jmrichardson_mlfinlab")
sys.path.append("/app/src/Model_Dev")

import pandas as pd
import numpy as np
from mlfinlab.microstructural_features.misc import vwap, get_avg_tick_size
from mlfinlab.microstructural_features.encoding import encode_tick_rule_array
from mlfinlab.microstructural_features.entropy import get_shannon_entropy, get_plug_in_entropy, get_lempel_ziv_entropy_fast#, get_konto_entropy_nb

from mlfinlab.ymws.tick_data_formatter import TickDataFormatter

from concurrent.futures import ProcessPoolExecutor, as_completed

# (Import other feature functions as needed)
# aggressor derivative?
# KCA?
# IsMaker?

def split_bars(tick_data: pd.DataFrame, tick_num_series: pd.Series) -> list:
    """
    Splits tick_data into a list of DataFrames, each corresponding to one bar.
    
    :param tick_data: DataFrame containing tick data.
    :param tick_num_series: Series containing tick indices where a bar is formed.
    :return: List of DataFrame slices, one per bar.
    """
    bars = []
    start_idx = 0
    # Ensure tick_num_series is sorted in ascending order
    for tick in sorted(tick_num_series):
        # Create a slice from start index up to the tick index.
        bar_slice = tick_data.iloc[start_idx:tick]
        if not bar_slice.empty:
            bars.append(bar_slice)
        start_idx = tick
    # Optionally, process any remaining ticks after the last boundary.
    # if start_idx < len(tick_data):
    #     bars.append(tick_data.iloc[start_idx:])
    return bars


def compute_features_for_bar(bar_df: pd.DataFrame, volume_encoding=None, pct_encoding=None) -> dict:
    """
    Compute intra-bar features for a given tick data slice.
    
    :param bar_df: DataFrame slice containing tick data for one bar.
    :param volume_encoding: Optional encoding dictionary for volume.
    :param pct_encoding: Optional encoding dictionary for log returns.
    :return: Dictionary of computed features.
    """
    # Extract raw arrays from the bar data.
    # Assuming columns are in order: date_time, price, volume
    prices = bar_df['price'].values
    volumes = bar_df['volume'].values
    date_time = bar_df['date_time'].iloc[-1]  # Timestamp for the bar (e.g., last tick)
    
    # Compute basic metrics.
    tick_sizes = volumes  # In this context, trade sizes are volumes.
    avg_tick_size = get_avg_tick_size(tick_sizes)
    # For tick rule, we need to compute the sign of price changes.
    tick_rule = []
    prev_price = prices[0]
    for price in prices:
        tick_rule.append(np.sign(price - prev_price))
        prev_price = price
    tick_rule_sum = sum(tick_rule)
    # VWAP calculation requires dollar volumes.
    dollar_volume = prices * volumes
    bar_vwap = vwap(dollar_volume, volumes)
    
    # Compute entropy features from tick rule.
    encoded_tick = encode_tick_rule_array(tick_rule)
    entropy_shannon = get_shannon_entropy(encoded_tick)
    entropy_plugin = get_plug_in_entropy(encoded_tick)
    entropy_lz = get_lempel_ziv_entropy_fast(encoded_tick)
    #entropy_konto = get_konto_entropy_nb(encoded_tick)
    
    # You would add additional feature computations (lambdas, etc.) similarly.
    
    # Collect features in a dictionary.
    features = {
        'date_time': date_time,
        'avg_tick_size': avg_tick_size,
        'tick_rule_sum': tick_rule_sum,
        'vwap': bar_vwap,
        'tick_rule_entropy_shannon': entropy_shannon,
        'tick_rule_entropy_plug_in': entropy_plugin,
        'tick_rule_entropy_lempel_ziv': entropy_lz,
        #'tick_rule_entropy_konto': entropy_konto

        # Add further features as computed.
    }
    return features


def compute_features_parallel(tick_data: pd.DataFrame, tick_num_series: pd.Series, volume_encoding=None, pct_encoding=None) -> pd.DataFrame:
    # Split the tick data into bar slices.
    bar_slices = split_bars(tick_data, tick_num_series)
    
    features_list = []
    # Use a ProcessPoolExecutor for parallel processing.
    with ProcessPoolExecutor() as executor:
        # Prepare a list of futures.
        futures = {executor.submit(compute_features_for_bar, bar, volume_encoding, pct_encoding): bar for bar in bar_slices}
        for future in as_completed(futures):
            try:
                result = future.result()
                features_list.append(result)
            except Exception as e:
                print(f"Error processing a bar: {e}")
    
    # Convert list of dictionaries to a DataFrame.
    return pd.DataFrame(features_list)


def generate_synthetic_tick_data(num_ticks=100):
    # Create a date_time column with 1-second intervals
    date_time = pd.date_range(start="2025-02-20 09:30:00", periods=num_ticks, freq='S')
    
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Simulate a random walk for price, starting at 100
    price = 100 + np.cumsum(np.random.randn(num_ticks))
    
    # Generate random volumes between 1 and 100
    volume = np.random.randint(1, 100, size=num_ticks)
    
    # Combine into a DataFrame
    df = pd.DataFrame({
        "date_time": date_time,
        "price": price,
        "volume": volume
    })
    return df


# Example usage:
if __name__ == "__main__":
    # Generate synthetic tick data (as previously shown or loaded from your source)
    tick_data = generate_synthetic_tick_data(1000)  # Replace with your data
    # Define tick_num_series boundaries (ensure they are in the correct range)
    tick_num_series = pd.Series(np.arange(50, len(tick_data)+1, 50))
    
    features_df = compute_features_parallel(tick_data, tick_num_series)
    print(features_df)

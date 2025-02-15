
# Imports
from typing import Union, Iterable, Optional
import sys
sys.path.append("/app/scripts/jmrichardson_mlfinlab") 
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import pandas as pd

from mlfinlab.data_structures.base_bars import BaseBars

import dask.dataframe as dd
from dask.distributed import Client
import cudf
import dask_cudf
import pandas as pd
import os
import time



def benchmark_processing(file_path_or_df):
    """
    # Batch processing of large datasets using Dask (o3-mini)
    # https://chatgpt.com/share/67adcf1f-056c-8000-a915-f142c1a42e74
    """    
    logging.info("Benchmarking CPU vs GPU performance...")
    sample_size = 100000  # Small batch for benchmarking
    
    # Load sample data
    if isinstance(file_path_or_df, str):
        sample_cpu = dd.read_csv(file_path_or_df, parse_dates=[0], assume_missing=True, blocksize=sample_size).head(sample_size)
        sample_gpu = cudf.DataFrame.from_pandas(sample_cpu)
    elif isinstance(file_path_or_df, pd.DataFrame):
        sample_cpu = file_path_or_df.head(sample_size)
        sample_gpu = cudf.DataFrame.from_pandas(sample_cpu)
    else:
        raise ValueError("Invalid input type")
    
    # CPU processing time
    start_time = time.time()
    _ = sample_cpu.mean()
    cpu_time = time.time() - start_time
    
    # GPU processing time
    start_time = time.time()
    _ = sample_gpu.mean()
    gpu_time = time.time() - start_time
    
    logging.info(f"CPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s")
    return gpu_time < cpu_time  # Return True if GPU is faster


class StandardBars(BaseBars):
    """
    Contains all of the logic to construct the standard bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_bars which will create an instance of this
    class and then construct the standard bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """
    def __init__(self, metric: str, threshold: int = 50000, batch_size: int = 20000000):
        """
        Constructor

        :param metric: (str) Type of run bar to create. Example: "dollar_run"
        :param threshold: (int) Threshold at which to sample
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        BaseBars.__init__(self, metric, batch_size)

        # Threshold at which to sample
        self.threshold = threshold

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for standard bars
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        """
        For loop which compiles the various bars: dollar, volume, or tick.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

        :param data: (tuple) Contains 3 columns - date_time, price, and volume.
        :return: (list) Extracted bars
        """

        # Iterate over rows
        list_bars = []

        for row in data:
            # Set variables
            date_time = row[0]
            self.tick_num += 1
            price = float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            if isinstance(self.threshold, (int, float)):
                # If the threshold is fixed, it's used for every sampling
                threshold = self.threshold
            else:
                # If the threshold is changing, then the threshold defined just before
                # sampling time is used
                threshold = self.threshold.iloc[self.threshold.index.get_loc(date_time, method='pad')]

            if self.open_price is None:
                self.open_price = price

            # Update high low prices
            self.high_price, self.low_price = self._update_high_low(price)

            # Calculations
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            # If threshold reached then take a sample
            if self.cum_statistics[self.metric] >= threshold:  # pylint: disable=eval-used
                self._create_bars(date_time, price,
                                  self.high_price, self.low_price, list_bars)

                # Reset cache
                self._reset_cache()
        return list_bars
    

    def batch_run(self, file_path_or_df, verbose=True, to_csv=False, output_path=None, use_gpu=None):
        """
        # Batch processing of large datasets using Dask (o3-mini)
        # https://chatgpt.com/share/67adcf1f-056c-8000-a915-f142c1a42e74
        """
        logging.info("Starting batch processing")
        
        if use_gpu is None:
            use_gpu = benchmark_processing(file_path_or_df)
            logging.info(f"Auto-selected {'GPU' if use_gpu else 'CPU'} for processing")
        
        client = Client()  # Start Dask distributed client
        
        if use_gpu:
            logging.info("Using GPU for processing")
            ddf = dask_cudf.read_csv(file_path_or_df, parse_dates=[0], assume_missing=True, blocksize="100MB")
        else:
            logging.info("Using CPU for processing")
            ddf = dd.read_csv(file_path_or_df, parse_dates=[0], assume_missing=True, blocksize="100MB")
        
        results = ddf.map_partitions(self.run, meta=pd.DataFrame()).compute()
        final_bars = pd.concat(results, ignore_index=True)
        
        if to_csv and output_path:
            final_bars.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))
        
        logging.info("Batch processing completed")
        client.close()
        return final_bars


# Modify get_dollar_bars, get_volume_bars, get_tick_bars accordingly
def get_dollar_bars(file_path_or_df, threshold=70000000, batch_size=20000000, verbose=True, to_csv=False, output_path=None, use_gpu=None):
    bars = StandardBars(metric='cum_dollar_value', threshold=threshold, batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose, to_csv, output_path, use_gpu)

def get_volume_bars(file_path_or_df, threshold=70000000, batch_size=20000000, verbose=True, to_csv=False, output_path=None, use_gpu=None):
    bars = StandardBars(metric='cum_volume', threshold=threshold, batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose, to_csv, output_path, use_gpu)

def get_tick_bars(file_path_or_df, threshold=70000000, batch_size=20000000, verbose=True, to_csv=False, output_path=None, use_gpu=None):
    bars = StandardBars(metric='cum_ticks', threshold=threshold, batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose, to_csv, output_path, use_gpu)
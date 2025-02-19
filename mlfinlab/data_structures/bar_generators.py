import os
import pandas as pd
import functools
from typing import Tuple, Union, Generator, Iterable, Optional, List

from mlfinlab import data_structures
from mlfinlab.data_structures.standard_data_structures import StandardBars
from mlfinlab.data_structures.time_data_structures import TimeBars
from mlfinlab.data_structures.imbalance_data_structures import EMAImbalanceBars, ConstImbalanceBars
from mlfinlab.data_structures.run_data_structures import EMARunBars, ConstRunBars # why working without mlfinlab.data_structures???

#https://chatgpt.com/share/67b2c8b7-f16c-8000-a2b5-bd09288c18cc

# -------------------------------
# Dollar Bars – Appending Version
# -------------------------------
# MODIFIED PART: Added a subclass to override batch_run so that output CSV is appended.
class AppendingStandardBars(StandardBars):
    """
    Subclass of StandardBars that appends to the CSV output instead of overwriting it.
    Modification: In batch_run, if the output file exists and is non-empty, do not clear it.
    """
    def batch_run(self, file_path_or_df, verbose=True, to_csv=False, output_path=None):
        if to_csv:
            # If the output file exists and is non-empty, do not clear it.
            if output_path and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                header = False
            else:
                header = True
                if output_path:
                    open(output_path, 'w').close()  # Clear file if it doesn't exist or is empty.
        count = 0
        final_bars = []
        cols = ['date_time', 'tick_num', 'open', 'high', 'low', 'close', 'volume', 'cum_buy_volume', 'cum_ticks', 'cum_dollar_value']
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:
                print('Batch number:', count)
            list_bars = self.run(data=batch)
            if to_csv:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path, header=header, index=False, mode='a')
                header = False  # Only write header on the first batch.
            else:
                final_bars += list_bars
            count += 1
        if final_bars:
            bars_df = pd.DataFrame(final_bars, columns=cols)
            return bars_df
        return None


# New helper function that uses our AppendingStandardBars instead of the original StandardBars.
def get_dollar_bars_appending(file_path_or_df, threshold=70000000, batch_size=20000000,
                              verbose=True, to_csv=False, output_path=None):
    bars = AppendingStandardBars(metric='cum_dollar_value', threshold=threshold, batch_size=batch_size)
    dollar_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return dollar_bars


# -------------------------------
# Below not constructed by o3-mini-high but hand-made!
# -------------------------------

def get_volume_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000,
                    batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):

    bars = AppendingStandardBars(metric='cum_volume', threshold=threshold, batch_size=batch_size)
    volume_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return volume_bars


def get_tick_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000,
                  batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):

    bars = AppendingStandardBars(metric='cum_ticks',
                        threshold=threshold, batch_size=batch_size)
    tick_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return tick_bars


# -------------------------------
# Time Bars – Appending Version
# -------------------------------
# Define AppendingTimeBars by subclassing TimeBars from time_data_structures.py
class AppendingTimeBars(TimeBars):
    """
    Subclass of TimeBars that appends to the CSV output instead of overwriting it.
    """
    def batch_run(self, file_path_or_df, verbose=True, to_csv=False, output_path=None):
        if to_csv:
            # Check if the file exists and is non-empty.
            if output_path and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                header = False
            else:
                header = True
                if output_path:
                    open(output_path, 'w').close()  # Clear the file if needed.
        count = 0
        final_bars = []
        # Expected columns for time bars (as documented in get_time_bars)
        cols = ['date_time', 'open', 'high', 'low', 'close', 'volume', 
                'cum_buy_volume', 'cum_ticks', 'cum_dollar_value']
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:
                print('Batch number:', count)
            list_bars = self.run(data=batch)
            if to_csv:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path,
                                                             header=header,
                                                             index=False,
                                                             mode='a')
                header = False  # Only write header for the first batch.
            else:
                final_bars += list_bars
            count += 1
        if final_bars:
            bars_df = pd.DataFrame(final_bars, columns=cols)
            return bars_df
        return None


def get_time_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                            resolution: str = 'D',
                            num_units: int = 1,
                            batch_size: int = 20000000,
                            verbose: bool = True,
                            to_csv: bool = False,
                            output_path: Optional[str] = None):
    bars = AppendingTimeBars(resolution=resolution,
                             num_units=num_units,
                             batch_size=batch_size)
    return bars.batch_run(file_path_or_df=file_path_or_df,
                          verbose=verbose,
                          to_csv=to_csv,
                          output_path=output_path)


# def get_time_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], resolution: str = 'D', num_units: int = 1, batch_size: int = 20000000,
#                   verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
#     bars = AppendingTimeBars(resolution=resolution, num_units=num_units, batch_size=batch_size)
#     time_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
#     return time_bars


# ---------------------------------------
# EMA Imbalance Bars – Appending Version
# ---------------------------------------
class AppendingEMAImbalanceBars(EMAImbalanceBars):
    """
    Subclass of EMAImbalanceBars that appends to CSV output instead of overwriting it.
    """
    def batch_run(self, file_path_or_df, verbose=True, to_csv=False, output_path=None):
        if to_csv:
            if output_path and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                header = False
            else:
                header = True
                if output_path:
                    open(output_path, 'w').close()
        count = 0
        final_bars = []
        cols = ['date_time', 'open', 'high', 'low', 'close', 'volume', 
                'cum_buy_volume', 'cum_ticks', 'cum_dollar_value']
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:
                print("EMA Imbalance Bars – Batch number:", count)
            list_bars = self.run(data=batch)
            if to_csv:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path,
                                                             header=header,
                                                             index=False,
                                                             mode='a')
                header = False
            else:
                final_bars += list_bars
            count += 1
        if final_bars:
            return pd.DataFrame(final_bars, columns=cols)
        return None

def get_ema_dollar_imbalance_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                            num_prev_bars: int = 3,
                                            expected_imbalance_window: int = 10000,
                                            exp_num_ticks_init: int = 20000,
                                            exp_num_ticks_constraints: List[float] = None,
                                            batch_size: int = 2e7,
                                            analyse_thresholds: bool = False,
                                            verbose: bool = True,
                                            to_csv: bool = False,
                                            output_path: Optional[str] = None):
    """
    Creates EMA dollar imbalance bars with CSV appending.
    """
    bars = AppendingEMAImbalanceBars(metric='dollar_imbalance',
                                     num_prev_bars=num_prev_bars,
                                     expected_imbalance_window=expected_imbalance_window,
                                     exp_num_ticks_init=exp_num_ticks_init,
                                     exp_num_ticks_constraints=exp_num_ticks_constraints,
                                     batch_size=batch_size,
                                     analyse_thresholds=analyse_thresholds)
    imbalance_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                    verbose=verbose,
                                    to_csv=to_csv,
                                    output_path=output_path)
    return imbalance_bars, pd.DataFrame(bars.bars_thresholds)

def get_ema_volume_imbalance_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                            num_prev_bars: int = 3,
                                            expected_imbalance_window: int = 10000,
                                            exp_num_ticks_init: int = 20000,
                                            exp_num_ticks_constraints: List[float] = None,
                                            batch_size: int = 2e7,
                                            analyse_thresholds: bool = False,
                                            verbose: bool = True,
                                            to_csv: bool = False,
                                            output_path: Optional[str] = None):
    bars = AppendingEMAImbalanceBars(metric='volume_imbalance',
                                     num_prev_bars=num_prev_bars,
                                     expected_imbalance_window=expected_imbalance_window,
                                     exp_num_ticks_init=exp_num_ticks_init,
                                     exp_num_ticks_constraints=exp_num_ticks_constraints,
                                     batch_size=batch_size,
                                     analyse_thresholds=analyse_thresholds)
    imbalance_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                    verbose=verbose,
                                    to_csv=to_csv,
                                    output_path=output_path)
    return imbalance_bars, pd.DataFrame(bars.bars_thresholds)

def get_ema_tick_imbalance_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                          num_prev_bars: int = 3,
                                          expected_imbalance_window: int = 10000,
                                          exp_num_ticks_init: int = 20000,
                                          exp_num_ticks_constraints: List[float] = None,
                                          batch_size: int = 2e7,
                                          analyse_thresholds: bool = False,
                                          verbose: bool = True,
                                          to_csv: bool = False,
                                          output_path: Optional[str] = None):
    bars = AppendingEMAImbalanceBars(metric='tick_imbalance',
                                     num_prev_bars=num_prev_bars,
                                     expected_imbalance_window=expected_imbalance_window,
                                     exp_num_ticks_init=exp_num_ticks_init,
                                     exp_num_ticks_constraints=exp_num_ticks_constraints,
                                     batch_size=batch_size,
                                     analyse_thresholds=analyse_thresholds)
    imbalance_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                    verbose=verbose,
                                    to_csv=to_csv,
                                    output_path=output_path)
    return imbalance_bars, pd.DataFrame(bars.bars_thresholds)

# ---------------------------------------
# Const Imbalance Bars – Appending Version
# ---------------------------------------
class AppendingConstImbalanceBars(ConstImbalanceBars):
    """
    Subclass of ConstImbalanceBars that appends to CSV output instead of overwriting it.
    """
    def batch_run(self, file_path_or_df, verbose=True, to_csv=False, output_path=None):
        if to_csv:
            if output_path and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                header = False
            else:
                header = True
                if output_path:
                    open(output_path, 'w').close()
        count = 0
        final_bars = []
        cols = ['date_time', 'open', 'high', 'low', 'close', 'volume',
                'cum_buy_volume', 'cum_ticks', 'cum_dollar_value']
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:
                print("Const Imbalance Bars – Batch number:", count)
            list_bars = self.run(data=batch)
            if to_csv:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path,
                                                             header=header,
                                                             index=False,
                                                             mode='a')
                header = False
            else:
                final_bars += list_bars
            count += 1
        if final_bars:
            return pd.DataFrame(final_bars, columns=cols)
        return None

def get_const_dollar_imbalance_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                              expected_imbalance_window: int = 10000,
                                              exp_num_ticks_init: int = 20000,
                                              batch_size: int = 2e7,
                                              analyse_thresholds: bool = False,
                                              verbose: bool = True,
                                              to_csv: bool = False,
                                              output_path: Optional[str] = None):
    bars = AppendingConstImbalanceBars(metric='dollar_imbalance',
                                       expected_imbalance_window=expected_imbalance_window,
                                       exp_num_ticks_init=exp_num_ticks_init,
                                       batch_size=batch_size,
                                       analyse_thresholds=analyse_thresholds)
    imbalance_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                    verbose=verbose,
                                    to_csv=to_csv,
                                    output_path=output_path)
    return imbalance_bars, pd.DataFrame(bars.bars_thresholds)

def get_const_volume_imbalance_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                              expected_imbalance_window: int = 10000,
                                              exp_num_ticks_init: int = 20000,
                                              batch_size: int = 2e7,
                                              analyse_thresholds: bool = False,
                                              verbose: bool = True,
                                              to_csv: bool = False,
                                              output_path: Optional[str] = None):
    bars = AppendingConstImbalanceBars(metric='volume_imbalance',
                                       expected_imbalance_window=expected_imbalance_window,
                                       exp_num_ticks_init=exp_num_ticks_init,
                                       batch_size=batch_size,
                                       analyse_thresholds=analyse_thresholds)
    imbalance_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                    verbose=verbose,
                                    to_csv=to_csv,
                                    output_path=output_path)
    return imbalance_bars, pd.DataFrame(bars.bars_thresholds)

def get_const_tick_imbalance_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                            expected_imbalance_window: int = 10000,
                                            exp_num_ticks_init: int = 20000,
                                            batch_size: int = 2e7,
                                            analyse_thresholds: bool = False,
                                            verbose: bool = True,
                                            to_csv: bool = False,
                                            output_path: Optional[str] = None):
    bars = AppendingConstImbalanceBars(metric='tick_imbalance',
                                       expected_imbalance_window=expected_imbalance_window,
                                       exp_num_ticks_init=exp_num_ticks_init,
                                       batch_size=batch_size,
                                       analyse_thresholds=analyse_thresholds)
    imbalance_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                    verbose=verbose,
                                    to_csv=to_csv,
                                    output_path=output_path)
    return imbalance_bars, pd.DataFrame(bars.bars_thresholds)



# -------------------------------
# EMA Run Bars – Appending Version
# -------------------------------
class AppendingEMARunBars(EMARunBars):
    """
    Subclass of EMARunBars that appends to the CSV output instead of overwriting it.
    """
    def batch_run(self, file_path_or_df, verbose=True, to_csv=False, output_path=None):
        if to_csv:
            # Check if file exists and is non-empty to decide header writing.
            if output_path and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                header = False
            else:
                header = True
                if output_path:
                    open(output_path, 'w').close()  # Clear file if necessary.
        count = 0
        final_bars = []
        cols = ['date_time', 'open', 'high', 'low', 'close', 'volume',
                'cum_buy_volume', 'cum_ticks', 'cum_dollar_value']
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:
                print("EMA Run Bars – Batch number:", count)
            list_bars = self.run(data=batch)
            if to_csv:
                pd.DataFrame(list_bars, columns=cols).to_csv(
                    output_path,
                    header=header,
                    index=False,
                    mode='a'
                )
                header = False  # Only write header once.
            else:
                final_bars += list_bars
            count += 1
        if final_bars:
            return pd.DataFrame(final_bars, columns=cols)
        return None

def get_ema_dollar_run_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                      num_prev_bars: int = 3,
                                      expected_imbalance_window: int = 10000,
                                      exp_num_ticks_init: int = 20000,
                                      exp_num_ticks_constraints: List[float] = None,
                                      batch_size: int = 2e7,
                                      analyse_thresholds: bool = False,
                                      verbose: bool = True,
                                      to_csv: bool = False,
                                      output_path: Optional[str] = None):
    bars = AppendingEMARunBars(metric='dollar_run',
                               num_prev_bars=num_prev_bars,
                               expected_imbalance_window=expected_imbalance_window,
                               exp_num_ticks_init=exp_num_ticks_init,
                               exp_num_ticks_constraints=exp_num_ticks_constraints,
                               batch_size=batch_size,
                               analyse_thresholds=analyse_thresholds)
    run_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                              verbose=verbose,
                              to_csv=to_csv,
                              output_path=output_path)
    return run_bars, pd.DataFrame(bars.bars_thresholds)

def get_ema_volume_run_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                      num_prev_bars: int = 3,
                                      expected_imbalance_window: int = 10000,
                                      exp_num_ticks_init: int = 20000,
                                      exp_num_ticks_constraints: List[float] = None,
                                      batch_size: int = 2e7,
                                      analyse_thresholds: bool = False,
                                      verbose: bool = True,
                                      to_csv: bool = False,
                                      output_path: Optional[str] = None):
    bars = AppendingEMARunBars(metric='volume_run',
                               num_prev_bars=num_prev_bars,
                               expected_imbalance_window=expected_imbalance_window,
                               exp_num_ticks_init=exp_num_ticks_init,
                               exp_num_ticks_constraints=exp_num_ticks_constraints,
                               batch_size=batch_size,
                               analyse_thresholds=analyse_thresholds)
    run_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                              verbose=verbose,
                              to_csv=to_csv,
                              output_path=output_path)
    return run_bars, pd.DataFrame(bars.bars_thresholds)

def get_ema_tick_run_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                    num_prev_bars: int = 3,
                                    expected_imbalance_window: int = 10000,
                                    exp_num_ticks_init: int = 20000,
                                    exp_num_ticks_constraints: List[float] = None,
                                    batch_size: int = 2e7,
                                    analyse_thresholds: bool = False,
                                    verbose: bool = True,
                                    to_csv: bool = False,
                                    output_path: Optional[str] = None):
    bars = AppendingEMARunBars(metric='tick_run',
                               num_prev_bars=num_prev_bars,
                               expected_imbalance_window=expected_imbalance_window,
                               exp_num_ticks_init=exp_num_ticks_init,
                               exp_num_ticks_constraints=exp_num_ticks_constraints,
                               batch_size=batch_size,
                               analyse_thresholds=analyse_thresholds)
    run_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                              verbose=verbose,
                              to_csv=to_csv,
                              output_path=output_path)
    return run_bars, pd.DataFrame(bars.bars_thresholds)

# -------------------------------
# Const Run Bars – Appending Version
# -------------------------------
class AppendingConstRunBars(ConstRunBars):
    """
    Subclass of ConstRunBars that appends to the CSV output instead of overwriting it.
    """
    def batch_run(self, file_path_or_df, verbose=True, to_csv=False, output_path=None):
        if to_csv:
            if output_path and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                header = False
            else:
                header = True
                if output_path:
                    open(output_path, 'w').close()
        count = 0
        final_bars = []
        cols = ['date_time', 'open', 'high', 'low', 'close', 'volume',
                'cum_buy_volume', 'cum_ticks', 'cum_dollar_value']
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:
                print("Const Run Bars – Batch number:", count)
            list_bars = self.run(data=batch)
            if to_csv:
                pd.DataFrame(list_bars, columns=cols).to_csv(
                    output_path,
                    header=header,
                    index=False,
                    mode='a'
                )
                header = False
            else:
                final_bars += list_bars
            count += 1
        if final_bars:
            return pd.DataFrame(final_bars, columns=cols)
        return None

def get_const_dollar_run_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                        num_prev_bars: int,
                                        expected_imbalance_window: int = 10000,
                                        exp_num_ticks_init: int = 20000,
                                        batch_size: int = 2e7,
                                        analyse_thresholds: bool = False,
                                        verbose: bool = True,
                                        to_csv: bool = False,
                                        output_path: Optional[str] = None):
    bars = AppendingConstRunBars(metric='dollar_run',
                                 num_prev_bars=num_prev_bars,
                                 expected_imbalance_window=expected_imbalance_window,
                                 exp_num_ticks_init=exp_num_ticks_init,
                                 batch_size=batch_size,
                                 analyse_thresholds=analyse_thresholds)
    run_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                              verbose=verbose,
                              to_csv=to_csv,
                              output_path=output_path)
    return run_bars, pd.DataFrame(bars.bars_thresholds)

def get_const_volume_run_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                        num_prev_bars: int,
                                        expected_imbalance_window: int = 10000,
                                        exp_num_ticks_init: int = 20000,
                                        batch_size: int = 2e7,
                                        analyse_thresholds: bool = False,
                                        verbose: bool = True,
                                        to_csv: bool = False,
                                        output_path: Optional[str] = None):
    bars = AppendingConstRunBars(metric='volume_run',
                                 num_prev_bars=num_prev_bars,
                                 expected_imbalance_window=expected_imbalance_window,
                                 exp_num_ticks_init=exp_num_ticks_init,
                                 batch_size=batch_size,
                                 analyse_thresholds=analyse_thresholds)
    run_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                              verbose=verbose,
                              to_csv=to_csv,
                              output_path=output_path)
    return run_bars, pd.DataFrame(bars.bars_thresholds)

def get_const_tick_run_bars_appending(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                      num_prev_bars: int,
                                      expected_imbalance_window: int = 10000,
                                      exp_num_ticks_init: int = 20000,
                                      batch_size: int = 2e7,
                                      analyse_thresholds: bool = False,
                                      verbose: bool = True,
                                      to_csv: bool = False,
                                      output_path: Optional[str] = None):
    bars = AppendingConstRunBars(metric='tick_run',
                                 num_prev_bars=num_prev_bars,
                                 expected_imbalance_window=expected_imbalance_window,
                                 exp_num_ticks_init=exp_num_ticks_init,
                                 batch_size=batch_size,
                                 analyse_thresholds=analyse_thresholds)
    run_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                              verbose=verbose,
                              to_csv=to_csv,
                              output_path=output_path)
    return run_bars, pd.DataFrame(bars.bars_thresholds)
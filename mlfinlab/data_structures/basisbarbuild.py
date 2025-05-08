import sys
import pandas as pd
#import numpy as np
import functools
from typing import Union, Iterable
from pathlib import Path
from datetime import datetime
# from mlfinlab import data_structures
# from mlfinlab.data_structures.standard_data_structures import StandardBars
#from mlfinlab.data_structures.bar_generators import get_tick_bars_appending, get_volume_bars_appending, get_dollar_bars_appending, get_time_bars_appending
#from icecream import ic
from mlfinlab.ymws.tick_data_formatter import TickDataFormatter


class BasisBarBuild():
    """
    Build bar data from CSV files using only the necessary columns.
    """
    # You can retain the default mappings as class attributes if desired.
    COLUMN_MAPPINGS = {
        "binance": {"timestamp": "date_time", "trade_price": "price", "trade_volume": "volume", "isBuyerMaker":"isBuyerMaker"},
        "oanda": {"time": "date_time", "ask": "price", "size": "volume"},
        "gmocoin": {"timestamp": "date_time", "price": "price", "size": "volume", "side":"side"},
    }

    # For CSV files without headers (i.e. columns are positional)
    COLUMN_POSITIONS = {
        "binance": {4: "date_time", 1: "price", 2: "volume", 5: "isBuyerMaker"},  # Map position-based indexing
        "binancefake": {4: "date_time", 1: "price", 2: "volume"},  # Map position-based indexing
        "oanda": {4: "date_time", 1: "price", 2: "volume"},  # Map position-based indexing
        "gmocoin": {5: "date_time", 4: "price", 3: "volume", 2: "side"},  # Map position-based indexing
        "gmocoinfake": {5: "date_time", 4: "price", 3: "volume"},  # Map position-based indexing
    }

    BAR_TYPE = {
        "time": "FHB",
        "standard": {"tick": "TB", "volume": "VB", "dollar": "DB"},
        "imbalance": {"tick": "TIB", "volume": "VIB", "dollar": "DIB"},
        "runs": {"tick": "TRB", "volume": "VRB", "dollar": "DRB"},
        "entropy": {"tick": "TEB", "volume": "VEB", "dollar": "DEB"},
    }

    def __init__(self, inputfilepath, period_start, period_end, batch_size, data_source="binance"):
        self.inputfilepath = inputfilepath
        self.period_start = period_start
        self.period_end = period_end
        self.batch_size = batch_size
        self.data_source = data_source        
        self.file_list = None
        # Create an instance of the formatter for the chosen data source.
        self.formatter = TickDataFormatter(
            data_source=self.data_source,
            column_mappings=BasisBarBuild.COLUMN_MAPPINGS,
            column_positions=BasisBarBuild.COLUMN_POSITIONS
        )

    def load_and_format_dataframe(self, file_path: str):
        """
        Delegate the CSV loading and formatting to the TickDataFormatter.
        """
        print(f"Processing file: {file_path}")
        return self.formatter.load_and_format_dataframe(file_path)

    def format_dataframe(self, data_source):
        """
        Decorator that uses load_and_format_dataframe before passing the data
        to the bar creation function.
        """
        def decorator(func):
            def wrapper(file_path: str, *args, **kwargs):
                df = self.load_and_format_dataframe(file_path)
                return func(df, *args, **kwargs)
            return wrapper
        return decorator

    def format_dataframe_time(self, data_source):
        """
        Specialized decorator for time bars that also converts the date_time column.
        -> time bar does not allow UTC (ymws)
        """
        def decorator(func):
            def wrapper(file_path: str, *args, **kwargs):
                df = self.load_and_format_dataframe(file_path)
                # Convert date_time column to datetime, adjusting the unit if needed.
                df['date_time'] = pd.to_datetime(df['date_time'], unit='ms', errors='coerce')
                return func(df, *args, **kwargs)
            return wrapper
        return decorator


    def readfile_list(self, start_date_str, end_date_str, date_format="%Y-%m-%d"):
        import glob, os
        from datetime import datetime

        try:
            file_list = glob.glob(os.path.join(self.inputfilepath, '*.csv'))
            
            def extract_date(file_path):
                filename = os.path.basename(file_path)
                name_no_ext = filename.replace('.csv', '')
                parts = name_no_ext.split('-')
                # Try to assume the last three parts form the date (e.g. "2024-11-08")
                if len(parts) >= 3:
                    candidate = "-".join(parts[-3:])
                    try:
                        return datetime.strptime(candidate, date_format)
                    except Exception:
                        pass
                # Fallback: if only year and month are present (e.g. "2024-11")
                if len(parts) >= 2:
                    candidate = "-".join(parts[-2:])
                    try:
                        return datetime.strptime(candidate, "%Y-%m")
                    except Exception:
                        pass
                raise ValueError(f"Unable to extract date from filename: {filename}")

            start_date = datetime.strptime(start_date_str, date_format)
            end_date = datetime.strptime(end_date_str, date_format)
            filtered_files = []
            for f in file_list:
                try:
                    file_date = extract_date(f)
                    if start_date <= file_date <= end_date:
                        filtered_files.append(f)
                except Exception as e:
                    print(f"An error occurred: {e}")
            self.file_list = filtered_files

        except FileNotFoundError:
            print(f"Error: Directory '{self.inputfilepath}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")


    def build_bars(self, bar_func, output_path, data_source="binance", **kwargs):
        """
        Generic method to build bars using a specified bar function.
        -> Use the data_source parameter here instead of hardcoding "binance"
        """
        decorated_bar_func = self.format_dataframe(data_source)(bar_func)
        self.readfile_list()
        if not self.file_list:
            print("No CSV files found.")
            return

        print(f"Found {len(self.file_list)} CSV file(s).")
        for file in self.file_list:
            decorated_bar_func(
                file,
                batch_size=self.batch_size,
                verbose=True,
                to_csv=True,
                output_path=output_path,
                **kwargs
            )

    # Specific methods for each bar type:
    def build_dollar_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_dollar_bars_appending
        self.build_bars(get_dollar_bars_appending, output_path, data_source, **kwargs)

    def build_volume_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_volume_bars_appending
        self.build_bars(get_volume_bars_appending, output_path, data_source, **kwargs)

    def build_tick_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_tick_bars_appending
        self.build_bars(get_tick_bars_appending, output_path, data_source, **kwargs)



    def build_ema_tick_imbalance_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_ema_tick_imbalance_bars_appending
        self.build_bars(get_ema_tick_imbalance_bars_appending, output_path, data_source, **kwargs)

    def build_const_tick_imbalance_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_const_tick_imbalance_bars_appending
        self.build_bars(get_const_tick_imbalance_bars_appending, output_path, data_source, **kwargs)

    def build_ema_volume_imbalance_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_ema_volume_imbalance_bars_appending
        self.build_bars(get_ema_volume_imbalance_bars_appending, output_path, data_source, **kwargs)

    def build_const_volume_imbalance_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_const_volume_imbalance_bars_appending
        self.build_bars(get_const_volume_imbalance_bars_appending, output_path, data_source, **kwargs)

    def build_ema_dollar_imbalance_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_ema_dollar_imbalance_bars_appending
        self.build_bars(get_ema_dollar_imbalance_bars_appending, output_path, data_source, **kwargs)

    def build_const_dollar_imbalance_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_const_dollar_imbalance_bars_appending
        self.build_bars(get_const_dollar_imbalance_bars_appending, output_path, data_source, **kwargs)



    def build_ema_tick_run_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_ema_tick_run_bars_appending
        self.build_bars(get_ema_tick_run_bars_appending, output_path, data_source, **kwargs)

    def build_const_tick_run_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_const_tick_run_bars_appending
        self.build_bars(get_const_tick_run_bars_appending, output_path, data_source, **kwargs)

    def build_ema_volume_run_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_ema_volume_run_bars_appending
        self.build_bars(get_ema_volume_run_bars_appending, output_path, data_source, **kwargs)

    def build_const_volume_run_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_const_volume_run_bars_appending
        self.build_bars(get_const_volume_run_bars_appending, output_path, data_source, **kwargs)

    def build_ema_dollar_run_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_ema_dollar_run_bars_appending
        self.build_bars(get_ema_dollar_run_bars_appending, output_path, data_source, **kwargs)

    def build_const_dollar_run_bars(self, output_path, data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_const_dollar_run_bars_appending
        self.build_bars(get_const_dollar_run_bars_appending, output_path, data_source, **kwargs)



    def build_time_bars(self, output_path, data_source="binance", **kwargs):
        """
        Builds time bars using the specialized decorator that converts date_time.
        """
        from mlfinlab.data_structures.bar_generators import get_time_bars_appending
        decorated_bar_func = self.format_dataframe_time(data_source)(get_time_bars_appending)
        self.readfile_list()
        if not self.file_list:
            print("No CSV files found.")
            return

        print(f"Found {len(self.file_list)} CSV file(s).")
        for file in self.file_list:
            decorated_bar_func(
                file,
                batch_size=self.batch_size,
                verbose=True,
                to_csv=True,
                output_path=output_path,
                **kwargs
            )



def generate_output_filename(inputfilepath, start_period, end_period, data_source, bar_type, **bar_params):
    """
    Generate a concise CSV file name that includes:
      - data_source (e.g. 'binance')
      - instrument (extracted from inputfilepath)
      - calendar period (start and end dates formatted as YYYYMMDD)
      - bar_type (e.g. 'time', 'tick', etc.)
      - bar-specific parameters (e.g. resolution, num_units, etc.)
    """
    path = Path(inputfilepath)
    # If the path ends with a slash, use the second-last part; otherwise, the last.
    instrument = path.parts[-2] if path.name == "" else path.name

    start_dt = datetime.strptime(start_period, "%Y-%m-%d")
    end_dt = datetime.strptime(end_period, "%Y-%m-%d")
    start_str = start_dt.strftime("%Y%m%d")
    end_str = end_dt.strftime("%Y%m%d")
    
    params_str = "_".join(f"{key}{value}" for key, value in bar_params.items() if value is not None)
    filename = f"{data_source}_{instrument}_{bar_type}_{start_str}-{end_str}"
    if params_str:
        filename += f"_{params_str}"
    filename += ".csv"
    return filename


def bar_pytest():
    """
    Intended to conduct a quick test on the csv output validity

    - excel check with tick num

    - time bar: utc regular intervals? -> OK (close price checked too)
        - also can compare with the distributed ones by Binance
    - dollar bar: match with binance raw data? -> have not confirmed yet (close price checked already)
    """
    return None


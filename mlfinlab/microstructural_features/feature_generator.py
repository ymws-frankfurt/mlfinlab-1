"""
Inter-bar feature generator which uses trades data and bars index to calculate inter-bar features
"""
import pandas as pd
import numpy as np
from itertools import chain

from mlfinlab.ymws.tick_data_formatter import TickDataFormatter
# from mlfinlab.ymws.KCA_raw import fitKCA
from mlfinlab.ymws.KCA_intra_posvelacc import generate_intra_kcapva

from mlfinlab.microstructural_features.entropy import get_shannon_entropy, get_plug_in_entropy, get_lempel_ziv_entropy_fast#, \
    #get_konto_entropy_nb
from mlfinlab.microstructural_features.encoding import encode_array
from mlfinlab.microstructural_features.second_generation import get_trades_based_kyle_lambda, \
    get_trades_based_amihud_lambda, get_trades_based_hasbrouck_lambda
from mlfinlab.microstructural_features.misc import get_avg_tick_size, vwap
from mlfinlab.microstructural_features.encoding import encode_tick_rule_array

# Import roll measure and roll impact functions from first_generation.py
from mlfinlab.microstructural_features.first_generation import get_roll_measure, get_roll_impact
from mlfinlab.microstructural_features.ww_runs_test import runs_z_score
from mlfinlab.features.fracdiff_opt import optimal_frac_diff
from mlfinlab.util.fast_ewma import ewma

from mlfinlab.structural_breaks.chow import get_chow_type_stat
from mlfinlab.structural_breaks.cusum import get_chu_stinchcombe_white_statistics
from mlfinlab.structural_breaks.sadf import get_sadf

from mlfinlab.util.misc import crop_data_frame_in_batches

# from concurrent.futures import ProcessPoolExecutor, as_completed
# Import the compute_features_for_bar function from your parallel_feature_extraction.py module
# from mlfinlab.microstructural_features.parallel_feature_extraction import compute_features_for_bar


# pylint: disable=too-many-instance-attributes


class MicrostructuralFeaturesGenerator:
    """
    Class which is used to generate inter-bar features when bars are already compressed.

    :param trades_input: (str or pd.DataFrame) Path to the csv file or Pandas DataFrame containing raw tick data
                                               in the format[date_time, price, volume]
    :param tick_num_series: (pd.Series) Series of tick number where bar was formed.
    :param batch_size: (int) Number of rows to read in from the csv, per batch.
    :param volume_encoding: (dict) Dictionary of encoding scheme for trades size used to calculate entropy on encoded messages
    :param pct_encoding: (dict) Dictionary of encoding scheme for log returns used to calculate entropy on encoded messages

    -> ### In case of no parallelization, this class is to be used (ymws), 
    in which case features (e.g. entropy, 1st generation, KCA, SB, WW, sumFFD) are rightly imported into this file as well ad to parallel_feature_extraction.py

    """

    def __init__(self, trades_input: (str, list, pd.DataFrame), tick_num_series: pd.Series, batch_size: int = 2e7,
                 volume_encoding: dict = None, pct_encoding: dict = None, data_source="binance", roll_window=1000,
                 # New parameters for intra–bar features:
                 intra_kca_fwd: list = None,     # list of forecast horizons for intra features
                 intra_kca_q: float = 0.001   # process noise parameter for KCA
                 ):
        """
        Constructor

        :param trades_input: (str or pd.DataFrame) Path to the csv file, a list of csv file paths or Pandas DataFrame containing raw tick data
                                                   in the format[date_time, price, volume]
        :param tick_num_series: (pd.Series) Series of tick number where bar was formed.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param volume_encoding: (dict) Dictionary of encoding scheme for trades size used to calculate entropy on encoded messages
        :param pct_encoding: (dict) Dictionary of encoding scheme for log returns used to calculate entropy on encoded messages
        :param data_source: (str) Identifier for the data source. (o3-mini-high)
        :param intra_kca_fwd: List of forecast horizon(s) for intra-bar features.
                          For example: [1, 10, 100, 250, 1000]. Defaults to [0] if not provided.
        :param intra_kca_q: Scalar to seed process noise for intra-KCA.        
        """
        self.tick_num_series = tick_num_series
        self.batch_size = int(batch_size)
        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding
        self.data_source = data_source
        self.roll_window = roll_window  # Expose the roll window size as a parameter

        # Initialize the formatter.
        self.formatter = TickDataFormatter(data_source=self.data_source)

        # Accept trades_input as a single file path, list of file paths, or a DataFrame.
        if isinstance(trades_input, str):
            self.generator_object = self.formatter.load_and_format_dataframe_in_batches(trades_input, self.batch_size)
        elif isinstance(trades_input, list):
            # Create a chained generator that loops over each file's batches sequentially.
            generators = []
            for file in trades_input:
                generators.append(self.formatter.load_and_format_dataframe_in_batches(str(file), self.batch_size))
            self.generator_object = chain(*generators)
        elif isinstance(trades_input, pd.DataFrame):
            self.generator_object = crop_data_frame_in_batches(trades_input, self.batch_size)
        else:
            raise ValueError('trades_input must be a file path, list of file paths, or a pandas DataFrame')
                
        # Setup tick number generator.
        self.tick_num_generator = iter(tick_num_series)
        self.current_bar_tick_num = next(self.tick_num_generator)

        # Initialize caches for features.
        self.price_diff = []
        self.trade_size = []
        self.tick_rule = []
        self.dollar_size = []
        self.log_ret = []
        self.prev_price = None
        self.prev_tick_rule = 0
        self.tick_num = 0

        # NEW: Cache to store raw tick prices for computing roll measure/impact.
        self.cum_prices = []

        #self.entropy_types = ['shannon', 'plug_in', 'lempel_ziv', 'konto']
        self.entropy_types = ['shannon', 'plug_in', 'lempel_ziv']

        # Batch_run properties
        self.prev_price = None
        self.prev_tick_rule = 0
        self.tick_num = 0

        # ---- New intra–bar parameters ----
        if intra_kca_fwd is None:
            self.intra_kca_fwd = [0]  # No forecast by default – or you can choose a default list.
        else:
            self.intra_kca_fwd = intra_kca_fwd
        self.intra_kca_q = intra_kca_q

        # NEW: Allow intra_kca_q to be provided as a list; if not, convert it to a single‐element list.
        if not isinstance(intra_kca_q, list):
            self.intra_kca_q = [intra_kca_q]
        else:
            self.intra_kca_q = intra_kca_q

    # --------------------------------------------------------------------------
    def _get_intra_kcafeatures(self, tick_series):
        """
        Compute intra–bar KCA features from the tick series (signed ticks)
        using generate_intra_kcapva from KCA_intra_posvelacc.py.
        
        Handles multiple values of the process noise parameter (q) by looping over each
        value in self.intra_kca_q and each forecast horizon in self.intra_kca_fwd.
        :param tick_series: List or array of signed ticks for the current bar.
        :return: Dictionary of intra features with keys such as 
                 'intra_kca_position_fwd_{f}_q_{q}', etc.
        """
        import numpy as np
        if len(tick_series) == 0:
            result = {}
            for q_val in self.intra_kca_q:
                for f in self.intra_kca_fwd:
                    for key in ['position', 'velocity', 'acceleration', 
                                'position_std', 'velocity_std', 'acceleration_std',
                                'position_t', 'velocity_t', 'acceleration_t']:
                        result[f"intra_kca_{key}_fwd_{f}_q_{q_val}"] = np.nan
            return result
        
        price_series = np.array(tick_series, dtype=float)
        n = len(price_series)
        # --- GUARD AGAINST TOO‐SHORT SERIES (avoids EM division by zero) ---
        if n <= 1:
            result = {}
            for q_val in self.intra_kca_q:
                for f in self.intra_kca_fwd:
                    for key in ['position', 'velocity', 'acceleration',
                                'position_std', 'velocity_std', 'acceleration_std',
                                'position_t', 'velocity_t', 'acceleration_t']:
                        result[f"intra_kca_{key}_fwd_{f}_q_{q_val}"] = np.nan
            return result
        # --- END GUARD ---
        result = {}
        
        # For each process noise parameter and forecast horizon, call KCA.
        for q_val in self.intra_kca_q:
            # For consistency, determine the maximum forecast horizon needed for this q.
            forecast_list = [f for f in self.intra_kca_fwd if f > 0]
            call_fwd = max(forecast_list) if forecast_list else 0

            # Call generate_intra_kcapva with the current q_val.
            features = generate_intra_kcapva(price_series, q_val, forecast_steps=call_fwd)
            
            for f in self.intra_kca_fwd:
                if f == 0:
                    idx = n - 1
                else:
                    idx = n + f - 1
                result[f"intra_kca_position_fwd_{f}_q_{q_val}"] = features['position'][idx]
                result[f"intra_kca_velocity_fwd_{f}_q_{q_val}"] = features['velocity'][idx]
                result[f"intra_kca_acceleration_fwd_{f}_q_{q_val}"] = features['acceleration'][idx]
                result[f"intra_kca_position_std_fwd_{f}_q_{q_val}"] = features['position_std'][idx]
                result[f"intra_kca_velocity_std_fwd_{f}_q_{q_val}"] = features['velocity_std'][idx]
                result[f"intra_kca_acceleration_std_fwd_{f}_q_{q_val}"] = features['acceleration_std'][idx]
                result[f"intra_kca_position_t_fwd_{f}_q_{q_val}"] = features['position_t'][idx]
                result[f"intra_kca_velocity_t_fwd_{f}_q_{q_val}"] = features['velocity_t'][idx]
                result[f"intra_kca_acceleration_t_fwd_{f}_q_{q_val}"] = features['acceleration_t'][idx]
        return result


    def get_features(self, verbose=True, to_csv=False, output_path=None):
        """
        Reads a csv file of ticks or pd.DataFrame in batches and then constructs corresponding microstructural intra-bar features:
        average tick size, tick rule sum, VWAP, Kyle lambda, Amihud lambda, Hasbrouck lambda, tick/volume/pct Shannon, Lempel-Ziv,
        Plug-in entropies if corresponding mapping dictionaries are provided (self.volume_encoding, self.pct_encoding).
        The csv file must have only 3 columns: date_time, price, & volume.

        :param verbose: (bool) Flag whether to print message on each processed batch or not
        :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
        :param output_path: (bool) Path to results file, if to_csv = True
        :return: (DataFrame or None) Microstructural features for bar index
        """
        # Define base columns
        cols = [
            'date_time', 
            'avg_tick_size', 
            'tick_rule_sum', 
            'vwap',
            'roll_measure',
            'roll_impact',            
            'kyle_lambda', 
            'kyle_lambda_t_value', 
            'amihud_lambda', 
            'amihud_lambda_t_value',
            'hasbrouck_lambda', 
            'hasbrouck_lambda_t_value',
            'runs_z_score',
        ]
        
        # Extend columns with entropy features for tick_rule
        for en_type in self.entropy_types:
            cols += ['tick_rule_entropy_' + en_type]

        # Extend columns for volume encoding if provided
        if self.volume_encoding is not None:
            for en_type in self.entropy_types:
                cols += ['volume_entropy_' + en_type]

        # Extend columns for percentage encoding if provided
        if self.pct_encoding is not None:
            for en_type in self.entropy_types:
                cols += ['pct_entropy_' + en_type]

      # --- NEW: Extend columns for intra–KCA features ---
        intra_keys = [
            'position','velocity','acceleration',
            'position_std','velocity_std','acceleration_std',
            'position_t','velocity_t','acceleration_t'
        ]
        for q_val in self.intra_kca_q:
            for f in self.intra_kca_fwd:
                for key in intra_keys:
                    cols.append(f"intra_kca_{key}_fwd_{f}_q_{q_val}")

        # Now cols has exactly as many names as values in each `features` list.


        # If output is to be written to CSV, prepare the file
        if to_csv:
            header = True
            open(output_path, 'w').close()  # Clear any previous content
        else:
            final_bars = []

        count = 0

        # Process each batch from the generator_object
        for batch in self.generator_object:
            if verbose:
                print('Processing batch:', count)
            
            # Process the current batch to extract bar features
            list_bars, stop_flag = self._extract_bars(batch)
            
            if to_csv:
                pd.DataFrame(list_bars, columns=cols).to_csv(
                    output_path, header=header, index=False, mode='a'
                )
                header = False  # Only write header for the first batch
            else:
                final_bars += list_bars
            
            count += 1
            
            # If _extract_bars signals no more data is needed, break early
            if stop_flag:
                break

        # Return the combined DataFrame if not writing to CSV
        if not to_csv and final_bars:
            return pd.DataFrame(final_bars, columns=cols)
        
        return None


    def _reset_cache(self):
        """
        Reset price_diff, trade_size, tick_rule, log_ret arrays to empty when bar is formed and features are
        calculated

        :return: None
        """
        self.price_diff = []
        self.trade_size = []
        self.tick_rule = []
        self.dollar_size = []
        self.log_ret = []
        self.cum_prices = []  # Reset cum_prices cache

    def _extract_bars(self, data):
        """
        For loop which calculates features for formed bars using trades data
    
        :param data: (tuple) Contains 3 or 4 columns – date_time, price, volume, and optionally isBuyerMaker.
        """

        # Iterate over rows
        list_bars = []

        for row in data.values:
            # Set variables
            date_time = row[0]
            price = float(row[1])
            volume = row[2]
            dollar_value = price * volume

            # Append current tick price to the cum_prices cache (NEW)
            self.cum_prices.append(price)

            # If using binance data and the isBuyerMaker flag is provided (4th column), override the tick rule.
            if self.data_source == "binance" and len(row) > 3:
                isBuyerMaker = row[3]
                # Convert string representations to boolean if needed.
                if isinstance(isBuyerMaker, str):
                    isBuyerMaker = isBuyerMaker.lower() == 'true'
                # Determine trade direction: True => seller-initiated (-1), False => buyer-initiated (+1)
                signed_tick = -1 if isBuyerMaker else 1
            else:
                signed_tick = self._apply_tick_rule(price)

            # signed_tick = self._apply_tick_rule(price)

            self.tick_num += 1

            # Derivative variables
            price_diff = self._get_price_diff(price)
            log_ret = self._get_log_ret(price)

            self.price_diff.append(price_diff)
            self.trade_size.append(volume)
            self.tick_rule.append(signed_tick)
            self.dollar_size.append(dollar_value)
            self.log_ret.append(log_ret)

            self.prev_price = price

            # If date_time reached bar index
            if self.tick_num >= self.current_bar_tick_num:
                self._get_bar_features(date_time, list_bars)

                # Take the next tick number
                try:
                    # self.current_bar_tick_num = self.tick_num_generator.__next__()
                    self.current_bar_tick_num = next(self.tick_num_generator)
                except StopIteration:
                    return list_bars, True  # Looped through all bar index
                # Reset cache
                self._reset_cache()
        return list_bars, False

    def _get_bar_features(self, date_time: pd.Timestamp, list_bars: list) -> list:
        """
        Calculate inter-bar features: lambdas, entropies, avg_tick_size, vwap

        :param date_time: (pd.Timestamp) When bar was formed
        :param list_bars: (list) Previously formed bars
        :return: (list) Inter-bar features
        """
        features = [date_time]

        # Tick rule sum, avg tick size, VWAP
        features.append(get_avg_tick_size(self.trade_size))
        features.append(sum(self.tick_rule))
        features.append(vwap(self.dollar_size, self.trade_size))

        # NEW: Compute roll measure and roll impact from tick data.
        # We create pandas Series from the cum_prices and dollar sizes lists.
        cum_prices_series = pd.Series(self.cum_prices)
        dollar_series = pd.Series(self.dollar_size)
        # Compute roll measure and roll impact from tick data using self.roll_window as default. If not enough ticks, assign adaptive_window (xNaN)
        if len(cum_prices_series) >= self.roll_window:
            roll_measure_series = get_roll_measure(cum_prices_series, window=self.roll_window)
            roll_impact_series = get_roll_impact(cum_prices_series, dollar_series, window=self.roll_window)

        else:
            # Adaptive window: use the entire available series -> https://chatgpt.com/c/67bcdab2-7c14-8000-bd1b-116302697bc8
            adaptive_window = len(cum_prices_series)
            roll_measure_series = get_roll_measure(cum_prices_series, window=adaptive_window)
            roll_impact_series = get_roll_impact(cum_prices_series, dollar_series, window=adaptive_window)

        # <Disable fixed window approach>
        # else:
        #     roll_measure_val = np.nan
        #     roll_impact_val = np.nan

        # <Disable dynamic window approach> -> all NaN
        # roll_measure_series = get_roll_measure(cum_prices_series, window=len(cum_prices_series))
        # roll_impact_series = get_roll_impact(cum_prices_series, dollar_series, window=len(cum_prices_series))

        # Take the last computed value (most recent)
        roll_measure_val = roll_measure_series.iloc[-1]
        roll_impact_val = roll_impact_series.iloc[-1]
        
        features.append(roll_measure_val)
        features.append(roll_impact_val)

        # Lambdas (using trades-based second generation functions)
        features.extend(get_trades_based_kyle_lambda(self.price_diff, self.trade_size, self.tick_rule))  # Kyle lambda
        features.extend(get_trades_based_amihud_lambda(self.log_ret, self.dollar_size))  # Amihud lambda
        features.extend(get_trades_based_hasbrouck_lambda(self.log_ret, self.dollar_size, self.tick_rule))  # Hasbrouck lambda

        # NEW: Compute the runs test z-score using the tick_rule series for the current bar.
        runs_z = runs_z_score(self.tick_rule)
        features.append(runs_z)

        # Entropy features for tick rule
        encoded_tick_rule_message = encode_tick_rule_array(self.tick_rule)
        features.append(get_shannon_entropy(encoded_tick_rule_message))
        features.append(get_plug_in_entropy(encoded_tick_rule_message))
        features.append(get_lempel_ziv_entropy_fast(encoded_tick_rule_message))
        #features.append(get_konto_entropy_nb(encoded_tick_rule_message))

        # Entropy features for volume encoding if provided
        if self.volume_encoding is not None:
            message = encode_array(self.trade_size, self.volume_encoding)
            features.append(get_shannon_entropy(message))
            features.append(get_plug_in_entropy(message))
            features.append(get_lempel_ziv_entropy_fast(message))
            #features.append(get_konto_entropy_nb(message))

        # Entropy features for percentage encoding if provided
        if self.pct_encoding is not None:
            message = encode_array(self.log_ret, self.pct_encoding)
            features.append(get_shannon_entropy(message))
            features.append(get_plug_in_entropy(message))
            features.append(get_lempel_ziv_entropy_fast(message))
            #features.append(get_konto_entropy_nb(message))

        # ---- NEW: Compute intra–bar features from the accumulated signed tick series.
        intra_feats = self._get_intra_kcafeatures(self.tick_rule)
        features.extend(list(intra_feats.values()))
        
        # Reset the tick_rule cache for the next bar. -> DISABLED
        # self.tick_rule = []

        # Return or store the complete features row. -> DISABLED
        # return bar_features

        list_bars.append(features)

    def _apply_tick_rule(self, price: float) -> int:
        """
        Advances in Financial Machine Learning, page 29.

        Applies the tick rule

        :param price: (float) Price at time t
        :return: (int) The signed tick
        """
        if self.prev_price is not None:
            tick_diff = price - self.prev_price
        else:
            tick_diff = 0

        if tick_diff != 0:
            signed_tick = np.sign(tick_diff)
            self.prev_tick_rule = signed_tick
        else:
            signed_tick = self.prev_tick_rule

        return signed_tick

    def _get_price_diff(self, price: float) -> float:
        """
        Get price difference between ticks

        :param price: (float) Price at time t
        :return: (float) Price difference
        """
        if self.prev_price is not None:
            price_diff = price - self.prev_price
        else:
            price_diff = 0  # First diff is assumed 0
        return price_diff

    def _get_log_ret(self, price: float) -> float:
        """
        Get log return between ticks

        :param price: (float) Price at time t
        :return: (float) Log return
        """
        if self.prev_price is not None:
            log_ret = np.log(price / self.prev_price)
        else:
            log_ret = 0  # First return is assumed 0
        return log_ret

    @staticmethod
    def _assert_csv(test_batch):
        """
        Tests that the csv file read has the format: date_time, price, and volume.
        or date_time, price, volume, isBuyerMaker (for binance data).
        If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

        :param test_batch: (pd.DataFrame) the first row of the dataset.
        :return: (None)
        """
        # assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
        assert test_batch.shape[1] in (3, 4), 'CSV must have 3 columns (date_time, price, volume) or 4 columns (with isBuyerMaker for binance).'
        assert isinstance(test_batch.iloc[0, 1], float), 'price column in csv not float.'
        assert not isinstance(test_batch.iloc[0, 2], str), 'volume column in csv not int or float.'

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            print('csv file, column 0, not a date time format:',
                  test_batch.iloc[0, 0])

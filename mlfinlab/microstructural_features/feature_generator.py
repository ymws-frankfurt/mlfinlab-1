"""
Inter-bar feature generator which uses trades data and bars index to calculate inter-bar features
"""
import pandas as pd
import numpy as np

from mlfinlab.ymws.tick_data_formatter import TickDataFormatter
from mlfinlab.ymws.KCA_composite import fitKCA

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

    def __init__(self, trades_input: (str, pd.DataFrame), tick_num_series: pd.Series, batch_size: int = 2e7,
                 volume_encoding: dict = None, pct_encoding: dict = None, data_source="binance", roll_window=1000):
        """
        Constructor

        :param trades_input: (str or pd.DataFrame) Path to the csv file or Pandas DataFrame containing raw tick data
                                                   in the format[date_time, price, volume]
        :param tick_num_series: (pd.Series) Series of tick number where bar was formed.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param volume_encoding: (dict) Dictionary of encoding scheme for trades size used to calculate entropy on encoded messages
        :param pct_encoding: (dict) Dictionary of encoding scheme for log returns used to calculate entropy on encoded messages
        :param data_source: (str) Identifier for the data source. (o3-mini-high)
        """
        self.tick_num_series = tick_num_series
        self.batch_size = batch_size
        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding
        self.data_source = data_source
        self.roll_window = roll_window  # Expose the roll window size as a parameter

        # Initialize the formatter.
        self.formatter = TickDataFormatter(data_source=self.data_source)

        # If trades_input is a file path, use batch processing.
        if isinstance(trades_input, str):
            self.generator_object = self.formatter.load_and_format_dataframe_in_batches(trades_input, self.batch_size)
        elif isinstance(trades_input, pd.DataFrame):
            # If it's already a DataFrame, you might decide whether to batch it:
            self.generator_object = crop_data_frame_in_batches(trades_input, self.batch_size)
        else:
            raise ValueError('trades_input must be a file path or a pandas DataFrame')
        
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

        # Entropy properties
        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding
        #self.entropy_types = ['shannon', 'plug_in', 'lempel_ziv', 'konto']
        self.entropy_types = ['shannon', 'plug_in', 'lempel_ziv']

        # Batch_run properties
        self.prev_price = None
        self.prev_tick_rule = 0
        self.tick_num = 0

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

        # NEW: Compute the runs test z-score using the raw tick prices for the current bar.
        runs_z = runs_z_score(self.cum_prices)
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



'''class MicrostructuralFeaturesGeneratorParallelBatched:
    """
    This class integrates batch loading/formatting of tick data with parallel feature extraction.
    It uses TickDataFormatter to read the CSV in chunks, accumulates rows until a bar boundary is reached,
    and then processes each bar concurrently to compute intra-bar features.
    """
    def __init__(self, file_path, data_source, tick_num_series, batch_size=10000, volume_encoding=None, pct_encoding=None):
        """
        :param file_path: Path to the raw tick CSV file.
        :param data_source: Identifier for the data source (e.g., 'binance') that determines column mapping.
        :param tick_num_series: A pandas Series containing the tick boundaries where a bar is formed (absolute tick counts).
        :param batch_size: Number of rows to read per batch (adjust based on available memory).
        :param volume_encoding: Optional encoding dictionary for volume (for additional feature computation).
        :param pct_encoding: Optional encoding dictionary for log returns.
        """
        self.file_path = file_path
        self.data_source = data_source
        # Convert tick boundaries to a list for easier sequential processing.
        self.tick_boundaries = list(tick_num_series)
        self.batch_size = batch_size
        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding
        # Initialize the TickDataFormatter using the given data source.
        self.formatter = TickDataFormatter(data_source=data_source)
        
    def get_features_parallel(self):
        """
        Reads the tick data in batches, accumulates rows until bar boundaries (tick counts) are met,
        splits the accumulated data into individual bars, and computes features for each bar in parallel.
        
        :return: A pandas DataFrame with computed intra-bar features.
        """
        # List to store each complete bar as a DataFrame.
        bar_list = []
        # Accumulator to hold rows across batches.
        accumulator = pd.DataFrame(columns=['date_time', 'price', 'volume'])
        
        # Create a generator that yields formatted DataFrame batches.
        batch_generator = self.formatter.load_and_format_dataframe_in_batches(self.file_path, self.batch_size)
        
        # Process each batch.
        for batch in batch_generator:
            # Reset index for consistent concatenation.
            batch = batch.reset_index(drop=True)
            # Append the new batch to the accumulator.
            accumulator = pd.concat([accumulator, batch], ignore_index=True)
            
            # While the accumulator contains enough rows for the next bar:
            while self.tick_boundaries and len(accumulator) >= self.tick_boundaries[0]:
                # The first tick boundary specifies the number of ticks in the bar.
                boundary = self.tick_boundaries.pop(0)
                # Extract the bar: the first 'boundary' rows from the accumulator.
                bar_df = accumulator.iloc[:boundary].copy()
                bar_list.append(bar_df)
                # Remove the processed rows from the accumulator.
                accumulator = accumulator.iloc[boundary:].reset_index(drop=True)
        
        # Optionally, process any leftover rows as a final (partial) bar.
        if not accumulator.empty:
            bar_list.append(accumulator)
        
        # Now, process each bar in parallel using ProcessPoolExecutor.
        features_list = []
        with ProcessPoolExecutor() as executor:
            # Submit each bar for feature computation.
            futures = {executor.submit(compute_features_for_bar, bar, self.volume_encoding, self.pct_encoding): bar for bar in bar_list}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    features_list.append(result)
                except Exception as e:
                    print(f"Error processing a bar: {e}")
        
        # Convert the list of feature dictionaries to a DataFrame.
        features_df = pd.DataFrame(features_list)
        return features_df'''
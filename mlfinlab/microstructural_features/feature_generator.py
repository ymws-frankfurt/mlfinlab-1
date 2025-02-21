"""
Inter-bar feature generator which uses trades data and bars index to calculate inter-bar features
"""
import pandas as pd
import numpy as np
from mlfinlab.microstructural_features.entropy import get_shannon_entropy, get_plug_in_entropy, get_lempel_ziv_entropy, \
    get_konto_entropy
from mlfinlab.microstructural_features.encoding import encode_array
from mlfinlab.microstructural_features.second_generation import get_trades_based_kyle_lambda, \
    get_trades_based_amihud_lambda, get_trades_based_hasbrouck_lambda
from mlfinlab.microstructural_features.misc import get_avg_tick_size, vwap
from mlfinlab.microstructural_features.encoding import encode_tick_rule_array

from tick_data_formatter import TickDataFormatter
from mlfinlab.util.misc import crop_data_frame_in_batches


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

    """

    def __init__(self, trades_input: (str, pd.DataFrame), tick_num_series: pd.Series, batch_size: int = 2e7,
                 volume_encoding: dict = None, pct_encoding: dict = None, data_source="binance"):
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
        self.data_source = data_source
        self.batch_size = batch_size
        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding
        self.tick_num_series = tick_num_series

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

    # The rest of your methods remain largely unchanged.
    # They will process each batch yielded by self.generator_object.

        # Entropy properties
        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding
        self.entropy_types = ['shannon', 'plug_in', 'lempel_ziv', 'konto']

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
            'kyle_lambda', 
            'kyle_lambda_t_value', 
            'amihud_lambda', 
            'amihud_lambda_t_value',
            'hasbrouck_lambda', 
            'hasbrouck_lambda_t_value'
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

    def _extract_bars(self, data):
        """
        For loop which calculates features for formed bars using trades data

        :param data: (tuple) Contains 3 columns - date_time, price, and volume.
        """

        # Iterate over rows
        list_bars = []

        for row in data.values:
            # Set variables
            date_time = row[0]
            price = float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

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
                    self.current_bar_tick_num = self.tick_num_generator.__next__()
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

        # Lambdas
        features.extend(get_trades_based_kyle_lambda(self.price_diff, self.trade_size, self.tick_rule))  # Kyle lambda
        features.extend(get_trades_based_amihud_lambda(self.log_ret, self.dollar_size))  # Amihud lambda
        features.extend(
            get_trades_based_hasbrouck_lambda(self.log_ret, self.dollar_size, self.tick_rule))  # Hasbrouck lambda

        # Entropy features
        encoded_tick_rule_message = encode_tick_rule_array(self.tick_rule)
        features.append(get_shannon_entropy(encoded_tick_rule_message))
        features.append(get_plug_in_entropy(encoded_tick_rule_message))
        features.append(get_lempel_ziv_entropy(encoded_tick_rule_message))
        features.append(get_konto_entropy(encoded_tick_rule_message))

        if self.volume_encoding is not None:
            message = encode_array(self.trade_size, self.volume_encoding)
            features.append(get_shannon_entropy(message))
            features.append(get_plug_in_entropy(message))
            features.append(get_lempel_ziv_entropy(message))
            features.append(get_konto_entropy(message))

        if self.pct_encoding is not None:
            message = encode_array(self.log_ret, self.pct_encoding)
            features.append(get_shannon_entropy(message))
            features.append(get_plug_in_entropy(message))
            features.append(get_lempel_ziv_entropy(message))
            features.append(get_konto_entropy(message))

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
        If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

        :param test_batch: (pd.DataFrame) the first row of the dataset.
        :return: (None)
        """
        assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
        assert isinstance(test_batch.iloc[0, 1], float), 'price column in csv not float.'
        assert not isinstance(test_batch.iloc[0, 2], str), 'volume column in csv not int or float.'

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            print('csv file, column 0, not a date time format:',
                  test_batch.iloc[0, 0])

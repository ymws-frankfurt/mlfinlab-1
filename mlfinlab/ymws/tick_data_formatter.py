import pandas as pd

class TickDataFormatter:
    """
    A class responsible for reading and formatting raw tick data CSV files.
    The formatted output is a DataFrame with columns: [date_time, price, volume].
    """
    def __init__(self, data_source: str, column_mappings: dict = None, column_positions: dict = None):
        """
        Initialize the formatter with mappings for the given data source.
        
        :param data_source: Identifier for the data source (e.g. 'binance', 'oanda').
        :param column_mappings: A dict mapping original column names to standard names.
        :param column_positions: A dict mapping column positions to standard names for headerless files.
        """
        self.data_source = data_source
        # Provide default mappings if none are supplied.
        self.column_mappings = column_mappings or {
            "binance": {"timestamp": "date_time", "trade_price": "price", "trade_volume": "volume"},
            "oanda": {"time": "date_time", "ask": "price", "size": "volume"}
        }
        self.column_positions = column_positions or {
            "binance": {4: "date_time", 1: "price", 2: "volume"}
        }

    def detect_header(self, file_path: str) -> bool:
        """
        Determines whether the CSV file at file_path contains a header.
        Returns True if a header is detected, False otherwise.
        """
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    first_line = line.strip()
                    break
            else:
                first_line = ""
        tokens = [token.strip() for token in first_line.split(',') if token.strip()]

        def is_data_token(token):
            try:
                float(token)
                return True
            except ValueError:
                # Allow boolean tokens as well.
                return token.lower() in {"true", "false"}

        # If every token can be interpreted as a number or boolean, assume headerless.
        return not (tokens and all(is_data_token(token) for token in tokens))

    def load_and_format_dataframe(self, file_path: str) -> pd.DataFrame:
        """
        Loads a CSV file, extracts the relevant columns, renames them, and reorders to [date_time, price, volume].
        
        :param file_path: Path to the CSV file.
        :return: A formatted pandas DataFrame.
        """
        has_header = self.detect_header(file_path)
        if has_header:
            mapping = self.column_mappings.get(self.data_source)
            if mapping is None:
                raise ValueError(f"No column mapping defined for data source '{self.data_source}'")
            usecols = list(mapping.keys())
            df = pd.read_csv(file_path, usecols=usecols, sep=",")
            df = df.rename(columns=mapping)
        else:
            positions = self.column_positions.get(self.data_source)
            if positions is None:
                raise ValueError(f"No column position mapping defined for data source '{self.data_source}'")
            usecols = list(positions.keys())
            df = pd.read_csv(file_path, usecols=usecols, header=None, sep=",")
            df = df.rename(columns=positions)
        # Reorder columns to the expected format.
        df = df[['date_time', 'price', 'volume']]
        return df


    def load_and_format_dataframe_in_batches(self, file_path: str, batch_size: int):
        """
        Loads a CSV file in batches (chunks), extracts the relevant columns,
        renames them, and reorders to [date_time, price, volume].
        
        :param file_path: Path to the CSV file.
        :param batch_size: Number of rows per batch.
        :return: A generator yielding formatted pandas DataFrames.
        """
        has_header = self.detect_header(file_path)
        if has_header:
            mapping = self.column_mappings.get(self.data_source)
            if mapping is None:
                raise ValueError(f"No column mapping defined for data source '{self.data_source}'")
            usecols = list(mapping.keys())
            reader = pd.read_csv(file_path, usecols=usecols, sep=",", chunksize=batch_size)
            for chunk in reader:
                chunk = chunk.rename(columns=mapping)
                yield chunk[['date_time', 'price', 'volume']]
        else:
            positions = self.column_positions.get(self.data_source)
            if positions is None:
                raise ValueError(f"No column position mapping defined for data source '{self.data_source}'")
            usecols = list(positions.keys())
            reader = pd.read_csv(file_path, usecols=usecols, header=None, sep=",", chunksize=batch_size)
            for chunk in reader:
                chunk = chunk.rename(columns=positions)
                yield chunk[['date_time', 'price', 'volume']]
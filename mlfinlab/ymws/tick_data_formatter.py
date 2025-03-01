import pandas as pd

class TickDataFormatter:
    """
    A class responsible for reading and formatting raw tick data CSV files.
    
    By default, the formatted output is a DataFrame with columns: [date_time, price, volume].
    For data_source 'binance', if preserve_aggressor is True, it will include the 'isBuyerMaker' column.
    """
    def __init__(self, data_source: str, preserve_aggressor: bool = True, column_mappings: dict = None, column_positions: dict = None):
        """
        Initialize the formatter.
        
        :param data_source: Identifier for the data source (e.g. 'binance', 'oanda').
        :param preserve_aggressor: If True and data_source is 'binance', include the 'isBuyerMaker' column. -> default True (ymws)
        :param column_mappings: Optional dict mapping original column names to standard names.
        :param column_positions: Optional dict mapping column positions to standard names for headerless files.
        """
        self.data_source = data_source
        self.preserve_aggressor = preserve_aggressor
        
        if data_source == "binance":
            if preserve_aggressor:
                self.column_mappings = column_mappings or {
                    "binance": {"time": "date_time", "price": "price", "qty": "volume", "isBuyerMaker": "isBuyerMaker"}
                }
                self.column_positions = column_positions or {
                    "binance": {4: "date_time", 1: "price", 2: "volume", 5: "isBuyerMaker"}
                }
            else:
                self.column_mappings = column_mappings or {
                    "binance": {"time": "date_time", "price": "price", "qty": "volume"}
                }
                self.column_positions = column_positions or {
                    "binance": {4: "date_time", 1: "price", 2: "volume"}
                }
        elif data_source == "oanda":
            self.column_mappings = column_mappings or {
                "oanda": {"time": "date_time", "ask": "price", "size": "volume"}
            }
            self.column_positions = column_positions or {
                "oanda": {}
            }
        else:
            # For other data sources, users can supply their own mappings.
            self.column_mappings = column_mappings or {}
            self.column_positions = column_positions or {}

    def detect_header(self, file_path: str) -> bool:
        """
        Determines whether the CSV file at file_path contains a header.
        
        :param file_path: Path to the CSV file.
        :return: True if a header is detected, False otherwise.
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
                return token.lower() in {"true", "false"}

        return not (tokens and all(is_data_token(token) for token in tokens))

    def load_and_format_dataframe(self, file_path: str) -> pd.DataFrame:
        """
        Loads a CSV file, extracts the relevant columns, renames them, and reorders them.
        
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
        
        # Reorder columns: include aggressor if requested and available.
        if self.data_source == "binance" and self.preserve_aggressor and "isBuyerMaker" in df.columns:
            df = df[['date_time', 'price', 'volume', 'isBuyerMaker']]
        else:
            df = df[['date_time', 'price', 'volume']]
        return df

    def load_and_format_dataframe_in_batches(self, file_path: str, batch_size: int):
        """
        Loads a CSV file in batches (chunks) and formats each batch.
        
        :param file_path: Path to the CSV file.
        :param batch_size: Number of rows per batch.
        :yield: Formatted pandas DataFrames for each batch.
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
                if self.data_source == "binance" and self.preserve_aggressor and "isBuyerMaker" in chunk.columns:
                    yield chunk[['date_time', 'price', 'volume', 'isBuyerMaker']]
                else:
                    yield chunk[['date_time', 'price', 'volume']]
        else:
            positions = self.column_positions.get(self.data_source)
            if positions is None:
                raise ValueError(f"No column position mapping defined for data source '{self.data_source}'")
            usecols = list(positions.keys())
            reader = pd.read_csv(file_path, usecols=usecols, header=None, sep=",", chunksize=batch_size)
            for chunk in reader:
                chunk = chunk.rename(columns=positions)
                if self.data_source == "binance" and self.preserve_aggressor and "isBuyerMaker" in chunk.columns:
                    yield chunk[['date_time', 'price', 'volume', 'isBuyerMaker']]
                else:
                    yield chunk[['date_time', 'price', 'volume']]

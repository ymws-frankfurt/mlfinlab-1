import pandas as pd
import numpy as np
import gzip
import io
from dateutil.parser import parse as _parse_dt

# ---------------------------------------------------------------------
#  (A)  *** THE ONLY COPY of column maps / positions lives right here ***
# ---------------------------------------------------------------------
COLUMN_MAPPINGS = {
    # Header‑based renaming
    "binance": {
        # "timestamp": "date_time", # For AggTrade https://chatgpt.com/share/681d3aa4-f344-8000-9769-ab5186c89a44
        "time":      "date_time",   # headered Spot dumps (default)
        "price": "price",
        "qty": "volume",
        "isBuyerMaker": "isBuyerMaker",
    },
    "gmocoin": {
        "timestamp": "date_time",   # e.g. 2025‑05‑01 09:00:00.123
        "price": "price",
        "size": "volume",
        "side": "side",
    },
}

COLUMN_POSITIONS = {
    # Position‑based renaming (CSV without header)
    "binance": {
        4: "date_time",
        1: "price",
        2: "volume",
        5: "isBuyerMaker",
    },
    "gmocoin": {
        5: "date_time",
        4: "price",
        3: "volume",
        2: "side",
    },
}

class TickDataFormatter:
    """
    A class responsible for reading and formatting raw tick data CSV files.
    
    Turns raw trade CSVs from multiple venues into one canonical layout:
    [date_time, price, volume]  (+ optional aggressor flag)
    """
    def __init__(self,
                data_source: str,
                preserve_aggressor: bool = True,
                column_mappings: dict | None = None,
                column_positions: dict | None = None):        
        """
        Initialize the formatter.
        
        :param data_source: Identifier for the data source (e.g. 'binance', 'oanda').
        :param preserve_aggressor: If True and data_source is 'binance', include the 'isBuyerMaker' column.
          
        :param column_mappings: Optional dict mapping original column names to standard names.
        :param column_positions: Optional dict mapping column positions to standard names for headerless files.
        """
        self.data_source = data_source.lower()
        self.preserve_aggressor = preserve_aggressor

        # -----------------------------------------------------------------
        # (B)  Pick the dicts: caller override **wins**, otherwise defaults
        # -----------------------------------------------------------------
        self.column_mappings = column_mappings or COLUMN_MAPPINGS
        self.column_positions = column_positions or COLUMN_POSITIONS

        # Guard: the chosen dicts *must* contain this source
        if self.data_source not in self.column_mappings \
           or self.data_source not in self.column_positions:
            raise ValueError(
                f"No mapping found for data_source='{self.data_source}'. "
                f"Add it to COLUMN_MAPPINGS / COLUMN_POSITIONS."
            )  

    def detect_header(self, file_path: str) -> bool:
        """
        Decide whether the first non-blank line in *file_path* is a header.

        Returns
        -------
        bool
            True  -> first non-blank line looks like column names
            False -> first non-blank line looks like data
        """
        # Use gzip.open for *.gz, regular open otherwise
        opener = gzip.open if file_path.endswith(".gz") else open

        # Helper: does this string look numeric / boolean?
        def _numeric(token: str) -> bool:
            try:
                float(token)
                return True
            except ValueError:
                return token.lower() in {"true", "false"}

        # Helper: does this string parse as a date/time?
        def _maybe_datetime(token: str) -> bool:
            try:
                _parse_dt(token)
                return True
            except Exception:       # broad → any parsing failure means “no”
                return False

        # Read one non-empty line and classify it
        with io.TextIOWrapper(opener(file_path, "rb"),
                            encoding="utf-8",
                            errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:                 # skip blank lines
                    continue
                tokens = [tok.strip() for tok in line.split(",")]

                all_numeric     = all(_numeric(tok) for tok in tokens)
                first_datetime  = _maybe_datetime(tokens[0])

                # A header is *not* purely numeric and *not* a datetime-first row
                return not (all_numeric or first_datetime)

        # Empty file → default to “has header” so downstream code won’t crash
        return True
        
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
        
        # ---- ① Make the column order identical for every feed ----
        if self.data_source == "binance" and self.preserve_aggressor and "isBuyerMaker" in df.columns:
            df["signed_tick"] = np.where(df["isBuyerMaker"], -1, 1)
        elif self.data_source == "gmocoin":
            if self.preserve_aggressor and "side" in df.columns:
                # side == BUY → +1, side == SELL → –1
                df["signed_tick"] = np.where(df["side"]
                                            .str.upper().str.startswith("B"), 1, -1)
            else:
                # leave it blank → later code will apply the tick-rule
                df["signed_tick"] = np.nan    

            # convert JST to Unix‑ms
            df["date_time"] = (
                pd.to_datetime(df["date_time"], utc=False)
                  .dt.tz_localize("Asia/Tokyo")
                  .dt.tz_convert("UTC")
                  .view("int64") // 10**6
            )
        else:
            # no aggressor column → fall back later to tick‑rule
            df["signed_tick"] = np.nan

        df = df[["date_time", "price", "volume", "signed_tick"]]            
        return df


    def load_and_format_dataframe_in_batches(self, file_path: str, batch_size: int):
        """
        Stream‑loads a CSV in fixed‑size chunks and returns each chunk in the
        canonical layout:

            date_time | price | volume | signed_tick

        * date_time  – Unix epoch milliseconds (JST→UTC conversion for GMO Coin)
        * signed_tick – +1 for buyer‑initiated, ‑1 for seller‑initiated trades
                        (derived from 'isBuyerMaker' or 'side' when available;
                        falls back to NaN → downstream tick‑rule)
        """
        # -------------------------------------------------------
        # 1.  Decide whether to read by header or by hard positions
        # -------------------------------------------------------
        has_header = self.detect_header(file_path)

        if has_header:
            mapping  = self.column_mappings[self.data_source]
            usecols  = list(mapping.keys())
            header   = 0
        else:
            mapping  = self.column_positions[self.data_source]
            usecols  = list(mapping.keys())
            header   = None

        # -------------------------------------------------------
        # 2.  Chunked reader
        # -------------------------------------------------------
        reader = pd.read_csv(
            file_path,
            usecols=usecols,
            header=header,
            sep=",",
            chunksize=batch_size,
        )

        # -------------------------------------------------------
        # 3.  Per‑chunk venue‑specific fixes + canonical reorder
        # -------------------------------------------------------
        for chunk in reader:
            chunk = chunk.rename(columns=mapping)

            if (
                self.data_source == "binance"
                and self.preserve_aggressor
                and "isBuyerMaker" in chunk.columns
            ):
                # Binance aggressor flag: True = seller, False = buyer
                chunk["signed_tick"] = np.where(chunk["isBuyerMaker"], -1, 1)

            elif self.data_source == "gmocoin":
                # GMO Coin side column: "BUY"/"SELL"
                if self.preserve_aggressor and "side" in chunk.columns:
                    chunk["signed_tick"] = np.where(
                        chunk["side"].str.upper().str.startswith("B"), 1, -1)
                else:
                    chunk["signed_tick"] = np.nan                
                # JST → UTC epoch‑ms
                chunk["date_time"] = (
                    pd.to_datetime(chunk["date_time"], utc=False)
                    .dt.tz_localize("Asia/Tokyo")
                    .dt.tz_convert("UTC")
                    .view("int64") // 10**6
                )

            else:
                # No aggressor flag — let downstream tick‑rule decide
                chunk["signed_tick"] = np.nan

            # Re‑order / select final columns
            yield chunk[["date_time", "price", "volume", "signed_tick"]]
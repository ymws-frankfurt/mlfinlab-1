# tick_data_formatter.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import re

class TickDataFormatter:
    """
    Normalises raw tick files from multiple venues into the canonical
    4‑column schema:  date_time | price | volume | signed_tick
    """

    # ------------------------------------------------------------------
    # ❶  Mapping dicts – single source‑of‑truth
    # ------------------------------------------------------------------
    column_mappings: Dict[str, Dict[str, str]] = {
        "binance": {
            "timestamp": "date_time", # legacy alias
            "time": "date_time",      # legacy alias
            "price": "price",
            "qty": "volume",
            "isBuyerMaker": "isBuyerMaker",   # keep for aggressor
        },
        "gmocoin": {
            "date_time": "date_time",
            "price": "price",
            "size": "volume",
            "side": "side",                   # keep for aggressor
        },
        # add new exchanges here
    }

    # NEW — exact positions for header‑less dumps
    column_positions: Dict[str, Dict[int, str]] = {
        "binance": {4: "date_time", 1: "price", 2: "volume", 5: "isBuyerMaker"},
        "gmocoin": {5: "date_time", 4: "price", 3: "volume", 2: "side"},
    }

    def __init__(self, data_source: str, preserve_aggressor: bool = True):
        if data_source not in self.column_mappings:
            raise ValueError(f"Unknown data_source '{data_source}'. "
                             f"Available: {list(self.column_mappings)}")
        self.data_source = data_source
        self.preserve_aggressor = preserve_aggressor

    # ------------------------------------------------------------------
    # ❷  Utility – does the file have a header row?
    # ------------------------------------------------------------------
    @staticmethod
    def detect_header(file_path: str) -> bool:
        """
        Return True if the first line looks like a header (non‑numeric first token).
        """
        with open(file_path, "r", encoding="utf-8") as fh:
            first_token = fh.readline().split(",", 1)[0].strip()

        numeric = re.fullmatch(r"-?\d+(\.\d+)?", first_token) is not None
        return not numeric

    # ------------------------------------------------------------------
    # ❸  Venue‑agnostic signed‑tick computation
    # ------------------------------------------------------------------
    def _compute_signed_tick(self, df: pd.DataFrame) -> pd.Series:
        if not self.preserve_aggressor:
            return pd.Series(np.nan, index=df.index)

        if self.data_source == "binance" and "isBuyerMaker" in df.columns:
            return np.where(df["isBuyerMaker"], -1, 1)

        if self.data_source == "gmocoin" and "side" in df.columns:
            return np.where(df["side"].str.upper().str.startswith("B"), 1, -1)

        return pd.Series(np.nan, index=df.index)   # fallback (tick‑rule later)

    # ------------------------------------------------------------------
    # ❹  Core loader  – whole file
    # ------------------------------------------------------------------
    def load_and_format_dataframe(self, file_path: str) -> pd.DataFrame:
        has_header = self.detect_header(file_path)
        mapping    = self.column_mappings[self.data_source]

        if has_header:
            try:
                df = pd.read_csv(file_path, usecols=list(mapping.keys()))
            except ValueError:
                # Header labels didn’t match → treat as headerless
                positions = self.column_positions[self.data_source]
                df = pd.read_csv(
                    file_path,
                    header=None,
                    usecols=list(positions.keys()),
                    names=list(positions.values()),
                )

        else:
            positions = self.column_positions[self.data_source]
            df = pd.read_csv(
                file_path,
                header=None,
                usecols=list(positions.keys()),   # pick the right columns
                names=list(positions.values()),   # canonical names
            )


        df = df.rename(columns=mapping)

        # venue‑specific timestamp normalisation
        if self.data_source == "gmocoin":
            df["date_time"] = (
                pd.to_datetime(df["date_time"], utc=False)
                  .dt.tz_localize("Asia/Tokyo")
                  .dt.tz_convert("UTC")
                  .view("int64") // 10**6
            )

        df["signed_tick"] = self._compute_signed_tick(df)
        return df[["date_time", "price", "volume", "signed_tick"]]

    # ------------------------------------------------------------------
    # ❺  Streaming loader – chunk iterator
    # ------------------------------------------------------------------
    def load_and_format_dataframe_in_batches(
        self, file_path: str, batch_size: int
    ):
        has_header = self.detect_header(file_path)
        mapping    = self.column_mappings[self.data_source]

        if has_header:
            try:
                reader = pd.read_csv(
                    file_path, usecols=list(mapping.keys()), chunksize=batch_size
                )
            except ValueError:
                positions = self.column_positions[self.data_source]
                reader = pd.read_csv(
                    file_path,
                    header=None,
                    usecols=list(positions.keys()),
                    names=list(positions.values()),
                    chunksize=batch_size,
                )
        else:
            positions = self.column_positions[self.data_source]
            reader = pd.read_csv(
                file_path,
                header=None,
                usecols=list(positions.keys()),
                names=list(positions.values()),
                chunksize=batch_size,
            )


        for chunk in reader:
            chunk = chunk.rename(columns=mapping)

            if self.data_source == "gmocoin":
                chunk["date_time"] = (
                    pd.to_datetime(chunk["date_time"], utc=False)
                      .dt.tz_localize("Asia/Tokyo")
                      .dt.tz_convert("UTC")
                      .view("int64") // 10**6
                )

            chunk["signed_tick"] = self._compute_signed_tick(chunk)
            yield chunk[["date_time", "price", "volume", "signed_tick"]]

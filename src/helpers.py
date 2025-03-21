from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import polars as pl

from src.columns import Cols


def timestamp_series_to_float(ser: pd.Series, ref_ts: pd.Timestamp | None = None) -> pd.Series:
    assert pd.api.types.is_datetime64_any_dtype(ser)
    ref_ts = ser.iloc[0] if ref_ts is not None else ref_ts
    return (ser - ref_ts).dt.total_seconds()


def load_parquet_timeseries(parquet_fp: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(parquet_fp)
        .filter(
            # pl.col(Cols.SYS_MODE) == "Running",
            pl.col(Cols.WIND_ESTIMATION) < 25,  # 25 seems like an error fallback value
        )
        .with_columns((pl.col(Cols.GEN_D_C_VOLTAGE) * pl.col(Cols.GEN_D_C_CURRENT)).alias("Power"))  # V * A = W
        .with_columns(
            (pl.col("Power") / (pl.col(Cols.ROTOR_SPEED1) * np.pi / 30)).alias("ProtoTorque")
        )  # W / rpm = W / (rad/s*pi/30) = N * m
    )


T = TypeVar("T", pl.DataFrame, pd.DataFrame)


def rename_cols(df: T) -> T:
    name_mapping = {v: k for k, v in Cols.__dict__.items() if not k.startswith("__")}
    if isinstance(df, pd.DataFrame):
        return df.rename(columns=name_mapping)
    return df.rename(name_mapping)

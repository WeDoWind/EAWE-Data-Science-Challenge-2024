# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Preprocessing data
#
# ## Downloading the data
# The data is located here: https://zenodo.org/records/8230330 and
# it must be downloaded and extracted to a `/data` folder
#
# ## Running preprocessing
# the following code loads the data and metadata and generates a parquet while also applying:
# - mapping `SysMode` accoding to the `Bjorko_modes_mapping.json`
# - mapping `Dig_IO_States` to booleans columns for each state according to `Bjorko_digital_io_states_mappings.csv`

# +
import json
from pathlib import Path

import polars as pl

# -

FLD = Path("../data/SHM")  # update as needed
OUT_FLD = Path("../data/SHM-outputs")  # update as needed
OUT_FLD.mkdir(exist_ok=True)

# +
# loading mappings
with open(FLD / "Bjorko_modes_mapping.json") as f:
    system_modes_mapping = {int(k): v for k, v in json.load(f).items()}

system_digital_io_states_mappings = {
    k.strip(): int(v)
    for k, v in (
        pl.read_csv(FLD / "Bjorko_digital_io_states_mappings.csv", n_rows=13)  # rest is notes
        .select("Digital signal", "Bit no.")
        .iter_rows()
    )
}

method_f_mapping = {
    0: "Manually selected Pwaste",
    1: "FCR-N",
    2: "FCR-D_upp",
    3: "FCR-D_ned",
}


# +
def convert_sys_modes_to_mapped_string(df: pl.LazyFrame, system_modes_mapping: dict) -> pl.LazyFrame:
    return df.with_columns(
        pl.col("SysMode")
        .map_elements(system_modes_mapping.get, return_dtype=pl.String)
        .cast(pl.Enum(system_modes_mapping.values()))
        .alias("MappedSysMode"),
    ).drop("SysMode")


def convert_dig_io_states_to_individual_states(
    df: pl.LazyFrame, system_digital_io_states_mappings: dict
) -> pl.LazyFrame:
    return (
        df.with_columns(
            pl.col("Dig_IO_States")
            .cast(pl.Int16)
            .map_elements(lambda x: bin(x), return_dtype=pl.String)
            .alias("_bit_IO_States")
        )
        .with_columns(
            (pl.col("_bit_IO_States").str.slice(-bit_position - 1, length=1) == "1").alias(signal)
            for signal, bit_position in system_digital_io_states_mappings.items()
        )
        .drop("Dig_IO_States", "_bit_IO_States")
    )


def convert_method_f_to_mapped_string(df: pl.LazyFrame, method_f_mapping: dict) -> pl.LazyFrame:
    return df.with_columns(pl.col("Method_f").cast(pl.Float32)).with_columns(
        pl.col("Method_f")
        .round(0)
        .map_elements(method_f_mapping.get, return_dtype=pl.String)
        .alias("ApproxMappedMethod_f"),
    )


def convert_datetime(df: pl.LazyFrame) -> pl.LazyFrame:
    _EPOCH_1904_TO_1970_SECONDS = 2082844800
    return df.with_columns(
        (1000 * (pl.col("Time") - _EPOCH_1904_TO_1970_SECONDS)).cast(pl.Datetime("ms")).alias("timestamp"),
    ).drop("Time")


def process_data_file(fp: Path) -> pl.LazyFrame:
    return (
        pl.scan_csv(source=fp)
        .pipe(convert_datetime)
        .pipe(convert_method_f_to_mapped_string, method_f_mapping=method_f_mapping)
        .pipe(convert_sys_modes_to_mapped_string, system_modes_mapping=system_modes_mapping)
        .pipe(
            convert_dig_io_states_to_individual_states,
            system_digital_io_states_mappings=system_digital_io_states_mappings,
        )
    )


# -

# %%time
process_data_file(FLD / "B1_CL4_20.csv").collect().write_parquet(OUT_FLD / "preprocessed_20hz.parquet")

# %%time
process_data_file(FLD / "B1_CL4_100.csv").collect().write_parquet(OUT_FLD / "preprocessed_100hz.parquet")

# %load_ext autoreload
# %autoreload 2

# +
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
from IPython.display import display

from src import helpers
from src.columns import Cols


def get_time_range(lazy_frame):
    """Extract first and last timestamp from the dataset."""
    time_col = Cols.TIMESTAMP

    # Get min and max timestamps efficiently using Polars
    time_range = lazy_frame.select(pl.min(time_col).alias("min_time"), pl.max(time_col).alias("max_time")).collect()

    min_time = time_range[0, "min_time"]
    max_time = time_range[0, "max_time"]

    print(f"Dataset spans from {min_time} to {max_time}")
    return min_time, max_time


def generate_chunk_ranges(start_dt, end_dt, time_chunk: pd.Timedelta = pd.Timedelta(days=1)):
    """Generate datetime ranges from start_date to end_date."""
    current = pd.Timestamp(start_dt.date())
    dt_ranges = []

    while current < end_dt:
        chunk_end = current + time_chunk
        dt_ranges.append((current, chunk_end))
        current = chunk_end

    return dt_ranges


def plot_chunk_data(chunk_df, start_dt, end_dt, plot_dir):
    """Create and save various plots for the chunk (eg week) data."""
    time_col = Cols.TIMESTAMP

    # Check if all required columns exist
    required_cols = ["Power", "WSN", "XTurbSpeed1", "PA_B1_SP"]
    missing_cols = [col for col in required_cols if col not in chunk_df.columns]

    if missing_cols:
        print(f"Warning: Missing columns for plotting: {missing_cols}")
        print(f"Available columns: {chunk_df.columns}")
        return

    # Check if time column exists
    if time_col not in chunk_df.columns:
        print(f"Warning: Time column '{time_col}' not found for temporal coloring")
        print(f"Available columns: {chunk_df.columns}")
        return

    # Convert to pandas for plotting with matplotlib
    try:
        pdf = chunk_df.to_pandas()
    except Exception as e:
        print(f"Error converting to pandas: {str(e)}")
        return

    # Ensure timestamp is in datetime format for coloring
    try:
        # If timestamp is already a datetime, this will work directly
        # If not, we try to convert it
        if not pd.api.types.is_datetime64_any_dtype(pdf[time_col]):
            pdf[time_col] = pd.to_datetime(pdf[time_col])

        # Convert timestamps to numerical values for coloring
        # Normalize to the range [0, 1] for the time period
        min_time = pdf[time_col].min()
        max_time = pdf[time_col].max()

        # Create a normalized time value for coloring (0 = start, 1 = end)
        if min_time != max_time:  # Avoid division by zero
            pdf["time_norm"] = (pdf[time_col] - min_time) / (max_time - min_time)
        else:
            pdf["time_norm"] = 0.5  # If all times are the same

    except Exception as e:
        print(f"Error processing timestamps for coloring: {str(e)}")
        pdf["time_norm"] = 0.5  # Default if we can't process timestamps

    # Create figure with subplots
    # We need a 2x3 grid but with the first column spanning both rows
    fig = plt.figure(figsize=(20, 20))

    # Define grid spec - 2 rows, 3 columns, with first column spanning both rows
    gs = fig.add_gridspec(3, 3)

    # Create the five axes objects
    ax1 = fig.add_subplot(gs[0, 0])  # first row, first column
    ax2 = fig.add_subplot(gs[0, 1])  # Top row, second column
    ax3 = fig.add_subplot(gs[0, 2])  # Top row, third column
    ax4 = fig.add_subplot(gs[1, :])  # middle row, all columns
    ax5 = fig.add_subplot(gs[2, :], sharex=ax4)  # Bottom row, all columns

    # Define a colormap for time
    cmap = plt.cm.viridis

    # remove silly wind speeds
    pdf.loc[pdf["WSN"] < 0, "WSN"] = np.nan
    pdf.loc[pdf["WSN"] > 50, "WSN"] = np.nan

    # Plot 1: Power vs WSN
    sc1 = ax1.scatter(pdf["WSN"], pdf["Power"], c=pdf["time_norm"], cmap=cmap, alpha=0.7, s=5)
    # add grey line showing normal rated power
    ax1.axhline(y=30000, color="darkgrey", lw=7, alpha=0.5, label="30kW")
    ax1.set_ylim(bottom=min(0, pdf["Power"].min() - 100), top=max(30000 * 1.05, pdf["Power"].max() + 100))
    ax1.set_xlabel("WSN [m/s]")
    ax1.set_ylabel("Power [W]")
    ax1.legend()
    ax1.set_title("Power vs WSN")
    ax1.grid(True, alpha=0.3)

    # Plot 2: GenTorqSP vs XTurbSpeed1
    normal_rpm = [
        54.42,
        59.16,
        68.67,
        71.24,
        78.23,
    ]
    normal_torque = [
        0.29,
        1327.68,
        1821.57,
        3988.41,
        3681.40,
    ]

    # pdf["ProtoTorque"] =pdf["Power"] / (pdf["XTurbSpeed1"].clip(lower=1) * np.pi / 30)
    # pdf.loc[pdf["Power"]<=0,"ProtoTorque"]=np.nan
    # pdf.loc[pdf["XTurbSpeed1"] <= 30, "ProtoTorque"] = np.nan
    sc2 = ax2.scatter(pdf["XTurbSpeed1"], pdf["GenTorqSP"], c=pdf["time_norm"], cmap=cmap, alpha=0.7, s=5)
    ax2.plot(normal_rpm, normal_torque, color="darkgrey", lw=7, alpha=0.5, label="normal")
    ax2.set_xlabel("XTurbSpeed1 [RPM]")
    ax2.set_ylabel("GenTorqSP [N-m]")
    ax2.set_xlim(left=50, right=90)
    ax2.legend()
    ax2.set_title("GenTorqSP vs XTurbSpeed1")
    ax2.grid(True, alpha=0.3)

    # Plot 3: PA_B1_SP vs Power
    sc3 = ax3.scatter(pdf["Power"], pdf["PA_B1_SP"], c=pdf["time_norm"], cmap=cmap, alpha=0.7, s=5)
    ax3.plot([0, 30000, 30000], [2, 2, 90], color="darkgrey", lw=7, alpha=0.5, label="normal")
    ax3.set_ylim(bottom=min(0, pdf["PA_B1_SP"].min() - 0.1), top=max(4, pdf["PA_B1_SP"].max() + 0.1))
    ax3.set_xlabel("Power [W]")
    ax3.set_ylabel("PA_B1_SP [deg]")
    ax3.legend()
    ax3.set_title("PA_B1_SP vs Power")
    ax3.grid(True, alpha=0.3)

    # Plot 4: timeline of power
    ax4.scatter(pdf[time_col], pdf["Power"], c=pdf["time_norm"], cmap=cmap, alpha=0.7, s=5, label="Power")
    ax4.axhline(y=30000, color="darkgrey", lw=7, alpha=0.5, label="30kW")
    ax4.set_ylim(bottom=min(0, pdf["Power"].min() - 100), top=max(30000 * 1.05, pdf["Power"].max() + 100))
    ax4.set_ylabel("Power [W]")
    ax4.set_title("Power vs time")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: timeline of torque residual and pitch residual
    pdf["NormalTorque"] = np.interp(pdf["XTurbSpeed1"], normal_rpm, normal_torque)
    pdf["TorqueResidual"] = pdf["GenTorqSP"] - pdf["NormalTorque"]
    pdf.loc[pdf["Power"] <= 0, "TorqueResidual"] = np.nan
    pdf["PitchResidual"] = pdf["PA_B1_SP"] - 2
    pdf.loc[pdf["Power"] <= 0, "PitchResidual"] = np.nan
    pdf.loc[pdf["Power"] >= (0.99 * 30000), "PitchResidual"] = np.nan

    ax5.plot(pdf[time_col], pdf["TorqueResidual"] / 4000, alpha=0.7, label="Normalized TorqueResidual")
    ax5.plot(
        pdf[time_col], (pdf["PitchResidual"] / 10).clip(lower=-1, upper=1), alpha=0.7, label="Normalized PitchResidual"
    )
    # ax4.scatter(pdf[time_col], (pdf["PitchResidual"]/10).clip(lower=-1,upper=1), c=pdf["time_norm"], cmap=cmap, alpha=0.7, s=5,
    #             label="Normalized PitchResidual")
    ax5.axhline(y=0, color="darkgrey", lw=7, alpha=0.5, label="normal")
    ax5.set_xlabel("timestamp")
    ax5.set_ylabel("Normalized residuals")
    ax5.legend()
    ax5.set_title("Normalized torque and pitch residual vs time")
    ax5.grid(True, alpha=0.3)

    # Add colorbar to show time progression
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label("Time Progression")
    cbar.set_ticklabels([])

    # Add time labels to the colorbar (start date and end date) with better positioning
    # Position the start date (bottom of colorbar)
    cbar.ax.text(
        1.5, 0.01, start_dt.strftime("%H:%M:%S"), transform=cbar.ax.transAxes, ha="left", va="bottom", rotation=90
    )
    # Position the end date (top of colorbar)
    cbar.ax.text(1.5, 0.99, end_dt.strftime("%H:%M:%S"), transform=cbar.ax.transAxes, ha="left", va="top", rotation=90)

    # Add overall title
    date_descr = (
        f"{start_dt.strftime('%Y-%m-%d %H:%M:%S')} to {end_dt.strftime('%H:%M:%S')}"
        if (end_dt - start_dt) <= pd.Timedelta(days=1)
        else f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
    )
    title = f"20Hz data {date_descr}"
    fig.suptitle(title, fontsize=16, y=0.98)

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.94, right=0.9)  # Make room for the title and colorbar

    # Save the figure
    try:
        plot_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_dir / f"{title.replace(':', '')}.png", dpi=150)
        print(f"Saved combined plot for chunk starting {start_dt.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Error saving plot: {str(e)}")
    finally:
        plt.close(fig)

    print(f"Saved plots for chunk starting {start_dt.strftime('%Y-%m-%d')}")


def process_and_plot_chunk(lazy_frame, start_dt, end_dt, plot_dir):
    """Process and create plots for data within the specified date range."""
    time_col = Cols.TIMESTAMP

    # Filter data for the specific chunk
    chunk_data = lazy_frame.filter((pl.col(time_col) >= start_dt) & (pl.col(time_col) < end_dt))

    if chunk_data.select(pl.len()).collect().item() == 0:
        print(f"No data for chunk {start_dt} to {end_dt}")
        return

    # Collect the data for the chunk - this brings it into memory
    # If still too large, consider sampling or aggregating
    try:
        chunk_df = chunk_data.collect()
    except:
        print(f"Exception on collect for chunk {start_dt} to {end_dt}")
        return

    print(f"Processing chunk {start_dt} to {end_dt} with {chunk_df.height} rows")

    # Generate plots for the chunk data
    plot_chunk_data(chunk_df, start_dt, end_dt, plot_dir)


if __name__ == "__main__":
    from src import helpers, modelling, plotting, constants

    FLD = constants.PROCESSED_DATA_DIR

    fp_100hz = FLD / "preprocessed_100hz.parquet"
    fp_20hz = FLD / "preprocessed_20hz.parquet"

    PLOT_DIR = FLD / "plots"

    df_20hz_lazy = helpers.load_parquet_timeseries(fp_20hz)

    min_time, max_time = get_time_range(df_20hz_lazy)
    datetime_ranges = generate_chunk_ranges(min_time, max_time, time_chunk=pd.Timedelta(hours=1))
    print(f"Processing {len(datetime_ranges)} chunks of data")

    # Process each chunk
    for i, (start, end) in enumerate(datetime_ranges):
        print(f"daterange {i + 1}/{len(datetime_ranges)}")
        process_and_plot_chunk(df_20hz_lazy, pd.Timestamp(start), pd.Timestamp(end), PLOT_DIR)

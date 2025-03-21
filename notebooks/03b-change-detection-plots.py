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


def generate_chunk_ranges(start_dt, end_dt, day_chunk: int = 7):
    """Generate date ranges from start_date to end_date."""
    current = pd.Timestamp(start_dt).date()
    date_ranges = []

    while current < (pd.Timestamp(end_dt).date()):
        chunk_end = current + pd.Timedelta(days=day_chunk)
        date_ranges.append((current, chunk_end))
        current = chunk_end

    return date_ranges


def plot_chunk_data(chunk_df, start_date, end_date, plot_dir):
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

    # Format date for filenames
    date_str = start_date.strftime("%Y%m%d")

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
    fig = plt.figure(figsize=(20, 12))

    # Define grid spec - 2 rows, 3 columns, with first column spanning both rows
    gs = fig.add_gridspec(2, 3)

    # Create the five axes objects
    ax1 = fig.add_subplot(gs[:, 0])  # Spans both rows in first column
    ax2 = fig.add_subplot(gs[0, 1])  # Top row, second column
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)  # Bottom row, second column
    ax4 = fig.add_subplot(gs[0, 2])  # Top row, third column
    ax5 = fig.add_subplot(gs[1, 2], sharex=ax4)  # Bottom row, third column

    # Define a colormap for time
    cmap = plt.cm.viridis

    # Plot 1: Power vs WSN (big scatter plot)
    sc1 = ax1.scatter(pdf["WSN"], pdf["Power"], c=pdf["time_norm"], cmap=cmap, alpha=0.7, s=5)
    ax1.set_xlabel("WSN")
    ax1.set_ylabel("Power")
    ax1.set_title("Power vs WSN")
    ax1.grid(True, alpha=0.3)

    # Plot 2: XTurbSpeed1 vs Power
    sc2 = ax2.scatter(pdf["Power"], pdf["XTurbSpeed1"], c=pdf["time_norm"], cmap=cmap, alpha=0.7, s=5)
    ax2.set_ylabel("XTurbSpeed1")
    ax2.set_title("XTurbSpeed1 vs Power")
    ax2.grid(True, alpha=0.3)
    ax2.label_outer()  # Hide xlabel since it's shared

    # Plot 3: PA_B1_SP vs Power
    sc3 = ax3.scatter(pdf["Power"], pdf["PA_B1_SP"], c=pdf["time_norm"], cmap=cmap, alpha=0.7, s=5)
    ax3.set_xlabel("Power")
    ax3.set_ylabel("PA_B1_SP")
    ax3.set_title("PA_B1_SP vs Power")
    ax3.grid(True, alpha=0.3)

    # Plot 4: XTurbSpeed1 vs WSN
    sc4 = ax4.scatter(pdf["WSN"], pdf["XTurbSpeed1"], c=pdf["time_norm"], cmap=cmap, alpha=0.7, s=5)
    ax4.set_ylabel("XTurbSpeed1")
    ax4.set_title("XTurbSpeed1 vs WSN")
    ax4.grid(True, alpha=0.3)
    ax4.label_outer()  # Hide xlabel since it's shared

    # Plot 5: PA_B1_SP vs WSN
    sc5 = ax5.scatter(pdf["WSN"], pdf["PA_B1_SP"], c=pdf["time_norm"], cmap=cmap, alpha=0.7, s=5)
    ax5.set_xlabel("WSN")
    ax5.set_ylabel("PA_B1_SP")
    ax5.set_title("PA_B1_SP vs WSN")
    ax5.grid(True, alpha=0.3)

    # Add colorbar to show time progression
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label("Time Progression")

    # Add time labels to the colorbar (start date and end date) with better positioning
    # Position the start date (bottom of colorbar)
    cbar.ax.text(
        1.5, 0.01, start_date.strftime("%Y-%m-%d"), transform=cbar.ax.transAxes, ha="left", va="bottom", rotation=90
    )
    # Position the end date (top of colorbar)
    cbar.ax.text(
        1.5, 0.99, end_date.strftime("%Y-%m-%d"), transform=cbar.ax.transAxes, ha="left", va="top", rotation=90
    )

    # Add overall title
    date_descr = (
        f"{start_date.strftime('%Y-%m-%d')}"
        if (end_date - start_date) <= pd.Timedelta(days=1)
        else f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    title = f"20Hz data {date_descr}"
    fig.suptitle(title, fontsize=16, y=0.98)

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.94, right=0.9)  # Make room for the title and colorbar

    # Save the figure
    try:
        plot_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_dir / f"{title}.png", dpi=150)
        print(f"Saved combined plot for chunk starting {start_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Error saving plot: {str(e)}")
    finally:
        plt.close(fig)

    print(f"Saved plots for chunk starting {start_date.strftime('%Y-%m-%d')}")


def process_and_plot_chunk(lazy_frame, start_date, end_date, plot_dir):
    """Process and create plots for data within the specified date range."""
    time_col = Cols.TIMESTAMP

    # Filter data for the specific chunk
    chunk_data = lazy_frame.filter((pl.col(time_col) >= start_date) & (pl.col(time_col) < end_date))

    # Collect the data for the chunk - this brings it into memory
    # If still too large, consider sampling or aggregating
    try:
        chunk_df = chunk_data.collect()
    except:
        print(f"Exception on collect for chunk {start_date} to {end_date}")
        return

    if chunk_df.height == 0:
        print(f"No data for chunk {start_date} to {end_date}")
        return

    print(f"Processing chunk {start_date} to {end_date} with {chunk_df.height} rows")

    # Generate plots for the chunk data
    plot_chunk_data(chunk_df, start_date, end_date, plot_dir)


if __name__ == "__main__":
    FLD = Path("../data/SHM-outputs")  # update as needed

    fp_100hz = FLD / "preprocessed_100hz.parquet"
    fp_20hz = FLD / "preprocessed_20hz.parquet"
    # -

    df_100hz_lazy = helpers.load_parquet_timeseries(fp_100hz)
    PLOT_DIR = Path(__file__).parent / Path(__file__).stem

    df_20hz_lazy = helpers.load_parquet_timeseries(fp_20hz)

    min_time, max_time = get_time_range(df_20hz_lazy)
    datetime_ranges = generate_chunk_ranges(min_time, max_time, day_chunk=1)
    print(f"Processing {len(datetime_ranges)} chunks of data")

    # Process each chunk
    for i, (start, end) in enumerate(datetime_ranges):
        print(f"daterange {i + 1}/{len(datetime_ranges)}")
        process_and_plot_chunk(df_20hz_lazy, pd.Timestamp(start), pd.Timestamp(end), PLOT_DIR)

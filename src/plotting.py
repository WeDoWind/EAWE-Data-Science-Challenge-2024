import ipywidgets
import numpy as np
import pandas as pd


def rolling_scatter_plot(
    df: pd.DataFrame, x_col, y_col, color_col: str, window_size: int = 10_000, starting_idx: int = 0
):
    xlim = df[x_col].min(), df[x_col].max()
    ylim = df[y_col].min(), df[y_col].max()
    n_splits = np.ceil(df.shape[0] / window_size).astype(int)
    color_kw = dict(c=color_col, cmap="viridis") if color_col else dict()

    def _f(i: int):
        idx_start, idx_end = i * window_size, (i * window_size) + window_size
        return df.iloc[idx_start:idx_end].plot.scatter(
            x=x_col, y=y_col, xlim=xlim, ylim=ylim, grid=True, title=f"data [{idx_start}:{idx_end}]", **color_kw
        )

    return ipywidgets.interact(
        _f,
        chunk_idx=ipywidgets.IntSlider(
            value=starting_idx, min=0, max=n_splits, step=1, layout=ipywidgets.Layout(width="1000px")
        ),
    )

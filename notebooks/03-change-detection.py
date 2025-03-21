# %load_ext autoreload
# %autoreload 2

# +
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import pwlf
from sklearn import ensemble, metrics
from tqdm.auto import tqdm

from src import helpers, modelling, plotting
from src.columns import Cols

pd.set_option("display.max_columns", 90)

RANDOM_SEED = 0
FLD = Path("../data/SHM-outputs")  # update as needed
PRED_TORQUE = "PredictedTorque"
fp_20hz = FLD / "preprocessed_20hz.parquet"
# fp_100hz = FLD / "preprocessed_100hz.parquet"
# -

# ### define curve from manually cleaned dataset

clean_df = pd.read_parquet(
    r"C:\Users\GCalvo\Documents\Github\_workbench_\labeller\__ignore__\labelled_data.parquet",
    filters=[("GEN_TORQUE_SET_POINT", ">", 100)],
)

# +
# ax = clean_df.plot.scatter(x="ROTOR_SPEED1", y="GEN_TORQUE_SET_POINT", label="GEN_TORQUE_SET_POINT")
# clean_df.plot.scatter(
#     x="ROTOR_SPEED1", y="ProtoTorque", ax=ax, c="orange", alpha=0.1, s=1, grid=True, label="ProtoTorque"
# )
# -

fit_model = modelling.build_fit_model(control_ser=clean_df.set_index("ROTOR_SPEED1")["ProtoTorque"])

pd.DataFrame(
    {
        "x": clean_df["ROTOR_SPEED1"],
        "actual": clean_df["ProtoTorque"],
        "predicted": fit_model.predict(clean_df["ROTOR_SPEED1"]),
    }
).set_index("x").plot(style=".")

# ## Setups

useful_cols = [
    # main
    Cols.TIMESTAMP,
    Cols.POWER,
    Cols.PITCH_ANGLE_BLADE1,
    Cols.ROTOR_SPEED1,
    Cols.ROTOR_SHAFT_TORQUE,
    # setpoints
    Cols.PITCH1_ANGLE_SET_POINT,
    Cols.OPTIMAL_RPM_REF,
    Cols.GEN_TORQUE_SET_POINT,
    # extra
    Cols.PROTO_TORQUE,
    Cols.GRID_FREQUENCY,
    Cols.WIND_ESTIMATION,
    Cols.POWER_PERCENTAGE,
    Cols.WIND_SPEED_NACELLE,
]
df_lazy = helpers.load_parquet_timeseries(fp_20hz)#.select(*useful_cols)


df_lazy.head(20).collect()

# ### measurement periods splitting

# +
tser = df_lazy.select(Cols.TIMESTAMP).collect().to_pandas()[Cols.TIMESTAMP]
is_late = tser.diff() > pd.Timedelta(seconds=1)
is_late_idxes = is_late[is_late].index
print(f"There are {len(is_late_idxes)} splits")

split_data = []
last_split_idx = None
for split_idx in is_late_idxes:
    ts = tser.iloc[slice(last_split_idx, split_idx - 1)]
    last_split_idx = split_idx
    split_data.append({"t_start": ts.min(), "t_end": ts.max()})

splits_df = pd.DataFrame(split_data).assign(t_delta=lambda d: (d["t_end"] - d["t_start"]).dt.total_seconds())
# display(splits_df['t_delta'].hist(bins=100))

long_periods_df = splits_df[splits_df["t_delta"] > 5000].sort_values("t_delta", ascending=False)
len(long_periods_df)
# -

# ### visualisation

# +
# t_start, t_end = long_periods_df.iloc[2][["t_start", "t_end"]]
t_start, t_end = pd.Timestamp("2022-09-22T08:24"), pd.Timestamp("2022-09-23")  # region 2

_d = (
    df_lazy.filter(pl.col(Cols.WIND_SPEED_NACELLE) > 3)
    .filter(pl.col(Cols.WIND_SPEED_NACELLE) < 6)
    # .filter(pl.col(Cols.POWER) < 28_000)
    # .filter((pl.col(Cols.TIMESTAMP) >= t_start) & (pl.col(Cols.TIMESTAMP) <= t_end))
    # .group_by_dynamic(Cols.TIMESTAMP, every="1s")
    # .agg(pl.all().mean())
    # .head(5_000)
    .collect()
    .to_pandas()
    .set_index(Cols.TIMESTAMP)
)
# -

# _d[Cols.PITCH_ANGLE_BLADE1].reset_index(drop=True).plot(figsize=(25,4), grid=True)
_d.sample(20_000).reset_index(drop=True).plot.scatter(
    x=Cols.WIND_SPEED_NACELLE, y=Cols.POWER, c=Cols.PITCH_ANGLE_BLADE1, figsize=(25, 4), grid=True, alpha=0.1, s=1
)
# px.line(_d)
# px.scatter(_d, x=Cols.WIND_ESTIMATION, y=Cols.POWER)
# px.scatter(_d, x=Cols.ROTOR_SPEED1, y=Cols.GEN_TORQUE_SET_POINT)  # region 2 logic


t_start, t_end = long_periods_df.iloc[2][["t_start", "t_end"]]

# ### collating long period data

# +
from sklearn import decomposition, pipeline, preprocessing

first_timestamp = df_lazy.select(Cols.TIMESTAMP).min().collect().item()

_dfs = []
for _, row in long_periods_df.iterrows():
    _dfs.append(
        df_lazy.filter((pl.col(Cols.TIMESTAMP) >= row["t_start"]) & (pl.col(Cols.TIMESTAMP) <= row["t_end"]))
        .filter(pl.col(Cols.WIND_SPEED_NACELLE) > 3.5)
        .filter(pl.col(Cols.WIND_SPEED_NACELLE) < 6)
        # .select(*useful_cols)
        .collect()
        .to_pandas()
    )
collated_long_periods_df = (
    pd.concat(_dfs)
    .assign(ts_float=lambda d: helpers.timestamp_series_to_float(d[Cols.TIMESTAMP], ref_ts=first_timestamp))
    .reset_index(drop=True)
)
# -

# ### pca + clustering plot

# +
import plotly.graph_objects as go
from sklearn import cluster

SAMPLE_N = 1_000
X = collated_long_periods_df.sample(SAMPLE_N, random_state=42)
Xt = m.fit_transform(X.drop(columns=Cols.TIMESTAMP).values)
hdb = cluster.HDBSCAN(alpha=2.0, min_cluster_size=SAMPLE_N // 20)
cluster_id_arr = hdb.fit_predict(Xt)

_plot_df = pd.DataFrame(
    np.concatenate([Xt, cluster_id_arr[:, None]], axis=1), columns=["c1", "c2", "c3", "cluster_id"]
).astype({"cluster_id": "category"})
if 1:
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=_plot_df["c1"],
                y=_plot_df["c2"],
                z=_plot_df["c3"],
                mode="markers",
                marker=dict(size=2, color=_plot_df["cluster_id"], colorscale="Viridis", opacity=0.8),
            )
        ]
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
    display(fig)
# -

(
    pd.concat([X.reset_index(drop=True), _plot_df], axis=1)
    # .query("cluster_id != -1")
    .plot.scatter(
        x=Cols.WIND_SPEED_NACELLE,
        y=Cols.POWER,
        c="cluster_id",
        figsize=(25, 4),
        grid=True,
        alpha=0.5,
        s=2,
        cmap="viridis",
    )
)

# ### rolling divergence detection

collated_long_periods_df

plotting.rolling_scatter_plot(
    collated_long_periods_df,
    x_col=Cols.ROTOR_SPEED1,
    y_col=Cols.GEN_TORQUE_SET_POINT,
    color_col=Cols.MET,
    window_size=1000,
    starting_idx=116,
)
collated_long_periods_df.head(5000).plot.scatter(x=Cols.ROTOR_SPEED1, y=Cols.PROTO_TORQUE, grid=True)

ref_y = Cols.GEN_TORQUE_SET_POINT  # Cols.PROTO_TORQUE
fit_model = modelling.build_fit_model(
    control_ser=collated_long_periods_df.head(5000).set_index(Cols.ROTOR_SPEED1)[ref_y]
)

_plt_df = collated_long_periods_df.assign(
    modelled=lambda d: fit_model.predict(d[Cols.ROTOR_SPEED1]),
    error=lambda d: d["modelled"] - d[ref_y],
    rolling_err=lambda d: d["error"].rolling(20, center=True).sum().abs(),
    rolling_err_smooth=lambda d: d["rolling_err"].rolling(100, center=True).mean().rolling(100, center=True).mean(),
)

# +
# _err_col = 'rolling_err_smooth' # 'rolling_err'
_err_col = "rolling_err"
_plot = ["error-hist", "omega-torque-curve", "error-timeseries"][2]

if _plot == "error-hist":  # helpful to choose threshold
    ax = _plt_df.query(f"{_err_col} < 20_000")[_err_col].hist(bins=100)
    ax.set_ylim(0, 100_000)
elif _plot == "omega-torque-curve":
    ax = _plt_df.query(f"{_err_col} > 6_000").plot.scatter(
        x=Cols.ROTOR_SPEED1, y=ref_y, c=_err_col, grid=True, s=1, alpha=0.2, figsize=(25, 4)
    )
    _x = np.linspace(55, 80)
    ax.plot(_x, fit_model.predict(_x), "-r", alpha=0.3)
elif _plot == "error-timeseries":
    _plt_df[_err_col].plot(figsize=(25, 4))
# -

# ### rolling curve detection


x_col = Cols.ROTOR_SPEED1
y_col = Cols.GEN_TORQUE_SET_POINT
# +
_df = collated_long_periods_df.set_index(Cols.TIMESTAMP)[[x_col, y_col]]
n_bins = 100
get_bins = lambda d: [i.mid for i in pd.cut(d[x_col], bins=np.linspace(_df[x_col].min(), _df[x_col].max(), n_bins))]

for _, _sub_df in _df.groupby(pd.Grouper(freq="10s")):
    if _sub_df.empty:
        continue
    _ser = _sub_df[y_col].groupby(get_bins(sub_df), observed=True).mean()
    break

# +

_model = pwlf.PiecewiseLinFit(
    x=_ser.index,
    y=_ser.values,
    seed=0,
)
# -

_model.fit()

mean_y.plot(style=".-")

sub_df.plot.scatter(x=x_col, y=y_col)

# ## OTHER

collated_long_periods_df.sample(20_000, random_state=10).plot.scatter(
    x=Cols.ROTOR_SPEED1, y=Cols.GEN_TORQUE_SET_POINT, xlim=(50, 80), ylim=(-200, 5000), grid=True
)

# +
import ipywidgets


def _f(i_start: int):
    _d = _df.iloc[i_start * 30_000 : (i_start * 30_000) + 30_000]
    ax = _d.plot.scatter(x=Cols.ROTOR_SPEED1, y=Cols.GEN_TORQUE_SET_POINT, xlim=(50, 80), ylim=(-200, 5000), grid=True)
    _d.plot.scatter(
        x=Cols.ROTOR_SPEED1,
        y=Cols.PROTO_TORQUE,
        xlim=(50, 80),
        ylim=(-200, 5000),
        grid=True,
        ax=ax,
        c="r",
        s=1,
        alpha=0.3,
    )

    # (
    #     _df.iloc[i_start*30_000: (i_start*30_000)+30_000]
    #     .set_index(Cols.TIMESTAMP)
    #     [[
    #         # Cols.POWER,

    #         # Cols.PITCH_ANGLE_BLADE1,

    #         Cols.ROTOR_SPEED1,
    #         Cols.OPTIMAL_RPM_REF,

    #         # Cols.ROTOR_SHAFT_TORQUE,
    #         # Cols.GEN_TORQUE_SET_POINT,
    #         # Cols.PROTO_TORQUE,

    #         # Cols.GRID_FREQUENCY,
    #         Cols.WIND_ESTIMATION,

    #         # Cols.PITCH1_ANGLE_SET_POINT,  # very close to PITCH_ANGLE_BLADE1
    #         # Cols.POWER_PERCENTAGE   # mostly flat
    #     ]]
    #     .transform(lambda x: (x-x.min())/(x.max()-x.min()) )
    #     .pipe(helpers.rename_cols)
    #     .plot(figsize=(25,4), grid=True)
    # )


ipywidgets.interact(_f, i_start=ipywidgets.IntSlider(min=0, max=(len(_df) // 30_000) + 1, step=1, value=8))
# +
t_start, t_end = long_periods_df.iloc[0][["t_start", "t_end"]]
setpoint_cols = [
    Cols.PITCH1_ANGLE_SET_POINT,
    Cols.OPTIMAL_RPM_REF,
    Cols.GEN_TORQUE_SET_POINT,
    Cols.GRID_FREQUENCY,
    Cols.POWER_PERCENTAGE,
]
_df = df.loc[t_start:t_end].iloc[15_000:30_000]

_df.loc[:, setpoint_cols].transform(lambda x: (x - x.min()) / (x.max() - x.min())).plot(figsize=(25, 3), grid=True)
_df.plot.scatter(
    x=Cols.PITCH_ANGLE_BLADE1, y=Cols.POWER, s=1, alpha=0.1, c=Cols.PITCH1_ANGLE_SET_POINT, figsize=(20, 5), grid=True
)
# -


df.drop(columns=["MappedSysMode", "ApproxMappedMethod_f", "Method_f"]).rename(columns=name_mapping).columns

(
    df.set_index(Cols.TIMESTAMP)
    .loc["2022-09-22":"2022-10-15"]
    .drop(columns=["MappedSysMode", "ApproxMappedMethod_f", "Method_f"])
    .reset_index()
    .rename(columns=name_mapping)
    .to_parquet(r"C:\Users\GCalvo\Documents\Github\_workbench_\labeller\.cache\tst.pq")
)

_d = (
    df.set_index(Cols.TIMESTAMP)
    .loc["2022-09-22":"2022-10-15"]
    .drop(columns=["MappedSysMode", "ApproxMappedMethod_f", "Method_f"])
    .assign(x=lambda d: d[Cols.GEN_TORQUE_SET_POINT] < 100)
)
_x = _d.query(f"{Cols.ROTOR_SPEED1} > 52").corr()["x"]

# +
# _x.dropna().sort_values().head(20)

# +
# df.set_index(Cols.TIMESTAMP).loc["2022-09-22":"2022-10-15"].query(f"{Cols.GEN_TORQUE_SET_POINT}>1000").iloc[:, -15:-3].mean()

# +
# df.set_index(Cols.TIMESTAMP).loc["2022-09-22":"2022-10-15"].plot.scatter(
#     x=Cols.ROTOR_SPEED1, y=Cols.GEN_TORQUE_SET_POINT,
#     figsize=(25,4), grid=True,
#     style=".", alpha=0.1,
#     c=Cols.PITCH_ANGLE_BLADE1, cmap="jet"
# )
# -

x = (
    df.set_index(Cols.TIMESTAMP)
    .loc["2022-09-22":"2022-10-15"]
    .query(f"{Cols.ROTOR_SPEED1} > 56")
    .dropna(subset=[Cols.GEN_TORQUE_SET_POINT])
)
ax = x.plot.scatter(
    x=Cols.PITCH_ANGLE_BLADE1,
    y=Cols.ROTOR_SHAFT_TORQUE,
    figsize=(25, 4),
    grid=True,
    style=".",
    alpha=0.1,
    label="ProtoTorque",
)
x.plot.scatter(
    x=Cols.PITCH_ANGLE_BLADE1,
    y=Cols.GEN_TORQUE_SET_POINT,
    figsize=(25, 4),
    grid=True,
    style=".",
    alpha=0.1,
    ax=ax,
    c="red",
    label="GEN_TORQUE_SET_POINT",
)

# i = 800_000
df.loc[900_000:1_200_000].assign(t=lambda d: d.reset_index().index).plot.scatter(
    x=Cols.ROTOR_SPEED1,
    y=Cols.GEN_TORQUE_SET_POINT,
    figsize=(25, 4),
    grid=True,
    style=".",
    alpha=0.1,
    c="t",
    cmap="jet",
)

model = ensemble.HistGradientBoostingRegressor(
    max_iter=100,
    l2_regularization=0.5,
    max_features=0.8,
)
model.fit()

# ## Explore


c[:, :-1]

# +
# import matplotlib.cm as cm
# c = df[Cols.OPTIMAL_RPM_REF].transform(lambda s: (s-s.min())/(s.max()-s.min()))

(
    df
    # .set_index(Cols.ROTOR_SPEED1)
    # .head(500_000)
    .sample(n=500_000)
    # [[
    #     Cols.GEN_TORQUE_SET_POINT,
    #     Cols.OPTIMAL_RPM_REF
    #     # "ProtoTorque",
    #     # PRED_TORQUE
    # ]]
).plot.scatter(
    x=Cols.ROTOR_SPEED1,
    y=Cols.GEN_TORQUE_SET_POINT,
    figsize=(25, 4),
    grid=True,
    style=".",
    alpha=0.1,
    c=Cols.OPTIMAL_RPM_REF,
    cmap="jet",
)
# -

df.sample(n=50_000).plot.scatter(
    x=Cols.ROTOR_SPEED1,
    y=Cols.OPTIMAL_RPM_REF,
    figsize=(25, 4),
    grid=True,
    style=".",
    alpha=0.1,
    c=Cols.GEN_TORQUE_SET_POINT,
    cmap="jet",
)


# # Plan
# 1. find change in control curve manually and add col that identifies the period
# 2. create a model (with feature importance) that can predict the period and look at the feature importance


def plot_reconstructed(
    control_ser: pd.Series,
    fit_model: pwlf.PiecewiseLinFit,
    xlim: tuple[float, float] = (50, 80),
    ylim: tuple[float, float] = (0, 4500),
) -> None:
    kw = dict(style=".", grid=True, figsize=(25, 4), alpha=0.5, xlim=xlim, ylim=ylim)
    pd.DataFrame(
        {"x": control_ser.index, "y": control_ser.values, "param_func": fit_model.predict(control_ser.index)}
    ).set_index("x").plot(**kw)


# ## Defining Base model that spans the whole range

# +
rolling_window = 15_000
rolling_step = rolling_window // 10
fit_n_bins = 100

print(f"""
{rolling_window=}
{rolling_step=}
{fit_n_bins=}
""")

# +
base_ser = control_df.iloc[:30_000]["y"]
base_model = build_fit_model(control_ser=base_ser, n_bins=fit_n_bins, n_breaks=5)
plot_reconstructed(control_ser=base_ser, fit_model=base_model, xlim=(50, 69), ylim=(0, 2500))
base_breaks = base_model.fit_breaks

print(f"""
{base_model.fit_breaks=}
{base_model.slopes=}
{base_model.intercepts=}
""")
# -

# ## Detecting change in configuration

parameters_over_time = []
for _chunk in tqdm(
    control_df.rolling(window=rolling_window, step=rolling_step), total=control_df.shape[0] // rolling_step + 1
):
    if _chunk.shape[0] < rolling_step:
        continue

    _n_breaks = int((base_breaks[:-1] <= _chunk.index.max()).sum())  # depends on how much the dataset extends
    row_info = {
        "ts_start": _chunk["ts"].min(),
        "ts_end": _chunk["ts"].max(),
        "n_breaks": _n_breaks,
    }

    # add a 0,0 to ensure lower end is always present
    _chunk = pd.concat([_chunk, pd.DataFrame({"ts": row_info["ts_start"], "y": 0}, index=[50, 51, 52])])

    _fit_model = build_fit_model(control_ser=_chunk["y"], n_bins=fit_n_bins, n_breaks=_n_breaks)
    _predicted = _fit_model.predict(_chunk.index)
    parameters_over_time.append(
        row_info
        | {
            "model": _fit_model,
            "rmse": metrics.root_mean_squared_error(_chunk["y"], _predicted),
            "mae": metrics.mean_absolute_error(_chunk["y"], _predicted),
            "r2": metrics.r2_score(_chunk["y"], _predicted),
        }
        | {f"break_{i}": v for i, v in enumerate(_fit_model.fit_breaks)}
        | {f"slope_{i}": v for i, v in enumerate(_fit_model.slopes)}
        | {f"intercept_{i}": v for i, v in enumerate(_fit_model.intercepts)}
    )

print(metrics.r2_score(_chunk["y"], _predicted))
pd.DataFrame({"true": _chunk["y"], "pred": _predicted}).plot.scatter(x="true", y="pred")

pd.DataFrame(parameters_over_time).query("ts_start <= '2022-09-23' and ts_end >= '2022-09-22'")

parameters_over_time

# +
_plt_df = pd.DataFrame(parameters_over_time).iloc[6:]


def plot_fitting_for_chunk(chunk_idx: int) -> None:
    _row = _plt_df.loc[chunk_idx]
    print(f"using {_row['n_breaks']} breaks")
    _is_chunk = (control_df["ts"] >= _row["ts_start"]) & (control_df["ts"] <= _row["ts_end"])
    _ser = control_df.loc[_is_chunk, "y"]
    plot_reconstructed(control_ser=_ser, fit_model=_row["model"])


# -

to_normalize = [
    "break_0",
    "break_1",
    "break_2",
    "break_3",
    "break_4",
    "break_5",
    "slope_0",
    "slope_1",
    "slope_2",
    "slope_3",
    "slope_4",
    "intercept_0",
    "intercept_1",
    "intercept_2",
    "intercept_3",
    "intercept_4",
]


def _norm_params(d: pd.DataFrame) -> pd.DataFrame:
    d[to_normalize] = d[to_normalize] / d[to_normalize].iloc[1]
    return d


px.line(_plt_df.drop(columns=["ts_start", "ts_end", "model"]).pipe(_norm_params), height=500)

# +
# to_compare = list(np.arange(19,25+1))
# _x = np.linspace(50, 80, 100)
# (
#     pd.DataFrame({"x": _x} | {f"y_{_idx}": _plt_df.loc[_idx, "model"].predict(_x) for _idx in to_compare})
#     .set_index("x").plot(grid=True, figsize=(25, 5)
# )

# +
from ipywidgets import IntSlider, Layout, interact

interact(
    plot_fitting_for_chunk,
    chunk_idx=IntSlider(value=19, min=_plt_df.index.min(), max=_plt_df.index.max(), layout=Layout(width="1000px")),
)

# +
# plot_fitting_for_chunk(19)

# +
# plot_fitting_for_chunk(40)

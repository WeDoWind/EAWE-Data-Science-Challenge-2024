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

RANDOM_SEED = 0
FLD = Path("../data/SHM-outputs")  # update as needed

fp_100hz = FLD / "preprocessed_100hz.parquet"
fp_20hz = FLD / "preprocessed_20hz.parquet"
# -

df_100hz_lazy = helpers.load_parquet_timeseries(fp_100hz)
categorical_cols = {
    i: j for i, j in df_100hz_lazy.collect_schema().items() if j not in (pl.Float64, pl.Float32, pl.Datetime)
}
name_mapping = {v: k for k, v in Cols.__dict__.items() if not k.startswith("__")}

# ## Plot exploration

# ### suggested time period
# the suggested period Sept 22-23, 2022 is not useful as the pitch is fixed at 2°.
# the period is only available in the 20Hz data.

df_20hz_lazy = helpers.load_parquet_timeseries(fp_20hz)

if 0:
    _df = (
        (
            df_20hz_lazy.filter(
                pl.col(Cols.TIMESTAMP) >= pd.Timestamp("2022-09-22"),
                pl.col(Cols.TIMESTAMP) < pd.Timestamp("2022-09-24"),
            )
        )
        .collect()
        .to_pandas()
    )

    ax = (
        _df.set_index(Cols.WIND_ESTIMATION)[[Cols.PITCH1_ANGLE_SET_POINT, Cols.PITCH_ANGLE_BLADE1]]
        .rename(columns=name_mapping)
        .plot(style=".", grid=True, figsize=(4, 2), alpha=0.1)
    )

# ### pitch angle values follow the setpoint pretty well
# So we can use them as direct signal to detect the control system parameters

# +
if 0:
    fig, axes = plt.subplots(1, 3, figsize=(25, 4))
    _df = df_100hz_lazy.collect().to_pandas().sample(50_000)
    for col, ax in zip([Cols.PITCH_ANGLE_BLADE1, Cols.PITCH_ANGLE_BLADE2, Cols.PITCH_ANGLE_BLADE3], axes):
        ax.scatter(x=_df[Cols.PITCH1_ANGLE_SET_POINT], y=_df[col], s=1, alpha=0.1)
        ax.set_ylabel(col)
        ax.set_xlabel("Pitch Angle SetPoint")
        ax.grid(True)

    df_100hz_lazy.head(5000).collect().to_pandas().set_index(Cols.TIMESTAMP)[
        [Cols.PITCH_ANGLE_BLADE1, Cols.PITCH_ANGLE_BLADE2, Cols.PITCH_ANGLE_BLADE3, Cols.PITCH1_ANGLE_SET_POINT]
    ].plot(alpha=0.3, figsize=(25, 3), grid=True)

if 0:
    px.line(
        df_100hz_lazy.head(2000)
        .collect()
        .to_pandas()
        .set_index(Cols.TIMESTAMP)[
            [Cols.PITCH_ANGLE_BLADE1, Cols.PITCH_ANGLE_BLADE2, Cols.PITCH_ANGLE_BLADE3, Cols.PITCH1_ANGLE_SET_POINT]
        ]
    ).show()  # .plot(alpha=0.3, figsize=(25,3))
# -

# ### exploring region 3 (rated power)

reg3_lazy_df = (
    df_100hz_lazy.filter(
        pl.col(Cols.WIND_ESTIMATION) > 6,
        pl.col(Cols.WIND_ESTIMATION) < 15,  # not many points over 15 and skew plots
        pl.col(Cols.WIND_ESTIMATION) < 24,  # value 25 seems data quality issue
        pl.col("Power") > 15_000,  # focus on approaching region 3
        # found interesting period manually in 20Hz data
        # pl.col(Cols.TIMESTAMP) >= pd.Timestamp("2022-07-05 12:08:39.400"),
        # pl.col(Cols.TIMESTAMP) < pd.Timestamp("2022-07-05 12:11:19.699"),
        # found interesting period manually in 100Hz data
        pl.col(Cols.TIMESTAMP) >= pd.Timestamp("2023-06-09 06:33:36.659000"),
        pl.col(Cols.TIMESTAMP) < pd.Timestamp("2023-06-09 08:37:02.460000"),
    )
    # .head(10_000)  # first points that show enough control behaviour
)
_df = reg3_lazy_df.collect().to_pandas()
_df.shape

# +
# px.line(
#     df_100hz_lazy.collect()
#     .to_pandas()
#     .set_index(Cols.TIMESTAMP)[Cols.PITCH_ANGLE_BLADE1]
#     .rolling(15, center=True)
#     .mean()
#     .diff()
#     .head(5000)
# )
# -

# code to slice data into interesting period
if 0:
    def show_period(d):
        print("time period", d[Cols.TIMESTAMP].min(), d[Cols.TIMESTAMP].max())
        return d

    (
        _df#.loc[59_000:80_000]
        .loc[3400:6600]
        .assign(t=lambda d: d.index)
        .pipe(show_period)
        .plot.scatter(x=Cols.WIND_ESTIMATION, y="Power", c="t", grid=True, figsize=(25, 4), alpha=0.1, cmap="jet")
    )

# ### Exploring dependency of Pitch on other variables

if 0:
    _df = (
        (
            df_100hz_lazy.filter(
                pl.col(Cols.WIND_ESTIMATION) > 6,
                pl.col(Cols.WIND_ESTIMATION) < 15,  # not many points over 15 and skew plots
                pl.col(Cols.WIND_ESTIMATION) < 24,  # value 25 seems data quality issue
                pl.col("Power") > 15_000,  # focus on approaching region 3
                pl.col(Cols.TIMESTAMP) >= pd.Timestamp("2023-06-09 06:33:36.659000"),
                pl.col(Cols.TIMESTAMP) < pd.Timestamp("2023-06-09 08:37:02.460000"),
            )
        )
        .collect()
        .to_pandas()
    )  # .sample(100_000)
    cols = [
        "Power",
        Cols.WIND_SPEED_NACELLE,  # Supersonic sensor on the nacelle
        Cols.WIND_SPEED30M_HEIGHT,  # Supersonic transducer in 30m Meteorological mast.
        Cols.WIND_ESTIMATION,
        Cols.DC_CURRENT_REF,
        Cols.GEN_D_C_CURRENT,
        Cols.GEN_D_C_VOLTAGE,
        Cols.GRID_FREQUENCY,
        Cols.OPTIMAL_RPM_REF,
        Cols.ROTOR_SHAFT_TORQUE,
        Cols.ROTOR_SPEED1,
        Cols.ROTOR_SPEED2,
    ]
    fig, axes = plt.subplots(4, 3, figsize=(25, 12))
    for col, ax in zip(cols, axes.ravel()):
        human_col = name_mapping.get(col, col)
        _df.rename(columns={col: human_col}).plot.scatter(
            x=human_col, y=Cols.PITCH_ANGLE_BLADE1, grid=True, s=1, alpha=0.1, ax=ax
        )

# ## Finding relationship between rotor speed and pitch

import ipywidgets as widgets

# #### finding good region 3 period
# good = lots of consecutive points in region 3

(
    df_100hz_lazy.with_columns((pl.col("Power") > 27_000).alias("at_rated"))
    .with_columns(pl.len().over(pl.col("at_rated").rle_id()).alias("consecutives"))
    .filter(pl.col("at_rated"))
    .group_by(pl.col(Cols.TIMESTAMP).dt.strftime("%Y-%m-%dT%H:%M"))
    .agg(pl.col("consecutives").max())
    .sort("consecutives")
    .collect()
)

# finding period start and end
if 0:
    df_100hz_lazy.filter(
        pl.col(Cols.WIND_ESTIMATION) > 6,
        pl.col(Cols.WIND_ESTIMATION) < 15,  # not many points over 15 and skew plots
        pl.col(Cols.WIND_ESTIMATION) < 24,  # value 25 seems data quality issue
        # pl.col("Power") > 27_000, # focus on approaching region 3
        pl.col(Cols.TIMESTAMP) >= pd.Timestamp("2023-06-09 06:33:27.210"),
        pl.col(Cols.TIMESTAMP) < pd.Timestamp("2023-06-09 06:34:19"),
    ).collect().to_pandas().plot.scatter(Cols.WIND_ESTIMATION, "Power")

_subset_df_lazy = (
    df_100hz_lazy.sort(Cols.TIMESTAMP).filter(
        # pl.col(Cols.TIMESTAMP) >= pd.Timestamp("2023-06-09 06:33:27.210"),
        # pl.col(Cols.TIMESTAMP) < pd.Timestamp("2023-06-09 06:34:19"),
        pl.col(Cols.TIMESTAMP) >= pd.Timestamp("2023-06-09 06:33:32"),
        pl.col(Cols.TIMESTAMP) < pd.Timestamp("2023-06-09 06:33:58"),
    )
).collect()


# #### define model calculation


def _calc(d: pl.DataFrame, Kp: float, Kd: float, Ki: float, Ki0: float, low_pass_filter_steps: float) -> pl.DataFrame:
    assert {Cols.ROTOR_SPEED1, Cols.PITCH1_ANGLE_SET_POINT} <= set(d.columns)
    return (
        d.select(Cols.TIMESTAMP, Cols.ROTOR_SPEED1, Cols.OPTIMAL_RPM_REF, Cols.PITCH1_ANGLE_SET_POINT)
        .with_columns(pl.col(Cols.ROTOR_SPEED1).rolling_mean(int(low_pass_filter_steps)))
        .with_columns((pl.col(Cols.ROTOR_SPEED1) - pl.col(Cols.OPTIMAL_RPM_REF)).alias("speed_err"))
        .with_columns(
            (Kp * pl.col("speed_err") + Kd * pl.col("speed_err").diff() + Ki * pl.col("speed_err").cum_sum() + Ki0).alias(
                "pid_reconstruction"
            )
        )
    )


# +
# df_100hz_lazy.select(Cols.OPTIMAL_RPM_REF).collect().to_pandas().hist(bins=100)
# -

# #### evaluate

# +
import sklearn.metrics

def _evaluate(vals):
    _plt_df = (
        _calc(_subset_df_lazy, *vals).drop_nulls().select(Cols.PITCH1_ANGLE_SET_POINT, "pid_reconstruction").to_pandas()
        # .transform( lambda x: (x-x.min()) / (x.max()/x.min()) )
    )
    print(
        f"RMSE: {sklearn.metrics.root_mean_squared_error(
            _plt_df[Cols.PITCH1_ANGLE_SET_POINT], _plt_df["pid_reconstruction"]
        )}"
    )
    px.line(_plt_df).show()
    px.scatter(_plt_df, x=Cols.PITCH1_ANGLE_SET_POINT, y="pid_reconstruction").show()


# +
from scipy.optimize import minimize


def _err(params) -> float:
    _df = _calc(_subset_df_lazy, *params).to_pandas()
    return ((_df["pid_reconstruction"] - _df[Cols.PITCH1_ANGLE_SET_POINT]) ** 2).sum()


init_params = (1.2250659997743292, 4.707864087033814, 0.008515762087036659, 2.783580614890669, 30.0)
param_bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (1, 100)]
opt_result = minimize(
    _err, init_params, tol=1e-20, bounds=param_bounds, options=dict(maxiter=1_000)
)  # , method="Nelder-Mead")
assert opt_result.success, f"Optimisation Failure: {opt_result.message}"
display(opt_result)

opt_vals = tuple(i.item() for i in opt_result.x)
print(opt_vals)
_calc_df = _evaluate(opt_vals)


# +
# Optuna proved not so great at finding better solutions and slow.

# if 0:
#     import optuna

#     def _objective(trial):
#         Kp = trial.suggest_float("Kp", -10_000, 10_000)
#         Kd = trial.suggest_float("Kd", -10_000, 10_000)
#         Ki = trial.suggest_float("Ki", -10_000, 10_000)
#         Ki0 = trial.suggest_float("Ki0", -10_000, 10_000)
#         low_pass_filter_steps = trial.suggest_int("low_pass_filter_steps", 1, 100)
#         _df = _calc(_subset_df_lazy, Kp=Kp, Kd=Kd, Ki=Ki, Ki0=Ki0,
#           low_pass_filter_steps=low_pass_filter_steps
#         ).to_pandas()
#         return ((_df["pid_reconstruction"]-_df[Cols.PITCH1_ANGLE_SET_POINT])**2).sum()

#     study = optuna.create_study(study_name="opt", direction="minimize")
#     init_params_vals = (1.0861219298124503, -10.303485565698683, 0.008531476394044316, 2.8571573095456286, 30.0)
#     init_params = dict(zip(["ω_sp", "Kp", "Kd", "Ki", "Ki0", "low_pass_filter_steps"], init_params_vals))
#     study.enqueue_trial(init_params)
#     optuna.logging.set_verbosity(optuna.logging.WARNING)
#     study.optimize(_objective, n_trials=1000, show_progress_bar=True, n_jobs=-1)
#     print(study.best_params)
# -

# #### interactive plotting

# +
def _f(Kp, Kd, Ki, Ki0, low_pass_filter_steps):
    _df = (
        _calc(_subset_df_lazy, Kp, Kd, Ki, Ki0, low_pass_filter_steps)
        .select(Cols.PITCH1_ANGLE_SET_POINT, "pid_reconstruction")
        .drop_nulls()
        .to_pandas()
    )
    rmse = sklearn.metrics.root_mean_squared_error(_df[Cols.PITCH1_ANGLE_SET_POINT], _df["pid_reconstruction"])
    print(f"rmse: {rmse:.5f}")
    _df.plot.scatter(x=Cols.PITCH1_ANGLE_SET_POINT, y="pid_reconstruction", grid=True, s=1, alpha=0.2)


def _make_fslider(x, multiplier=2):
    _min, _max = -multiplier * x, multiplier * x
    if _min > _max:
        _min, _max = _max, _min
    return widgets.FloatSlider(min=_min, max=_max, step=abs(x) / 100, value=x)


Kp, Kd, Ki, Ki0, low_pass_filter_steps = (1.2250659997743292, 4.707864087033814, 0.008515762087036659, 2.783580614890669, 30.0)
widgets.interact(
    _f,
    Kp=_make_fslider(Kp),
    Kd=_make_fslider(Kd),
    Ki=_make_fslider(Ki),
    Ki0=_make_fslider(Ki0),
    low_pass_filter_steps=widgets.IntSlider(min=1, max=100, step=1, value=low_pass_filter_steps),
    # rolling_period_ms=widgets.IntSlider(min=0, max=100, step=1, value=opt_rolling_period_ms),
    # rolling_period_offset=widgets.IntSlider(min=0, max=100, step=1, value=opt_rolling_period_offset),
)
# -

# #### write submission csv

_calc(_subset_df_lazy, Kp, Kd, Ki, Ki0, low_pass_filter_steps).to_pandas().to_csv("04-pitch-strategy-evaluate.csv")

# +
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pwlf
from sklearn import ensemble, metrics, model_selection

from src.columns import Cols

RANDOM_SEED = 0
FLD = Path("../data/SHM-outputs")  # update as needed

fp_20hz = FLD / "preprocessed_20hz.parquet"
fp_100hz = FLD / "preprocessed_100hz.parquet"

# +
x_col = Cols.ROTOR_SPEED1
y_hidden_col = Cols.GEN_TORQUE_SET_POINT
y_col = Cols.ROTOR_SHAFT_TORQUE

# df_lazy = (
#     pl.scan_parquet(fp_100hz)
#     .filter(pl.col(Cols.SYS_MODE) == "Running", pl.col(Cols.ROTOR_SPEED1) > 50)
# .with_columns((pl.col(Cols.GEN_D_C_VOLTAGE) * pl.col(Cols.GEN_D_C_CURRENT)).alias("Power"))  # V * A = W
# .with_columns(
#     (pl.col("Power") / (pl.col(Cols.ROTOR_SPEED1) * np.pi / 30)).alias("ProtoTorque")
# )  # W / rpm = W / (rad/s*pi/30) = N * m
# )
df_lazy = (
    pl.scan_parquet(fp_20hz)
    .filter(
        pl.col(Cols.SYS_MODE) == "Running",
        pl.col(Cols.ROTOR_SPEED1) > 50,
        # pl.col(Cols.TIMESTAMP) > pd.Timestamp("2022-09-22"),
        # pl.col(Cols.TIMESTAMP) < pd.Timestamp("2022-09-28"),
        pl.col(Cols.TIMESTAMP) >= pd.Timestamp("2023-06-09 06:33:36.659000"), # same as pitch strategy study
        pl.col(Cols.TIMESTAMP) < pd.Timestamp("2023-06-09 08:37:02.460000"), # same as pitch strategy study
    )
    .with_columns((pl.col(Cols.GEN_D_C_VOLTAGE) * pl.col(Cols.GEN_D_C_CURRENT)).alias("Power"))  # V * A = W
    .with_columns(
        (pl.col("Power") / (pl.col(Cols.ROTOR_SPEED1) * np.pi / 30)).alias("ProtoTorque")
    )  # W / rpm = W / (rad/s*pi/30) = N * m
)

categorical_cols = {i: j for i, j in df_lazy.collect_schema().items() if j not in (pl.Float64, pl.Float32, pl.Datetime)}
name_mapping = {v: k for k, v in Cols.__dict__.items() if not k.startswith("__")}
df = df_lazy.collect().to_pandas()
# -


# ## Plot exploration

ax = (
    df.set_index('WSN')
    .rename(columns=name_mapping)[['POWER']]
    .plot(style=".", grid=True, figsize=(25, 4), alpha=0.1)
)

ax = (
    df.set_index('Power')
    .rename(columns=name_mapping)[['PITCH1_ANGLE_SET_POINT']]
    .plot(style=".", grid=True, figsize=(25, 4), alpha=0.1)
)


ax = (
    df.set_index('Power')
    .rename(columns=name_mapping)[['ROTOR_SPEED1']]
    .plot(style=".", grid=True, figsize=(25, 4), alpha=0.1)
)

ax = (
    df.set_index(x_col)
    .rename(columns=name_mapping)[["GEN_TORQUE_SET_POINT", "PROTO_TORQUE"]]
    .plot(style=".", grid=True, figsize=(25, 4), alpha=0.1)
)
# df.to_pandas().plot.scatter(x=x_col, y=y_col, grid=True, figsize=(25, 4), s=1, alpha=0.1)

# # Plan
# 1. create a model that predicts GEN_TORQUE_SET_POINT from the rest of the variables
# 2. create a model that defines the map based on the predicted GEN_TORQUE_SET_POINT

# ## Stage 1: predict GEN_TORQUE_SET_POINT


# +
target = Cols.GEN_TORQUE_SET_POINT
features = [
    c
    for c in df.columns
    if c
    not in (
        target,
        Cols.TIMESTAMP,  # could cause data leakage
        # are also setpoints
        Cols.DC_CURRENT_REF,
        Cols.OPTIMAL_RPM_REF,
        Cols.PITCH1_ANGLE_SET_POINT,
        Cols.YAW_SPEED_HYDRAULIC_REF,
        Cols.YAW_DAMPING_L_REF,
        Cols.YAW_DAMPING_R_REF,
    )
]

# model built-in alternative to converting manually .astype({k: 'category' for k in categorical_cols})
categorical_features = [i for i in categorical_cols if i in features]


# -


def model_eval(model: Any, X_test: pd.DataFrame, y_test: pd.Series, x_col: str | None = Cols.ROTOR_SPEED1) -> None:
    y_pred = model.predict(X_test)
    _d = pd.DataFrame({"actual": y_test, "predicted": y_pred})
    if x_col is not None:
        _d = _d.assign(x=X_test[x_col]).set_index("x")
        ax = _d["actual"].plot(style=".", grid=True, figsize=(25, 4), alpha=0.1, c="green")
        _d["predicted"].plot(style=".", grid=True, figsize=(25, 4), alpha=0.1, ax=ax, c="red")

    print(f"""
    R2  : {metrics.r2_score(_d["actual"], _d["predicted"]):>8.3f}
    RMSE: {metrics.root_mean_squared_error(_d["actual"], _d["predicted"]):>8.3f}
    MAE : {metrics.median_absolute_error(_d["actual"], _d["predicted"]):>8.3f}
    """)


# +
X = df[features]  # .drop(columns=categorical_features) #.astype({k: 'category' for k in categorical_cols})
y = df[target]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, shuffle=False)
# -

# ### trying to estimate feature importance with other model

# +
# _model = ensemble.RandomForestRegressor(n_estimators=10, n_jobs=-1)
# _model.fit(X_train, y_train)
# pd.Series(_model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10).plot.bar()

# +
# model_eval(_model, X_test, y_test)
# -

# ### training model

model = ensemble.HistGradientBoostingRegressor(
    max_iter=100,
    l2_regularization=0.5,
    max_features=0.8,
    categorical_features=categorical_features,
    random_state=RANDOM_SEED,
)
model.fit(X_train, y_train)
model_eval(model, X_test, y_test)

# ### estimate feature importance permutation_importance

if 0:  # very compute intensive
    from sklearn.inspection import permutation_importance

    importance_results = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=0, n_jobs=-1)
    fig, ax = plt.subplots(figsize=(25, 4))
    (  # plot with errors (usually very small)
        pd.DataFrame(
            {"mean": importance_results.importances_mean, "std": importance_results.importances_std},
            index=X_train.columns,
        )
        .assign(upper=lambda d: d["mean"] + d["std"], lower=lambda d: d["mean"] - d["std"])
        .sort_values("upper")
        .tail(10)
        .pipe(lambda d: ax.barh(d.index, d["mean"], xerr=3 * d["std"]))
    )

# ## Stage 2: extract lookup points

control_df = pd.DataFrame(
    {"ts": df[Cols.TIMESTAMP].values, "y": model.predict(X)}, index=df[Cols.ROTOR_SPEED1].rename("x").values
).sort_values("ts")


def build_fit_model(
    control_ser: pd.Series, n_bins: int = 100, n_breaks: int = 4, seed: int = RANDOM_SEED
) -> pwlf.PiecewiseLinFit:
    _bins = pd.cut(control_ser.index, bins=np.linspace(control_ser.index.min(), control_ser.index.max(), n_bins))
    _binned_ser = control_ser.groupby(_bins, observed=True).mean()
    linear_fit_model = pwlf.PiecewiseLinFit(
        x=[i.mid for i in _binned_ser.index],
        y=_binned_ser.values,
        seed=seed,
    )
    linear_fit_model.fit(n_breaks)
    return linear_fit_model


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


# +
# _ser = control_df["y"].iloc[:30_000]
# linear_fit_model = build_fit_model(control_ser=_ser, n_bins=100, n_breaks=5)
# print(linear_fit_model.fit_breaks)
# print(linear_fit_model.slopes)
# print(linear_fit_model.intercepts)
# (
#     pd.DataFrame({"x": _ser.index, "y": _ser.values, "param_func": linear_fit_model.predict(_ser.index)})
#     .set_index("x")
#     .plot(style=".", grid=True, figsize=(25, 4), alpha=0.5)
# )
# -

# ## Defining Base model that spans the whole range

rolling_window = 15_000
rolling_step = rolling_window // 10
fit_n_bins = 100

# +
base_ser = control_df["y"]
base_model = build_fit_model(control_ser=base_ser, n_bins=fit_n_bins, n_breaks=5)
plot_reconstructed(control_ser=base_ser, fit_model=base_model, xlim=(50, 80), ylim=(0, 4500))
base_breaks = base_model.fit_breaks

print(f"""
{base_model.fit_breaks=}
{base_model.slopes=}
{base_model.intercepts=}
{base_model.predict(base_model.fit_breaks)=}
""")
# -

# ### Summarizing plot

# +
_d = (
    df.head(20_000).assign(
        predicted_setpoint = lambda d : model.predict(d[features]),
        fitted_predicted_setpoint = lambda d : base_model.predict(d[Cols.ROTOR_SPEED1]),
    )
)

import plotly.express as px

fig = px.scatter(
    pd.DataFrame({
        Cols.ROTOR_SPEED1: _d[Cols.ROTOR_SPEED1],
        "actual_setpoint": _d[Cols.GEN_TORQUE_SET_POINT],
        "predicted_setpoint": _d['predicted_setpoint'],
        "fitted_predicted_setpoint": _d['fitted_predicted_setpoint'],
    }),
    x=Cols.ROTOR_SPEED1,
    y=[
        "actual_setpoint",
        "predicted_setpoint",
        "fitted_predicted_setpoint"
    ]
)
fig.update_xaxes(title_text=f"{Cols.ROTOR_SPEED1} [RPM]")
fig.update_yaxes(title_text="Torque [N-m]")
# -

# Store the base model
with open(FLD / "base_model.pickle", "wb") as f:
    pickle.dump(base_model, f, pickle.HIGHEST_PROTOCOL)

# Try to figure out VS_MaxTq
print(f"{df['GenTorqSP'].max()=}")
print(f"{df['GenTorqSP'].quantile(0.9999)=}")
print(f"{df['GenTorqSP'].quantile(0.999)=}")
print(f"{df['GenTorqSP'].quantile(0.99)=}")

# Try to figure out VS_MaxRat
trq_sp_abs_diffs=df['GenTorqSP'].diff().abs()
print(f"{trq_sp_abs_diffs.max()=}")
print(f"{trq_sp_abs_diffs.quantile(0.9999)=}")
print(f"{trq_sp_abs_diffs.quantile(0.999)=}")
print(f"{trq_sp_abs_diffs.quantile(0.99)=}")

# Try to figure out VS_RtGnSp
print(f"{df['OptRpm'].max()=}")

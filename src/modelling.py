import numpy as np
import pandas as pd
import pwlf

RANDOM_SEED = 42


def build_fit_model(
    control_ser: pd.Series, n_bins: int = 100, n_breaks: int = 5, seed: int = RANDOM_SEED
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


class StateModel: ...

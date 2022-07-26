import numpy as np


def get_min_floor(vals, step=1000):
    try:
        return np.round(np.floor(np.nanmin(vals) * step) / step, 3)
    except ValueError:
        return np.nan


def get_max_floor(vals, step=1000):
    try:
        return np.round(np.ceil(np.nanmax(vals) * step) / step, 3)
    except ValueError:
        return np.nan

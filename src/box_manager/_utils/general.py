import numpy as np


def get_min_floor(vals, step=1000):
    return np.round(np.floor(np.min(vals) * step) / step, 3)


def get_max_floor(vals, step=1000):
    return np.round(np.ceil(np.max(vals) * step) / step, 3)

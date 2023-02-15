import napari
import numpy as np
import pandas


def get_min_floor(vals, step=1000):
    try:
        if isinstance(vals, pandas.Series):
            if pandas.isnull(vals).all():
                return np.nan
            else:
                return np.round(np.floor(np.nanmin(vals) * step) / step, 3)
        else:
            return np.round(np.floor(np.nanmin(vals) * step) / step, 3)
    except ValueError:
        return np.nan


def get_max_floor(vals, step=1000):
    try:
        if isinstance(vals, pandas.Series):
            if pandas.isnull(vals).all():
                return np.nan
            else:
                return np.round(np.floor(np.nanmax(vals) * step) / step, 3)
        else:
            return np.round(np.floor(np.nanmax(vals) * step) / step, 3)
    except ValueError:
        return np.nan


def get_identifier(layer, cur_slice):
    if isinstance(layer, napari.layers.Points):
        return layer.data[:, cur_slice]
    elif isinstance(layer, napari.layers.Shapes):
        return np.array([entry[0, cur_slice] for entry in layer.data])
    else:
        assert False, (layer, type(layer))

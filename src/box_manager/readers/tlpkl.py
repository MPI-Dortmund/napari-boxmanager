import typing
import warnings

import pandas as pd

if typing.TYPE_CHECKING:
    import pathlib

    import numpy as np


class DimZMissingWarning(Warning):
    pass


def read(path: "pathlib.Path") -> pd.DataFrame:
    pandas_data: pd.DataFrame = pd.read_pickle(path)

    if "dim_z" not in pandas_data.attrs:
        warnings.warn(
            "dim_z attribute missing in pkl file! Please invert your Z dimension values manually!",  # noqa: E501
            DimZMissingWarning,
        )
    else:
        pandas_data["Z"] = pandas_data.attrs["dim_z"] - pandas_data["Z"]

    pandas_data["X"] = pandas_data["X"].astype(int)
    pandas_data["Y"] = pandas_data["Y"].astype(int)
    pandas_data["Z"] = pandas_data["Z"].astype(int)

    pandas_data.rename(
        columns={
            "X": "z",
            "Y": "x",
            "Z": "y",
            "width": "box_z",
            "height": "box_x",
            "depth": "box_y",
        },
        inplace=True,
    )

    return pandas_data[
        ["x", "y", "z", "metric_best", "size", "box_x", "box_y", "box_z"]
    ]


def to_napari(
    path: "pathlib.Path",
) -> "tuple[np.ndarray, dict[str, typing.Any], dict[str, typing.Any], str]":
    return read(path), {}, {}, "points"

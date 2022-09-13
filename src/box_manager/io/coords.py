import os
import typing

import numpy.typing as npt
import pandas as pd

from .interface import NapariLayerData
from . import io_utils as coordsio


class BoxFileNumberOfColumnsError(pd.errors.IntCastingNaNError):
    pass


class UnknownFormatException(Exception):
    ...


DEFAULT_BOXSIZE: int = 10


def get_valid_extensions():

    return ["coords"]


def read(path: os.PathLike) -> pd.DataFrame:

    names = ["x", "y", "z"]
    box_data: pd.DataFrame = pd.read_csv(
        path,
        delim_whitespace=True,
        index_col=False,
        header=None,
        dtype=float,
        names=names,
        usecols=range(len(names)),
    )  # type: ignore
    try:
        box_data.astype(int)
    except pd.errors.IntCastingNaNError:
        raise BoxFileNumberOfColumnsError

    return box_data


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[NapariLayerData]":

    return coordsio.to_napari(
        path=path,
        read_func=read,
        prepare_napari_func=_prepare_napari_coords,
        meta_columns=[],
        feature_columns=[],
    )


def _prepare_napari_coords(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    output_data: pd.DataFrame = pd.DataFrame(columns=["x", "y", "z"])

    output_data["z"] = input_df["x"]
    output_data["y"] = input_df["y"]
    output_data["x"] = input_df["z"]
    output_data["boxsize"] = DEFAULT_BOXSIZE

    return output_data


def _write_coords(path: os.PathLike, df: pd.DataFrame):
    df[["x", "y", "z"]].to_csv(path, sep=" ", header=None, index=None)


def _make_df_data(
    coordinates: pd.DataFrame, box_size: npt.ArrayLike, features: dict
) -> pd.DataFrame:
    data = {"x": [], "y": [], "z": [], "boxsize": []}
    for (z, y, x), boxsize in zip(
        coordinates,
        box_size,
    ):
        data["x"].append(x)
        data["y"].append(y)
        data["z"].append(z)
        data["boxsize"].append(boxsize)
    return pd.DataFrame(data)


def from_napari(
    path: os.PathLike, layer_data: list[NapariLayerData]
):
    path = coordsio.from_napari(
        path=path,
        layer_data=layer_data,
        write_func=_write_coords,
        format_func=_make_df_data,
    )
    return path

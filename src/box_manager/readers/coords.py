import os
import typing
from collections.abc import Callable

import numpy as np
import pandas as pd
from .coordinate_reader import to_napari_generic_coordinates

if typing.TYPE_CHECKING:
    import numpy.typing as npt


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
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":

    return to_napari_generic_coordinates(
        path=path,
        read_func=read,
        prepare_napari_func=_prepare_napari_coords,
        meta_columns=_get_meta_idx(),
        feature_columns=_get_meta_idx() + _get_hidden_meta_idx()
    )


def _get_meta_idx():
    return []


def _get_hidden_meta_idx():
    return []


def _prepare_napari_coords(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    output_data: pd.DataFrame = pd.DataFrame(
        columns=["x", "y", "z"] + _get_meta_idx() + _get_hidden_meta_idx()
    )

    output_data["z"] = input_df["x"]
    output_data["y"] = input_df["y"]
    output_data["x"] = input_df["z"]
    output_data["boxsize"] = DEFAULT_BOXSIZE

    return output_data


def _make_df_data(coordinates, box_size):
    data = {
        "x": [],
        "y": [],
        "z": [],
        "boxsize": []
    }
    for (z, y, x), boxsize in zip(
            coordinates,
            box_size,
    ):
        data["x"].append(x)
        data["y"].append(y)
        data["z"].append(z)
        data["boxsize"].append(boxsize)
    return data

def _write_coords(path : os.PathLike, df: pd.DataFrame):
    df[['x','y','z']].to_csv(path,sep=' ', header=None, index=None)

def from_napari(
    path: os.PathLike,
    layer_data: list[tuple[typing.Any, dict, str]]
):

    for data, meta, layer in layer_data:

        if data.shape[1] == 2:
            data = np.insert(data, 0, 0, axis=1)

        coordinates = data[meta["shown"]]
        boxsize = meta['size'][meta["shown"]][:,0]
        export_data = {}

        coords_writer=_write_coords
        export_data[path] = _make_df_data(coordinates, boxsize)


        for outpth in export_data:
            df = pd.DataFrame(export_data[outpth])
            coords_writer(outpth,df)


    return path

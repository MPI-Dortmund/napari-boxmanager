import pandas as pd
from pyStarDB import sp_pystardb as star
import os
from . import io_utils as coordsio
import typing
from .interface import NapariLayerData
import numpy as np
import numpy.typing as npt

DEFAULT_BOXSIZE=200

def get_valid_extensions():
    return ["star"]

###################
# READ FUNCTIONS
###################

def read(path: "os.PathLike") -> pd.DataFrame:
    sfile = star.StarFile(path)
    box_data = sfile['']

    return box_data

def _prepare_napari_coords(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    is_3d = '_rlnCoordinateZ' in input_df.columns

    columns = ["z", "y"]
    if is_3d:
        columns.append("x")
    output_data: pd.DataFrame = pd.DataFrame(columns=columns)

    output_data["z"] = input_df["_rlnCoordinateX"]
    output_data["y"] = input_df["_rlnCoordinateY"]
    if is_3d:
        output_data["x"] = input_df["_rlnCoordinateZ"]

    output_data["boxsize"] = DEFAULT_BOXSIZE

    return output_data


def to_napari(
    path: typing.Union[os.PathLike, list[os.PathLike]],
) -> "list[NapariLayerData]":

    return coordsio.to_napari(
        path=path,
        read_func=read,
        prepare_napari_func=_prepare_napari_coords,
        meta_columns=[],
        feature_columns=[],
    )

###################
# WRITE FUNCTIONS
###################
def _make_df_data(
    coordinates: pd.DataFrame, box_size: npt.ArrayLike, features: pd.DataFrame
) -> pd.DataFrame:
    data = {
        "_rlnCoordinateX": [],
        "_rlnCoordinateY": [],
    }
    for i in range(len(coordinates)):
        coords = coordinates[i]

        is_3d = True

        if len(coords) == 2:
            is_3d = False
            y, x = coords
            z = np.nan
        else:
            z, y, x = coords

        data["_rlnCoordinateX"].append(x)
        data["_rlnCoordinateY"].append(y)
        if is_3d:
            data["_rlnCoordinateZ"].append(z)

    return pd.DataFrame(data)

def _write_star(path: os.PathLike, df: pd.DataFrame):
    sfile = star.StarFile(path)

    sfile.update("", df, True)

    sfile.write_star_file(
        overwrite=True, tags=[""]
    )

def from_napari(
    path: os.PathLike, layer_data: list[NapariLayerData]
):
    path = coordsio.from_napari(
        path=path,
        layer_data=layer_data,
        write_func=_write_star,
        format_func=_make_df_data,
    )

    return path
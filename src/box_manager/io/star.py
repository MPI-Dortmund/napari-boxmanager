import os
import tempfile
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
from pyStarDB import sp_pystardb as star

from . import io_utils as coordsio
from .interface import NapariLayerData

DEFAULT_BOXSIZE = 200


def get_valid_extensions():
    return ["star", "cs"]


###################
# READ FUNCTIONS
###################


def read(path: "os.PathLike") -> pd.DataFrame:
    sfile = star.StarFile(path)
    if "particles" in sfile:
        # relion 3.1
        box_data = sfile["particles"]
    elif "" in sfile:
        box_data = sfile[""]
    else:
        return pd.DataFrame()

    return box_data


def _prepare_napari_coords(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    is_3d = "_rlnCoordinateZ" in input_df.columns
    is_filament = "_rlnHelicalTubeID" in input_df.columns
    columns = ["z", "y"]
    if is_3d:
        columns.append("x")
    output_data: pd.DataFrame = pd.DataFrame(columns=columns)

    output_data["z"] = input_df["_rlnCoordinateX"]
    output_data["y"] = input_df["_rlnCoordinateY"]
    if is_3d:
        output_data["x"] = input_df["_rlnCoordinateZ"]

    if is_filament:
        output_data["fid"] = input_df["_rlnHelicalTubeID"]

    if "_rlnAutopickFigureOfMerit" in input_df.columns:
        output_data["confidence"] = input_df["_rlnAutopickFigureOfMerit"]

    output_data["boxsize"] = DEFAULT_BOXSIZE

    return output_data


def _split_star(
    path_star: str, output_dir: str
) -> typing.Union[typing.List[str], str]:
    """
    Splits star file in case it contains coordinates for multiple micrographs
    """
    data = read(path_star)
    if "_rlnMicrographName" not in data.columns:
        return path_star

    unique_micrographs = np.unique(data["_rlnMicrographName"]).tolist()
    if len(unique_micrographs) == 1:
        return path_star
    os.makedirs(output_dir, exist_ok=True)
    file_extension = os.path.splitext(os.path.basename(path_star))[1]
    new_paths = []
    for mic in unique_micrographs:
        mask = data["_rlnMicrographName"] == mic
        mic_data = data[mask]
        pth = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(mic))[0] + file_extension,
        )
        _write_star(pth, mic_data)
        new_paths.append(pth)
    return new_paths


def to_napari(
    path: typing.Union[os.PathLike, list[os.PathLike]],
) -> "list[NapariLayerData]":

    with tempfile.TemporaryDirectory() as tmpdir:
        if not isinstance(path, list):
            path = _split_star(path, tmpdir)

        r = coordsio.to_napari_coordinates(
            path=path,
            read_func=read,
            prepare_napari_func=_prepare_napari_coords,
            meta_columns=["confidence"],
            feature_columns=["fid", "boxsize"],
            valid_extensions=get_valid_extensions(),
        )

    return r


###################
# WRITE FUNCTIONS
###################
def _make_df_data_particle(
    coordinates: pd.DataFrame, **kwargs
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


def _make_df_data_filament(
    coordinates: pd.DataFrame,
    box_size: npt.ArrayLike,
    filament_spacing: int,
    **kwargs
) -> pd.DataFrame:
    data = {
        "_rlnCoordinateX": [],
        "_rlnCoordinateY": [],
        "_rlnHelicalTubeID": [],
    }
    filaments = []
    box_size_per_filament = []
    last_box_size = -1
    for (y, x, fid), boxsize in zip(
        coordinates,
        box_size,
    ):
        if (
            len(data["_rlnHelicalTubeID"]) > 0
            and data["_rlnHelicalTubeID"][-1] != fid
        ):
            filaments.append(pd.DataFrame(data))
            box_size_per_filament.append(last_box_size)
            data = {
                "_rlnCoordinateX": [],
                "_rlnCoordinateY": [],
                "_rlnHelicalTubeID": [],
            }

        data["_rlnCoordinateX"].append(x)
        data["_rlnCoordinateY"].append(y)
        data["_rlnHelicalTubeID"].append(fid)
        last_box_size = boxsize

    # Resampling

    filaments.append(pd.DataFrame(data))
    box_size_per_filament.append(last_box_size)

    ## Resampling
    for index_fil, fil in enumerate(filaments):
        if not filament_spacing:
            distance = int(box_size_per_filament[index_fil] * 0.2)
        else:
            distance = filament_spacing
        filaments[index_fil] = coordsio.resample_filament(
            fil,
            distance,
            coordinate_columns=["_rlnCoordinateX", "_rlnCoordinateY"],
            constant_columns=["_rlnHelicalTubeID"],
        )

    return pd.concat(filaments)


def _write_star(path: os.PathLike, df: pd.DataFrame, **kwargs):

    sfile = star.StarFile(path)

    sfile.update("", df, True)

    sfile.write_star_file(overwrite=True, tags=[""])


def from_napari(
    path: os.PathLike,
    layer_data: list[NapariLayerData],
    suffix: str,
    filament_spacing: int,
):
    is_filament = coordsio.is_filament_layer(layer_data)
    if is_filament:
        format_func = _make_df_data_filament
    else:
        format_func = _make_df_data_particle

    path = coordsio.from_napari(
        path=path,
        layer_data=layer_data,
        write_func=_write_star,
        format_func=format_func,
        suffix=suffix,
        filament_spacing=filament_spacing,
    )

    return path

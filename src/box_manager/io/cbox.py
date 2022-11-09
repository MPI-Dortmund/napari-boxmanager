import os
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
from pyStarDB import sp_pystardb as star
from .interface import NapariLayerData
from . import io_utils as coordsio

valid_extensions = ["cbox"]
coords_3d_idx = ["x", "y", "z"]
coords_2d_idx = ["y", "z"]
meta_columns = []

#########################
# GENERAL STUFF
#########################

### READING ####
################

def get_valid_extensions() -> list[str]:
    return valid_extensions

def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[NapariLayerData]":
    r = coordsio.to_napari(
        path=path,
        read_func=read,
        prepare_napari_func=_prepare_napari,
        meta_columns=_get_meta_columns(),
        feature_columns=_get_meta_columns(),
    )

    return r

def _get_meta_columns():
    return meta_columns

def read(path: os.PathLike) -> pd.DataFrame:
    try:
        starfile = star.StarFile(path)
        data_dict = starfile["cryolo"]
    except TypeError:
        return None
    return pd.DataFrame(data_dict)

### Writing ####
################

def from_napari(
    path: os.PathLike | list[os.PathLike] | pd.DataFrame,
    layer_data: list[NapariLayerData],
):

    output_path = coordsio.from_napari(
        path=path,
        layer_data=layer_data,
        write_func=write_cbox,
        format_func=_make_df_data,
    )
    return output_path

#########################
# FILAMENT STUFF
#########################

### READING ####
################

#########################
# PARTICLE STUFF
#########################

### READING ####
################

def _prepare_napari(input_df: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    input_df Dataframe with raw data from the read function

    Returns
    -------
    Dataframe with centered coordinates and additional metadate if necessary.

    """

    cryolo_data = input_df

    _fill_meta_idx(cryolo_data)
    is_3d = True
    if cryolo_data["_CoordinateZ"].isnull().values.any():
        is_3d = False

    columns = ["z", "y"]
    if is_3d:
        columns.append("x")

    output_data: pd.DataFrame = pd.DataFrame(
        columns=columns + _get_meta_columns()
    )

    output_data["z"] = np.array(cryolo_data["_CoordinateX"]) + np.array(
        cryolo_data["_Width"] / 2
    )
    output_data["y"] = np.array(cryolo_data["_CoordinateY"]) + np.array(
        cryolo_data["_Height"] / 2
    )
    if is_3d:
        output_data["x"] = cryolo_data["_CoordinateZ"]

    output_data["boxsize"] = (
        np.array(cryolo_data["_Width"]) + np.array(cryolo_data["_Height"])
    ) / 2

    if "size" in meta_columns:
        output_data["size"] = (
            np.array(cryolo_data["_EstWidth"])
            + np.array(cryolo_data["_EstHeight"])
        ) / 2

    if "num_boxes" in meta_columns:
        output_data["num_boxes"] = cryolo_data["_NumBoxes"]

    if "confidence" in meta_columns:
        output_data["confidence"] = cryolo_data["_Confidence"]

    if "fid" in meta_columns:
        output_data["fid"] = cryolo_data["_filamentid"]

    return output_data


def _fill_meta_idx(input_df: pd.DataFrame) -> None:
    """
    Fills the meta idx array.

    Parameters
    ----------
    input_dict Raw input data

    Returns
    -------
    None

    """
    global meta_columns

    if (
        not input_df["_EstWidth"].isnull().values.any()
    ) and "size" not in meta_columns:
        meta_columns.append("size")
    if (
        not input_df["_Confidence"].isnull().values.any()
    ) and "confidence" not in meta_columns:
        meta_columns.append("confidence")
    if (
        not input_df["_NumBoxes"].isnull().values.any()
    ) and "num_boxes" not in meta_columns:
        meta_columns.append("num_boxes")
    if (
        not input_df["_filamentid"].isnull().values.any()
    ) and "fid" not in meta_columns:
        meta_columns.append("fid")


### Writing ####
################

def write_cbox(path: os.PathLike, df: pd.DataFrame):
    sfile = star.StarFile(path)

    version_df = pd.DataFrame([["1.0"]], columns=["_cbox_format_version"])
    sfile.update("global", version_df, False)
    include_slices = []
    if not df["_CoordinateZ"].isnull().values.any():
        include_slices = [
            a
            for a in np.unique(df["_CoordinateZ"]).tolist()
            if not np.isnan(a)
        ]

    sfile.update("cryolo", df, True)

    include_df = pd.DataFrame(include_slices, columns=["_slice_index"])
    sfile.update("cryolo_include", include_df, True)

    sfile.write_star_file(
        overwrite=True, tags=["global", "cryolo", "cryolo_include"]
    )

def _make_df_data(
    coordinates: pd.DataFrame, box_size: npt.ArrayLike, features: pd.DataFrame
) -> pd.DataFrame:
    data = {
        "_CoordinateX": [],
        "_CoordinateY": [],
        "_CoordinateZ": [],
        "_Width": [],
        "_Height": [],
        "_Depth": [],
        "_EstWidth": [],
        "_EstHeight": [],
        "_Confidence": [],
        "_NumBoxes": [],
        "_Angle": [],
    }
    for i in range(len(coordinates)):
        coords = coordinates[i]
        boxsize = box_size[i]

        is_3d = True

        if len(coords) == 2:
            is_3d = False
            y, x = coords
            z = np.nan
        else:
            z, y, x = coords

        data["_CoordinateX"].append(x - boxsize / 2)
        data["_CoordinateY"].append(y - boxsize / 2)
        data["_CoordinateZ"].append(z)
        data["_Width"].append(boxsize)
        data["_Height"].append(boxsize)
        if is_3d:
            data["_Depth"].append(boxsize)
        else:
            data["_Depth"].append(np.nan)

        if "size" in features:
            data["_EstWidth"].append(features["size"].iloc[i])
            data["_EstHeight"].append(features["size"].iloc[i])
        else:
            data["_EstWidth"].append(np.nan)
            data["_EstHeight"].append(np.nan)

        if "confidence" in features:
            data["_Confidence"].append(features["confidence"].iloc[i])
        else:
            data["_Confidence"].append(np.nan)

        if "numboxes" in features:
            data["_NumBoxes"].append(features["numboxes"].iloc[i])
        else:
            data["_NumBoxes"].append(np.nan)

        if "angle" in features:
            data["_Angle"].append(features["angle"].iloc[i])
        else:
            data["_Angle"].append(np.nan)

    return pd.DataFrame(data)











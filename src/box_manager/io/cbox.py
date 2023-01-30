import copy
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from pyStarDB import sp_pystardb as star

from . import io_utils as coordsio
from .interface import NapariLayerData

valid_extensions = ["cbox"]
coords_3d_idx = ["x", "y", "z"]
coords_2d_idx = ["y", "z"]
meta_columns = []
feature_columns = []

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
    r = coordsio.to_napari_coordinates(
        path=path,
        read_func=read,
        prepare_napari_func=_prepare_napari,
        meta_columns=_get_meta_columns(),
        feature_columns=_get_feature_columns(),
        valid_extensions=get_valid_extensions(),
    )

    return r


def _get_meta_columns():
    return meta_columns


def _get_feature_columns():
    return feature_columns


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
    suffix: str,
    filament_spacing: int,
):
    is_filament = coordsio.is_filament_layer(layer_data)

    if is_filament:
        format_func = _make_df_data_filament
    else:
        format_func = _make_df_data

    output_path = coordsio.from_napari(
        path=path,
        layer_data=layer_data,
        write_func=write_cbox,
        format_func=format_func,
        suffix=suffix,
        filament_spacing=filament_spacing,
    )
    return output_path


#########################
# FILAMENT STUFF
#########################

### WRITING ####
################


def _make_df_data_filament(
    coordinates: pd.DataFrame,
    box_size: npt.ArrayLike,
    features: pd.DataFrame,
    filament_spacing: int,
    **kwargs,
) -> pd.DataFrame:

    is_3d = coordinates.shape[1] == 4
    data = {}
    data["_CoordinateX"] = []
    data["_CoordinateY"] = []
    if is_3d:
        data["_CoordinateZ"] = []

    data["_Width"] = []
    data["_Height"] = []
    if is_3d:
        data["_Depth"] = []

    data["_filamentid"] = []

    feature_map = {}
    other_interpolation_cols = []
    coord_columns = ["_CoordinateX", "_CoordinateY"]
    constant_columns = ["_filamentid", "_Width", "_Height"]
    if is_3d:
        coord_columns.append("_CoordinateZ")
        constant_columns.append("_Depth")

    if "confidence" in features:
        data["_Confidence"] = []
        feature_map["confidence"] = "_Confidence"
        other_interpolation_cols.append("_Confidence")
    if "angle" in features:
        data["_Angle"] = []
        feature_map["angle"] = "_Angle"
        other_interpolation_cols.append("_Angle")

    empty_data = copy.deepcopy(data)
    filaments = []
    entry = 0
    for coords_and_fid, boxsize in zip(
        coordinates,
        box_size,
    ):
        if is_3d:
            z, y, x, fid = coords_and_fid
        else:
            y, x, fid = coords_and_fid
        if len(data["_filamentid"]) > 0 and data["_filamentid"][-1] != fid:
            filaments.append(pd.DataFrame(data))
            data = copy.deepcopy(empty_data)

        data["_CoordinateX"].append(x - boxsize / 2)
        data["_CoordinateY"].append(y - boxsize / 2)
        data["_filamentid"].append(fid)
        data["_Width"].append(boxsize)
        data["_Height"].append(boxsize)
        if is_3d:
            data["_Depth"].append(boxsize)
            data["_CoordinateZ"].append(z)
        for key in feature_map:
            data[feature_map[key]].append(features[key].iloc[entry])
        entry = entry + 1

    filaments.append(pd.DataFrame(data))

    ## Resampling
    for index_fil, fil in enumerate(filaments):
        if not filament_spacing:
            distance = int(fil["_Width"][0] * 0.2)
        else:
            distance = filament_spacing
        filaments[index_fil] = coordsio.resample_filament(
            fil,
            distance,
            coordinate_columns=coord_columns,
            constant_columns=constant_columns,
            other_interpolation_col=other_interpolation_cols,
        )

    return pd.concat(filaments)


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

    _fill_meta_features_idx(cryolo_data)
    is_3d = True
    if (
        "_CoordinateZ" not in cryolo_data
        or cryolo_data["_CoordinateZ"].isnull().values.any()
    ):
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

    if "angle" in feature_columns:
        output_data["angle"] = cryolo_data["_Angle"]

    if "fid" in feature_columns:
        output_data["fid"] = cryolo_data["_filamentid"]

    return output_data


def _fill_meta_features_idx(input_df: pd.DataFrame) -> None:
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
    global feature_columns

    if (
        "_EstWidth" in input_df.columns
        and not input_df["_EstWidth"].isnull().values.any()
    ) and "size" not in meta_columns:
        meta_columns.append("size")

    if (
        "_Confidence" in input_df.columns
        and not input_df["_Confidence"].isnull().values.any()
    ) and "confidence" not in meta_columns:
        meta_columns.append("confidence")

    if (
        "_NumBoxes" in input_df.columns
        and not input_df["_NumBoxes"].isnull().values.any()
    ) and "num_boxes" not in meta_columns:
        meta_columns.append("num_boxes")

    if (
        "_Angle" in input_df.columns
        and not input_df["_Angle"].isnull().values.any()
    ) and "angle" not in feature_columns:
        feature_columns.append("angle")

    if (
        "_filamentid" in input_df.columns
        and "_filamentid" in input_df
        and not input_df["_filamentid"].isnull().values.any()
    ) and "fid" not in feature_columns:
        feature_columns.append("fid")


### Writing ####
################


def write_cbox(path: os.PathLike, df: pd.DataFrame, **kwargs):

    sfile = star.StarFile(path)

    version_df = pd.DataFrame([["1.0"]], columns=["_cbox_format_version"])
    sfile.update("global", version_df, False)
    include_slices = []
    if "_CoordinateZ" in df.columns:
        if not df["_CoordinateZ"].isnull().values.any():
            include_slices = [
                a
                for a in np.unique(df["_CoordinateZ"]).tolist()
                if not np.isnan(a)
            ]

    if "empty_slices" in kwargs:
        include_slices.extend(kwargs["empty_slices"])
    include_slices.sort()

    sfile.update("cryolo", df, True)

    include_df = pd.DataFrame(include_slices, columns=["_slice_index"])
    sfile.update("cryolo_include", include_df, True)

    sfile.write_star_file(
        overwrite=True, tags=["global", "cryolo", "cryolo_include"]
    )


def _make_df_data(
    coordinates: pd.DataFrame,
    box_size: npt.ArrayLike,
    features: pd.DataFrame,
    **kwargs,
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

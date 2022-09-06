import os
import typing
from pyStarDB import sp_pystardb as star
import pandas as pd
import numpy as np
from .coordinate_reader import to_napari_generic_coordinates

valid_extensions = ['cbox']
coords_3d_idx = ["x", "y", "z"]
coords_2d_idx = ["y", "z"]
meta_idx = []

def read(path: os.PathLike) -> pd.DataFrame:
    starfile = star.StarFile(path)
    data_dict = starfile['cryolo']
    return pd.DataFrame(data_dict)


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":

    return to_napari_generic_coordinates(
        path=path,
        read_func=read,
        prepare_napari_func=_prepare_napari,
        meta_columns=_get_meta_idx(),
        feature_columns=_get_meta_idx()
    )


def get_valid_extensions() -> list[str]:
    return valid_extensions

def from_napari(
    path: os.PathLike | list[os.PathLike] | pd.DataFrame,
    data: typing.Any,
    meta: dict,
):
    pass


def _update_meta_idx(input_dict: typing.Dict):
    global meta_idx

    if not input_dict['_EstWidth'].isnull().values.any():
        meta_idx.append('size')
    if not input_dict['_Confidence'].isnull().values.any():
        meta_idx.append('confidence')
    if not input_dict['_NumBoxes'].isnull().values.any():
        meta_idx.append('num_boxes')


def _prepare_napari(input_dict: typing.Dict) -> pd.DataFrame:

    output_data: pd.DataFrame = pd.DataFrame(
        columns=_get_3d_coords_idx()
        + _get_meta_idx()
    )
    cryolo_data = input_dict
    _update_meta_idx(cryolo_data)
    is_3d=True
    if cryolo_data['_CoordinateX'].isnull().values.any():
        is_3d=False
    output_data.attrs['is_3d'] = is_3d
    output_data['z'] = cryolo_data['_CoordinateX']
    output_data['y'] = cryolo_data['_CoordinateY']
    if is_3d:
        output_data['x'] = cryolo_data['_CoordinateZ']


    output_data["boxsize"] = (np.array(cryolo_data["_Width"]) + np.array(cryolo_data["_Height"]))/2

    if 'size' in meta_idx:
        output_data["size"] = (np.array(cryolo_data["_EstWidth"]) + np.array(cryolo_data["_EstHeight"])) / 2

    if 'num_boxes' in meta_idx:
        output_data["num_boxes"] = cryolo_data["_NumBoxes"]

    if 'confidence' in meta_idx:
        output_data["confidence"] = cryolo_data["_Confidence"]

    return output_data



def _get_3d_coords_idx():
    return coords_3d_idx

def _get_2d_coords_idx():
    return coords_2d_idx


def _get_meta_idx():
    return meta_idx


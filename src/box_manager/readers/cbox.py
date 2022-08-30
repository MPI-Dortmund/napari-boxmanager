import os
import typing
from pyStarDB import sp_pystardb as star
import pandas as pd
import numpy as np

valid_extensions = ['cbox']
coords_3d_idx = ["x", "y", "z"]
meta_idx = ["confidence", "size"]

def read(path: os.PathLike) -> typing.Dict:
    starfile = star.StarFile(path)
    data_dict = starfile['cryolo']
    return data_dict

def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":

    if not isinstance(path, list):
        path = sorted(glob.glob(path))  # type: ignore

    for file_name in path:
        pass

def get_valid_extensions() -> list[str]:
    return valid_extensions

def from_napari(
    path: os.PathLike | list[os.PathLike] | pd.DataFrame,
    data: typing.Any,
    meta: dict,
):
    data_dict = read(path)
    napari_df = _prepare_napari(data_dict)


def _prepare_napari(input_dict: typing.Dict) -> pd.DataFrame:

    output_data: pd.DataFrame = pd.DataFrame(
        columns=_get_3d_coords_idx()
        + _get_meta_idx()
    )
    cryolo_data = input_dict['cryolo']

    output_data['x'] = cryolo_data['_CoordinateZ']
    output_data['y'] = cryolo_data['_CoordinateY']
    output_data['z'] = cryolo_data['_CoordinateX']
    output_data["boxsize"] = (np.array(cryolo_data["_Width"]) + np.array(cryolo_data["_Height"]))/2

    print(cryolo_data['_CoordinateZ'])

def _get_3d_coords_idx():
    return coords_3d_idx


def _get_meta_idx():
    return meta_idx


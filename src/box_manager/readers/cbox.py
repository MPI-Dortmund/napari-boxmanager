import os
import typing
from pyStarDB import sp_pystardb as star
import pandas as pd
import numpy as np
from . import _MAX_LAYER_NAME
from .._qt import OrganizeBox as orgbox

valid_extensions = ['cbox']
coords_3d_idx = ["x", "y", "z"]
coords_2d_idx = ["y", "z"]
meta_idx = []

def read(path: os.PathLike) -> typing.Dict:
    starfile = star.StarFile(path)
    data_dict = starfile['cryolo']
    return data_dict

def get_layer_name(path: os.PathLike | list[os.PathLike]) -> str:
    if isinstance(path, list) and len(path) > 1:
        name = "Coordinates"
    elif isinstance(path, list):
        if len(path[0]) >= _MAX_LAYER_NAME + 3:
            name = f"...{path[0][-_MAX_LAYER_NAME:]}"  # type: ignore
        else:
            name = path[0]  # type: ignore
    else:
        assert False, path

    return name

def _is_3d(pth: list[os.PathLike]):
    cryolo_data = read(pth[0])
    is_3d = True
    if cryolo_data['_CoordinateX'].isnull().values.any():
        is_3d = False

    return is_3d

def _get_coords_idx_func(pth: os.PathLike | list[os.PathLike]):

    if isinstance(pth, list) and len(pth) > 1:
        return _get_3d_coords_idx  # Happens for --stack option and '*.ext'

    if _is_3d(pth):
        return _get_3d_coords_idx
    else:
        return _get_2d_coords_idx

def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":

    if not isinstance(path, list):
        path = sorted(glob.glob(path))  # type: ignore

    input_df, metadata = _prepare_df(
        path
    )

    metadata.update(orgbox.get_metadata(path))

    features = {
        entry: input_df[entry].to_numpy()
        for entry in _get_meta_idx()
    }

    layer_name = get_layer_name(path)

    is_3d_data = _is_3d(path)

    kwargs = {
        "edge_color": "blue",
        "face_color": "transparent",
        "symbol": "disc" if is_3d_data else "square",
        "edge_width": 2,
        "edge_width_is_relative": False,
        "size": input_df["boxsize"],
        "out_of_slice_display": True if is_3d_data else False,
        "opacity": 0.5,
        "name": layer_name,
        "metadata": metadata,
        "features": features,
    }

    idx_func = _get_coords_idx_func(path)
    return [(input_df[idx_func()], kwargs, "points")]


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

def _prepare_df(
    path: list[os.PathLike],
) -> tuple[pd.DataFrame, dict[int, os.PathLike]]:

    data_df: list[pd.DataFrame] = []
    metadata: dict = {}

    for idx, entry in enumerate(path):
        input_data = read(entry)
        box_napari_data = _prepare_napari(input_data, entry_index=idx)
        data_df.append(box_napari_data)

        metadata[idx] = {}
        metadata[idx]["path"] = entry
        metadata[idx]["name"] = os.path.basename(entry)
        metadata[idx]["write"] = None
        metadata[idx].update(
            {
                f"{entry}_{func.__name__}": func(data_df[idx][entry])
                for func in [min, max]
                for entry in _get_meta_idx()
            }
        )
    return pd.concat(data_df, ignore_index=True), metadata

def _prepare_napari(input_dict: typing.Dict, **kwargs) -> pd.DataFrame:

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

    output_data['x'] = cryolo_data['_CoordinateZ']
    output_data['y'] = cryolo_data['_CoordinateY']
    if is_3d:
        output_data['z'] = cryolo_data['_CoordinateX']
    else:
        output_data['z'] = kwargs['image_index']

    output_data["boxsize"] = (np.array(cryolo_data["_Width"]) + np.array(cryolo_data["_Height"]))/2

    if 'size' in meta_idx:
        output_data["size"] = (np.array(cryolo_data["_EstWidth"]) + np.array(cryolo_data["_EstHeight"])) / 2

    if 'num_boxes' in meta_idx:
        output_data["num_boxes"] = cryolo_data["_NumBoxes"]

    if 'confidence' in meta_idx:
        output_data["confidence"] = cryolo_data["_confidence"]

    return output_data



def _get_3d_coords_idx():
    return coords_3d_idx

def _get_2d_coords_idx():
    return coords_2d_idx


def _get_meta_idx():
    return meta_idx


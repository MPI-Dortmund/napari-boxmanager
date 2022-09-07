import os
import typing
from .._qt import OrganizeBox as orgbox
from collections.abc import Callable
import pandas as pd

if typing.TYPE_CHECKING:
    import numpy.typing as npt

_MAX_LAYER_NAME = 30

def _prepare_coords_df(
    path: list[os.PathLike],
    read_func: Callable[[os.PathLike], pd.DataFrame],
    prepare_napari_func: Callable,
    meta_columns: typing.List[str] = []

) -> tuple[pd.DataFrame, dict[int, os.PathLike], bool]:

    data_df: list[pd.DataFrame] = []
    metadata: dict = {}
    is_3d=True
    for idx, entry in enumerate(path):
        input_data = read_func(entry)
        box_napari_data = prepare_napari_func(input_data)

        if 'x' not in box_napari_data:
            is_3d = False
            box_napari_data["x"] = idx
        data_df.append(box_napari_data)

        metadata[idx] = {}
        metadata[idx]["path"] = entry
        metadata[idx]["name"] = os.path.basename(entry)
        metadata[idx]["write"] = None
        metadata[idx].update(
            {
                f"{entry}_{func.__name__}": func(data_df[idx][entry])
                for func in [min, max]
                for entry in meta_columns
            }
        )

    return pd.concat(data_df, ignore_index=True), metadata, is_3d

def get_coords_layer_name(path: os.PathLike | list[os.PathLike]) -> str:
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

def to_napari_generic_coordinates(
        path: os.PathLike | list[os.PathLike],
        read_func: Callable[[os.PathLike], pd.DataFrame],
        prepare_napari_func: Callable,
        meta_columns: typing.List[str] = [],
        feature_columns: typing.List[str] = [],

) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
    input_df: pd.DataFrame
    features: dict[str, typing.Any]

    if not isinstance(path, list):
        path = sorted(glob.glob(path))  # type: ignore

    input_df, metadata, is_3d = _prepare_coords_df(
        path,
        read_func=read_func,
        prepare_napari_func=prepare_napari_func,
        meta_columns=meta_columns
    )

    metadata.update(orgbox.get_metadata(path))

    features = {
        entry: input_df[entry].to_numpy()
        for entry in feature_columns#_get_meta_idx() + _get_hidden_meta_idx()
    }

    layer_name = get_coords_layer_name(path)

    kwargs = {
        "edge_color": "blue",
        "face_color": "transparent",
        "symbol": "disc" if is_3d else "square",
        "edge_width": 2,
        "edge_width_is_relative": False,
        "size": input_df["boxsize"],
        "out_of_slice_display": True if is_3d else False,
        "opacity": 0.5,
        "name": layer_name,
        "metadata": metadata,
        "features": features,
    }

    if (isinstance(path, list) and len(path) > 1) or is_3d:
        coord_columns = ["x", "y", "z"]  # Happens for --stack option and '*.ext'
    else:
        coord_columns = ["y", "z"]

    return [(input_df[coord_columns], kwargs, "points")]
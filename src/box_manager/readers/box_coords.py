import glob
import os
import pathlib
import typing
from collections.abc import Callable

import numpy as np
import pandas as pd
from . import _MAX_LAYER_NAME

if typing.TYPE_CHECKING:
    import numpy.typing as npt

from .._qt import OrganizeBox as orgbox


class BoxFileNumberOfColumnsError(pd.errors.IntCastingNaNError):
    pass

class UnknownFormatException(Exception):
    ...

DEFAULT_BOXSIZE: int = 10


def get_valid_extensions():

    return ["box", "coords"]

def read(path: "os.PathLike") -> pd.DataFrame:
    names =["x", "y", "box_x", "box_y"]
    if os.path.splitext(path)[1]==".coords":
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

def get_idx_func(pth: os.PathLike | list[os.PathLike]):

    if isinstance(pth, list) and len(pth) > 1:
        return _get_3d_coords_idx  # Happens for --stack option and '*.ext'

    if is_3d(pth):
        return _get_3d_coords_idx
    else:
        return _get_2d_coords_idx

def is_3d(pth: list[os.PathLike]):
    if os.path.splitext(pth[0])[1] == ".coords":
        return True
    else:
        return False

def get_data_prepare(pth: os.PathLike) -> Callable:
    if os.path.splitext(pth)[1] == ".coords":
        return _prepare_napari_coords
    elif os.path.splitext(pth)[1] == ".box":
        return _prepare_napari_box
    else:
        raise UnknownFormatException(f"{os.path.splitext(pth[0])[1]} is not supported.")


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

def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
    input_df: pd.DataFrame
    name: str
    features: dict[str, typing.Any]

    if not isinstance(path, list):
        original_path = path
        path = sorted(glob.glob(path))  # type: ignore
    else:
        original_path = path[0]


    is_3d_data = is_3d(path)


    input_df, metadata = _prepare_df(
        path
    )
    metadata.update(orgbox.get_metadata(path))

    features = {
        entry: input_df[entry].to_numpy()
        for entry in _get_meta_idx() + _get_hidden_meta_idx()
    }

    layer_name = get_layer_name(path)
    idx_func = get_idx_func(path)

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

    return [(input_df[idx_func()], kwargs, "points")]


def _get_3d_coords_idx():
    return ["x", "y", "z"]


def _get_2d_coords_idx():
    return ["y", "z"]


def _get_meta_idx():
    return []


def _get_hidden_meta_idx():
    return []


def _prepare_napari_box(
    input_df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    output_data: pd.DataFrame = pd.DataFrame(
        columns=_get_3d_coords_idx() + _get_meta_idx() + _get_hidden_meta_idx()
    )

    output_data["z"] = input_df["x"] + input_df["box_x"] // 2
    output_data["y"] = input_df["y"] + input_df["box_y"] // 2
    output_data["x"] = kwargs['entry_index']
    output_data["boxsize"] = np.maximum(
        input_df[["box_x", "box_y"]].mean(axis=1), DEFAULT_BOXSIZE
    ).astype(int)

    return output_data

def _prepare_napari_coords(
    input_df: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    output_data: pd.DataFrame = pd.DataFrame(
        columns=_get_3d_coords_idx() + _get_meta_idx() + _get_hidden_meta_idx()
    )

    output_data["z"] = input_df["x"]
    output_data["y"] = input_df["y"]
    output_data["x"] = input_df["z"]
    output_data["boxsize"] = DEFAULT_BOXSIZE

    return output_data


def _prepare_df(
    path: list[os.PathLike],
) -> tuple[pd.DataFrame, dict[int, os.PathLike]]:
    data_df: list[pd.DataFrame] = []
    metadata: dict = {}
    for idx, entry in enumerate(path):

        conv = get_data_prepare(entry)
        data_df.append(conv(read(entry), entry_index=idx))
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



def _write_box(path : os.PathLike, df: pd.DataFrame):
    df['x'] = df['x'] - df['boxsize'] // 2
    df['y'] = df['y'] - df['boxsize'] // 2
    df[['x','y','boxsize','boxsize']].to_csv(path,sep = " ", index=None,header=None)


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
        ext = os.path.splitext(path)[1]
        export_data = {}
        if ext==".coords":
            coords_writer=_write_coords
            export_data[path] = _make_df_data(coordinates, boxsize)
        elif ext==".box":
            for z in np.unique(coordinates[:,0]):
                z = int(z)
                mask = coordinates[:,0]==z
                filename = meta['metadata'][z]['name']
                dirname = os.path.dirname(path)
                basename, extension = os.path.splitext(os.path.basename(path))
                if not extension:
                    basename, extension = extension, basename

                file_base = os.path.splitext(os.path.basename(filename))[0]
                output_file = pathlib.Path(
                    dirname, file_base+extension
                )
                export_data[output_file] = _make_df_data(coordinates[mask], boxsize[mask])
            coords_writer = _write_box

        for outpth in export_data:
            df = pd.DataFrame(export_data[outpth])
            coords_writer(outpth,df)


    return path

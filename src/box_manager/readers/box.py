import os
import pathlib
import typing

import numpy as np
import pandas as pd
from .coordinate_io import to_napari_generic_coordinates


if typing.TYPE_CHECKING:
    import numpy.typing as npt


class BoxFileNumberOfColumnsError(pd.errors.IntCastingNaNError):
    pass

class UnknownFormatException(Exception):
    ...

DEFAULT_BOXSIZE: int = 10


def get_valid_extensions():

    return ["box"]

def read(path: "os.PathLike") -> pd.DataFrame:
    names =["x", "y", "box_x", "box_y"]
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
    r = to_napari_generic_coordinates(
        path=path,
        read_func=read,
        prepare_napari_func=_prepare_napari_box,
        meta_columns=_get_meta_idx(),
        feature_columns=_get_meta_idx() + _get_hidden_meta_idx()
    )
    return r

def _get_meta_idx():
    return []


def _get_hidden_meta_idx():
    return []


def _prepare_napari_box(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    output_data: pd.DataFrame = pd.DataFrame(
        columns=["y", "z"] + _get_meta_idx() + _get_hidden_meta_idx()
    )

    output_data["z"] = input_df["x"] + input_df["box_x"] // 2
    output_data["y"] = input_df["y"] + input_df["box_y"] // 2

    output_data["boxsize"] = np.maximum(
        input_df[["box_x", "box_y"]].mean(axis=1), DEFAULT_BOXSIZE
    ).astype(int)


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

def _write_box(path : os.PathLike, df: pd.DataFrame):

    df['x'] = df['x'] - df['boxsize'] // 2
    df['y'] = df['y'] - df['boxsize'] // 2
    df[['x','y','boxsize','boxsize']].to_csv(path,sep = " ", index=None,header=None)

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

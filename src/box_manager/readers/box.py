import glob
import os
import pathlib
import typing
from collections.abc import Callable

import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    import numpy.typing as npt


class BoxFileNumberOfColumnsError(pd.errors.IntCastingNaNError):
    pass


def get_valid_extensions():
    return ["box"]


def read(path: "os.PathLike") -> pd.DataFrame:
    box_data: pd.DataFrame = pd.read_csv(
        path,
        delim_whitespace=True,
        index_col=False,
        header=None,
        dtype=float,
        names=["x", "y", "box_x", "box_y"],
        usecols=range(4),
    )  # type: ignore
    try:
        box_data.astype(int)
    except pd.errors.IntCastingNaNError:
        raise BoxFileNumberOfColumnsError

    return box_data


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
    input_df: pd.DataFrame
    name: str
    features: dict[str, typing.Any]

    if not isinstance(path, list):
        path = glob.glob(path)  # type: ignore

    if isinstance(path, list) and len(path) > 1:
        idx_func: Callable[[], list[str]] = _get_3d_coords_idx
        name = "boxfiles"
    elif isinstance(path, list):
        idx_func: Callable[[], list[str]] = _get_2d_coords_idx
        name = path[0]  # type: ignore
    else:
        assert False, path
    input_df, metadata = _prepare_df(
        path if isinstance(path, list) else [path]
    )

    features = {
        entry: input_df[entry].to_numpy()
        for entry in _get_meta_idx() + _get_hidden_meta_idx()
    }
    kwargs = {
        "edge_color": "blue",
        "face_color": "transparent",
        "symbol": "square",
        "edge_width": 2,
        "edge_width_is_relative": False,
        "size": input_df["boxsize"],
        "out_of_slice_display": False,
        "opacity": 0.5,
        "name": name,
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


def _prepare_napari(
    input_df: pd.DataFrame,
    z_value: int = 1,
) -> pd.DataFrame:
    output_data: pd.DataFrame = pd.DataFrame(
        columns=_get_3d_coords_idx() + _get_meta_idx() + _get_hidden_meta_idx()
    )

    output_data["z"] = input_df["x"] + input_df["box_x"] // 2
    output_data["y"] = input_df["y"] + input_df["box_y"] // 2
    output_data["x"] = z_value
    output_data["boxsize"] = np.maximum(
        input_df[["box_x", "box_y"]].mean(axis=1), 10
    ).astype(int)

    return output_data


def _prepare_df(
    path: list[os.PathLike],
) -> tuple[pd.DataFrame, dict[int, os.PathLike]]:
    data_df: list[pd.DataFrame] = []
    metadata: dict = {}
    for idx, entry in enumerate(path):
        data_df.append(_prepare_napari(read(entry), idx))
        metadata[idx] = {}
        metadata[idx]["path"] = entry
        metadata[idx]["name"] = os.path.basename(entry)
        metadata[idx].update(
            {
                f"{entry}_{func.__name__}": func(data_df[idx][entry])
                for func in [min, max]
                for entry in _get_meta_idx()
            }
        )

    return pd.concat(data_df, ignore_index=True), metadata


def from_napari(
    path: os.PathLike, layer_data: list[tuple[typing.Any, dict, str]]
):
    dirname = os.path.dirname(path)
    basename, extension = os.path.splitext(os.path.dirname(path))
    if not extension:
        basename, extension = extension, basename

    for data, meta, layer in layer_data:
        # lines: list[str] = []

        if data.shape[1] == 2:
            data = np.insert(data, 0, 0, axis=1)
        else:
            assert False, data

        output_lines = {}
        for (z, y, x), boxsize, file_name in zip(
            data[meta["shown"]],
            meta["features"]["boxsize"][meta["shown"]],
            meta["features"]["identifier"][meta["shown"]],
        ):
            if len(layer_data) == 1:
                output_file = path
            else:
                file_base = os.path.splitext(os.path.basename(file_name))[0]
                output_file = pathlib.Path(
                    dirname, basename, file_base, extension
                )

            output_lines.setdefault(output_file, []).append(
                f"{x-boxsize//2}  {y-boxsize//2}  {boxsize}  {boxsize}\n"
            )

        for file_name, box_line in output_lines.items():
            with open(file_name, "w") as write:
                write.writelines(box_line)

    return path

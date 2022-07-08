import glob
import os
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
        box_data = box_data.astype(int)
    except pd.errors.IntCastingNaNError:
        raise BoxFileNumberOfColumnsError

    box_data["filename"] = path

    return box_data


def to_napari(
    path: os.PathLike | list[os.PathLike] | pd.DataFrame,
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
    input_df: pd.DataFrame
    name: str
    features: dict[str, typing.Any]

    if not isinstance(path, pd.DataFrame):
        if not isinstance(path, list):
            path = glob.glob(path)

        if isinstance(path, list) and len(path) > 1:
            idx_func: Callable[[], list[str]] = _get_3d_coords_idx
            name = "boxfiles"
        elif isinstance(path, list):
            idx_func: Callable[[], list[str]] = _get_2d_coords_idx
            name = path[0]
        else:
            assert False, path
        input_df = _prepare_df(path if isinstance(path, list) else [path])
    else:
        input_df = path
        name = "boxfiles"

    features = {entry: input_df[entry].to_numpy() for entry in _get_meta_idx()}
    features["identifier"] = input_df["sort_idx"]
    features["group"] = input_df["grp_idx"]
    metadata = {
        f"{entry}_{func.__name__}": func(input_df[entry])
        for func in [min, max]
        for entry in _get_meta_idx()
    }
    kwargs = {
        "edge_color": "blue",
        "face_color": "transparent",
        "symbol": "disc",
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
    return ["boxsize"]


def _prepare_napari(
    input_df: pd.DataFrame,
    z_value: int = 1,
) -> pd.DataFrame:
    coords_idx = _get_3d_coords_idx()
    metric_idx = _get_meta_idx()
    util_idx = ["sort_idx", "grp_idx"]
    output_data: pd.DataFrame = pd.DataFrame(
        columns=coords_idx + metric_idx + util_idx
    )

    output_data["z"] = input_df["x"] + input_df["box_x"] // 2
    output_data["y"] = input_df["y"] + input_df["box_y"] // 2
    output_data["x"] = z_value
    output_data["boxsize"] = np.maximum(
        input_df[["box_x", "box_y"]].mean(axis=1), 10
    ).astype(int)
    output_data["sort_idx"] = input_df["filename"]
    output_data["grp_idx"] = input_df["filename"]

    return output_data


def _prepare_df(path: list[os.PathLike]) -> pd.DataFrame:
    data_df: list[pd.DataFrame] = []
    for idx, entry in enumerate(path):
        data_df.append(_prepare_napari(read(entry), idx))
    return pd.concat(data_df, ignore_index=True)


def from_napari(path: str, data: typing.Any, meta: dict):
    lines: list[str] = []
    shown: list[bool] = meta["shown"]

    if data.shape[1] == 2:
        for (y, x), boxsize in zip(
            data[shown], meta["features"]["boxsize"][shown]
        ):
            lines.append(
                f"{x-boxsize//2}  {y-boxsize//2}  {boxsize}  {boxsize}\n"
            )

        with open(path, "w") as write:
            write.writelines(lines)

    elif data.shape[1] == 3:
        print(data)
    else:
        assert False, data

    return path

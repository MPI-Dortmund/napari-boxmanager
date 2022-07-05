import os
import typing

import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    pass


class BoxFileNumberOfColumnsError(pd.errors.IntCastingNaNError):
    pass


def read(path: "os.PathLike") -> pd.DataFrame:
    box_data: pd.DataFrame = pd.read_csv(
        path,
        delim_whitespace=True,
        index_col=False,
        header=None,
        dtype=float,
        names=["x", "y", "box_x", "box_y"],
        usecols=range(4),
    )
    try:
        box_data = box_data.astype(int)
    except pd.errors.IntCastingNaNError:
        raise BoxFileNumberOfColumnsError

    box_data["filename"] = path

    return box_data


def prepare_napari(
    input_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, typing.Any]]:
    coords_idx = ["x", "y"]
    metric_idx = ["boxsize"]
    util_idx = ["sort_idx", "grp_idx"]
    output_data: pd.DataFrame = pd.DataFrame(
        columns=coords_idx + metric_idx + util_idx
    )

    output_data["x"] = input_df["y"] + input_df["box_y"] // 2
    output_data["y"] = input_df["x"] + input_df["box_x"] // 2
    output_data["boxsize"] = np.maximum(
        input_df[["box_x", "box_y"]].mean(axis=1), 10
    ).astype(int)
    output_data["sort_idx"] = input_df["filename"]
    output_data["grp_idx"] = input_df["filename"]

    return output_data, coords_idx, metric_idx, {"out_of_slice_display": False}

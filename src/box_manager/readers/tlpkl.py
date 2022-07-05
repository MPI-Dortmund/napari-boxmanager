import os
import typing

import pandas as pd

if typing.TYPE_CHECKING:
    pass


class DimZMissingWarning(Warning):
    pass


def read(path: "os.PathLike") -> pd.DataFrame:
    """
    Read a tlpkl conform file into memory.
    tlpkl files need to have their coordinates for the Z and X dimension inverted.

    :param path: Path of the file to read information from.
    :type path: Path object or str

    :return: DataFrame containing the coordinates, box size, metrics, and cluster size
    :rtype: Pandas DataFrame
    """
    pandas_data: pd.DataFrame = pd.read_pickle(path)

    pandas_data["X"] = pandas_data["X"].astype(int)
    pandas_data["Y"] = pandas_data["Y"].astype(int)
    pandas_data["Z"] = pandas_data["Z"].astype(int)

    return pandas_data


def prepare_napari(
    input_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], typing.Any]:
    coords_idx = ["x", "y", "z"]
    metric_idx = ["metric", "size", "boxsize"]
    util_idx = ["sort_idx", "grp_idx"]
    output_data: pd.DataFrame = pd.DataFrame(
        columns=coords_idx + metric_idx + util_idx
    )

    output_data["x"] = input_df["Z"]
    output_data["y"] = input_df["Y"]
    output_data["z"] = input_df["X"]
    output_data["metric"] = input_df["metric_best"]
    output_data["size"] = input_df["size"]
    output_data["boxsize"] = input_df[["height", "width", "depth"]].mean(
        axis=1
    )
    output_data["sort_idx"] = input_df["predicted_class"]
    output_data["grp_idx"] = input_df["predicted_class_name"]

    return output_data, coords_idx, metric_idx, {}

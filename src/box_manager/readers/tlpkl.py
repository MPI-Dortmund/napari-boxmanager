import os
import typing
import warnings

import matplotlib.cm as mcm
import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:

    import numpy.typing as npt


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


def _prepare_napari(
    input_data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rename_columns = {
        "X": "z",
        "Y": "y",
        "Z": "x",
        "width": "boxsize",
        "metric_best": "metric",
    }
    output_data: pd.DataFrame | None = input_data.rename(
        columns=rename_columns
    )

    if output_data is None:
        assert False, "Inplace option cannot be enabled"

    if "dim_z" not in output_data.attrs:
        warnings.warn(
            "dim_z attribute missing in pkl file! Please invert your x dimension values manually!",  # noqa: E501
            DimZMissingWarning,
        )
    else:
        output_data["x"] = output_data.attrs["dim_z"] - output_data["x"]
    output_data["boxsize"] = (
        output_data[["height", "boxsize", "depth"]].mean().mean()
    )

    return output_data, ["x", "y", "z"], ["metric", "size", "boxsize"]


def to_napari(
    input_data: "os.PathLike | pd.DataFrame",
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
    """
    Read a tlpkl conform file into memory to use within napari.

    :param path: Path of the file to read information from.
    :type path: pathlib.Path or str

    :return: Data to create a point, i.e., coords, point_kwargs, and type
    :rtype: list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]
    """
    if isinstance(input_data, os.PathLike):
        input_data = read(input_data)
    data, coords_idx, metadata_idx = _prepare_napari(input_data)

    group_name = "predicted_class_name"
    colors = mcm.get_cmap("gist_rainbow")
    n_classes = np.unique(input_data[group_name]).size

    output_list: "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]" = []
    for idx, (cluster_name, cluster_df) in enumerate(
        data.sort_values(by=["predicted_class", "metric"]).groupby(
            group_name, sort=False
        )
    ):
        color = colors(idx / (n_classes - 1))
        metadata = {
            f"{entry}_{func.__name__}": func(cluster_df[entry])
            for func in [min, max]
            for entry in metadata_idx
        }
        metadata["id"] = cluster_df["predicted_class"].iloc[0]

        # to_numpy currently needed. Should be fixed in 0.4.16rc8
        features = {
            entry: cluster_df[entry].to_numpy() for entry in metadata_idx
        }

        point_kwargs = {
            "edge_color": [color],
            "face_color": "transparent",
            "symbol": "disc",
            "edge_width": 2,
            "edge_width_is_relative": False,
            "size": cluster_df["boxsize"],
            "out_of_slice_display": True,
            "opacity": 0.5,
            "name": cluster_name,
            "metadata": metadata,
            "features": features,
        }
        output_list.append((cluster_df[coords_idx], point_kwargs, "points"))

    return output_list

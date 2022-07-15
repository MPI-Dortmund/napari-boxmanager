import glob
import os
import typing

import matplotlib.cm as mcm
import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def read(path: "os.PathLike") -> pd.DataFrame:
    """
    Read a tloc conform file into memory.
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


def get_valid_extensions():
    return ["tloc"]


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
    input_df: pd.DataFrame
    name: str
    features: dict[str, typing.Any]

    if not isinstance(path, list):
        path = glob.glob(path)  # type: ignore

    output_dfs = []
    for file_name in path:  # type: ignore
        input_df = read(file_name)
        napari_df = _prepare_napari(input_df)
        for cluster_name, cluster_df in napari_df.groupby(
            "grp_idx", sort=False
        ):
            output_dfs.append((file_name, cluster_name, cluster_df))

    colors = mcm.get_cmap("gist_rainbow")
    n_layers = np.maximum(len(output_dfs), 2)  # Avoid zero division

    output_layers = []
    for idx, (file_name, cluster_name, cluster_df) in enumerate(output_dfs):
        cur_color = colors(idx / (n_layers - 1))
        metadata = {}
        for idx in range(cluster_df["x"].max() + 1):
            metadata[idx] = {"path": file_name, "name": f"slice {idx}"}
            idx_view_df = cluster_df[cluster_df["x"] == idx]
            metadata[idx].update(
                {
                    f"{entry}_{func.__name__}": func(
                        idx_view_df[entry], default=0
                    )
                    for func in [min, max]
                    for entry in _get_meta_idx()
                }
            )
        features = {
            entry: cluster_df[entry].to_numpy()
            for entry in _get_meta_idx() + _get_hidden_meta_idx()
        }
        kwargs = {
            "edge_color": [cur_color],
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
        output_layers.append(
            (cluster_df[_get_3d_coords_idx()], kwargs, "points")
        )

    return output_layers


def from_napari(
    path: os.PathLike | list[os.PathLike] | pd.DataFrame,
    data: typing.Any,
    meta: dict,
):
    raise NotImplementedError


def _prepare_napari(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    output_data: pd.DataFrame = pd.DataFrame(
        columns=_get_3d_coords_idx()
        + _get_meta_idx()
        + _get_hidden_meta_idx()
        + _get_util_idx()
    )

    output_data["x"] = input_df["Z"]
    output_data["y"] = input_df["Y"]
    output_data["z"] = input_df["X"]
    output_data["metric"] = input_df["metric_best"]
    output_data["size"] = input_df["size"]
    output_data["boxsize"] = input_df[["height", "width", "depth"]].mean(
        axis=1
    )
    output_data["grp_idx"] = input_df["predicted_class_name"]

    return output_data


def _prepare_df(input_df: pd.DataFrame):
    pass


def _get_3d_coords_idx():
    return ["x", "y", "z"]


def _get_meta_idx():
    return ["metric", "size"]


def _get_hidden_meta_idx():
    return ["boxsize"]


def _get_util_idx():
    return ["grp_idx"]

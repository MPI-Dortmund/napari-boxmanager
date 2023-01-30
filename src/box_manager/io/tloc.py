import glob
import os
import typing

import matplotlib.cm as mcm
import numpy as np
import numpy.typing as npt
import pandas as pd

from .._utils import general
from . import io_utils as coordsio
from .interface import NapariLayerData
from .io_utils import MAX_LAYER_NAME

if typing.TYPE_CHECKING:
    pass


def write(path: "os.PathLike", output_df: pd.DataFrame):
    output_df.to_pickle(path)


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

    pandas_data["X"] = pandas_data["X"]
    pandas_data["Y"] = pandas_data["Y"]
    pandas_data["Z"] = pandas_data["Z"]

    return pandas_data


def get_valid_extensions():
    return ["tloc"]


def _get_meta_idx():
    return ["metric", "size"]


def _get_feature_idx():
    return ["grp_idx"]


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[NapariLayerData]":
    file_name: str

    if not isinstance(path, list):
        path = sorted(glob.glob(path))  # type: ignore

    output_dfs = []
    for file_name in path:  # type: ignore
        input_df = read(file_name)
        r = coordsio.to_napari_coordinates(
            path=path,
            read_func=read,
            prepare_napari_func=_prepare_napari_box,
            meta_columns=_get_meta_idx(),
            feature_columns=_get_feature_idx(),
            valid_extensions=get_valid_extensions(),
        )

        data, kwargs, napari_type = r[0]
        metadata = kwargs["metadata"]
        features = pd.DataFrame(kwargs["features"])
        for cluster_id, cluster_df in features.groupby("grp_idx", sort=False):
            new_kwargs = kwargs.copy()
            new_metadata = {}
            for key, value in metadata.items():
                if isinstance(key, int):
                    if key in set(cluster_df.index):
                        new_metadata[key] = value
                else:
                    new_metadata[key] = value

            path = input_df.attrs["references"][cluster_id]
            if len(file_name) >= MAX_LAYER_NAME + 3:
                name = f"...{path[-MAX_LAYER_NAME:]}"  # type: ignore
            else:
                name = path  # type: ignore

            new_metadata["is_2d_stack"] = False
            new_metadata["references"] = input_df.attrs
            new_metadata["predicted_class"] = cluster_id

            new_kwargs["name"] = name
            new_kwargs["metadata"] = new_metadata
            new_kwargs["features"] = cluster_df.drop(
                columns=["grp_idx"]
            ).reset_index(drop=True)
            output_dfs.append(
                (
                    data.iloc[cluster_df.index].reset_index(drop=True),
                    new_kwargs,
                    napari_type,
                )
            )

    colors = mcm.get_cmap("gist_rainbow")
    n_layers = np.maximum(len(output_dfs), 2)  # Avoid zero division

    for cidx, (data, kwargs, _) in enumerate(output_dfs):
        metadata = kwargs["metadata"]
        for idx in range(int(data[["x", "y", "z"]].max().max().round(0)) + 1):
            idx_view_df = kwargs["features"].loc[data["x"].round(0) == idx, :]
            metadata[idx] = {
                "path": kwargs["metadata"]["original_path"],
                "name": "slice",
                "write": None,
            }
            metadata[idx].update(
                {
                    f"{entry}_{'min' if 'min' in func.__name__ else 'max'}": func(
                        idx_view_df[entry]
                    )
                    for func in [general.get_min_floor, general.get_max_floor]
                    for entry in _get_meta_idx()
                }
            )

        cur_color = colors(cidx / (n_layers - 1))
        kwargs["edge_color"] = [cur_color]

    return output_dfs


# def to_napari_old(
#    path: os.PathLike | list[os.PathLike],
# ) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
#    print("OLD")
#    input_df: pd.DataFrame
#    name: str
#    features: dict[str, typing.Any]
#
#    if not isinstance(path, list):
#        path = sorted(glob.glob(path))  # type: ignore
#
#    output_dfs = []
#    for file_name in path:  # type: ignore
#        input_df = read(file_name)
#        napari_df = _prepare_napari(input_df)
#        for cluster_id, cluster_df in napari_df.groupby("grp_idx", sort=False):
#            path = input_df.attrs["references"][cluster_id]
#            if len(path) >= MAX_LAYER_NAME + 3:
#                name = f"...{path[-MAX_LAYER_NAME:]}"  # type: ignore
#            else:
#                name = path  # type: ignore
#            output_dfs.append(
#                (
#                    cluster_id,
#                    file_name,
#                    name,
#                    cluster_df,
#                    input_df.attrs,
#                )
#            )
#
#    colors = mcm.get_cmap("gist_rainbow")
#    n_layers = np.maximum(len(output_dfs), 2)  # Avoid zero division
#
#    output_layers = []
#    for idx, (
#        cluster_id,
#        file_name,
#        cluster_name,
#        cluster_df,
#        attrs,
#    ) in enumerate(output_dfs):
#        cur_color = colors(idx / (n_layers - 1))
#        metadata = {
#            "input_attrs": attrs,
#            "predicted_class": cluster_id,
#            "set_lock": True,
#        }
#        for idx in range(
#            int(cluster_df[["x", "y", "z"]].max().max().round(0)) + 1
#        ):
#            idx_view_df = cluster_df.loc[cluster_df["x"].round(0) == idx, :]
#            metadata[idx] = {
#                "path": file_name,
#                "name": "slice",
#                "write": None,
#            }
#            metadata[idx].update(
#                {
#                    f"{entry}_{'min' if 'min' in func.__name__ else 'max'}": func(
#                        idx_view_df[entry]
#                    )
#                    for func in [general.get_min_floor, general.get_max_floor]
#                    for entry in _get_meta_idx()
#                }
#            )
#        features = {
#            entry: cluster_df[entry].to_numpy()
#            for entry in _get_meta_idx() + _get_hidden_meta_idx()
#        }
#        kwargs = {
#            "edge_color": [cur_color],
#            "face_color": "transparent",
#            "symbol": "disc",
#            "edge_width": 2,
#            "edge_width_is_relative": False,
#            "size": cluster_df["boxsize"],
#            "out_of_slice_display": True,
#            "opacity": 0.5,
#            "name": cluster_name,
#            "metadata": metadata,
#            "features": features,
#        }
#        output_layers.append(
#            (cluster_df[_get_3d_coords_idx()], kwargs, "points")
#        )
#
#    return output_layers
#
#


def _format_tloc(
    coordinates: pd.DataFrame,
    box_size: npt.ArrayLike,
    features: pd.DataFrame,
    metadata: dict[str, typing.Any],
    **kwargs,
) -> pd.DataFrame:
    column_names = [
        "X",
        "Y",
        "Z",
        "predicted_class",
        "size",
        "metric_best",
        "width",
        "height",
        "depth",
    ]

    data_df = pd.DataFrame(columns=column_names)
    data_df["X"] = coordinates[..., 2]
    data_df["Y"] = coordinates[..., 1]
    data_df["Z"] = coordinates[..., 0]

    data_df["size"] = features["size"].to_numpy()
    data_df["metric_best"] = features["metric"].to_numpy()

    data_df["predicted_class"] = metadata["predicted_class"]
    data_df["width"] = box_size
    data_df["height"] = box_size
    data_df["depth"] = box_size
    data_df.attrs = metadata["references"]

    return data_df


def _write_tloc(path: os.PathLike, df: pd.DataFrame, **kwargs):
    write(path, df)


def from_napari(
    path: os.PathLike,
    layer_data: list[NapariLayerData],
    suffix: str,
    filament_spacing: int,
):
    path = coordsio.from_napari(
        path=path,
        layer_data=layer_data,
        write_func=_write_tloc,
        format_func=_format_tloc,
        suffix=suffix,
        filament_spacing=filament_spacing,
    )
    return path


# def from_napari_old(
#    path: os.PathLike,
#    layer_data: typing.Any,
# ):
#    column_names = [
#        "X",
#        "Y",
#        "Z",
#        "predicted_class",
#        "size",
#        "metric_best",
#        "width",
#        "height",
#        "depth",
#    ]
#    output_dfs = []
#    for coords, meta, _ in layer_data:
#        shown_mask = meta["shown"]
#        data_df = pd.DataFrame(columns=column_names)
#        features = meta["features"]
#        data_df["X"] = coords[shown_mask, 2]
#        data_df["Y"] = coords[shown_mask, 1]
#        data_df["Z"] = coords[shown_mask, 0]
#        data_df["predicted_class"] = meta["metadata"]["predicted_class"]
#        data_df["size"] = features.loc[shown_mask, "size"].to_numpy()
#        data_df["metric_best"] = features.loc[shown_mask, "metric"].to_numpy()
#        data_df["width"] = meta["size"][shown_mask, 2]
#        data_df["height"] = meta["size"][shown_mask, 1]
#        data_df["depth"] = meta["size"][shown_mask, 0]
#        data_df.attrs = meta["metadata"]["input_attrs"]
#        output_dfs.append(data_df)
#
#    output_df = pd.concat(output_dfs, ignore_index=True, axis=0)
#    write(path, output_df)
#
#    return path


def _prepare_napari_box(
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
    output_data["grp_idx"] = input_df["predicted_class"]

    return output_data


def _get_3d_coords_idx():
    return ["x", "y", "z"]


def _get_hidden_meta_idx():
    return []


def _get_util_idx():
    return ["grp_idx"]

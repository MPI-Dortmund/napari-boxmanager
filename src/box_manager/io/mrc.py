import glob
import os
import typing

import mrcfile
import numpy as np
import pandas as pd

from .io_utils import MAX_LAYER_NAME, PROXY_THRESHOLD_GB, LoaderProxy

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def load_mrc(path):
    with mrcfile.open(path, "r") as mrc:
        data = mrc.data
    return data


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
    if not isinstance(path, list):
        original_path = path
        if len(path) >= MAX_LAYER_NAME + 3:
            name = f"...{path[-MAX_LAYER_NAME:]}"  # type: ignore
        else:
            name = path  # type: ignore
        path = sorted(glob.glob(path))  # type: ignore
        if len(path) > 1:
            name = "mrcfiles"
    else:
        original_path = path[0]
        name = "mrcfiles"

    # arrays = []
    voxel_size = 1
    metadata: dict = {
        "pixel_spacing": voxel_size,
        "original_path": original_path,
    }
    for idx, file_name in enumerate(path):
        metadata[idx] = {}
        metadata[idx]["path"] = file_name
        metadata[idx]["name"] = os.path.basename(file_name)

    with mrcfile.open(path[0], permissive=True, header_only=True) as mrc:
        metadata["pixel_spacing"] = (
            mrc.voxel_size.x if mrc.voxel_size.x != 0 else 1
        )

    file_size = (
        sum(os.stat(file_name).st_size for file_name in path) / 1024**3
    )
    if len(path) > 1 and file_size > PROXY_THRESHOLD_GB:
        data = LoaderProxy(path, load_mrc)
    else:
        data_list = []
        for file_name in path:
            with mrcfile.open(file_name, permissive=True) as mrc:
                tmp_data = mrc.data
                tmp_data = (tmp_data - np.mean(tmp_data)) / np.std(tmp_data)
                data_list.append(tmp_data)
        data = np.squeeze(np.stack(data_list))


    metadata["is_3d"] = len(path) == 1 and data.ndim == 3
    metadata["is_2d_stack"] = len(path) > 1 and data.ndim == 3
    add_kwargs = {"metadata": metadata, "name": name}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]


def get_valid_extensions():
    return ["mrc", "mrcs", "st", "rec"]


def from_napari(
    path: os.PathLike | list[os.PathLike] | pd.DataFrame,
    data: typing.Any,
    meta: dict,
):
    raise NotImplementedError

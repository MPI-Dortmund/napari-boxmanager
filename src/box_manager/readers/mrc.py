import glob
import os
import typing

import mrcfile
import numpy as np
import pandas as pd

from . import _MAX_LAYER_NAME

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
    if not isinstance(path, list):
        original_path = path
        if "*" in path:
            name = "mrcfiles"
        elif len(path) >= _MAX_LAYER_NAME + 3:
            name = f"...{path[-_MAX_LAYER_NAME:]}"  # type: ignore
        else:
            name = path  # type: ignore
        path = sorted(glob.glob(path))  # type: ignore
    elif len(path[0]) >= _MAX_LAYER_NAME + 3:
        original_path = path[0]
        name = f"...{path[0][-_MAX_LAYER_NAME:]}"  # type: ignore
    else:
        name = path[0]  # type: ignore
        original_path = path[0]

    arrays = []
    voxel_size = 1
    metadata: dict = {
        "pixel_spacing": voxel_size,
        "original_path": original_path,
    }
    for idx, file_name in enumerate(path):
        metadata[idx] = {}
        metadata[idx]["path"] = file_name
        metadata[idx]["name"] = os.path.basename(file_name)
        with mrcfile.open(file_name, permissive=True) as mrc:
            data = mrc.data
            metadata["pixel_spacing"] = (
                mrc.voxel_size.x if mrc.voxel_size.x != 0 else 1
            )
        arrays.append(data)

    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))

    add_kwargs = {"metadata": metadata, "name": name}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]


def get_valid_extensions():
    return ["mrc", "mrcs", "st"]


def from_napari(
    path: os.PathLike | list[os.PathLike] | pd.DataFrame,
    data: typing.Any,
    meta: dict,
):
    raise NotImplementedError

import glob
import os
import typing

import mrcfile
import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
    if not isinstance(path, list):
        path = glob.glob(path)  # type: ignore

    arrays = []
    voxel_size = 1
    for file_name in path:
        with mrcfile.open(file_name, permissive=True) as mrc:
            data = mrc.data
            voxel_size = mrc.voxel_size.x
        arrays.append(data)

    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))

    add_kwargs = {"metadata": {"pixel_spacing": voxel_size}}

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

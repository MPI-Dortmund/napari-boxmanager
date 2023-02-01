import os
import typing

import mrcfile
import numpy as np
import pandas as pd

from .io_utils import to_napari_image

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def load_image(path: str) -> np.array:

    extension = os.path.splitext(path)[1]

    with mrcfile.open(path, "r", permissive=True) as mrc:
        data = mrc.data
    if extension == ".mrci":
        return data.astype(np.int32)
    else:
        return data


def get_pixel_size(path: str) -> float:
    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        return mrc.voxel_size.x if mrc.voxel_size.x != 0 else 1

    return 1


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":

    if isinstance(path, list):
        extension = os.path.splitext(path[0])[1]
    else:
        extension = os.path.splitext(path)[1]

    if extension == ".mrci":
        do_normalize = False
    else:
        do_normalize = True

    layer_data = to_napari_image(
        path,
        load_image=load_image,
        get_pixel_size=get_pixel_size,
        do_normalize=do_normalize,
    )

    if extension == ".mrci":
        layer_data = [(_[0], _[1], "labels") for _ in layer_data]

    return layer_data


def get_valid_extensions():
    return ["mrc", "mrcs", "st", "rec", "mrci"]


def from_napari(
    path: os.PathLike | list[os.PathLike] | pd.DataFrame,
    data: typing.Any,
    meta: dict,
):
    raise NotImplementedError

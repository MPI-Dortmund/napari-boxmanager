import glob
import os
import typing

import mrcfile
import numpy as np
import pandas as pd
from .io_utils import MAX_LAYER_NAME, PROXY_THRESHOLD_GB, LoaderProxy, to_napari_image

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def load_image(path: str) -> np.array:
    with mrcfile.open(path, "r", permissive=True) as mrc:
        data = mrc.data
    return data


def get_pixel_size(path: str) -> float:
    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        return mrc.voxel_size.x if mrc.voxel_size.x != 0 else 1

    return 1


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":

    return to_napari_image(path, load_image=load_image, get_pixel_size=get_pixel_size)


def get_valid_extensions():
    return ["mrc", "mrcs", "st", "rec"]


def from_napari(
    path: os.PathLike | list[os.PathLike] | pd.DataFrame,
    data: typing.Any,
    meta: dict,
):
    raise NotImplementedError

import os
import typing
import tifffile
import numpy as np
import pandas as pd
from .io_utils import to_napari_image

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def load_image(path: str) -> np.ndarray:

    with tifffile.TiffFile(path) as tif:
        img = tif.pages[0].asarray()
        if len(img.shape) == 3:
            img = np.flip(img, 1)
        else:
            img = np.flip(img, 0)
        return img
    return None


def get_pixel_size(path: str) -> float:
    with tifffile.TiffFile(path) as tif:
        try:
            return tif.pages[0].tags['TVIPS'].value['PixelSizeX']*10
        except KeyError:
            print("Can't find pixelsize. Use default (1).")

    return 1


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":

    return to_napari_image(path, load_image=load_image, get_pixel_size=get_pixel_size)


def get_valid_extensions() -> typing.List[str]:
    return ["tif", "tiff"]


def from_napari(
    path: os.PathLike | list[os.PathLike] | pd.DataFrame,
    data: typing.Any,
    meta: dict,
):
    raise NotImplementedError

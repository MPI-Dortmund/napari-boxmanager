import glob
import os
import sys
import typing
from collections.abc import Callable
import numpy as np

import pandas as pd

from . import io as bm_readers

if typing.TYPE_CHECKING:
    import numpy.typing as npt

def is_tomo(path: str, reader: Callable):
    img = reader(path)
    num_dim = np.squeeze(img[0][0]).ndim
    return num_dim== 3

def get_dir(path):
    layers = []
    for file_ext in bm_readers._VALID_IOS.keys():
        files = glob.glob(os.path.join(path, f"*.{file_ext}"))
        reader = bm_readers.get_reader(file_ext)
        if not files:
            continue

        is_first_file_tomo=is_tomo(files[0],reader)
        if is_first_file_tomo:
            for file in files:
                layers.extend(reader(file))
        else:
            layers.extend(reader(sorted(files)))
    return layers


def napari_get_reader(
    path: os.PathLike | list[os.PathLike],
) -> "Callable[[os.PathLike | list[os.PathLike] | pd.DataFrame], list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]] | None":
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : os.PathLike or list of os.PathLike
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    if os.path.isdir(path):
        return get_dir
    else:
        load_type = os.path.splitext(path)[-1][1:]

    return bm_readers.get_reader(load_type)

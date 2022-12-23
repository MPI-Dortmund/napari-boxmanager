import glob
import os
import typing
from collections.abc import Callable

import pandas as pd

from . import io as bm_readers

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def get_dir(path):
    layers = []
    for file_ext in bm_readers._VALID_IOS.keys():
        files = glob.glob(os.path.join(path, f"*.{file_ext}"))
        if not files:
            continue
        print("ext", file_ext)
        layers.extend(bm_readers.get_reader(file_ext)(sorted(files)))
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

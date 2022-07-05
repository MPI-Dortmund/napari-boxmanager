import os
import typing
from collections.abc import Callable

import pandas as pd

from . import readers as bm_readers

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def check_reader(path: os.PathLike) -> bool:
    if path.endswith(".pkl"):  # type: ignore
        load_type = pd.read_pickle(path).attrs["boxread_identifier"]
    else:
        load_type = os.path.splitext(path)[-1][1:]
    return bm_readers.check_reader(load_type)


def get_reader(
    path: os.PathLike,
) -> "Callable[[list[os.PathLike] | pd.DataFrame], tuple[tuple[npt.ArrayLike, dict[str, typing.Any], str]]]":
    if path.endswith(".pkl"):  # type: ignore
        load_type = pd.read_pickle(path).attrs["boxread_identifier"]
    else:
        load_type = os.path.splitext(path)[-1][1:]
    return bm_readers.get_reader(load_type)


def napari_get_reader(path: os.PathLike | list[os.PathLike]):
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

    return reader_function if check_reader(path) else None


def reader_function(path: list[os.PathLike] | os.PathLike):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """

    reader_func: "Callable[[list[os.PathLike] | pd.DataFrame], tuple[tuple[npt.ArrayLike, dict[str, typing.Any], str]]]" = get_reader(
        path[0] if isinstance(path, list) else path
    )
    return reader_func(path if isinstance(path, list) else [path])

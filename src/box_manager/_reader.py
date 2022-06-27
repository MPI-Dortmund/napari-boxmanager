import os
import pathlib
import typing
from collections.abc import Callable

import pandas as pd

from . import readers as bm_readers

if typing.TYPE_CHECKING:
    import numpy as np


class ReaderClass:
    def __init__(self, paths: list[str] | str):
        self.paths: list[str] = paths if isinstance(paths, list) else [paths]
        self.readers: "list[Callable[[pathlib.Path], tuple[np.ndarray, dict[str, typing.Any], str]]]" = (
            []
        )

        for path in self.paths:
            if path.endswith(".pkl"):
                load_type = pd.read_pickle(path).attrs["boxread_identifier"]
            else:
                load_type = os.path.splitext(path)[-1]
            self.readers.append(bm_readers.valid_readers[load_type])

    def items(
        self,
    ) -> "list[tuple[str, Callable[[pathlib.Path], tuple[np.ndarray, dict[str, typing.Any], str]]]]":
        return list(zip(self.paths, self.readers))


def napari_get_reader(input_path: str | list[str]):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    path: str
    if isinstance(input_path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = input_path[0]
    else:
        path = input_path

    try:
        ReaderClass(path)
    except KeyError:
        # if we know we cannot read the file, we immediately return None.
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(input_path: list | str):
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

    reader_class = ReaderClass(input_path)

    output_data = []
    for path, func in reader_class.items():
        output_data.append(func(pathlib.Path(path)))

    return output_data

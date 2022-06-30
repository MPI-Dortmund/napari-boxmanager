import os
import typing
from collections.abc import Callable

import pandas as pd

from . import readers as bm_readers

if typing.TYPE_CHECKING:
    import numpy.typing as npt

    from .readers import interface


class ReaderClass:
    def __init__(self, paths: list[os.PathLike] | os.PathLike):
        self.paths: list[os.PathLike] = (
            paths if isinstance(paths, list) else [paths]
        )
        self.readers: "list[interface.ReaderInterface]" = []

        load_type: str
        for path in self.paths:
            if path.endswith(".pkl"):
                load_type = pd.read_pickle(path).attrs["boxread_identifier"]
            else:
                load_type = os.path.splitext(path)[-1]
            self.readers.append(bm_readers.valid_readers[load_type])

    def items(
        self,
    ) -> "list[tuple[os.PathLike, Callable[[os.PathLike], tuple[npt.ArrayLike, dict[str, typing.Any], str]]]]":
        return list(zip(self.paths, self.readers))


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

    try:
        ReaderClass(path)
    except KeyError:
        # if we know we cannot read the file, we immediately return None.
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


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

    reader_class = ReaderClass(path)

    output_data = []
    for cur_path, module in reader_class.items():
        output_data.extend(bm_readers.to_napari(cur_path, module))

    return output_data

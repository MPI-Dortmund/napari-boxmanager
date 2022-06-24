"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/plugins/guides.html?#readers
"""

import os
import pathlib
from collections.abc import Callable

import numpy as np
import pandas as pd

from .readers import box, cbox, star, tepkl, tlpkl, tmpkl


class ReaderClass:
    valid_file_endings = (
        ".pkl",
        ".tlpkl",
        ".tepkl",
        ".tmpkl",
        ".cbox",
        ".box",
        ".star",
    )

    def __init__(self, paths: list[str] | str):
        self.paths: list[str] = paths if isinstance(paths, list) else [paths]

    def is_valid(self) -> list[bool]:
        return [
            bool(
                [
                    ending
                    for ending in self.valid_file_endings
                    if path.endswith(ending)
                ]
            )
            for path in self.paths
        ]

    def is_all_valid(self) -> bool:
        return all(self.is_valid())

    def load_functions(self) -> list[pd.DataFrame]:
        data_list = []
        for path in self.paths:
            if path.endswith("pkl"):
                list_val = self.load_pkl(path)
            elif path.endswith(".box"):
                list_val = box.read
            elif path.endswith(".cbox"):
                list_val = cbox.read
            elif path.endswith(".star"):
                list_val = star.read
            else:
                assert False, path
            data_list.append(list_val)
        return data_list

    @staticmethod
    def load_pkl(path: str) -> Callable[[pathlib.Path], pd.DataFrame]:
        if path.endswith(".pkl"):
            identifier = pd.read_pickle(path).attrs["boxread_identifier"]
        else:
            identifier = os.path.splitext(path)[-1]

        if identifier == ".tlpkl":
            return tlpkl.read
        elif identifier == ".tmpkl":
            return tmpkl.read
        elif identifier == ".tepkl":
            return tepkl.read
        else:
            assert False, identifier


def napari_get_reader(path):
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
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    if not ReaderClass(path).is_all_valid():
        # if we know we cannot read the file, we immediately return None.
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path: list | str):
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
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    # load all files into array
    arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]

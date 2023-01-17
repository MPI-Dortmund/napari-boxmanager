import os
import typing

from . import io as bm_readers

if typing.TYPE_CHECKING:
    pass


def napari_get_writer(
    path: os.PathLike,
    data: list[tuple[typing.Any, dict, str]],
    provided_extension=None,
    suffix="",
    cur_spacing=0,
):
    if provided_extension:
        extension = provided_extension
    else:
        basename, extension = os.path.splitext(os.path.basename(path))
        if not extension:
            extension = basename
    load_type = extension[1:]
    writer = bm_readers.get_writer(load_type)
    if not writer:
        return None
    else:
        return writer(path, data, suffix, cur_spacing)


# """
# This module is an example of a barebones writer plugin for napari.
#
# It implements the Writer specification.
# see: https://napari.org/plugins/guides.html?#writers
#
# Replace code below according to your needs.
# """
# from __future__ import annotations
#
# from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union
#
# if TYPE_CHECKING:
#    DataType = Union[Any, Sequence[Any]]
#    FullLayerData = Tuple[DataType, dict, str]
#
#
# def write_single_image(path: str, data: Any, meta: dict):
#    """Writes a single image layer"""
#
#
# def write_multiple(path: str, data: List[FullLayerData]):
#    """Writes multiple layers of different types."""

import glob
import importlib
import os
import typing
from collections.abc import Callable

import pandas as pd

from .interface import IOInterface

__all__ = ["get_reader", "IOInterface"]

if typing.TYPE_CHECKING:
    import numpy.typing as npt

_IGNORE_LIST = ["interface.py", "io_utils.py", os.path.basename(__file__)]

_VALID_IOS: dict[
    str,
    IOInterface,
] = {}
for module in glob.iglob(f"{os.path.dirname(__file__)}/*.py"):
    if os.path.basename(module) in _IGNORE_LIST:
        continue

    _name: str = (
        f"box_manager.io.{os.path.basename(os.path.splitext(module)[0])}"
    )
    package: IOInterface = importlib.import_module(_name)  # type:ignore

    for extension in package.get_valid_extensions():
        _VALID_IOS[extension] = package


def get_reader(
    key: str,
) -> "Callable[[os.PathLike | list[os.PathLike] | pd.DataFrame], list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]] | None":
    return _VALID_IOS[key].to_napari if key in _VALID_IOS else None


def get_writer(
    key: str,
) -> "Callable[[os.PathLike | list[os.PathLike], list, dict]], None":
    return _VALID_IOS[key].from_napari if key in _VALID_IOS else None

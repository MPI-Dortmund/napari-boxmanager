import glob
import importlib
import os
import typing
from collections.abc import Callable

import pandas as pd

from .interface import ReaderInterface

__all__ = ["check_reader", "get_reader", "ReaderInterface"]

if typing.TYPE_CHECKING:
    import numpy.typing as npt

_IGNORE_LIST = ["interface.py", os.path.basename(__file__)]

_VALID_READERS: dict[
    str,
    str,
] = {}
for module in glob.iglob(f"{os.path.dirname(__file__)}/*.py"):
    if os.path.basename(module) in _IGNORE_LIST:
        continue

    _name: str = f"{os.path.basename(os.path.splitext(module)[0])}"
    package: ReaderInterface = importlib.import_module(_name)  # type:ignore

    for extension in package.get_valid_extensions():
        _VALID_READERS[extension] = f"box_manager.readers.{_name}"


def check_reader(key: str) -> bool:
    return key in _VALID_READERS


def get_reader(
    key: str,
) -> "Callable[[list[os.PathLike] | pd.DataFrame], tuple[tuple[npt.ArrayLike, dict[str, typing.Any], str]]]":
    package: ReaderInterface = importlib.import_module(
        _VALID_READERS[key]
    )  # type:ignore
    return package.to_napari

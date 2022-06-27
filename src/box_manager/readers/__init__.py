import glob
import importlib
import os
import typing
import warnings
from collections.abc import Callable

from . import interface

if typing.TYPE_CHECKING:
    import pathlib

    import numpy as np


class ReaderMissingToNapariFunction(Warning):
    pass


_ignore_list = ["interface.py", os.path.basename(__file__)]

valid_readers: dict[
    str,
    "list[Callable[[pathlib.Path], tuple[np.ndarray, dict[str, typing.Any], str]]]",
] = {}
for module in glob.iglob(f"{os.path.dirname(__file__)}/*.py"):
    if os.path.basename(module) in _ignore_list:
        continue

    _name: str = f".{os.path.basename(os.path.splitext(module)[0])}"
    _package: interface.ReaderInterface = importlib.import_module(
        f"box_manager.readers{_name}"
    )

    _function_name: str = "to_napari"
    try:
        valid_readers[_name] = getattr(_package, _function_name)
    except AttributeError:
        warnings.warn(
            ReaderMissingToNapariFunction(
                f"Reader {_name} does not have a function {_function_name}."
            )
        )

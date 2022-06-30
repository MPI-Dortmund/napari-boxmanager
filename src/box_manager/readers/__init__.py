import glob
import importlib
import inspect
import os
import typing

from . import interface
from .interface import to_napari

__all__ = ["to_napari"]

if typing.TYPE_CHECKING:
    pass


class ReaderMissingToNapariFunction(Warning):
    pass


_ignore_list = ["interface.py", os.path.basename(__file__)]

valid_readers: dict[
    str,
    "list[interface.ReaderInterface]",
] = {}
for module in glob.iglob(f"{os.path.dirname(__file__)}/*.py"):
    if os.path.basename(module) in _ignore_list:
        continue

    _name: str = f".{os.path.basename(os.path.splitext(module)[0])}"
    _package: interface.ReaderInterface = importlib.import_module(
        f"box_manager.readers{_name}"
    )

    if all(
        [
            hasattr(_package, name)
            for name, _ in inspect.getmembers(
                interface.ReaderInterface, predicate=inspect.isfunction
            )
            if not name.startswith("__")
        ]
    ):
        valid_readers[_name] = _package

import glob
import importlib
import os
import warnings

from . import interface


class ReaderMissingToNapariFunction(Warning):
    pass


ignore_list = ["interface.py", os.path.basename(__file__)]

valid_readers: dict[str, interface.ReaderInterface] = {}
for module in glob.iglob(f"{os.path.dirname(__file__)}/*.py"):
    if os.path.basename(module) in ignore_list:
        continue

    name = os.path.basename(os.path.splitext(module)[0])
    package = importlib.import_module(f"box_manager.readers.{name}")

    _function_name: str = "to_napari"
    try:
        valid_readers[name] = getattr(package, _function_name)
    except AttributeError:
        warnings.warn(
            ReaderMissingToNapariFunction(
                f"Reader {name} does not have a function {_function_name}."
            )
        )

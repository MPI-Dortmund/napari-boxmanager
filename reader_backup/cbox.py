import typing

import pandas as pd

if typing.TYPE_CHECKING:
    import pathlib

    import numpy as np


def read(path: "pathlib.Path") -> pd.DataFrame:
    raise NotImplementedError


def prepare_napari(
    path: "pathlib.Path",
) -> "tuple[np.ndarray, dict[str, typing.Any], dict[str, typing.Any], str]":
    raise NotImplementedError

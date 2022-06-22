from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import pathlib


class BoxFileNumberOfColumnsError(pd.errors.IntCastingNaNError):
    pass


def read(path: "pathlib.Path") -> pd.DataFrame:
    try:
        box_data: pd.DataFrame = pd.read_csv(
            path,
            delim_whitespace=True,
            index_col=False,
            header=None,
            dtype=float,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame([], dtype=int)

    box_data.iloc[:, 0] += box_data[2] // 2
    box_data.iloc[:, 1] += box_data[3] // 2

    try:
        return box_data[[1, 0]].astype(int)
    except pd.errors.IntCastingNaNError:
        raise BoxFileNumberOfColumnsError

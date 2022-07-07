import os
import typing

import pandas as pd

if typing.TYPE_CHECKING:
    import numpy.typing as npt


class ReaderInterface(typing.Protocol):
    def to_napari(
        self,
        path: os.PathLike | list[os.PathLike] | pd.DataFrame,
    ) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
        ...

    def get_valid_extensions(self) -> list[str]:
        ...

    def from_napari(
        self,
        path: os.PathLike | list[os.PathLike] | pd.DataFrame,
        data: typing.Any,
        meta: dict,
    ):
        ...

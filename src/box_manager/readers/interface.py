import os
import typing

import pandas as pd

if typing.TYPE_CHECKING:
    import numpy.typing as npt


class ReaderInterface(typing.Protocol):
    def to_napari(
        self,
        path: list[os.PathLike] | pd.DataFrame,
    ) -> "tuple[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
        ...

    def get_valid_extensions(self) -> list[str]:
        ...

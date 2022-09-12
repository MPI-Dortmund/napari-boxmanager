import os
import typing

import pandas as pd

if typing.TYPE_CHECKING:
    import numpy.typing as npt

NapariMetaData = dict[str, typing.Any]
NapariLayerData = tuple[npt.ArrayLike, NapariMetaData, str]

class IOInterface(typing.Protocol):
    def to_napari(
        self,
        path: os.PathLike | list[os.PathLike],
    ) -> "list[NapariLayerData]":
        ...

    def get_valid_extensions(self) -> list[str]:
        ...

    def from_napari(
        self,
        path: os.PathLike | list[os.PathLike] | pd.DataFrame,
        data: list[NapariLayerData],
    ):
        ...

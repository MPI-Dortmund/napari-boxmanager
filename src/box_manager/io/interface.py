import os
import typing

import numpy.typing as npt
import pandas as pd

NapariMetaData = dict[str, typing.Any]
NapariLayerData = tuple[
    npt.ArrayLike | list[npt.ArrayLike], NapariMetaData, str
]


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
        layer_data: list[NapariLayerData],
        suffix: str,
        filament_spacing: int,
    ):
        ...

import typing

if typing.TYPE_CHECKING:
    import numpy as np


class ReaderInterface(typing.Protocol):
    def to_napari(self) -> "tuple[np.ndarray, dict[str, typing.Any], str]":
        ...

import glob
import io
import os
import typing

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
from numpy.array_api._array_object import Array

from .coordinate_io import _MAX_LAYER_NAME, _PROXY_THRESHOLD_GB

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def load_mrc(path):
    with mrcfile.open(path, "r") as mrc:
        data = mrc.data
    return data


class LoaderProxy(Array):
    def __init__(self, files, reader_func):
        self.reader_func = reader_func
        self.files = files
        if len(self.files) == 0:
            raise AttributeError("Cannot provide empty files list")

        _data = self.load_image(0)
        self._array = np.empty((len(self.files), *_data.shape), dtype=bool)

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        return obj

    def load_image(self, index) -> Array:
        data = self.reader_func(self.files[index])
        return (data - np.mean(data)) / np.std(data)

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return (self[idx] for idx in range(len(self.files)))

    def __getitem__(self, key):
        try:
            super()._validate_index(key, None)
        except TypeError:
            super()._validate_index(key)

        if isinstance(key, Array):
            key._array

        try:
            _key = key[0]
        except TypeError:
            _key = key

        if isinstance(_key, (int, np.integer)):
            return self.load_image(_key)
        else:
            return self.get_dummy_image()

    def get_dummy_image(self):
        size = self.shape[-1]
        fig = plt.figure(figsize=(size, size), dpi=1)
        new_shape = (int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
        plt.text(
            0.5,
            0.5,
            "\U00002639",
            va="center_baseline",
            ha="center",
            fontsize=size * 50,
        )
        plt.axis("off")

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format="raw")
        io_buf.seek(0)
        img_arr = np.reshape(
            np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
            newshape=new_shape,
        )[..., 0]
        io_buf.close()
        plt.close(fig)
        return img_arr


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
    if not isinstance(path, list):
        original_path = path
        if "*" in path:
            name = "mrcfiles"
        elif len(path) >= _MAX_LAYER_NAME + 3:
            name = f"...{path[-_MAX_LAYER_NAME:]}"  # type: ignore
        else:
            name = path  # type: ignore
        path = sorted(glob.glob(path))  # type: ignore
    elif len(path[0]) >= _MAX_LAYER_NAME + 3:
        original_path = path[0]
        name = f"...{path[0][-_MAX_LAYER_NAME:]}"  # type: ignore
    else:
        name = path[0]  # type: ignore
        original_path = path[0]

    # arrays = []
    voxel_size = 1
    metadata: dict = {
        "pixel_spacing": voxel_size,
        "original_path": original_path,
    }
    for idx, file_name in enumerate(path):
        metadata[idx] = {}
        metadata[idx]["path"] = file_name
        metadata[idx]["name"] = os.path.basename(file_name)

    with mrcfile.open(path[0], permissive=True, header_only=True) as mrc:
        metadata["pixel_spacing"] = (
            mrc.voxel_size.x if mrc.voxel_size.x != 0 else 1
        )

    file_size = (
        sum(os.stat(file_name).st_size for file_name in path) / 1024**3
    )
    if len(path) > 1 and file_size > _PROXY_THRESHOLD_GB:
        data = LoaderProxy(path, load_mrc)
    else:
        data_list = []
        for file_name in path:
            with mrcfile.open(file_name, permissive=True) as mrc:
                tmp_data = mrc.data
                data_list.append(tmp_data)
        data = np.squeeze(np.stack(data_list))
        data = (data - np.mean(data)) / np.std(data)

    metadata["is_3d"] = len(path) == 1 and data.ndim == 3
    add_kwargs = {"metadata": metadata, "name": name}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]


def get_valid_extensions():
    return ["mrc", "mrcs", "st", "rec"]


def from_napari(
    path: os.PathLike | list[os.PathLike] | pd.DataFrame,
    data: typing.Any,
    meta: dict,
):
    raise NotImplementedError

import glob
import io
import os
import pathlib
import sys
import typing
import warnings
from collections.abc import Callable
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from .interface import NapariLayerData, NapariMetaData

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        r"The numpy.array_api submodule is still experimental. See NEP 47.",
    )
    from numpy.array_api._array_object import Array

from .._qt import OrganizeBox as orgbox

MAX_LAYER_NAME: int = 30
PROXY_THRESHOLD_GB: float = (
    float(os.environ["BOXMANAGER_PROXY_GB"])
    if "BOXMANAGER_PROXY_GB" in os.environ
    else 2
)


class FormatFunc(Protocol):
    def __call__(
        self,
        coordinates: pd.DataFrame,
        boxsize: npt.ArrayLike,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        ...

def _split_filaments(data: pd.DataFrame) -> typing.List[pd.DataFrame]:
    filament_ids = np.unique(data['fid'])
    filaments = []
    for id in filament_ids:
        mask = data['fid']==id
        filaments.append(data[mask])
    return filaments

def _prepare_coords_df(
    path: list[os.PathLike],
    read_func: Callable[[os.PathLike], pd.DataFrame],
    prepare_napari_func: Callable,
    meta_columns: typing.List[str] = [],
) -> tuple[typing.List[pd.DataFrame], dict, bool, bool]:

    data_df: list[pd.DataFrame] = []
    metadata: dict = {}
    is_3d = True
    is_filament = False
    for idx, entry in enumerate(path):
        input_data = read_func(entry)

        box_napari_data = pd.DataFrame(columns=meta_columns)

        if input_data is not None:
            box_napari_data = prepare_napari_func(input_data)
            if "x" not in box_napari_data:
                is_3d = False
                box_napari_data["x"] = idx
            if 'fid' in box_napari_data:
                is_filament = True
                box_napari_data = _split_filaments(box_napari_data)


        checkbox = None
        if is_filament:
            if len(box_napari_data) == 0 or np.all([b.empty for b in box_napari_data]):
                checkbox = True
        elif box_napari_data.empty:
            checkbox = True

        if is_filament:
            data_df.extend(box_napari_data)
        else:
            data_df.append(box_napari_data)


        metadata[idx] = {}
        metadata[idx]["path"] = entry
        metadata[idx]["name"] = os.path.basename(entry)
        metadata[idx]["write"] = checkbox
        try:
            metadata[idx].update(
                {
                    f"{entry}_{func.__name__}": func(data_df[idx][entry])
                    for func in [min, max]
                    for entry in meta_columns
                }
            )
        except ValueError:
            pass

    return data_df, metadata, is_3d, is_filament


def get_coords_layer_name(path: os.PathLike | list[os.PathLike]) -> str:
    if isinstance(path, list) and len(path) > 1:
        name = "Coordinates"
    elif isinstance(path, list):
        if len(path[0]) >= MAX_LAYER_NAME + 3:
            name = f"...{path[0][-MAX_LAYER_NAME:]}"  # type: ignore
        else:
            name = path[0]  # type: ignore
    else:
        assert False, path

    return name


def _to_napari_filament(input_df, coord_columns):
    boxsize = [np.mean(fil['boxsize']) for fil in input_df]
    input_df = [fil[coord_columns] for fil in input_df]
    color = [np.random.choice(range(256), size=3) for _ in range(len(input_df))] #RGB
    color = ['#%02x%02x%02x' % (r, g, b) for r,g,b in color]
    kwargs: NapariMetaData = {
        "edge_color": color,
        "face_color": "transparent",
        "edge_width": boxsize,
        "opacity": 0.4,
        "shape_type": "path",

    }
    dat = input_df
    layer_type = "shapes"

    return dat, kwargs, layer_type

def _to_napari_particle(input_df, coord_columns, is_3d):
    input_df = pd.concat(input_df, ignore_index=True)


    kwargs: NapariMetaData = {
        "edge_color": "red",
        "face_color": "transparent",
        "symbol": "disc",
        "edge_width": 0.05,
        "edge_width_is_relative": True,
        "size": np.average(input_df["boxsize"]),
        "out_of_slice_display": True if is_3d else False,
        "opacity": 0.8,
    }
    dat = input_df[coord_columns]
    layer_type = "points"

    return dat, kwargs, layer_type

def to_napari(
    path: os.PathLike | list[os.PathLike],
    read_func: Callable[[os.PathLike], pd.DataFrame],
    prepare_napari_func: Callable,
    meta_columns: typing.List[str] = [],
    feature_columns: typing.List[str] = [],
) -> "list[NapariLayerData]":

    input_df: pd.DataFrame
    features: dict[str, typing.Any]

    orgbox_meta = orgbox.get_metadata(path)

    if not isinstance(path, list):
        path = sorted(glob.glob(path))  # type: ignore

    input_df, metadata, is_3d, is_filament = _prepare_coords_df(
        path,
        read_func=read_func,
        prepare_napari_func=prepare_napari_func,
        meta_columns=meta_columns,
    )

    metadata["is_2d_stack"] = len(path) > 1
    metadata.update(orgbox_meta)

    features = {
        entry: input_df[entry].to_numpy()
        for entry in feature_columns  # _get_meta_idx() + _get_hidden_meta_idx()
    }

    layer_name = get_coords_layer_name(path)

    if (isinstance(path, list) and len(path) > 1) or is_3d:
        # Happens for --stack option and '*.ext'
        coord_columns = ["x", "y", "z"]
    else:
        coord_columns = ["y", "z"]

    if is_filament:
        dat, kwargs, layer_type = _to_napari_filament(input_df, coord_columns)
    else:
        dat, kwargs, layer_type = _to_napari_particle(input_df, coord_columns, is_3d)

    kwargs["name"] = layer_name
    kwargs["metadata"] = metadata
    kwargs["features"] = features

    return [(dat, kwargs, layer_type)]


def _generate_output_filename(orignal_filename: str, output_path: os.PathLike):
    dirname = os.path.dirname(output_path)
    basename, extension = os.path.splitext(os.path.basename(output_path))
    if not extension:  # in case '.box' is provided as output path.
        basename, extension = extension, basename

    file_base = os.path.splitext(os.path.basename(orignal_filename))[0]
    output_file = pathlib.Path(dirname, file_base + extension)
    return output_file


def _write_particle_data(
        path: os.PathLike,
        data: NapariLayerData,
        meta: NapariLayerData,
        format_func: FormatFunc,
        write_func: Callable[[os.PathLike, pd.DataFrame], typing.Any],
    ):
    if data.shape[1] == 2:
        data = np.insert(data, 0, 0, axis=1)


    if 'shown' in meta:
        mask = meta["shown"]
    else:
        # For filaments
        mask = np.ones(len(data), dtype=int)==1

    coordinates = data[mask]

    if 'size' in meta:
        boxsize = meta["size"][mask][:, 0]
    else:
        #For filaments
        boxsize = np.array(meta["edge_width"])[mask]
    export_data = {}
    try:
        is_2d_stacked = meta["metadata"]["is_2d_stack"]
    except KeyError:
        is_2d_stacked = False

    if is_2d_stacked:
        for z in np.unique(coordinates[:, 0]):
            z = int(z)
            mask = coordinates[:, 0] == z
            filename = meta["metadata"][z]["name"]
            output_file = _generate_output_filename(
                orignal_filename=filename, output_path=path
            )

            export_data[output_file] = format_func(
                coordinates[mask, 1:],
                boxsize[mask],
                meta["features"].loc[mask, :],
            )
    else:
        export_data[path] = format_func(
            coordinates, boxsize, meta["features"]
        )

    for outpth in export_data:
        df = export_data[outpth]
        write_func(outpth, df)
    last_file = outpth
    return str(last_file)


def from_napari(
    path: os.PathLike,
    layer_data: list[NapariLayerData],
    format_func: FormatFunc,
    write_func: Callable[[os.PathLike, pd.DataFrame | list[pd.DataFrame]], typing.Any],
) -> os.PathLike:

    last_file = ""
    for data, meta, layer in layer_data:

        if isinstance(data,list):
            boxsize = []
            repeat = []
            fid = []
            for fil_index, fil in enumerate(data):
                boxsize.extend([meta['edge_width'][fil_index]]*len(fil))
                repeat.append(len(fil))
                fid.extend([fil_index+1]*len(fil))
            meta['features'] = meta['features'].loc[meta['features'].index.repeat(repeat)]
            meta['edge_width'] = boxsize
            data_list = np.concatenate(data)
            fid = np.array(fid,dtype=int)
            data_list = np.append(data_list, np.atleast_2d(np.array(fid)).T, axis=1)
            last_file = _write_particle_data(path, data_list, meta, format_func, write_func)
        else:
            last_file = _write_particle_data(path,data,meta,format_func,write_func)


    return str(last_file)


class LoaderProxy(Array):
    def __init__(self, files, reader_func):
        self.reader_func = reader_func
        self.files = files
        if len(self.files) == 0:
            raise AttributeError("Cannot provide empty files list")

        self._array = None
        self._shape = self.load_image(0).shape

    @property
    def shape(self):
        return (len(self.files), *self._shape)

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        return obj

    def load_image(self, index) -> Array:
        data = self.reader_func(self.files[index])
        self._array = np.empty((1, 1, 1), dtype=data.dtype)
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

    def __copy__(self):
        return LoaderProxy(self.files, self.reader_func)

    def __deepcopy__(self, _):
        return self.__copy__()

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

import glob
import io
import itertools
import os
import pathlib
import typing
import warnings
from collections.abc import Callable
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from .._qt import OrganizeBox as orgbox
from .interface import NapariLayerData, NapariMetaData


class IllegalFormatException(Exception):
    pass


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        r"The numpy.array_api submodule is still experimental. See NEP 47.",
    )
    from numpy.array_api._array_object import Array

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
        metadata: dict,
        filament_spacing: int,
    ) -> pd.DataFrame:
        ...


def _split_filaments(data: pd.DataFrame) -> typing.List[pd.DataFrame]:
    filament_ids = np.unique(data["fid"])
    filaments = []
    for id in filament_ids:
        mask = data["fid"] == id
        filaments.append(data[mask])
    return filaments


def _prepare_coords_df(
    path: list[os.PathLike],
    read_func: Callable[[os.PathLike], pd.DataFrame],
    prepare_napari_func: Callable,
    meta_columns: typing.List[str] = [],
) -> tuple[typing.List[pd.DataFrame], dict, bool]:

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
            if "fid" in box_napari_data:
                is_filament = True
                box_napari_data = _split_filaments(box_napari_data)
        checkbox = None
        if is_filament:
            if len(box_napari_data) == 0 or np.all(
                [b.empty for b in box_napari_data]
            ):
                checkbox = True
        elif box_napari_data.empty:
            checkbox = True

        if checkbox:
            # needs initialization if input file is empty
            min_max_data = set()
        elif is_filament:
            data_df.extend(box_napari_data)
            min_max_data = pd.concat(box_napari_data)
        else:
            min_max_data = box_napari_data
            data_df.append(box_napari_data)

        metadata[idx] = {}
        metadata[idx]["path"] = entry
        metadata[idx]["name"] = os.path.basename(entry)
        metadata[idx]["write"] = checkbox

        try:
            metadata[idx].update(
                {
                    f"{col}_{func.__name__}": func(min_max_data[col])
                    for func in [min, max]
                    for col in meta_columns
                    if col in min_max_data
                }
            )
        except ValueError:
            pass
    metadata["is_filament_layer"] = is_filament

    return data_df, metadata, is_3d


def is_filament_layer(
    layer_data: typing.Union[NapariLayerData, list[NapariLayerData]]
) -> bool:
    def check(ldat: NapariLayerData) -> bool:
        is_filament = (
            "is_filament_layer" in ldat[1]["metadata"]
            and ldat[1]["metadata"]["is_filament_layer"]
        )
        return is_filament

    if isinstance(layer_data, list):
        if np.sum([check(layer) for layer in layer_data]) not in (
            0,
            len(layer_data),
        ):
            raise IllegalFormatException(
                "Only layers of the same type (either particle or filament can be saved."
            )
        is_filament = check(layer_data[0])
    else:
        is_filament = check(layer_data)
    return is_filament


def get_coords_layer_name(path: os.PathLike | list[os.PathLike]) -> str:
    if isinstance(path, list) and len(path) > 1:
        name = (os.path.splitext(path[0])[1]).upper()[1:]
    elif isinstance(path, list):
        if len(path[0]) >= MAX_LAYER_NAME + 3:
            name = f"...{path[0][-MAX_LAYER_NAME:]}"  # type: ignore
        else:
            name = path[0]  # type: ignore
    else:
        assert False, path

    return name


def _to_napari_filament(input_df: list[pd.DataFrame], coord_columns, is_3d):
    # boxsize_ = [np.mean(fil['boxsize']) for fil in input_df]

    color = []
    boxsize = []
    total = 0

    for fil in input_df:
        bs = np.mean(fil["boxsize"])
        c = np.random.choice(range(256), size=3)
        total += len(fil)
        color.extend([c] * len(fil))
        boxsize.extend([bs] * len(fil))
    input_df = pd.concat([fil[coord_columns] for fil in input_df])
    color = [(r, g, b, 255) for r, g, b in color]

    kwargs: NapariMetaData = {
        "edge_color": color,
        "face_color": "transparent",
        "symbol": "disc",
        "edge_width": 0.05,
        "edge_width_is_relative": True,
        "size": np.average(boxsize),
        "out_of_slice_display": True if is_3d else False,
        "opacity": 0.8,
    }
    dat = input_df
    layer_type = "points"

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


def to_napari_image(
    path: os.PathLike | list[os.PathLike],
    load_image: typing.Callable[[str], np.array],
    get_pixel_size: typing.Callable[[str], float],
    do_normalize: bool = True,
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":

    is_2d_stack = isinstance(path, list) or "*" in path

    if not isinstance(path, list):
        original_path = path
        if len(path) >= MAX_LAYER_NAME + 3:
            name = f"...{path[-MAX_LAYER_NAME:]}"  # type: ignore
        else:
            name = path  # type: ignore
        path = sorted(glob.glob(path))  # type: ignore
        if len(path) > 1:
            name = "images"
    else:
        original_path = path[0]
        name = "images"

    # arrays = []
    voxel_size = 1
    metadata: dict = {
        "pixel_spacing": voxel_size,
        "original_path": original_path,
    }
    metadata["pixel_spacing"] = get_pixel_size(path[0])

    file_size = (
        sum(os.stat(file_name).st_size for file_name in path) / 1024**3
    )
    if len(path) > 1 and file_size > PROXY_THRESHOLD_GB:
        data = LoaderProxy(path, load_image, do_normalize)
    else:
        data_list = []
        for file_name in path:
            tmp_data = load_image(file_name)
            if do_normalize:
                tmp_data = (tmp_data - np.mean(tmp_data)) / np.std(tmp_data)
            data_list.append(tmp_data)
        data = np.squeeze(np.stack(data_list))

    metadata["is_3d"] = len(path) == 1 and data.ndim == 3
    metadata["is_2d_stack"] = is_2d_stack

    if (metadata["is_3d"] or metadata["is_2d_stack"]) and data.ndim == 2:
        data = np.expand_dims(data, axis=0)

    if not metadata["is_3d"]:
        for idx, file_name in enumerate(path):
            metadata[idx] = {}
            metadata[idx]["path"] = file_name
            metadata[idx]["name"] = os.path.basename(file_name)

    add_kwargs = {"metadata": metadata, "name": name}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]


def to_napari_coordinates(
    path: os.PathLike | list[os.PathLike],
    read_func: Callable[[os.PathLike], pd.DataFrame],
    prepare_napari_func: Callable,
    meta_columns: typing.List[str] = [],
    feature_columns: typing.List[str] = [],
    valid_extensions: typing.List[str] = [],
) -> "list[NapariLayerData]":

    input_df_list: list[pd.DataFrame]
    features: dict[str, typing.Any]

    is_2d_stack = isinstance(path, list) or "*" in path
    orgbox_meta = orgbox.get_metadata(path)

    if not isinstance(path, list):
        path = sorted(glob.glob(path))  # type: ignore

    input_df_list, metadata, is_3d = _prepare_coords_df(
        path,
        read_func=read_func,
        prepare_napari_func=prepare_napari_func,
        meta_columns=meta_columns,
    )
    metadata.update(orgbox_meta)
    metadata["is_2d_stack"] = is_2d_stack
    metadata["ignore_idx"] = feature_columns
    features = {}
    for entry in feature_columns + meta_columns:
        if metadata["is_filament_layer"]:
            for fil in input_df_list:
                if entry in features:
                    features[entry] = np.concatenate(
                        [features[entry], fil[entry].to_numpy()]
                    )
                else:
                    features[entry] = fil[entry].to_numpy()
        else:
            all = pd.concat(input_df_list)
            if entry in all:
                features[entry] = all[entry].to_numpy()
    layer_name = get_coords_layer_name(path)

    if is_2d_stack or is_3d:
        # Happens for --stack option and '*.ext'
        coord_columns = ["x", "y", "z"]
    else:
        coord_columns = ["y", "z"]

    if metadata["is_filament_layer"]:
        dat, kwargs, layer_type = _to_napari_filament(
            input_df_list, coord_columns, is_3d
        )
    else:
        dat, kwargs, layer_type = _to_napari_particle(
            input_df_list, coord_columns, is_3d
        )
    kwargs["name"] = layer_name
    kwargs["metadata"] = metadata
    kwargs["features"] = features

    return [(dat, kwargs, layer_type)]


def _generate_output_filename(
    orignal_filename: str, output_path: os.PathLike, suffix=""
):
    if not os.path.isdir(output_path):
        dirname = os.path.dirname(output_path)
        file_base, extension = os.path.splitext(os.path.basename(output_path))
        if not extension:  # in case '.box' is provided as output path.
            file_base, extension = extension, file_base
    else:
        extension = suffix
        dirname = output_path
        file_base = os.path.splitext(os.path.basename(orignal_filename))[0]

    output_file = pathlib.Path(dirname, file_base + extension)
    return output_file


def resample_filament(
    filament: pd.DataFrame,
    new_distance: float,
    coordinate_columns,
    constant_columns=[],
    other_interpolation_col=[],
):
    if len(filament) <= 1:
        return filament
    import numpy as np
    from scipy.interpolate import interp1d

    new_distance = np.sqrt(new_distance**2 + new_distance**2)

    constants = {}
    for col in constant_columns:
        if col in filament:
            constants[col] = filament[col][0]

    sqsum = 0
    for col in coordinate_columns:
        coord_dat = filament[col].to_list()
        sqsum = sqsum + np.ediff1d(coord_dat, to_begin=0) ** 2

    distance_elem = np.cumsum(np.sqrt(sqsum))
    total_elength = distance_elem[-1]

    distance = distance_elem / distance_elem[-1]  # norm to 1
    interpolators = []
    for col in coordinate_columns + other_interpolation_col:
        interp = interp1d(distance, filament[col].to_list())
        interpolators.append(interp)

    num = int(total_elength / (new_distance)) + 1

    alpha = np.linspace(0, 1, num)

    interpolated = []
    for interp in interpolators:
        interpolated.append(interp(alpha))

    ####
    # Generate new boxes
    ####
    new_boxes = {}
    for col_i, col in enumerate(coordinate_columns + other_interpolation_col):
        new_boxes[col] = interpolated[col_i]

    for col in constant_columns:
        new_boxes[col] = [constants[col]] * len(interpolated[0])
    new_fil = pd.DataFrame(new_boxes)

    return new_fil


def _write_particle_data(
    path: os.PathLike,
    data: NapariLayerData,
    meta: NapariLayerData,
    format_func: FormatFunc,
    write_func: Callable[[os.PathLike, pd.DataFrame, ...], typing.Any],
    suffix: str = "",
    filament_spacing: float = 0,
):
    if data.shape[1] == 2:
        data = np.insert(data, 0, 0, axis=1)

    kwargs = {}

    if "shown" in meta:
        mask = meta["shown"]
    else:
        # For filaments
        mask = np.ones(len(data), dtype=int) == 1

    coordinates = data[mask]

    if "size" in meta:
        boxsize = meta["size"][mask][:, 0]
    else:
        # For filaments
        boxsize = np.array(meta["edge_width"])[mask]
    export_data = {}
    try:
        is_2d_stacked = meta["metadata"]["is_2d_stack"]
    except KeyError:
        is_2d_stacked = False

    if is_2d_stacked:
        for z in meta["metadata"]:
            if not isinstance(z, int) or meta["metadata"][z]["write"] is False:
                continue
            mask = coordinates[:, 0] == z
            if np.sum(mask) == 0 and meta["metadata"][z]["write"] is not True:
                continue

            filename = meta["metadata"][z]["name"]
            kwargs["image_name"] = filename
            output_file = _generate_output_filename(
                orignal_filename=filename, output_path=path, suffix=suffix
            )

            d = format_func(
                coordinates=coordinates[mask, 1:],
                box_size=boxsize[mask],
                features=meta["features"].loc[mask, :],
                metadata=meta["metadata"],
                filament_spacing=filament_spacing,
            )
            export_data[output_file] = (d, {})

    else:
        filename = meta["metadata"]["original_path"]
        output_file = _generate_output_filename(
            orignal_filename=filename, output_path=path, suffix=suffix
        )
        empty_slices = []
        slices_with_coords = np.unique(coordinates[:, 0]).tolist()
        for z in meta["metadata"]:
            if (
                not isinstance(z, int)
                or meta["metadata"][z]["write"] is not True
            ):
                continue
            if z in slices_with_coords:
                continue
            empty_slices.append(z)

        export_data[output_file] = (
            format_func(
                coordinates=coordinates,
                box_size=boxsize,
                features=meta["features"].loc[mask, :],
                metadata=meta["metadata"],
                filament_spacing=filament_spacing,
            ),
            {"empty_slices": empty_slices},
        )

    for outpth in export_data:
        df = export_data[outpth][0]
        write_func(outpth, df, **export_data[outpth][1])
    last_file = outpth
    return str(last_file)


def convert_shape_filament_layer_to_boxlayer(
    data: list[np.array], meta: dict
) -> (np.array, dict):
    repeat = []
    for fil in data:
        repeat.append(len(fil))
    meta["features"] = meta["features"].loc[
        meta["features"].index.repeat(repeat)
    ]

    meta["edge_width"] = list(
        itertools.chain.from_iterable(
            [
                [meta["edge_width"][filid]] * len(d)
                for filid, d in enumerate(data)
            ]
        )
    )

    meta["features"]["fid"] = list(
        itertools.chain.from_iterable(
            [[filid] * len(d) for filid, d in enumerate(data)]
        )
    )

    data = np.concatenate(data)

    return data, meta


def from_napari(
    path: os.PathLike,
    layer_data: list[NapariLayerData],
    format_func: FormatFunc,
    write_func: Callable[
        [os.PathLike, pd.DataFrame | list[pd.DataFrame]], typing.Any
    ],
    suffix: str = "",
    filament_spacing: int = 0,
) -> os.PathLike:

    last_file = ""
    for data, meta, layer in layer_data:
        is_filament_data = is_filament_layer(layer_data)
        if isinstance(data, list):
            data, meta = convert_shape_filament_layer_to_boxlayer(data, meta)

        if is_filament_data:

            if "edge_width" not in meta:
                boxsize = meta["features"]["boxsize"]
                meta["edge_width"] = boxsize

            fid = meta["features"]["fid"]

            fid = np.array(fid, dtype=int)

            data_list = np.append(data, np.atleast_2d(np.array(fid)).T, axis=1)
            last_file = _write_particle_data(
                path,
                data_list,
                meta,
                format_func,
                write_func,
                suffix,
                filament_spacing,
            )
        else:
            last_file = _write_particle_data(
                path,
                data,
                meta,
                format_func,
                write_func,
                suffix,
                filament_spacing,
            )

    return str(last_file)


class LoaderProxy(Array):
    def __init__(self, files, reader_func, do_normalize):
        self.reader_func = reader_func
        self.files = files
        self.do_normalize = do_normalize
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
        if self.do_normalize:
            return (data - np.mean(data)) / np.std(data)
        else:
            return data

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

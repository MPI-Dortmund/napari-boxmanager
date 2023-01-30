import os

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import io_utils as coordsio
from .interface import NapariLayerData


class BoxFileNumberOfColumnsError(pd.errors.IntCastingNaNError):
    pass


class UnknownFormatException(Exception):
    ...


DEFAULT_BOXSIZE: int = 10
DEFAULT_RESAMPLING_FILAMENT: int = -1  # 20% of boxsize


def get_valid_extensions():
    return ["box"]


def is_helicon_with_particle_coords(path):
    try:
        with open(path) as f:
            first_line = f.readline()
            f.close()
        return "#micrograph" in first_line
    except ValueError:
        return False


def read_boxfile(path: "os.PathLike") -> pd.DataFrame:
    names = ["x", "y", "box_x", "box_y"]
    box_data: pd.DataFrame = pd.read_csv(
        path,
        delim_whitespace=True,
        index_col=False,
        header=None,
        dtype=float,
        names=names,
        usecols=range(len(names)),
    )  # type: ignore
    try:
        box_data.astype(int)
    except pd.errors.IntCastingNaNError:
        raise BoxFileNumberOfColumnsError
    return box_data


def read_helicon_boxfile(path: "os.PathLike") -> pd.DataFrame:
    def get_first_and_last_coord(helix_line):
        if not helix_line.startswith("#helix"):
            raise ValueError("Line does not start with '#helix'")
        import re

        result = re.findall(r"\d+.\d*", helix_line)
        allnumbers = [float(r) for r in result]
        return allnumbers[:4]

    if os.stat(path).st_size != 0:
        split_indicis = []
        boxsize = 0
        index_first_helix = -1
        csvlines = None
        with open(path) as csvfile:
            csvlines = csvfile.readlines()
            helixlines_indicies = []
            for index, row in enumerate(csvlines):
                if row.startswith("#segment"):
                    boxsize = int(float(row.split()[2]))
                elif row.startswith("#helix"):
                    boxsize = int(float(row[(row.rfind(",") + 1) :]))
                    if index_first_helix == -1:
                        index_first_helix = index
                    else:

                        split_indicis.append(
                            index
                            - index_first_helix
                            - (len(split_indicis) + 1)
                        )
                    helixlines_indicies.append(index)

        coordinates = np.atleast_2d(np.genfromtxt(path))
        # coordinates_lowleftcorner = coordinates - boxsize / 2
        coord_filaments = np.split(coordinates, split_indicis)

        data_dict = {"x": [], "y": [], "box_x": [], "box_y": [], "fid": []}
        for filament_index, filament in enumerate(coord_filaments):
            first_and_last_coord = get_first_and_last_coord(
                csvlines[helixlines_indicies[filament_index]]
            )

            # first
            data_dict["x"].append(first_and_last_coord[0])
            data_dict["y"].append(first_and_last_coord[1])
            data_dict["box_x"].append(boxsize)
            data_dict["box_y"].append(boxsize)
            data_dict["fid"].append(filament_index + 1)

            # in between
            for coords in filament:
                if len(coords) > 0:
                    data_dict["x"].append(coords[0])
                    data_dict["y"].append(coords[1])
                    data_dict["box_x"].append(boxsize)
                    data_dict["box_y"].append(boxsize)
                    data_dict["fid"].append(filament_index + 1)

            # last
            data_dict["x"].append(first_and_last_coord[2])
            data_dict["y"].append(first_and_last_coord[3])
            data_dict["box_x"].append(boxsize)
            data_dict["box_y"].append(boxsize)
            data_dict["fid"].append(filament_index + 1)

        data_df = pd.DataFrame(data_dict)

        ## make lower left corner to be compatible with prepare function
        data_df["x"] = data_df["x"] - boxsize // 2
        data_df["y"] = data_df["y"] - boxsize // 2

        return data_df

    return None


def write_eman1_helicon(path: str, filaments: list[pd.DataFrame], **kwargs):

    import csv

    image_filename = "NA"
    if "image_name" in kwargs:
        image_filename = kwargs["image_name"]
    with open(path, "w", newline="", encoding="utf-8") as boxfile:
        boxwriter = csv.writer(
            boxfile,
            delimiter="\t",
            quotechar="|",
            quoting=csv.QUOTE_NONE,
            lineterminator="\n",
        )
        # micrograph: actin_cAla_1_corrfull.mrc
        # segment length: 384
        # segment width: 384

        if filaments is not None and len(filaments) > 0:

            boxsize = filaments[0]["boxsize"][0]

            boxwriter.writerow(["#micrograph: " + image_filename])
            boxwriter.writerow(["#segment length: " + str(int(boxsize))])
            boxwriter.writerow(["#segment width: " + str(int(boxsize))])

            for fil in filaments:
                if len(fil) > 0:
                    boxwriter.writerow(
                        [
                            "#helix: ("
                            + str(fil["x"][0] + boxsize / 2)
                            + ", "
                            + str(fil["y"][0] + boxsize / 2)
                            + "),"
                            + "("
                            + str(fil["x"][len(fil) - 1] + boxsize / 2)
                            + ", "
                            + str(fil["y"][len(fil) - 1] + boxsize / 2)
                            + "),"
                            + str(int(boxsize))
                        ]
                    )
                    for index, box in fil.iterrows():
                        boxwriter.writerow(
                            [
                                ""
                                + str(box["x"] + boxsize / 2)
                                + " "
                                + str(box["y"] + boxsize / 2)
                            ]
                        )


def read(path: "os.PathLike") -> pd.DataFrame:
    if is_helicon_with_particle_coords(path):
        box_data: pd.DataFrame = read_helicon_boxfile(path)
    else:
        box_data: pd.DataFrame = read_boxfile(path)
    return box_data


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[NapariLayerData]":
    r = coordsio.to_napari_coordinates(
        path=path,
        read_func=read,
        prepare_napari_func=_prepare_napari_box,
        meta_columns=_get_meta_idx(),
        feature_columns=_get_feature_idx(),
        valid_extensions=get_valid_extensions(),
    )
    return r


def _get_meta_idx():
    return []


def _get_feature_idx():
    return ["fid", "boxsize"]


def _prepare_napari_box(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    input_df Raw data from read function

    Returns
    -------
    Dataframe with centered coordinates with box size. Returns a list of dataframes in case of filaments

    """
    output_data: pd.DataFrame = pd.DataFrame(
        columns=["y", "z"] + _get_meta_idx()
    )

    output_data["z"] = input_df["x"] + input_df["box_x"] // 2
    output_data["y"] = input_df["y"] + input_df["box_y"] // 2

    output_data["boxsize"] = np.maximum(
        input_df[["box_x", "box_y"]].mean(axis=1), DEFAULT_BOXSIZE
    ).astype(int)

    if "fid" in input_df:
        output_data["fid"] = input_df["fid"]

    return output_data


def _write_box(path: os.PathLike, df: pd.DataFrame, **kwargs):
    df[["x", "y", "boxsize", "boxsize"]].to_csv(
        path, sep=" ", index=None, header=None
    )


def _make_df_data_particle(
    coordinates: pd.DataFrame, box_size: npt.ArrayLike, **kwargs
) -> pd.DataFrame:
    data = {"x": [], "y": [], "boxsize": []}
    for (y, x), boxsize in zip(
        coordinates,
        box_size,
    ):
        data["x"].append(x - boxsize / 2)
        data["y"].append(y - boxsize / 2)
        data["boxsize"].append(boxsize)
    return pd.DataFrame(data)


def _make_df_data_filament(
    coordinates: pd.DataFrame,
    box_size: npt.ArrayLike,
    filament_spacing: int,
    **kwargs,
) -> list[pd.DataFrame]:
    data = {"x": [], "y": [], "fid": [], "boxsize": []}
    filaments = []

    for (y, x, fid), boxsize in zip(
        coordinates,
        box_size,
    ):
        if len(data["fid"]) > 0 and data["fid"][-1] != fid:
            filaments.append(pd.DataFrame(data))
            data = {"x": [], "y": [], "fid": [], "boxsize": []}

        data["x"].append(x - boxsize / 2)
        data["y"].append(y - boxsize / 2)
        data["fid"].append(fid)
        data["boxsize"].append(boxsize)
    filaments.append(pd.DataFrame(data))

    ## Resampling
    for index_fil, fil in enumerate(filaments):
        if not filament_spacing:
            distance = int(fil["boxsize"][0] * 0.2)
        else:
            distance = filament_spacing
        filaments[index_fil] = coordsio.resample_filament(
            fil,
            distance,
            coordinate_columns=["x", "y"],
            constant_columns=["boxsize"],
        )

    return filaments


def from_napari(
    path: os.PathLike,
    layer_data: list[NapariLayerData],
    suffix: str,
    filament_spacing: int,
):
    is_filament = coordsio.is_filament_layer(layer_data)
    if is_filament:
        format_func = _make_df_data_filament
        write_func = write_eman1_helicon
    else:
        format_func = _make_df_data_particle
        write_func = _write_box
    path = coordsio.from_napari(
        path=path,
        layer_data=layer_data,
        write_func=write_func,
        format_func=format_func,
        suffix=suffix,
        filament_spacing=filament_spacing,
    )

    return path

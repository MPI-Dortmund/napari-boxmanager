import os
import typing
from pyStarDB import sp_pystardb as star
import pandas as pd
import numpy as np
from . import coordinate_io as coordsio
import numpy.typing as npt

valid_extensions = ['cbox']
coords_3d_idx = ["x", "y", "z"]
coords_2d_idx = ["y", "z"]
meta_columns = []

def read(path: os.PathLike) -> pd.DataFrame:
    starfile = star.StarFile(path)
    data_dict = starfile['cryolo']
    return pd.DataFrame(data_dict)


def to_napari(
    path: os.PathLike | list[os.PathLike],
) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":
    r = coordsio.to_napari(
        path=path,
        read_func=read,
        prepare_napari_func=_prepare_napari,
        meta_columns=_get_meta_columns(),
        feature_columns=_get_meta_columns()
    )

    return r

def write_cbox(path : os.PathLike, df: pd.DataFrame):
    columns = []
    columns.append('_CoordinateX')
    columns.append('_CoordinateY')
    columns.append('_CoordinateZ')
    columns.append('_Width')
    columns.append('_Height')
    columns.append('_Depth')
    columns.append('_EstWidth')
    columns.append('_EstHeight')
    columns.append('_Confidence')
    columns.append('_NumBoxes')
    columns.append('_Angle')
    print("write write write")
    #df = pd.DataFrame(coords, columns=columns)



def get_valid_extensions() -> list[str]:
    return valid_extensions


def _make_df_data(coordinates: pd.DataFrame,
                  box_size: npt.ArrayLike,
                  features: pd.DataFrame) -> pd.DataFrame:
    data = {
        "_CoordinateX": [],
        "_CoordinateY": [],
        "_CoordinateZ": [],
        "_Width": [],
        "_Height": [],
        "_Depth": [],
        "_EstWidth": [],
        "_EstHeight": [],
        "_Confidence": [],
        "_NumBoxes": [],
        "_Angle": []
    }

    for coords, boxsize in zip(
            coordinates,
            box_size,
    ):
        is_3d = True

        if len(coords) == 2:
            is_3d = False
            y,x = coords
            z = None
        else:
            z, y, x = coords

        data["_CoordinateX"].append(x-boxsize/2)
        data["_CoordinateY"].append(y-boxsize/2)
        data["_CoordinateZ"].append(z)
        data["_Width"].append(boxsize)
        data["_Height"].append(boxsize)
        if is_3d:
            data["_Depth"].append(boxsize)



def from_napari(
    path: os.PathLike | list[os.PathLike] | pd.DataFrame,
    layer_data: list[tuple[typing.Any, dict, str]],
):
    coordsio.from_napari(
        path=path,
        layer_data=layer_data,
        write_func=write_cbox,
        format_func=_make_df_data,
        is_2d_stacked=True
    )


def _fill_meta_idx(input_df: pd.DataFrame) -> None:
    '''
    Fills the meta idx array.

    Parameters
    ----------
    input_dict Raw input data

    Returns
    -------
    None

    '''
    global meta_columns

    if (not input_df['_EstWidth'].isnull().values.any()) and 'size' not in meta_columns :
        meta_columns.append('size')
    if (not input_df['_Confidence'].isnull().values.any()) and 'confidence' not in meta_columns:
        meta_columns.append('confidence')
    if (not input_df['_NumBoxes'].isnull().values.any()) and 'num_boxes' not in meta_columns:
        meta_columns.append('num_boxes')


def _prepare_napari(input_df: pd.DataFrame) -> pd.DataFrame:
    '''

    Parameters
    ----------
    input_df Dataframe with raw data from the read function

    Returns
    -------
    Dataframe with centered coordinates and additional metadate if necessary.

    '''

    cryolo_data = input_df

    _fill_meta_idx(cryolo_data)
    is_3d=True
    if cryolo_data['_CoordinateZ'].isnull().values.any():
        is_3d=False

    columns = ['z', 'y']
    if is_3d:
        columns.append('x')

    output_data: pd.DataFrame = pd.DataFrame(
        columns=columns
                + _get_meta_columns()
    )

    output_data['z'] = np.array(cryolo_data['_CoordinateX']) + np.array(cryolo_data["_Width"]/2)
    output_data['y'] = np.array(cryolo_data['_CoordinateY']) + np.array(cryolo_data["_Height"]/2)
    if is_3d:
        output_data['x'] = cryolo_data['_CoordinateZ']



    output_data["boxsize"] = (np.array(cryolo_data["_Width"]) + np.array(cryolo_data["_Height"]))/2

    if 'size' in meta_columns:
        output_data["size"] = (np.array(cryolo_data["_EstWidth"]) + np.array(cryolo_data["_EstHeight"])) / 2

    if 'num_boxes' in meta_columns:
        output_data["num_boxes"] = cryolo_data["_NumBoxes"]

    if 'confidence' in meta_columns:
        output_data["confidence"] = cryolo_data["_Confidence"]

    return output_data


def _get_meta_columns():
    return meta_columns


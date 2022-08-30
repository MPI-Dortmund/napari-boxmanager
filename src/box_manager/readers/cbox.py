import os
import typing
from pyStarDB import sp_pystardb as star
import pandas as pd
import numpy as np
from interface import ReaderInterface



class CBOXReader(ReaderInterface):

    def __init__(self):
        self.valid_extensions = ['cbox']
        self.coords_3d_idx = ["x", "y", "z"]
        self.meta_idx = ["confidence", "size"]
        self.idx_cbox_dict = {
            'x': '_CoordinateZ',
            'y': '_CoordinateY',
            'z': '_CoordinateX',
            'boxsize': '_Width',
            'confidence': '_Confidence',
            'size': ['_EstWidth','_EstHeight'],
            'numboxes': ['_NumBoxes'],
            'angle': ['_Angle']
        }

    def read(self, path: os.PathLike) -> typing.Dict:
        starfile = star.StarFile(path)
        data_dict = starfile['cryolo']
        return data_dict

    def to_napari(
        self,
        path: os.PathLike | list[os.PathLike],
    ) -> "list[tuple[npt.ArrayLike, dict[str, typing.Any], str]]":

        if not isinstance(path, list):
            path = sorted(glob.glob(path))  # type: ignore

        for file_name in path:
            pass

    def get_valid_extensions(self) -> list[str]:
        return self.valid_extensions

    def from_napari(
        self,
        path: os.PathLike | list[os.PathLike] | pd.DataFrame,
        data: typing.Any,
        meta: dict,
    ):
        data_dict = self.read(path)
        napari_df = self._prepare_napari(data_dict)
        print(napari_df)

    def _prepare_napari(self, input_dict: typing.Dict) -> pd.DataFrame:

        output_data: pd.DataFrame = pd.DataFrame(
            columns=self._get_3d_coords_idx()
            + self._get_meta_idx()
        )
        cryolo_data = input_dict['cryolo']

        output_data['x'] = cryolo_data['_CoordinateZ']
        output_data['y'] = cryolo_data['_CoordinateY']
        output_data['z'] = cryolo_data['_CoordinateX']
        output_data["boxsize"] = (np.array(cryolo_data["_Width"]) + np.array(cryolo_data["_Height"]))/2

        print(cryolo_data['_CoordinateZ'])

    def _get_3d_coords_idx(self):
        return COORDS_3D_IDX


    def _get_meta_idx(self):
        return META_IDX


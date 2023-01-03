import numpy as np
import pandas as pd
from numpy import testing

from box_manager.io.io_utils import resample_filament


def test_resampling_simple_distance_1():
    input = {"x": [1, 5, 7, 15, 16], "y": [1, 5, 7, 15, 16]}
    expected_output = {
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    }

    fil = pd.DataFrame(input)

    output = resample_filament(fil, 1, ["x", "y"])

    exp_x_float = np.array(expected_output["x"]).astype(float)
    exp_y_float = np.array(expected_output["y"]).astype(float)
    out_x_float = np.array(output["x"]).astype(float)
    out_y_float = np.array(output["y"]).astype(float)
    testing.assert_allclose(exp_x_float, out_x_float)
    testing.assert_allclose(exp_y_float, out_y_float)


def test_resampling_simple_distance_2():
    input = {"x": [1, 5, 7, 15], "y": [1, 5, 7, 15]}
    expected_output = {
        "x": [1, 3, 5, 7, 9, 11, 13, 15],
        "y": [1, 3, 5, 7, 9, 11, 13, 15],
    }

    fil = pd.DataFrame(input)

    output = resample_filament(fil, 2, ["x", "y"])

    exp_x_float = np.array(expected_output["x"]).astype(float)
    exp_y_float = np.array(expected_output["y"]).astype(float)
    out_x_float = np.array(output["x"]).astype(float)
    out_y_float = np.array(output["y"]).astype(float)
    testing.assert_allclose(exp_x_float, out_x_float)
    testing.assert_allclose(exp_y_float, out_y_float)

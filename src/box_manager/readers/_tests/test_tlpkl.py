import os
import pathlib

import numpy
import pandas

import box_manager.readers.tlpkl as nrt

TEST_DATA = pathlib.Path(os.path.dirname(__file__), "test_data")


def test_correct_read():
    tlpkl_file = pathlib.Path(TEST_DATA, "valid.tlpkl")
    test_data = nrt.read(tlpkl_file)

    x = [86, 88, 6, 24, 78, 46, 30, 60, 76, 76, 70, 66, 30, 76]
    y = [26, 2, 94, 40, 52, 58, 62, 86, 16, 60, 82, 44, 18, 94]
    z = [98, 44, 92, 50, 48, 72, 64, 44, 92, 58, 48, 18, 28, 10]
    size = [18, 40, 90, 67, 89, 88, 82, 82, 76, 51, 11, 282, 248, 244]
    metric_best = numpy.array(
        [
            0.639648,
            0.945801,
            0.947754,
            0.951172,
            0.953125,
            0.955566,
            0.956055,
            0.956543,
            0.534180,
            0.538574,
            0.550781,
            0.880371,
            0.885742,
            0.888184,
        ],
        dtype=numpy.float32,
    )
    width = [37] * len(x)
    height = [40] * len(x)
    depth = [50] * len(x)
    predicted_class = [1] * 8 + [2] * 6
    predicted_class_name = ["cluster_2.pkl"] * 8 + ["cluster_3.pkl"] * 6
    expected = pandas.DataFrame(
        {
            "X": x,
            "Y": y,
            "Z": z,
            "size": size,
            "metric_best": metric_best,
            "predicted_class": predicted_class,
            "predicted_class_name": predicted_class_name,
            "width": width,
            "height": height,
            "depth": depth,
        }
    )
    pandas.testing.assert_frame_equal(test_data[expected.columns], expected)


def test_correct_prepare_napari():
    x = [86, 88, 6]
    y = [26, 2, 94]
    z = [98, 44, 92]
    size = [18, 40, 90]
    metric_best = numpy.array(
        [0.639648, 0.945801, 0.947754], dtype=numpy.float32
    )
    width = [37] * len(x)
    height = [40] * len(x)
    depth = [50] * len(x)
    input_df = pandas.DataFrame(
        {
            "Z": z,
            "Y": y,
            "X": x,
            "size": size,
            "metric_best": metric_best,
            "width": width,
            "height": height,
            "depth": depth,
            "predicted_class_name": ["classx.pkl"] * 3,
            "predicted_class": [0] * 3,
        }
    )
    expected = pandas.DataFrame(
        {
            "z": x,
            "y": y,
            "x": z,
            "size": size,
            "metric": metric_best,
            "boxsize": numpy.mean(numpy.array([width, height, depth])),
            "grp_idx": ["classx.pkl"] * 3,
            "sort_idx": [0] * 3,
        }
    )
    test_df, test_coords, test_metadata, kwargs = nrt.prepare_napari(input_df)
    assert test_coords == ["x", "y", "z"]
    assert test_metadata == ["metric", "size", "boxsize"]
    pandas.testing.assert_frame_equal(
        test_df[expected.columns.values], expected
    )
    assert kwargs == {}

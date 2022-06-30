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
        }
    )
    input_df.attrs["dim_z"] = 100
    expected = pandas.DataFrame(
        {
            "z": x,
            "y": y,
            "x": z,
            "size": size,
            "metric": metric_best,
            "boxsize": numpy.mean(numpy.array([width, height, depth])),
        }
    )
    test_df, test_coords, test_metadata = nrt._prepare_napari(input_df)
    assert test_coords == ["x", "y", "z"]
    assert test_metadata == ["metric", "size", "boxsize"]
    pandas.testing.assert_frame_equal(
        test_df[expected.columns.values], expected
    )


def test_to_napari_file_correct_returns():
    tlpkl_file = pathlib.Path(TEST_DATA, "valid.tlpkl")

    points1, points2 = nrt.to_napari(tlpkl_file)
    assert all([len(entry) == 3 for entry in (points1, points2)])

    x = [86, 88, 6, 24, 78, 46, 30, 60, 76, 76, 70, 66, 30, 76]
    y = [26, 2, 94, 40, 52, 58, 62, 86, 16, 60, 82, 44, 18, 94]
    z = [98, 44, 92, 50, 48, 72, 64, 44, 92, 58, 48, 18, 28, 10]
    points = pandas.DataFrame(
        {
            "x": z,
            "y": y,
            "z": x,
        }
    )
    pandas.testing.assert_frame_equal(points1[0], points.iloc[:8])
    pandas.testing.assert_frame_equal(points2[0], points.iloc[8:])

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

    boxsize = [numpy.array([width, height, depth]).mean().mean()] * len(x)

    metadata = {
        "id": 1,
        "size_min": min(size[:8]),
        "size_max": max(size[:8]),
        "metric_min": min(metric_best[:8]),
        "metric_max": max(metric_best[:8]),
        "boxsize_min": min(boxsize[:8]),
        "boxsize_max": max(boxsize[:8]),
    }
    assert list(sorted(metadata.keys())) == list(
        sorted(points1[1]["metadata"].keys())
    )
    for key in metadata.keys():
        val1 = metadata[key]
        val2 = points1[1]["metadata"][key]
        numpy.testing.assert_array_almost_equal(val1, val2)

    metadata = {
        "id": 2,
        "size_min": min(size[8:]),
        "size_max": max(size[8:]),
        "metric_min": min(metric_best[8:]),
        "metric_max": max(metric_best[8:]),
        "boxsize_min": min(boxsize[8:]),
        "boxsize_max": max(boxsize[8:]),
    }
    assert list(sorted(metadata.keys())) == list(
        sorted(points2[1]["metadata"].keys())
    )
    for key in metadata.keys():
        val1 = metadata[key]
        val2 = points2[1]["metadata"][key]
        numpy.testing.assert_array_almost_equal(val1, val2)

    features = {
        "size": size[:8],
        "metric": metric_best[:8],
        "boxsize": boxsize[:8],
    }
    assert list(sorted(features.keys())) == list(
        sorted(points1[1]["features"].keys())
    )
    for key in features.keys():
        val1 = features[key]
        val2 = points1[1]["features"][key]
        numpy.testing.assert_array_almost_equal(val1, val2)

    features = {
        "size": size[8:],
        "metric": metric_best[8:],
        "boxsize": boxsize[8:],
    }
    assert list(sorted(features.keys())) == list(
        sorted(points2[1]["features"].keys())
    )
    for key in features.keys():
        val1 = features[key]
        val2 = points2[1]["features"][key]
        numpy.testing.assert_array_almost_equal(val1, val2)

    numpy.testing.assert_array_equal(boxsize[:8], points1[1]["size"])
    numpy.testing.assert_array_equal(boxsize[8:], points2[1]["size"])

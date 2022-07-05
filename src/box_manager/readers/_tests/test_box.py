import os
import pathlib

import pandas
import pytest

import box_manager.readers.box as brb

TEST_DATA = pathlib.Path(os.path.dirname(__file__), "test_data")


@pytest.mark.parametrize("file_name", ["2d_center.box", "2d_center_float.box"])
def test_read_box_file_center_correct_coord_values(file_name):
    box_file = pathlib.Path(TEST_DATA, file_name)
    box_data = brb.read(box_file)

    expected = pandas.DataFrame(
        {
            "x": [24, 36, 78],
            "y": [40, 62, 51],
            "box_x": [0, 0, 0],
            "box_y": [0, 0, 0],
            "filename": box_file,
        }
    )
    pandas.testing.assert_frame_equal(box_data, expected)


def test_read_box_file_left_correct_coord_values():
    file_name = "2d_left.box"
    box_file = pathlib.Path(TEST_DATA, file_name)
    box_data = brb.read(box_file)

    expected = pandas.DataFrame(
        {
            "x": [14, 26, 68],
            "y": [35, 57, 46],
            "box_x": [20, 20, 20],
            "box_y": [10, 10, 10],
            "filename": box_file,
        }
    )
    pandas.testing.assert_frame_equal(box_data, expected)


def test_read_box_file_empty_expect_empty_list():
    box_file = pathlib.Path(TEST_DATA, "2d_empty.box")
    box_data = brb.read(box_file)

    expected = pandas.DataFrame(
        columns=["x", "y", "box_x", "box_y", "filename"]
    )
    pandas.testing.assert_frame_equal(box_data, expected, check_dtype=False)


def test_read_box_file_corrupt_column_expect_error():
    box_file = pathlib.Path(TEST_DATA, "2d_corrupt_unqual_columns.box")
    with pytest.raises(brb.BoxFileNumberOfColumnsError):
        brb.read(box_file)


def test_read_box_file_corrupt_string_expect_error():
    box_file = pathlib.Path(TEST_DATA, "2d_corrupt_has_string.box")
    with pytest.raises(ValueError):
        brb.read(box_file)


@pytest.mark.parametrize(
    "params", [[[10, 10, 10], [0, 0, 0]], [[30, 35, 40]] * 2]
)
def test_prepare_napari_box_file_center_correct_coord_values(params):
    expected_size, box_size = params
    print
    input_df = pandas.DataFrame(
        {
            "x": [
                24 - box_size[0] // 2,
                36 - box_size[1] // 2,
                78 - box_size[2] // 2,
            ],
            "y": [
                40 - box_size[0] // 2,
                62 - box_size[1] // 2,
                51 - box_size[2] // 2,
            ],
            "box_x": box_size,
            "box_y": box_size,
            "filename": "test",
        }
    )
    expected_df = pandas.DataFrame(
        {
            "x": [40, 62, 51],
            "y": [24, 36, 78],
            "boxsize": expected_size,
            "sort_idx": "test",
            "grp_idx": "test",
        }
    )
    data, coords_idx, metrics_idx, kwargs = brb.prepare_napari(input_df)
    pandas.testing.assert_frame_equal(data, expected_df)
    assert coords_idx == ["x", "y"]
    assert metrics_idx == ["boxsize"]
    assert kwargs == {"out_of_slice_display": False}

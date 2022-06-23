import os
import pathlib

import numpy
import pytest

import box_manager.readers.box as brb

TEST_DATA = pathlib.Path(os.path.dirname(__file__), "test_data")


def test_read_box_file_center_expect_correct_values():
    box_file = pathlib.Path(TEST_DATA, "2d_center.box")
    box_data = brb.read(box_file)

    expected = numpy.array([[40, 24], [62, 36], [51, 78]])
    numpy.testing.assert_equal(box_data, expected)


def test_read_box_file_center_float_expect_correct_int_values():
    box_file = pathlib.Path(TEST_DATA, "2d_center_float.box")
    box_data = brb.read(box_file)

    expected = numpy.array([[40, 24], [62, 36], [51, 78]])
    numpy.testing.assert_equal(box_data, expected)


def test_read_box_file_left_expect_correct_values():
    box_file = pathlib.Path(TEST_DATA, "2d_left.box")
    box_data = brb.read(box_file)

    expected = numpy.array([[40, 24], [62, 36], [51, 78]])
    numpy.testing.assert_equal(box_data, expected)


def test_read_box_file_empty_expect_empty_list():
    box_file = pathlib.Path(TEST_DATA, "2d_empty.box")
    box_data = brb.read(box_file)

    expected = numpy.empty((0, 0), dtype=int)
    numpy.testing.assert_equal(box_data, expected)


def test_read_box_file_corrupt_column_expect_error():
    box_file = pathlib.Path(TEST_DATA, "2d_corrupt_unqual_columns.box")
    with pytest.raises(brb.BoxFileNumberOfColumnsError):
        brb.read(box_file)


def test_read_box_file_corrupt_string_expect_error():
    box_file = pathlib.Path(TEST_DATA, "2d_corrupt_has_string.box")
    with pytest.raises(ValueError):
        brb.read(box_file)

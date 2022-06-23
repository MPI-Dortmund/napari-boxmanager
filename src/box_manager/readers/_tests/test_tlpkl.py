import os
import pathlib

import numpy
import pytest

import box_manager.readers.tlpkl as nrt

TEST_DATA = pathlib.Path(os.path.dirname(__file__), "test_data")


@pytest.mark.parametrize(
    "params",
    [
        [
            2,
            [
                86,
                88,
                6,
                24,
                78,
                46,
                30,
                60,
                76,
                76,
                70,
                66,
                30,
                76,
            ],
        ],
        [
            0,
            [
                26,
                2,
                94,
                40,
                52,
                58,
                62,
                86,
                16,
                60,
                82,
                44,
                18,
                94,
            ],
        ],
    ],
)
def test_read_tlpkl_is_valid_x_y(params):
    idx, values = params
    tlpkl_file = pathlib.Path(TEST_DATA, "valid.tlpkl")
    test_data = nrt.read(tlpkl_file)

    numpy.testing.assert_array_equal(test_data.iloc[:, idx], values)


def test_read_tlpkl_is_valid_z():
    tlpkl_file = pathlib.Path(TEST_DATA, "valid.tlpkl")
    test_data = nrt.read(tlpkl_file)

    dim = 100
    z_values = [
        dim - entry
        for entry in [
            98,
            44,
            92,
            50,
            48,
            72,
            64,
            44,
            92,
            58,
            48,
            18,
            28,
            10,
        ]
    ]
    numpy.testing.assert_array_equal(test_data.iloc[:, 1], z_values)


def test_read_tlp_is_valid_size():
    tlpkl_file = pathlib.Path(TEST_DATA, "valid.tlpkl")
    test_data = nrt.read(tlpkl_file)

    size = [
        18,
        40,
        90,
        67,
        89,
        88,
        82,
        82,
        76,
        51,
        11,
        282,
        248,
        244,
    ]
    numpy.testing.assert_array_equal(test_data.iloc[:, 4], size)


def test_read_tlp_is_valid_metric():
    tlpkl_file = pathlib.Path(TEST_DATA, "valid.tlpkl")
    test_data = nrt.read(tlpkl_file)

    metric = [
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
    ]
    numpy.testing.assert_array_almost_equal(test_data.iloc[:, 3], metric)


def test_read_tlpkl_missing_dim_z_leads_to_warning():
    tlpkl_file = pathlib.Path(TEST_DATA, "valid_missing_dim_z.tlpkl")
    with pytest.warns(nrt.DimZMissingWarning):
        nrt.read(tlpkl_file)


@pytest.mark.filterwarnings(
    "ignore::box_manager.readers.tlpkl.DimZMissingWarning"
)
def test_read_tlpkl_missing_dim_z_attr_yields_normal_z():
    tlpkl_file = pathlib.Path(TEST_DATA, "valid_missing_dim_z.tlpkl")
    test_data = nrt.read(tlpkl_file)

    z_values = [
        98,
        44,
        92,
        50,
        48,
        72,
        64,
        44,
        92,
        58,
        48,
        18,
        28,
        10,
    ]
    numpy.testing.assert_array_equal(test_data.iloc[:, 1], z_values)


@pytest.mark.parametrize("params", [[5, 40], [6, 50], [7, 37]])
def test_read_tlp_is_correct_boxsize(params):
    idx, box_size = params
    tlpkl_file = pathlib.Path(TEST_DATA, "valid.tlpkl")
    test_data = nrt.read(tlpkl_file)

    numpy.testing.assert_array_equal(
        test_data.iloc[:, idx], [box_size] * len(test_data)
    )

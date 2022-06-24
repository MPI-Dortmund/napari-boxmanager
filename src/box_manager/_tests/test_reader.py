# import numpy as np

import os

import pytest

import box_manager._reader as br


# tmp_path is a pytest fixture
@pytest.mark.parametrize(
    "file_ending",
    [".pkl", ".tlpkl", ".tepkl", ".tmpkl", ".cbox", ".box", ".star"],
)
def test_read_valid_files_return_func(tmp_path, file_ending):
    file_path = os.path.join(tmp_path, f"tmp{file_ending}")
    with open(file_path, "w"):
        pass

    assert br.napari_get_reader(file_path) == br.reader_function


@pytest.mark.parametrize(
    "file_ending", [".invalid", ".tlpkl2", ".tpkl", ".ok"]
)
def test_read_invalid_files_returns_none(tmp_path, file_ending):
    file_path = os.path.join(tmp_path, f"tmp{file_ending}")
    with open(file_path, "w"):
        pass

    assert br.napari_get_reader(file_path) is None


# tmp_path is a pytest fixture
# def test_reader(tmp_path):
#    """An example of how you might test your plugin."""
#
#    # write some fake data using your supported file format
#    my_test_file = str(tmp_path / "myfile.npy")
#    original_data = np.random.rand(20, 20)
#    np.save(my_test_file, original_data)
#
#    # try to read it back in
#    reader = napari_get_reader(my_test_file)
#    assert callable(reader)
#
#    # make sure we're delivering the right format
#    layer_data_list = reader(my_test_file)
#    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
#    layer_data_tuple = layer_data_list[0]
#    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0
#
#    # make sure it's the same as it started
#    np.testing.assert_allclose(original_data, layer_data_tuple[0])
#
#
# def test_get_reader_pass():
#    reader = napari_get_reader("fake.file")
#    assert reader is None

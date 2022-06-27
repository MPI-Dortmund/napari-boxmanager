# import numpy as np

import os

import pandas as pd
import pytest

import box_manager._reader as br
from box_manager.readers import tepkl, tlpkl, tmpkl

HERE = os.path.dirname(__file__)

VALID_PKL = [".tlpkl", ".tepkl", ".tmpkl"]
VALID_PKL_FUNC = [tlpkl.to_napari, tepkl.to_napari, tmpkl.to_napari]
VALID_BOX = [".cbox", ".box", ".star"]
VALID_BOX_FUNC = [br.cbox.to_napari, br.box.to_napari, br.star.to_napari]
VALID_FILE_ENDINGS = VALID_PKL + VALID_BOX + [".pkl"]
INVALID_FILE_ENDINGS = [".pkl2", ".tlpkl2", ".tepkl2"]
INVALID_FILE_FUNC = [None, None, None]

VALID_FILES = [
    f"{HERE}/test_data/2d_left.box",
    f"{HERE}/test_data/valid.tlpkl",
]


# tmp_path is a pytest fixture
@pytest.mark.parametrize("file_ending", VALID_FILE_ENDINGS)
def test_read_valid_files_return_func(file_ending):
    file_path = f"tmp{file_ending}"
    assert br.napari_get_reader(file_path) == br.reader_function


def test_read_first_valid_is_None():
    assert (
        br.napari_get_reader(VALID_FILE_ENDINGS + INVALID_FILE_ENDINGS)
        == br.reader_function
    )


def test_read_first_invalid_is_None():
    assert (
        br.napari_get_reader(INVALID_FILE_ENDINGS + VALID_FILE_ENDINGS) is None
    )


@pytest.mark.parametrize("file_ending", INVALID_FILE_ENDINGS)
def test_read_invalid_files_returns_none(file_ending):
    file_path = f"tmp{file_ending}"
    assert br.napari_get_reader(file_path) is None


@pytest.mark.parametrize("file_ending", VALID_FILE_ENDINGS)
def test_readclass_read_str_valid_files_return_func(file_ending):
    file_path = f"tmp{file_ending}"
    reader_class = br.ReaderClass(file_path)
    assert reader_class.paths == [file_path]


@pytest.mark.parametrize("file_ending", VALID_FILE_ENDINGS)
def test_readclass_read_list_valid_files_return_func(file_ending):
    file_path = f"tmp{file_ending}"
    reader_class = br.ReaderClass([file_path])
    assert reader_class.paths == [file_path]


@pytest.mark.parametrize("file_path", VALID_FILES)
def test_readclass_read_file_is_valid_true(file_path):
    reader_class = br.ReaderClass(file_path)
    assert reader_class.is_valid() == [True]


@pytest.mark.parametrize("file_path", VALID_FILES)
def test_readclass_read_file_list_is_valid_true(file_path):
    reader_class = br.ReaderClass([file_path])
    assert reader_class.is_valid() == [True]


@pytest.mark.parametrize("file_ending", VALID_FILE_ENDINGS)
def test_readclass_is_valid_true(file_ending):
    file_path = f"tmp{file_ending}"
    reader_class = br.ReaderClass(file_path)
    assert reader_class.is_valid() == [True]


@pytest.mark.parametrize("file_ending", INVALID_FILE_ENDINGS)
def test_readclass_is_valid_false(file_ending):
    file_path = f"tmp{file_ending}"
    reader_class = br.ReaderClass(file_path)
    assert reader_class.is_valid() == [False]


@pytest.mark.parametrize("file_ending", VALID_FILE_ENDINGS)
def test_readclass_is_valid_list_true(file_ending):
    file_path = f"tmp{file_ending}"
    reader_class = br.ReaderClass([file_path, file_path])
    assert reader_class.is_valid() == [True, True]


@pytest.mark.parametrize("file_ending", INVALID_FILE_ENDINGS)
def test_readclass_is_valid_list_false(file_ending):
    file_path = f"tmp{file_ending}"
    reader_class = br.ReaderClass([file_path, file_path])
    assert reader_class.is_valid() == [False, False]


@pytest.mark.parametrize("file_ending", VALID_FILE_ENDINGS)
def test_readclass_is_all_valid_list_true(file_ending):
    file_path = f"tmp{file_ending}"
    reader_class = br.ReaderClass([file_path, file_path])
    assert reader_class.is_all_valid()


@pytest.mark.parametrize("file_ending", INVALID_FILE_ENDINGS)
def test_readclass_is_all_valid_list_false(file_ending):
    file_path = f"tmp{file_ending}"
    reader_class = br.ReaderClass([file_path, file_path])
    assert not reader_class.is_all_valid()


@pytest.mark.parametrize("params", zip(VALID_PKL, VALID_PKL_FUNC))
def test_readclass_load_pkl_ending_returns_correct_functions(params):
    file_ending, return_func = params
    file_path = f"tmp{file_ending}"
    assert br.ReaderClass.load_pkl(file_path) == return_func


@pytest.mark.parametrize("params", zip(VALID_PKL, VALID_PKL_FUNC))
def test_readclass_load_pkl_attrs_returns_correct_functions(tmp_path, params):
    file_ending, return_func = params
    file_path = os.path.join(tmp_path, "tmp.pkl")
    test_data = pd.DataFrame()
    test_data.attrs["boxread_identifier"] = file_ending
    test_data.to_pickle(file_path)
    assert br.ReaderClass.load_pkl(file_path) == return_func


@pytest.mark.parametrize("file_ending", INVALID_FILE_ENDINGS)
def test_readclass_load_pkl_invalid_raises_assert(file_ending):
    file_path = f"tmp{file_ending}"
    with pytest.raises(AssertionError):
        br.ReaderClass.load_pkl(file_path)


@pytest.mark.parametrize("params", zip(VALID_PKL, VALID_PKL_FUNC))
def test_readclass_load_functions_attrs_returns_correct_functions(
    tmp_path, params
):
    file_ending, return_func = params
    file_path = os.path.join(tmp_path, "tmp.pkl")
    test_data = pd.DataFrame()
    test_data.attrs["boxread_identifier"] = file_ending
    test_data.to_pickle(file_path)
    assert br.ReaderClass(file_path).load_functions() == [return_func]


@pytest.mark.parametrize(
    "params", zip(VALID_PKL + VALID_BOX, VALID_PKL_FUNC + VALID_BOX_FUNC)
)
def test_readclass_load_functions_ending_returns_correct_functions(params):
    file_ending, return_func = params
    file_path = f"tmp{file_ending}"
    assert br.ReaderClass(file_path).load_functions() == [return_func]


@pytest.mark.parametrize(
    "params",
    zip(
        VALID_PKL + VALID_BOX + INVALID_FILE_ENDINGS,
        VALID_PKL_FUNC + VALID_BOX_FUNC + INVALID_FILE_FUNC,
    ),
)
def test_readclass_load_mixed_functions_ending_returns_correct_functions(
    params,
):
    file_ending, return_func = params
    file_path = f"tmp{file_ending}"
    assert br.ReaderClass(file_path).load_functions() == [return_func]


def test_readclass_load_functions_ending_returns_correct_functions_list():
    files = [
        f"tmp{entry}" for entry in VALID_PKL + VALID_BOX + INVALID_FILE_ENDINGS
    ]
    assert (
        br.ReaderClass(files).load_functions()
        == VALID_PKL_FUNC + VALID_BOX_FUNC + INVALID_FILE_FUNC
    )


def test_valid_but_not_specified_raises_assertion():
    new_valid_ending = ".asfasdfaw34r23sdfasfa34w5w3rsfd"
    file_path = f"tmp{new_valid_ending}"
    reader = br.ReaderClass(file_path)
    reader.valid_file_endings = (new_valid_ending,)
    with pytest.raises(AssertionError):
        reader.load_functions()


@pytest.mark.parametrize("file_path", INVALID_FILE_ENDINGS)
def test_reader_function_invalid_string_returns_empty_list(file_path):
    assert br.reader_function(file_path) == []


@pytest.mark.parametrize("file_path", INVALID_FILE_ENDINGS)
def test_reader_function_invalid_list_returns_empty_list(file_path):
    assert br.reader_function([file_path]) == []


@pytest.mark.parametrize("file_path", VALID_FILES)
def test_reader_function_valid_list_returns_empty_list(file_path):
    assert len(br.reader_function([file_path])) == 1


@pytest.mark.parametrize("file_path", VALID_FILES)
def test_reader_function_valid_string_returns_empty_list(file_path):
    assert len(br.reader_function(file_path)) == 1


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

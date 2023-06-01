import box_manager._reader as br
from box_manager import io as bm_readers
import pytest
import os
HERE = os.path.dirname(__file__)
def test_open_2D_filament_cbox_files_including_empty():

    reader = bm_readers.get_reader('cbox')
    reader(f"{HERE}/../../../test_data/filament_cbox_2d/*.cbox")

@pytest.mark.parametrize("input", [
f"{HERE}/../../../test_data/filament_cbox_2d/A*.cbox",
])
def test_open_2D_filament_cbox_files_without_empty(input):
    file_ext = os.path.splitext(input)[1][1:]
    reader = bm_readers.get_reader(file_ext)
    reader(input)
import box_manager._reader as br
import pytest

def test_open_2D_filament_cbox_files_including_empty():
    reader = br.napari_get_reader("fake.cbox")
    reader("/home/twagner/Projects/napari/src/napari-boxmanager-github/test_data/filament_cbox_2d/*.cbox")

@pytest.mark.parametrize("input", [
"/home/twagner/Projects/napari/src/napari-boxmanager-github/test_data/filament_cbox_2d/A*.cbox",
])
def test_open_2D_filament_cbox_files_without_empty(input):
    reader = br.napari_get_reader(input)
    reader(input)
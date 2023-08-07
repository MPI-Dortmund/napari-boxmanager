from unittest.mock import MagicMock, patch

import napari
import numpy as np
import pytest

import mrcfile
import tempfile

@pytest.fixture(scope="function")
def napari_viewer():
    yield napari.Viewer()

@pytest.fixture(scope="function")
def organize_layer_widget_tomo(napari_viewer):
    rand_vol = np.random.randn(500,500,500).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdirname:
        with mrcfile.new(f"{tmpdirname}/tmp.mrc") as mrc:
            mrc.set_data(rand_vol)
        napari_viewer.open(plugin='napari-boxmanager',
                    path=f"{tmpdirname}/tmp.mrc")
        widget, _ = napari_viewer.window.add_plugin_dock_widget('napari-boxmanager', widget_name='organize_layer', tabify=True)
        widget.widget()._new_shapes()
        yield widget.widget()
@patch(
        "qtpy.QtWidgets.QFileDialog.getExistingDirectory",
        MagicMock(
            return_value="/tmp/blub/"
        ),
    )
class Test__run_save:
    def test_save_tomo_filament(self, napari_viewer, organize_layer_widget_tomo):
        # generate random tomogram
        fila = [[77, 50, 50], [77, 100, 100], [77, 200, 200]]
        assert len(napari_viewer.layers) == 2
        napari_viewer.layers[1]._add_shapes(fila, shape_type='path', edge_width=100)
        organize_layer_widget_tomo._run_save()
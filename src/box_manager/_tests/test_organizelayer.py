import pytest
import napari
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="function")
def napari_viewer():
    yield napari.Viewer()

@pytest.fixture(scope="function")
def organize_layer_widget_tomo(napari_viewer):
    napari_viewer.open(plugin='napari-boxmanager',
                path='/mnt/data/twagner/Projects/TomoTwin/results/202208_YenT_step3/tomo_a15/d01t15_a15.mrc')
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
# """
# This module is an example of a barebones QWidget plugin for napari
#
# It implements the Widget specification.
# see: https://napari.org/plugins/guides.html?#widgets
#
# Replace code below according to your needs.
# """
from typing import TYPE_CHECKING

# from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from ._utils import filters

if TYPE_CHECKING:
    import napari
    import napari.layers


def filter_layer(
    napari_viewer: "napari.Viewer",
    napari_layer: "napari.layers.Image",
    lp_filter_resolution: float = 30,
    hp_filter_resolution: float = 1000,
    pixel_size: float = -1,
    show_mask: bool = False,
):

    if pixel_size == -1:
        try:
            pixel_size = napari_layer.metadata["pixel_spacing"]
        except KeyError:
            print("No pixel spacing set in metadata")
            return None

    filtered_image, mask = filters.bandpass_filter(
        napari_layer.data,
        lp_filter_resolution,
        hp_filter_resolution,
        pixel_size,
    )

    napari_layer.visible = False
    if show_mask:
        napari_viewer.add_image(
            mask,
            name=f"MASK LP {int(lp_filter_resolution)} HP {int(hp_filter_resolution)} - {napari_layer.name}",
        )

    napari_viewer.add_image(
        filtered_image,
        name=f"LP {int(lp_filter_resolution)} HP {int(hp_filter_resolution)} - {napari_layer.name}",
        metadata=napari_layer.metadata,
    )


# def example_magic_widget(img_layer: "napari.layers.Points"):
#    print(f"you have selected {img_layer}")
#
#
#
#
# class ExampleQWidget(QWidget):
#    # your QWidget.__init__ can optionally request the napari viewer instance
#    # in one of two ways:
#    # 1. use a parameter called `napari_viewer`, as done here
#    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
#    def __init__(self, napari_viewer):
#        super().__init__()
#        self.viewer = napari_viewer
#
#        btn = QPushButton("Click me!")
#        btn.clicked.connect(self._on_click)
#
#        self.setLayout(QHBoxLayout())
#        self.layout().addWidget(btn)
#
#    def _on_click(self):
#        print("napari has", len(self.viewer.layers), "layers")
#
#
# @magic_factory
# def example_magic_widget(img_layer: "napari.layers.Image"):
#    print(f"you have selected {img_layer}")
#
#
## Uses the `autogenerate: true` flag in the plugin manifest
## to indicate it should be wrapped as a magicgui to autogenerate
## a widget.
# def example_function_widget(img_layer: "napari.layers.Image"):
#    print(f"you have selected {img_layer}")

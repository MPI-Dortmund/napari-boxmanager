import os
import pathlib

import napari
import napari.layers
import numpy as np
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

ICON_DIR = f"{os.path.dirname(napari.__file__)}/resources/icons"


class AddLayerWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.napari_viewer = napari_viewer
        self.napari_viewer.events.theme.connect(self._apply_icons)

        self._add_point = QPushButton(self)
        self._add_shape = QPushButton(self)
        self._add_label = QPushButton(self)

        self._add_point.clicked.connect(self._new_points)
        self._add_shape.clicked.connect(self._new_shapes)
        self._add_label.clicked.connect(self._new_labels)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self._add_point)
        self.layout().addWidget(self._add_shape)
        self.layout().addWidget(self._add_label)

        self._apply_icons()

    def _apply_icons(self, *_):
        theme_dir = pathlib.Path(
            ICON_DIR, f"_themes/{self.napari_viewer.theme}"
        )

        point_icon = QIcon(os.path.join(theme_dir, "new_points.svg"))
        self._add_point.setIcon(point_icon)

        point_icon = QIcon(os.path.join(theme_dir, "new_shapes.svg"))
        self._add_shape.setIcon(point_icon)

        point_icon = QIcon(os.path.join(theme_dir, "new_labels.svg"))
        self._add_label.setIcon(point_icon)

    def _new_points(self):
        kwargs = {
            "edge_color": "red",
            "face_color": "transparent",
            "symbol": "disc",
            "edge_width": 2,
            "edge_width_is_relative": False,
            "size": 128,
            "out_of_slice_display": False,
            "opacity": 0.5,
        }
        layer = self.napari_viewer.add_points(
            ndim=max(self.napari_viewer.dims.ndim, 2),
            scale=self.napari_viewer.layers.extent.step,
            **kwargs,
        )
        layer.events.size()

    def _new_labels(self):
        layers_extent = self.napari_viewer.layers.extent
        extent = layers_extent.world
        scale = layers_extent.step
        scene_size = extent[1] - extent[0]
        corner = extent[0] + 0.5 * layers_extent.step
        shape = [
            np.round(s / sc).astype("int") if s > 0 else 1
            for s, sc in zip(scene_size, scale)
        ]
        empty_labels = np.zeros(shape, dtype=int)
        self.napari_viewer.add_labels(
            empty_labels, translate=np.array(corner), scale=scale
        )

    def _new_shapes(self):
        kwargs = {}
        self.napari_viewer.add_shapes(
            ndim=max(self.napari_viewer.dims.ndim, 2),
            scale=self.napari_viewer.layers.extent.step,
            **kwargs,
        )

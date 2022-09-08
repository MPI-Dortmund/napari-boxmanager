import os
import pathlib

import napari
import napari.layers
import numpy as np
from qtpy.QtCore import Slot
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

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

        self._layer = QComboBox(self)

        self.setLayout(QVBoxLayout())

        layout = QFormLayout()
        layout.addRow("Copy metadata from:", self._layer)
        self.layout().addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(self._add_point)
        layout.addWidget(self._add_shape)
        layout.addWidget(self._add_label)
        self.layout().addLayout(layout)

        self.layout().addStretch(True)

        self._apply_icons()

        self.napari_viewer.layers.events.inserted.connect(self._update_combo)
        self.napari_viewer.layers.events.removed.connect(self._update_combo)

        self._update_combo()

    @Slot(object)
    def _update_combo(self, *_):
        layer_names = sorted(
            entry.name
            for entry in self.napari_viewer.layers
            if isinstance(entry, napari.layers.Image)
        )
        current_text = self._layer.currentText()
        self._layer.clear()
        self._layer.addItems(layer_names)
        self._layer.setCurrentText(current_text)

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

    def _get_metadata(self):
        layer_name = self._layer.currentText()
        if not layer_name:
            return {}
        layer_meta = self.napari_viewer.layers[layer_name].metadata
        metadata = {}
        for key, value in layer_meta.items():
            if isinstance(key, int):
                metadata[key] = {}
                metadata[key][
                    "path"
                ] = f"{os.path.splitext(value['path'])[0]}.box"
                metadata[key][
                    "name"
                ] = f"{os.path.splitext(value['name'])[0]}.box"
            elif key in ("original_path",):
                metadata[key] = value
        return metadata

    def _new_points(self):
        metadata = self._get_metadata()
        kwargs = {
            "edge_color": "red",
            "face_color": "transparent",
            "symbol": "disc",
            "edge_width": 4,
            "edge_width_is_relative": False,
            "size": 128,
            "name": "coordinates",
            "out_of_slice_display": False,
            "opacity": 0.8,
            "metadata": metadata,
        }
        layer = self.napari_viewer.add_points(
            ndim=max(self.napari_viewer.dims.ndim, 2),
            scale=self.napari_viewer.layers.extent.step,
            **kwargs,
        )
        layer.events.size()

    def _new_labels(self):
        metadata = self._get_metadata()
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
            empty_labels,
            translate=np.array(corner),
            scale=scale,
            metadata=metadata,
        )

    def _new_shapes(self):
        metadata = self._get_metadata()
        kwargs = {
            "metadata": metadata,
        }
        self.napari_viewer.add_shapes(
            ndim=max(self.napari_viewer.dims.ndim, 2),
            scale=self.napari_viewer.layers.extent.step,
            **kwargs,
        )
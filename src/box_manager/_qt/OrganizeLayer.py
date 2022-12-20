import os
import pathlib

import napari
import napari.layers
import numpy as np
from napari.layers.shapes._shapes_constants import Mode
from qtpy.QtCore import Slot
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .._writer import napari_get_writer

ICON_DIR = f"{os.path.dirname(napari.__file__)}/resources/icons"


class OrganizeLayerWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.napari_viewer = napari_viewer
        self.napari_viewer.events.theme.connect(self._apply_icons)

        self.setLayout(QVBoxLayout())
        self.separators = []

        self._add_ui()
        self._add_seperator()
        self._save_ui()

        self._apply_icons()
        self.layout().addStretch(True)

    def _save_ui(self):
        self.napari_viewer.layers.selection.events.changed.connect(
            self._save_layer_changed
        )
        self.napari_viewer.layers.events.inserted.connect(
            self._update_layer_combo
        )
        self.napari_viewer.layers.events.removed.connect(
            self._update_layer_combo
        )

        inner_layout = QVBoxLayout()
        inner_layout.setContentsMargins(0, 0, 0, 0)
        self.layout().addLayout(inner_layout)

        self.save_run_btn = QPushButton("Save to dir", self)
        self.save_run_btn.clicked.connect(self._run_save)

        inner_layout.addWidget(QLabel("Write layer", self))
        self.save_layers = {
            "layer": QComboBox(self),
            "dimension": QComboBox(self),
            "type": QComboBox(self),
            "format": QComboBox(self),
            "suffix": QLineEdit(self),
        }
        self.save_layers["layer"].addItems([""])
        self.save_layers["dimension"].addItems(["", "2D", "3D"])
        self.save_layers["type"].addItems(["", "Particles", "Filaments"])
        self.save_layers["suffix"].setPlaceholderText("e.g., _suffix")

        self.formats = {
            "2D": {
                "Particles": [
                    ".cbox",
                    ".box",
                    ".star (Relion)",
                    # ".cs (cryoSPARC)",
                ],
                "Filaments": [".cbox", ".box (helicon)", ".star (Relion)"],
            },
            "3D": {
                "Particles": [".coords", ".tloc", ".cbox"],
                "Filaments": [".box", ".coords"],
            },
        }

        self.save_layers["dimension"].currentTextChanged.connect(
            self._update_format
        )
        self.save_layers["type"].currentTextChanged.connect(
            self._update_format
        )
        self.save_layers["layer"].currentTextChanged.connect(
            self._update_format
        )

        layout = QFormLayout()
        for name, widget in self.save_layers.items():
            layout.addRow(name.capitalize(), widget)
        inner_layout.addLayout(layout)
        inner_layout.addWidget(self.save_run_btn)

        self._update_layer_combo()

    def _add_seperator(self):
        _ = QWidget(self)
        _.setFixedHeight(10)
        self.layout().addWidget(_)

        line = QWidget(self)
        line.setFixedHeight(2)
        self.layout().addWidget(line)
        self.separators.append(line)

        _ = QWidget(self)
        _.setFixedHeight(10)
        self.layout().addWidget(_)

    def _add_ui(self):
        inner_layout = QVBoxLayout()
        inner_layout.setContentsMargins(0, 0, 0, 0)
        self.layout().addLayout(inner_layout)

        inner_layout.addWidget(QLabel("Add a new layer", self))

        self._add_point = QPushButton(self)
        self._add_shape = QPushButton(self)
        # self._add_label = QPushButton(self)

        self._add_point.clicked.connect(self._new_points)
        self._add_shape.clicked.connect(self._new_shapes)
        # self._add_label.clicked.connect(self._new_labels)

        self._layer = QComboBox(self)

        layout = QFormLayout()
        layout.addRow("Target image layer:", self._layer)
        layout.addRow("Create particle layer:", self._add_point)
        layout.addRow("Create filament layer:", self._add_shape)
        # layout.addRow("Create label layer:", self._add_label)
        inner_layout.addLayout(layout)

        self.napari_viewer.layers.events.inserted.connect(
            self._update_add_combo
        )
        self.napari_viewer.layers.events.removed.connect(
            self._update_add_combo
        )

        self._update_add_combo()

    @Slot()
    def _run_save(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Open directory",
            os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not dir_path:
            return

        cur_format = self.save_layers["format"].currentText().split(" ", 1)[0]
        cur_layer = self.napari_viewer.layers[
            self.save_layers["layer"].currentText()
        ]
        napari_get_writer(
            dir_path,
            [cur_layer.as_layer_data_tuple()],
            cur_format,
            self.save_layers["suffix"].text() + cur_format,
        )

    @Slot(object)
    def _update_layer_combo(self, *_):
        names = [
            entry.name
            for entry in self.napari_viewer.layers
            if isinstance(entry, (napari.layers.Shapes, napari.layers.Points))
        ]
        self.save_layers["layer"].clear()
        self.save_layers["layer"].addItems([""] + names)
        try:
            self.save_layers["layer"].setCurrentText(
                self.napari_viewer.layers.selection.active.name
            )
        except AttributeError:
            pass

    @Slot(object)
    def _save_layer_changed(self, _):
        if len(self.napari_viewer.layers.selection) != 1:
            self.save_layers["layer"].setCurrentIndex(0)
        else:
            current_layer = self.napari_viewer.layers.selection.active
            self.save_layers["layer"].setCurrentText(current_layer.name)

    @Slot(str)
    def _update_format(self, _=None):
        cur_layer = self.save_layers["layer"].currentText()

        if not cur_layer:
            self.save_run_btn.setEnabled(False)
            for name, widget in self.save_layers.items():
                if name != "layer":
                    widget.setEnabled(False)
            return

        for name, widget in self.save_layers.items():
            if name != "layer":
                widget.setEnabled(True)

        layer = self.napari_viewer.layers[cur_layer]
        ndim = layer.ndim
        try:
            is_2d_stack = layer.metadata["is_2d_stack"]
        except KeyError:
            is_2d_stack = False

        try:
            is_filament_layer = layer.metadata["is_filament_layer"]
        except KeyError:
            is_filament_layer = False

        if ndim == 2 or is_2d_stack:
            cur_dim = "2D"
        elif ndim == 3:
            cur_dim = "3D"
        else:
            cur_dim = ""
        prev_signal = self.save_layers["dimension"].blockSignals(True)
        self.save_layers["dimension"].setCurrentText(cur_dim)
        self.save_layers["dimension"].blockSignals(prev_signal)

        if is_filament_layer:
            cur_type = "Filaments"
        else:
            cur_type = "Particles"
        prev_signal = self.save_layers["type"].blockSignals(True)
        self.save_layers["type"].setCurrentText(cur_type)
        self.save_layers["type"].blockSignals(prev_signal)

        self.save_layers["format"].clear()
        try:
            self.save_layers["format"].addItems(
                self.formats[cur_dim][cur_type]
            )
        except KeyError:
            self.save_run_btn.setEnabled(False)
            self.save_layers["suffix"].setEnabled(False)
        else:
            self.save_run_btn.setEnabled(True)
            self.save_layers["suffix"].setEnabled(True)

    @Slot(object)
    def _update_add_combo(self, *_):
        layer_names = sorted(
            entry.name
            for entry in self.napari_viewer.layers
            if isinstance(entry, napari.layers.Image)
        )
        current_text = self._layer.currentText()
        self._layer.clear()
        self._layer.addItems(layer_names)
        self._layer.setCurrentText(current_text)

        enabled = bool(layer_names)
        self._add_point.setEnabled(enabled)
        self._add_shape.setEnabled(enabled)
        # self._add_label.setEnabled(enabled)

    def _apply_icons(self, *_):
        theme_dir = pathlib.Path(
            ICON_DIR, f"_themes/{self.napari_viewer.theme}"
        )

        point_icon = QIcon(os.path.join(theme_dir, "new_points.svg"))
        self._add_point.setIcon(point_icon)

        point_icon = QIcon(os.path.join(theme_dir, "new_shapes.svg"))
        self._add_shape.setIcon(point_icon)

        if self.napari_viewer.theme.lower() == "light":
            # argb
            color = "#7D000000"
        else:
            # argb
            color = "#7DFFFFFF"
        for separator in self.separators:
            separator.setStyleSheet(f"background-color: {color}")

        # point_icon = QIcon(os.path.join(theme_dir, "new_labels.svg"))
        # self._add_label.setIcon(point_icon)

    def _get_metadata(self):
        layer_name = self._layer.currentText()
        metadata = {
            "do_activate_on_insert": True,
        }
        if not layer_name:
            return metadata
        layer_meta = self.napari_viewer.layers[layer_name].metadata
        for key, value in layer_meta.items():
            if isinstance(key, int):
                metadata[key] = {}
                metadata[key][
                    "path"
                ] = f"{os.path.splitext(value['path'])[0]}.box"
                metadata[key][
                    "name"
                ] = f"{os.path.splitext(value['name'])[0]}.box"
                metadata[key]["image_name"] = value["name"]
                metadata[key]["real"] = False
                metadata[key]["write"] = None
            elif key in ("original_path", "is_2d_stack"):
                metadata[key] = value
        return metadata

    def _get_out_of_slice_display(self):
        layer_name = self._layer.currentText()
        layer_meta = self.napari_viewer.layers[layer_name].metadata
        if "is_3d" in layer_meta and "is_2d_stack" in layer_meta:
            if layer_meta["is_3d"] and not layer_meta["is_2d_stack"]:
                return True
            else:
                return False
        else:
            return False

    def _new_points(self):
        # if len(self.napari_viewer.layers) == 0:
        #    return
        metadata = self._get_metadata()
        kwargs = {
            "edge_color": "red",
            "face_color": "transparent",
            "symbol": "disc",
            "edge_width": 0.05,
            "edge_width_is_relative": True,
            "size": 128,
            "name": "coordinates",
            "out_of_slice_display": self._get_out_of_slice_display(),
            "opacity": 0.8,
            "metadata": metadata,
        }
        layer = self.napari_viewer.add_points(
            ndim=max(self.napari_viewer.dims.ndim, 2),
            scale=self.napari_viewer.layers.extent.step,
            **kwargs,
        )
        layer.events.size()
        layer.mode = "add"

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
        metadata["is_filament_layer"] = True
        kwargs = {
            "metadata": metadata,
            "face_color": "transparent",
            "edge_color": "red",
            "edge_width": 20,
            "opacity": 0.4,
            "name": "filaments",
            "shape_type": "path",
        }
        shape = self.napari_viewer.add_shapes(
            ndim=max(self.napari_viewer.dims.ndim, 2),
            scale=self.napari_viewer.layers.extent.step,
            **kwargs,
        )
        shape.mode = Mode.ADD_PATH

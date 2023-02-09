import os

import napari
import napari.layers
import numpy as np
from napari._qt.qt_resources._svg import QColoredSVGIcon
from napari.layers.shapes._shapes_constants import Mode
from napari.utils.notifications import show_error, show_info
from pathlib import Path
from qtpy.QtCore import Slot
from qtpy.QtGui import QIntValidator
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

        self._save_form_layout = QFormLayout()
        self._save_form_layout.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow
        )
        self._add_ui()
        self._add_seperator()
        self._save_ui()
        self._add_seperator()
        self._link_ui()

        self._apply_icons()
        self.layout().addStretch(True)

    def _link_ui(self):
        self.napari_viewer.layers.events.inserted.connect(
            self._update_link_combo
        )
        self.napari_viewer.layers.events.removed.connect(
            self._update_link_combo
        )

        self.link_layers = {"image": QComboBox(self), "layer": QComboBox(self)}

        self.link_run_btn = QPushButton("Link layers", self)
        self.link_run_btn.clicked.connect(self._link_layers)

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addRow("Target image layer:", self.link_layers["image"])
        layout.addRow("Target other layer:", self.link_layers["layer"])
        # layout.addRow("Create label layer:", self._add_label)

        inner_layout = QVBoxLayout()
        inner_layout.setContentsMargins(0, 0, 0, 0)
        self.layout().addLayout(inner_layout)

        inner_layout.addWidget(QLabel("Link layers", self))
        inner_layout.addLayout(layout)
        inner_layout.addWidget(self.link_run_btn)

    @Slot()
    def _link_layers(self):
        inner_layout = QVBoxLayout()
        inner_layout.setContentsMargins(0, 0, 0, 0)
        self.layout().addLayout(inner_layout)

        image_name = self.link_layers["image"].currentText()
        layer_name = self.link_layers["layer"].currentText()
        self.napari_viewer.layers[layer_name].metadata[
            "layer_name"
        ] = image_name
        show_info("link succesfull")

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
            "filament spacing": QLineEdit(self),
            "dummyf": QLabel(self),
            "suffix": QLineEdit(self),
        }
        self.save_layers["layer"].addItems([""])
        self.save_layers["dimension"].addItems(["", "2D", "3D"])
        self.save_layers["type"].addItems(["", "Particles", "Filaments"])
        self.save_layers["suffix"].setPlaceholderText("e.g., _suffix")
        self.save_layers["filament spacing"].setPlaceholderText(
            "spacing in pixel"
        )

        int_validator = QIntValidator()
        int_validator.setBottom(0)
        self.save_layers["filament spacing"].setValidator(int_validator)
        self.save_layers["filament spacing"].setText("0")
        self.save_layers["filament spacing"].setToolTip(
            "If 0, an overlap of 80% is used."
        )

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
                "Filaments": [".cbox", ".coords"],
            },
        }

        self.save_layers["dimension"].currentTextChanged.connect(
            self._update_format
        )
        self.save_layers["type"].currentTextChanged.connect(
            self._update_format
        )
        self.save_layers["layer"].currentTextChanged.connect(
            lambda x: self._update_format(x, is_layer=True)
        )

        max_length = max(len(x) for x in self.save_layers)
        for name, widget in self.save_layers.items():
            self._save_form_layout.addRow(
                f"<pre>{name.capitalize():<{max_length}s}</pre>", widget
            )
            # self._save_form_layout.addRow(f"<pre>{name.capitalize()}</pre>", widget)
        inner_layout.addLayout(self._save_form_layout)
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
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
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
        cur_format = self.save_layers["format"].currentText().split(" ", 1)[0]
        cur_layer = self.napari_viewer.layers[
            self.save_layers["layer"].currentText()
        ]

        cur_spacing = self.save_layers["filament spacing"].text()
        cur_type = self.save_layers["type"].currentText()
        if cur_type == "Filaments" and not cur_spacing:
            show_error("Filament format requires Filament spacing")
            return

        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Open directory",
            os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not dir_path:
            return

        napari_get_writer(
            dir_path,
            [cur_layer.as_layer_data_tuple()],
            cur_format,
            self.save_layers["suffix"].text() + cur_format,
            cur_spacing=int(cur_spacing),
        )

    @Slot(object)
    def _update_link_combo(self, *_):
        for entry in self.napari_viewer.layers:
            entry.events.name.disconnect(self._update_link_combo)
            entry.events.name.connect(self._update_link_combo)

        names_image = [
            entry.name
            for entry in self.napari_viewer.layers
            if isinstance(entry, napari.layers.Image)
        ]
        names_others = [
            entry.name
            for entry in self.napari_viewer.layers
            if not isinstance(entry, napari.layers.Image)
        ]
        self.link_layers["image"].clear()
        self.link_layers["image"].addItems(names_image)
        self.link_layers["layer"].clear()
        self.link_layers["layer"].addItems(names_others)

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
    def _update_format(self, _=None, is_layer=False):
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

        if not is_layer:
            cur_dim = self.save_layers["dimension"].currentText()
        elif ndim == 2 or is_2d_stack:
            cur_dim = "2D"
        elif ndim == 3:
            cur_dim = "3D"
        else:
            cur_dim = ""
        prev_signal = self.save_layers["dimension"].blockSignals(True)
        self.save_layers["dimension"].setCurrentText(cur_dim)
        self.save_layers["dimension"].blockSignals(prev_signal)

        if not is_layer:
            cur_type = self.save_layers["type"].currentText()
        elif is_filament_layer:
            cur_type = "Filaments"
        else:
            cur_type = "Particles"
        prev_signal = self.save_layers["type"].blockSignals(True)
        self.save_layers["type"].setCurrentText(cur_type)
        self.save_layers["type"].blockSignals(prev_signal)

        type_field_d = self.save_layers["dummyf"]
        type_label_d = self._save_form_layout.labelForField(type_field_d)
        type_field = self.save_layers["filament spacing"]
        type_label = self._save_form_layout.labelForField(type_field)
        if self.save_layers["type"].currentText() == "Filaments":
            type_field.show()
            type_label.show()
            type_field_d.hide()
            type_label_d.hide()
        else:
            type_field.hide()
            type_label.hide()
            type_field_d.show()
            type_label_d.hide()

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

    def get_theme_color(self):
        if self.napari_viewer.theme.lower() == "light":
            # argb
            color = "#7D000000"
        else:
            # argb
            color = "#7DFFFFFF"
        return color

    def _apply_icons(self, *_):
        point_icon = QColoredSVGIcon.from_resources("new_points").colored(
            theme=self.napari_viewer.theme
        )
        self._add_point.setIcon(point_icon)

        point_icon = QColoredSVGIcon.from_resources("new_shapes").colored(
            theme=self.napari_viewer.theme
        )
        self._add_shape.setIcon(point_icon)

        color = self.get_theme_color()

        for separator in self.separators:
            separator.setStyleSheet(f"background-color: {color}")

        # point_icon = QColoredSVGIcon.from_resources('new_labels').colored(theme=self.napari_viewer.theme)
        # self._add_label.setIcon(point_icon)

    def _get_metadata(self):
        layer_name = self._layer.currentText()
        metadata = {
            "do_activate_on_insert": True,
            "layer_name": layer_name,
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
            "name": f"coordinates ({Path(*Path(metadata['original_path']).parts[-2:])})",
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

            "name": f"filaments ({Path(*Path(metadata['original_path']).parts[-2:])})",
            "shape_type": "path",
        }
        shape = self.napari_viewer.add_shapes(
            ndim=max(self.napari_viewer.dims.ndim, 2),
            scale=self.napari_viewer.layers.extent.step,
            **kwargs,
        )
        shape.mode = Mode.ADD_PATH

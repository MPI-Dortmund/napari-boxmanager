import os
from pathlib import Path

import napari
import napari.layers
import numpy as np
from napari._qt.qt_resources._svg import QColoredSVGIcon
from napari.layers.shapes._shapes_constants import Mode
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import Slot
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .._utils import general
from .._writer import napari_get_writer

ICON_DIR = f"{os.path.dirname(napari.__file__)}/resources/icons"


class VerifyDialog(QDialog):
    def __init__(self, entries: list[tuple[str, str]]):
        super().__init__()

        self.setLayout(QVBoxLayout())

        widget = QWidget(self)
        widget.setMinimumSize(500, 500)

        widget.setLayout(QVBoxLayout())

        b1 = QPushButton("Accept", self)
        b1.clicked.connect(self.accept)
        b2 = QPushButton("Reject", self)
        b2.clicked.connect(self.reject)

        tmp_layout = QHBoxLayout()
        tmp_layout.setContentsMargins(0, 0, 0, 0)
        tmp_layout.addWidget(b1)
        tmp_layout.addWidget(b2)
        self.layout().addLayout(tmp_layout)

        area = QScrollArea(self)

        for l1, l2 in entries:
            tmp_layout = QHBoxLayout()
            tmp_layout.setContentsMargins(0, 0, 0, 0)
            tmp_layout.addWidget(QLabel(l1, self))
            tmp_layout.addWidget(QLabel(l2, self))
            widget.layout().addLayout(tmp_layout)

        widget.layout().addStretch(1)

        area.setWidget(widget)
        self.layout().addWidget(area)


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
        self._add_seperator()
        self._link_auto_ui()

        self._apply_icons()
        self.layout().addStretch(True)
        self.saved_dir_path = os.getcwd()

    def _link_auto_ui(self):
        inner_layout = QVBoxLayout()
        inner_layout.setContentsMargins(0, 0, 0, 0)

        self.link_auto_layers = {
            "prefix image": QLineEdit(self),
            "suffix image": QLineEdit(self),
            "prefix layer": QLineEdit(self),
            "suffix layer": QLineEdit(self),
        }
        self.link_run_auto_btn = QPushButton("Link", self)
        layout = QFormLayout()
        for name, widget in self.link_auto_layers.items():
            layout.addRow(name + "(optional)", widget)

        self.link_run_auto_btn.clicked.connect(self._link_auto_layers)

        inner_layout.addWidget(QLabel("Automatic layer linking", self))
        inner_layout.addLayout(layout)
        inner_layout.addWidget(self.link_run_auto_btn)
        self.layout().addLayout(inner_layout)

    @Slot()
    def _link_auto_layers(self):
        prefix_image = self.link_auto_layers["prefix image"].text()
        suffix_image = self.link_auto_layers["suffix image"].text()
        prefix_layer = self.link_auto_layers["prefix layer"].text()
        suffix_layer = self.link_auto_layers["suffix layer"].text()

        def get_unique(layer: napari.layers.Layer, prefix: str, suffix: str):
            return (
                os.path.splitext(
                    os.path.basename(layer.metadata["original_path"])
                )[0]
                .removeprefix(prefix)
                .removesuffix(suffix)
            )

        image_layers = {
            get_unique(_, prefix_image, suffix_image): _.name
            for _ in self.napari_viewer.layers
            if isinstance(_, napari.layers.Image)
        }
        layer_layers = {
            get_unique(_, prefix_layer, suffix_layer): _.name
            for _ in self.napari_viewer.layers
            if not isinstance(_, napari.layers.Image)
        }

        data = []
        for abbreviation, name in image_layers.items():
            try:
                data.append((name, layer_layers[abbreviation]))
            except KeyError:
                pass
        a = VerifyDialog(data)
        result = a.exec()
        if result == QDialog.Accepted:
            for image_name, layer_name in data:
                self._link_layers(image_name, layer_name)

    def _link_ui(self):
        self.napari_viewer.layers.events.inserted.connect(
            self._update_link_combo
        )
        self.napari_viewer.layers.events.removed.connect(
            self._update_link_combo
        )

        self.link_layers = {"image": QComboBox(self), "layer": QComboBox(self)}

        self.link_run_btn = QPushButton("Link", self)
        self.link_run_btn.clicked.connect(self._link_layers)

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addRow("Target image layer:", self.link_layers["image"])
        layout.addRow("Target other layer:", self.link_layers["layer"])
        # layout.addRow("Create label layer:", self._add_label)

        inner_layout = QVBoxLayout()
        inner_layout.setContentsMargins(0, 0, 0, 0)
        self.layout().addLayout(inner_layout)

        inner_layout.addWidget(QLabel("Manual layer Linking", self))
        inner_layout.addLayout(layout)
        inner_layout.addWidget(self.link_run_btn)

        self._update_link_combo()

    @Slot()
    def _link_layers(self, image_name=None, layer_name=None):

        if image_name is None:
            image_name = self.link_layers["image"].currentText()
        if layer_name is None:
            layer_name = self.link_layers["layer"].currentText()
        image_id = general.get_layer_id(
            self.napari_viewer, self.napari_viewer.layers[image_name]
        )
        self.napari_viewer.layers[layer_name].metadata.setdefault(
            "linked_image_layers", []
        ).append(image_id)

        for layer in self.napari_viewer.layers:
            layer.visible = False
        self.napari_viewer.layers[image_name].visible = True
        self.napari_viewer.layers[layer_name].visible = True

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
            "inter-box distance": QLineEdit(self),
            "dummyf": QLabel(self),
            "suffix": QLineEdit(self),
        }
        self.save_layers["layer"].addItems(["Selected"])
        self.save_layers["dimension"].addItems(["", "2D", "3D"])
        self.save_layers["type"].addItems(["", "Particles", "Filaments"])
        self.save_layers["suffix"].setPlaceholderText("e.g., _suffix")
        self.save_layers["inter-box distance"].setPlaceholderText(
            "spacing in pixel"
        )

        int_validator = QIntValidator()
        int_validator.setBottom(0)
        self.save_layers["inter-box distance"].setValidator(int_validator)
        self.save_layers["inter-box distance"].setText("0")
        self.save_layers["inter-box distance"].setToolTip(
            "Inter-box distance in pixel. If 0, an overlap of 80% of box-size is used."
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

        self._add_point = QPushButton("Create particle layer", self)
        self._add_shape = QPushButton("Create filament layer", self)
        # self._add_label = QPushButton(self)

        self._add_point.clicked.connect(self._new_points)
        self._add_shape.clicked.connect(self._new_shapes)
        # self._add_label.clicked.connect(self._new_labels)

        self._layer = QComboBox(self)

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addRow("Target image layer:", self._layer)
        layout.addRow(self._add_point)
        layout.addRow(self._add_shape)
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
    def _run_save(self,path=None):


        cur_spacing = self.save_layers["inter-box distance"].text()
        cur_type = self.save_layers["type"].currentText()
        if cur_type == "Filaments" and not cur_spacing:
            show_error("Filament format requires inter-box distance")
            return


        self.saved_dir_path = QFileDialog.getExistingDirectory(
            self,
            "Open directory",
            self.saved_dir_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )

        if not self.saved_dir_path:
            self.saved_dir_path = os.getcwd()
            return


        cur_format = self.save_layers["format"].currentText().split(" ", 1)[0]
        sel_layer = self.save_layers["layer"].currentText()
        layers_to_write = []
        if sel_layer == "Selected":
            target_type = napari.layers.Points
            if cur_type == "Filaments":
                target_type = napari.layers.Shapes
            relevant_layers = [x for x in self.napari_viewer.layers.selection if isinstance(x, target_type)]
            if len(relevant_layers)==0:
                show_error("No coordinate layers selected")
                return
            for cur_layer in relevant_layers:
                if len(cur_layer.data) == 0:
                    continue
                layers_to_write.append(cur_layer)
        else:
            cur_layer = self.napari_viewer.layers[sel_layer]
            layers_to_write.append(cur_layer)

        for cur_layer in layers_to_write:
            napari_get_writer(
                self.saved_dir_path,
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
        self.save_layers["layer"].addItems(["Selected"] + names)
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

        if cur_layer == "Selected":
            return

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
        type_field = self.save_layers["inter-box distance"]
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
            "linked_image_layers": [
                general.get_layer_id(
                    self.napari_viewer, self.napari_viewer.layers[layer_name]
                )
            ],
            "skip_match": None,
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

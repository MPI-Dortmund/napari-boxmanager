import os
import typing
from copy import deepcopy

import napari.layers
import numpy as np
from napari.layers.base.base import Layer
from napari.utils.notifications import show_error
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .._utils import general

if typing.TYPE_CHECKING:
    import napari


class PrefixSuffixCount(QWidget):
    editingFinished = Signal()

    def __init__(self, name: str, parent=None):
        super().__init__(parent)

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.setLayout(layout)
        self.dirname = ""

        self.combo = QComboBox(self)
        self.prefix_edit = QLineEdit(self)
        self.prefix_edit.textEdited.connect(
            lambda x: self.editingFinished.emit()
        )
        self.suffix_edit = QLineEdit(self)
        self.suffix_edit.textEdited.connect(
            lambda x: self.editingFinished.emit()
        )

        self.layout().addRow(f"{name.capitalize()} layer", self.combo)
        self.layout().addRow(f"{name.capitalize()} prefix", self.prefix_edit)
        self.layout().addRow(f"{name.capitalize()} suffix", self.suffix_edit)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.register_combo_functions()

    def register_combo_functions(self):
        funcs = [
            "currentTextChanged",
            "currentText",
            "itemText",
            "count",
            "clear",
            "addItems",
            "setCurrentText",
            "currentIndex",
        ]
        for func in funcs:
            setattr(self, func, getattr(self.combo, func))

    @property
    def prefix(self):
        return self.prefix_edit.text()

    @prefix.setter
    def prefix(self, value):
        self.prefix_edit.setText(value)

    @property
    def suffix(self):
        return self.suffix_edit.text()

    @suffix.setter
    def suffix(self, value):
        self.suffix_edit.setText(value)


class OrganizeBoxWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.napari_viewer = napari_viewer
        self.loadable_layers = (napari.layers.Points, napari.layers.Shapes)

        self.napari_viewer.layers.events.inserted.connect(self._update_combo)
        self.napari_viewer.layers.events.removed.connect(self._update_combo)

        self.image_layer = PrefixSuffixCount("image", self)
        self.coord_layer = PrefixSuffixCount("coords", self)

        self.image_layer.currentTextChanged.connect(self._update_combo)
        self.image_layer.editingFinished.connect(
            lambda: self._update_table(create_layer=False)
        )
        self.coord_layer.currentTextChanged.connect(self._update_combo)
        self.coord_layer.editingFinished.connect(
            lambda: self._update_table(create_layer=False)
        )

        self.table_widget = QTableWidget(self)
        button = QPushButton("Create layer")
        button.clicked.connect(
            lambda *_: self._update_table(create_layer=True)
        )

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.image_layer)
        self.layout().addWidget(self.coord_layer)
        self.layout().addWidget(button)
        self.layout().addWidget(self.table_widget)

        self._update_combo()

        if self.image_layer.count() == 1 and self.coord_layer.count() >= 1:
            layers = []
            for idx in range(self.coord_layer.count()):
                layers.append(self.coord_layer.itemText(idx))
            for layer in layers:
                self.coord_layer.setCurrentText(layer)
                self._update_table()

    @Slot(object)
    @Slot(str)
    def _update_combo(self, *_):

        for widget, valid_types in (
            (self.coord_layer, self.loadable_layers),
            (self.image_layer, napari.layers.Image),
        ):
            valid_layers: list[napari.layers.Layer] = sorted(
                (
                    entry
                    for entry in self.napari_viewer.layers
                    if isinstance(entry, valid_types)
                    and entry.ndim == 3
                    and "is_2d_stack" in entry.metadata
                    and entry.metadata["is_2d_stack"]
                    and "original_path" in entry.metadata
                ),
                key=lambda x: x.name,
            )

            current_text = [widget.itemText(i) for i in range(widget.count())]

            if valid_layers != current_text:
                prev_text = widget.currentText()
                widget.currentTextChanged.disconnect(self._update_combo)
                widget.clear()
                widget.addItems([entry.name for entry in valid_layers])
                widget.setCurrentText(prev_text)
                widget.currentTextChanged.connect(self._update_combo)

            if widget.currentIndex() == -1:
                return

            layer = valid_layers[widget.currentIndex()]
            layer_metadata = layer.metadata
            if (
                "original_path" in layer_metadata
                and "*" in layer_metadata["original_path"]
            ):
                dirname = os.path.dirname(layer_metadata["original_path"])
                prefix, suffix = os.path.basename(
                    layer_metadata["original_path"]
                ).split("*")
            elif "original_path" in layer_metadata:
                dirname = os.path.dirname(layer_metadata["original_path"])
                _, suffix = os.path.splitext(
                    os.path.basename(layer_metadata["original_path"])
                )
                prefix = ""
            else:
                dirname = os.path.dirname(layer.name)
                _, suffix = os.path.splitext(os.path.basename(layer.name))
                prefix = ""

            if "prefix" in layer_metadata:
                prefix = layer_metadata["prefix"]
            if "suffix" in layer_metadata:
                suffix = layer_metadata["suffix"]

            widget.prefix = prefix
            widget.suffix = suffix
            widget.dirname = dirname

        self._update_table(create_layer=False)

    def _update_table(self, create_layer=True):
        self.table_widget.clear()
        if not self.image_layer.currentText():
            return
        if not self.coord_layer.currentText():
            return
        layer_image = self.napari_viewer.layers[self.image_layer.currentText()]
        prefix_image = self.image_layer.prefix
        suffix_image = self.image_layer.suffix
        image_dict = {
            (
                os.path.basename(value["path"])
                .removesuffix(suffix_image)
                .removeprefix(prefix_image)
            ): idx
            for idx, value in layer_image.metadata.items()
            if isinstance(idx, int) and "path" in value
        }

        layer_coord = self.napari_viewer.layers[self.coord_layer.currentText()]
        prefix_coord = self.coord_layer.prefix
        suffix_coord = self.coord_layer.suffix
        coord_dict = {
            (
                os.path.basename(value["path"])
                .removesuffix(suffix_coord)
                .removeprefix(prefix_coord)
            ): idx
            for idx, value in layer_coord.metadata.items()
            if isinstance(idx, int) and "path" in value
        }

        old_data, old_state, old_type_str = layer_coord.as_layer_data_tuple()
        new_data = deepcopy(old_data)
        ident_data = general.get_identifier(layer_coord, 0)

        new_meta = {
            key: value
            for key, value in layer_coord.metadata.items()
            if not isinstance(key, int)
        }
        new_meta["prefix"] = prefix_coord
        new_meta["suffix"] = suffix_coord
        new_meta["do_activate_on_insert"] = True

        total_mask = np.zeros(len(ident_data), dtype=bool)
        table_list = []
        for key, image_idx in image_dict.items():
            mic_name = layer_image.metadata[image_idx]["name"]
            try:
                box_name = layer_coord.metadata[image_idx]["name"]
            except KeyError:
                box_name = "-"

            try:
                coord_idx = coord_dict[key]
            except KeyError:
                name = f"{prefix_coord}{key}{suffix_coord}"
                new_meta[image_idx] = {
                    "path": os.path.join(self.coord_layer.dirname, name),
                    "name": name,
                    "image_name": mic_name,
                    "write": None,
                    "real": False,
                }
                new_coord_name = "-"
                table_list.append((mic_name, box_name, new_coord_name))
                continue
            else:
                new_meta[image_idx] = layer_coord.metadata[coord_idx]
                new_meta[image_idx]["image_name"] = mic_name
                try:
                    new_meta[image_idx]["real"]
                except KeyError:
                    new_meta[image_idx]["real"] = True
                if new_meta[image_idx]["real"]:
                    new_coord_name = new_meta[image_idx]["name"]
                else:
                    new_coord_name = "-"
            table_list.append((mic_name, box_name, new_coord_name))

            slice_mask = np.round(ident_data, 0) == coord_idx
            total_mask = total_mask | slice_mask
            new_data[slice_mask, 0] = image_idx

        try:
            new_data = new_data[total_mask, :]
        except TypeError:
            new_data = [
                entry for entry, keep in zip(new_data, total_mask) if keep
            ]

        self.table_widget.setColumnCount(3)
        self.table_widget.setRowCount(len(table_list))

        self.table_widget.setHorizontalHeaderLabels(
            [
                layer_image.name,
                layer_coord.name,
                f"{layer_coord.name} (matched)",
            ]
        )
        for row_idx, row_entries in enumerate(table_list):
            for col_idx, value in enumerate(row_entries):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemIsSelectable)
                self.table_widget.setItem(row_idx, col_idx, item)
        self.table_widget.resizeColumnsToContents()

        if create_layer and len(new_data) != 0:
            new_state = {}
            new_meta["matched"] = True
            new_meta["layer_name"] = layer_image.name
            for key, value in old_state.items():
                if key == "visible":
                    new_state[key] = True
                elif key == "name":
                    new_state[key] = f"{layer_coord.name} (matched)"
                elif key == "metadata":
                    new_state[key] = new_meta
                elif key in (
                    "edge_width",
                    "face_color",
                    "edge_color",
                    "size",
                    "features",
                    "shown",
                ):
                    try:
                        new_state[key] = value[total_mask]
                    except TypeError:
                        new_state[key] = value
                else:
                    new_state[key] = value

            layer_coord.visible = False
            new = Layer.create(new_data, new_state, old_type_str)
            self.napari_viewer.layers.insert(
                self.napari_viewer.layers.index(layer_coord) + 1, new
            )
            self.coord_layer.setCurrentText(new.name)
            self.coord_layer.currentTextChanged.emit(new.name)
        elif create_layer:
            show_error("No matching entries for match_mics plugin")


def get_metadata(path: os.PathLike | list[os.PathLike]):
    return {"original_path": path[0] if isinstance(path, list) else path}

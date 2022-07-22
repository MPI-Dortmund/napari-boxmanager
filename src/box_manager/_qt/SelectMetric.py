import enum
import typing

import napari.layers
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvas
from qtpy.QtCore import QModelIndex, QRegularExpression, Qt, Signal, Slot
from qtpy.QtGui import (
    QRegularExpressionValidator,
    QStandardItem,
    QStandardItemModel,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

if typing.TYPE_CHECKING:
    import napari


def _get_min_floor(vals, step=1000):
    return np.round(np.floor(np.min(vals) * step) / step, 3)


def _get_max_floor(vals, step=1000):
    return np.round(np.ceil(np.max(vals) * step) / step, 3)


class ButtonActions(enum.Enum):
    ADD = 0
    DEL = 1
    UPDATE = 2


class GroupModel(QStandardItemModel):
    def __init__(self, read_only, parent=None):
        super().__init__(parent)
        self.read_only = read_only
        self.group_items = {}
        self.label_dict = {}
        self.label_dict_rev = {}
        self.default_labels = [""]
        self._update_labels(self.default_labels)

    def update_model(self, rows_candidates, value, col_idx):
        parents = {entry[1] for entry in rows_candidates if entry[0] == -1}

        layer_dict = {}
        for parent_idx, row_idx in rows_candidates:
            if parent_idx in parents:
                continue
            layer_dict.setdefault(parent_idx, []).append(row_idx)

        self.blockSignals(True)
        for parent_idx, rows_idx in layer_dict.items():
            if parent_idx == -1:
                parent_item = self.invisibleRootItem()
                change_children = True
            else:
                parent_item = self.item(parent_idx, 0)
                change_children = False

            for row_idx in rows_idx:
                child_item = parent_item.child(row_idx, col_idx)
                child_item.setText(str(value))
                if change_children:
                    self.change_children(row_idx, col_idx)
        self.blockSignals(False)
        self.layoutChanged.emit()
        return layer_dict

    def change_children(self, row, column):

        if self.label_dict_rev[column] in self.read_only:
            return

        root_item = self.invisibleRootItem().child(row, 0)
        value = self.invisibleRootItem().child(row, column).text()
        prev_signal = self.blockSignals(True)
        for grandchild_idx in range(root_item.rowCount()):
            grandchild_item = root_item.child(grandchild_idx, column)
            grandchild_item.setText(str(value))
        self.blockSignals(prev_signal)

    def _update_labels(self, columns):
        self.label_dict = {}
        self.label_dict_rev = {}
        i_label = -1
        for i_label in range(self.columnCount()):
            label = self.horizontalHeaderItem(i_label).text()
            self.label_dict[label] = i_label
            self.label_dict_rev[i_label] = label

        for i_label, new_label in enumerate(columns, i_label + 1):
            if new_label not in self.label_dict:
                self.label_dict[new_label] = i_label
                self.label_dict_rev[i_label] = new_label

        self.setHorizontalHeaderLabels(self.label_dict)

    def sort(self):
        self.invisibleRootItem().sortChildren(self.label_dict["name"])

    def set_values(self, parent_idx, rows_idx, col_name, value):
        for row in rows_idx:
            self.set_value(parent_idx, row, col_name, value)

    def set_value(self, parent_idx, row_idx, col_name, value):
        root_element = self.invisibleRootItem()
        if parent_idx == -1:
            child_item = root_element
        else:
            child_item = root_element.child(parent_idx, 0)
        child_item.child(row_idx, self.label_dict[col_name]).setText(
            str(value)
        )

    def get_values(self, parent_idx, rows_idx, col_name):
        return [self.get_value(parent_idx, row, col_name) for row in rows_idx]

    def get_value(self, parent_idx, row_idx, col_name):
        root_element = self.invisibleRootItem()
        if parent_idx == -1:
            child_item = root_element
        else:
            child_item = root_element.child(parent_idx, 0)
        return child_item.child(row_idx, self.label_dict[col_name]).text()

    def add_group(self, group_name, columns: dict) -> bool:
        if group_name in self.group_items:
            return False

        self._update_labels(columns.keys())

        item_root = QStandardItem()
        root_element = self.invisibleRootItem()
        row_idx = root_element.rowCount()

        self.append_to_row(
            root_element, columns, row_idx, first_item=item_root
        )
        self.group_items[group_name] = item_root
        return True

    def remove_group(self, group_name):
        if group_name not in self.group_items:
            return

        index = self.indexFromItem(self.group_items[group_name])
        self.takeRow(index.row())
        del self.group_items[group_name]

    def append_to_row(self, root_element, columns, row_idx, first_item=None):
        for col_idx in range(self.columnCount()):
            cur_label = self.horizontalHeaderItem(col_idx).text()
            if col_idx == 0:
                col_item = first_item or QStandardItem()
                col_item.setEditable(False)
            else:
                col_item = QStandardItem(
                    columns[cur_label] if cur_label in columns else "-"
                )
                col_item.setEditable(cur_label not in self.read_only)
            root_element.setChild(row_idx, col_idx, col_item)

    def append_element_to_group(self, group_name, columns):
        group_item = self.group_items[group_name]
        row_idx = group_item.rowCount()
        item_icon = QStandardItem()
        item_icon.setEditable(False)
        group_item.setChild(row_idx, 0, item_icon)

        self.append_to_row(group_item, columns, row_idx)


class GroupDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._plus_icon = "\U0000002B"  # QIcon("plus.png")
        self._minus_icon = "\U00002212"  # QIcon("minus.png")

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if not index.parent().isValid():
            is_open = bool(option.state & QStyle.State_Open)
            option.features |= QStyleOptionViewItem.HasDecoration
            option.text = self._minus_icon if is_open else self._plus_icon


class GroupView(QTreeView):
    elementsUpdated = Signal(dict, str)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setIndentation(0)
        self.setExpandsOnDoubleClick(False)
        self.clicked.connect(self.on_clicked)
        delegate = GroupDelegate(self)
        self.setItemDelegateForColumn(0, delegate)
        self.model = model
        self.setModel(self.model)
        self.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    @Slot(QModelIndex)
    def on_clicked(self, index):
        if not index.parent().isValid() and index.column() == 0:
            self.setExpanded(index, not self.isExpanded(index))

    @Slot(float, int)
    def update_elements(self, value, col_idx):
        rows_candidates = {
            (entry.parent().row(), entry.row())
            for entry in self.selectedIndexes()
        }
        if not rows_candidates:
            return

        update_dict = self.model.update_model(rows_candidates, value, col_idx)
        layer_dict = {}
        for parent_idx, rows_idx in update_dict.items():
            if parent_idx == -1:
                for row in rows_idx:
                    root_item = self.model.invisibleRootItem().child(row, 0)
                    rows = list(range(root_item.rowCount()))
                    layer_dict[row] = rows
            else:
                layer_dict[parent_idx] = rows_idx

        col_name = self.model.label_dict_rev[col_idx]
        self.elementsUpdated.emit(layer_dict, col_name)


class SelectMetricWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.napari_viewer = napari_viewer
        self.metrics: dict[str, typing.Any] = {}
        self.metric_dict: dict = {}
        self.prev_points: list[str] = []

        self.read_only = [
            "",
            "identifier",
            "shown",
            "n_selected",
            "n_boxes",
            "name",
            "slice",
        ]
        self.ignore_idx = [
            "boxsize",
        ] + self.read_only

        self.napari_viewer.layers.events.inserted.connect(self.reset_choices)
        self.napari_viewer.layers.events.removed.connect(self.reset_choices)

        self.layer_input = QComboBox(self)
        self.reset_choices(None)

        self.table_model = GroupModel(self.read_only, self)
        self.table_widget = GroupView(self.table_model, self)
        self.table_widget.elementsUpdated.connect(self._update_view)
        self.metric_area = QVBoxLayout()

        layout_input = QHBoxLayout()
        btn_add = QPushButton("Add", self)
        btn_add.clicked.connect(
            lambda _, name=ButtonActions.ADD: self._add_remove_table(name)
        )
        btn_del = QPushButton("Del", self)
        btn_del.clicked.connect(
            lambda _, name=ButtonActions.DEL: self._add_remove_table(name)
        )
        btn_update = QPushButton("Update", self)
        btn_update.clicked.connect(
            lambda _, name=ButtonActions.UPDATE: self._add_remove_table(name)
        )
        layout_input.addWidget(self.layer_input, stretch=1)
        layout_input.addWidget(btn_add)
        layout_input.addWidget(btn_del)
        layout_input.addWidget(btn_update)

        self.setLayout(QVBoxLayout())
        self.layout().addLayout(layout_input, stretch=0)  # type: ignore
        self.layout().addWidget(self.table_widget, stretch=1)
        self.layout().addLayout(self.metric_area, stretch=0)  # type: ignore

    @Slot(dict, str)
    def _update_view(self, layer_dict, col_name):
        metric_name, is_min_max = self.trim_suffix(col_name)

        for parent_idx, rows_idx in layer_dict.items():
            layer_name = self.table_model.get_value(-1, parent_idx, "name")
            layer = self.napari_viewer.layers[layer_name]  # type: ignore
            do_update = not layer.visible

            slice_idx = list(
                map(
                    int,
                    self.table_model.get_values(parent_idx, rows_idx, "slice"),
                )
            )
            layer_vals = float(
                self.table_model.get_value(parent_idx, rows_idx[0], col_name)
            )

            if layer.data.shape[1] == 3:
                mask_dimension = np.isin(layer.data[:, 0], slice_idx)
            elif layer.data.shape[1] == 2:
                mask_dimension = np.ones(layer.data.shape[0], dtype=bool)
            else:
                assert False, layer

            if is_min_max:
                old_shown = layer.shown.copy()
                mask_metric = np.ones(mask_dimension.shape)
                for metric_name in layer.features.columns:
                    if metric_name in self.ignore_idx:
                        continue
                    min_val = min(
                        map(
                            float,
                            self.table_model.get_values(
                                parent_idx, rows_idx, f"{metric_name}_min"
                            ),
                        )
                    )
                    max_val = max(
                        map(
                            float,
                            self.table_model.get_values(
                                parent_idx, rows_idx, f"{metric_name}_max"
                            ),
                        )
                    )
                    mask_metric = (
                        mask_metric
                        & (min_val <= layer.features[metric_name])
                        & (layer.features[metric_name] <= max_val)
                    )
                layer.shown[mask_dimension & mask_metric] = True
                layer.shown[mask_dimension & ~mask_metric] = False

                if not np.array_equal(old_shown, layer.shown):
                    for idx, row in enumerate(rows_idx):
                        if layer.data.shape[1] == 3:
                            slice_mask = layer.data[:, 0] == slice_idx[idx]
                        elif layer.data.shape[1] == 2:
                            slice_mask = np.ones(
                                layer.data.shape[0], dtype=bool
                            )
                        else:
                            assert False, layer
                        self.table_model.set_value(
                            parent_idx,
                            row,
                            "n_selected",
                            np.count_nonzero(
                                mask_dimension & mask_metric & slice_mask
                            ),
                        )
                        self.table_model.set_value(
                            parent_idx,
                            row,
                            "n_boxes",
                            np.count_nonzero(mask_dimension & slice_mask),
                        )

                    self.table_model.set_value(
                        -1,
                        parent_idx,
                        "n_selected",
                        np.count_nonzero(layer.shown),
                    )
                    self.table_model.set_value(
                        -1, parent_idx, "n_boxes", len(layer.shown)
                    )
                    do_update = True
            elif metric_name == "boxsize":
                do_update = True
                layer.size[mask_dimension] = layer_vals
            else:
                assert False

            if do_update:
                layer.visible = True

    def _update_view_old(self, top_idx, _, idx):
        if not idx:
            return

        parent_idx = top_idx.parent().row()
        if parent_idx == -1:
            return

        row_idx = top_idx.row()
        col_idx = top_idx.column()

        col_name = self.table_model.label_dict_rev[col_idx]
        if col_name in self.read_only:
            return

        metric_name, is_min_max = self.trim_suffix(col_name)
        layer_name = self.table_model.get_value(-1, parent_idx, "name")
        layer = self.napari_viewer.layers[layer_name]  # type: ignore

        layer_val = float(
            self.table_model.get_value(parent_idx, row_idx, col_name)
        )
        slice_idx = int(
            self.table_model.get_value(parent_idx, row_idx, "slice")
        )

        if layer.data.shape[1] == 3:
            mask_dimension = layer.data[:, 0] == slice_idx
        elif layer.data.shape[1] == 2:
            mask_dimension = np.ones(layer.data.shape[0])
        else:
            assert False, layer

        if is_min_max:
            mask_metric = np.ones(mask_dimension.shape)
            for metric_name in layer.features.columns:
                if metric_name in self.ignore_idx:
                    continue
                min_val = float(
                    self.table_model.get_value(
                        parent_idx, row_idx, f"{metric_name}_min"
                    )
                )
                max_val = float(
                    self.table_model.get_value(
                        parent_idx, row_idx, f"{metric_name}_max"
                    )
                )
                mask_metric = (
                    mask_metric
                    & (min_val <= layer.features[metric_name])
                    & (layer.features[metric_name] <= max_val)
                )

            layer.shown[mask_dimension & mask_metric] = True
            layer.shown[mask_dimension & ~mask_metric] = False

            layer.metadata[slice_idx][col_name] = layer_val
            self.table_model.set_value(
                parent_idx,
                row_idx,
                "n_selected",
                np.count_nonzero(mask_dimension & mask_metric),
            )
            self.table_model.set_value(
                parent_idx,
                row_idx,
                "n_boxes",
                np.count_nonzero(mask_dimension),
            )

            self.table_model.set_value(
                -1, parent_idx, "n_selected", np.count_nonzero(layer.shown)
            )
            self.table_model.set_value(
                -1, parent_idx, "n_boxes", len(layer.shown)
            )
        elif metric_name == "boxsize":
            layer.size[mask_dimension] = layer_val
        else:
            assert False, (layer_name, metric_name)

        layer.visible = layer.visible

    @staticmethod
    def trim_suffix(label_name):
        if label_name.endswith("_min"):
            metric_name = label_name.removesuffix("_min")
            min_max = True
        elif label_name.endswith("_max"):
            metric_name = label_name.removesuffix("_max")
            min_max = True
        else:
            metric_name = label_name
            min_max = False
        return metric_name, min_max

    def reset_choices(self, _):
        point_layers: list[str] = sorted(
            entry.name
            for entry in self.napari_viewer.layers
            if isinstance(entry, napari.layers.Points)
        )

        if point_layers != self.prev_points:
            current_item = self.layer_input.currentText()
            self.layer_input.clear()
            self.layer_input.addItems(point_layers)
            self.layer_input.setCurrentText(current_item)
            self.prev_points = point_layers

    def _prepare_entries(self, layer, name=None) -> list:
        output_list = []
        features_copy = layer.features.copy()
        if layer.data.shape[1] == 3:
            features_copy["identifier"] = name or layer.data[:, 0].astype(int)
        elif layer.data.shape[1] == 2:
            features_copy["identifier"] = name or 0
        else:
            assert False, layer.data

        features_copy["shown"] = layer.shown
        for identifier, ident_df in features_copy.groupby(
            "identifier", sort=False
        ):
            cur_name = name or layer.metadata[identifier]["name"]
            output_list.append(
                self._prepare_columns(ident_df, cur_name, identifier)
            )

        return output_list

    def _prepare_columns(self, features, name, slice_idx) -> dict:
        output_dict = {}
        output_dict["name"] = name
        output_dict["slice"] = str(slice_idx)
        output_dict["n_boxes"] = str(len(features))
        output_dict["n_selected"] = str(np.count_nonzero(features["shown"]))
        output_dict["boxsize"] = (
            10
            if "boxsize" not in features
            else str(int(np.mean(features["boxsize"])))
        )
        for col_name in features.columns:
            if col_name in self.ignore_idx:
                continue

            output_dict[f"{col_name}_min"] = str(
                _get_min_floor(features[col_name])
            )
            output_dict[f"{col_name}_max"] = str(
                _get_max_floor(features[col_name])
            )
        return output_dict

    def _add_remove_table(self, action: "ButtonActions", update=True):
        layer_name = self.layer_input.currentText()
        if not layer_name.strip():
            return None

        if action == ButtonActions.ADD:
            layer = self.napari_viewer.layers[layer_name]  # type: ignore
            if self.table_model.add_group(
                layer_name, self._prepare_entries(layer, layer_name)[0]
            ):
                entries = self._prepare_entries(layer)
                for entry in entries:
                    self.table_model.append_element_to_group(layer_name, entry)
        elif action == ButtonActions.DEL:
            self.table_model.remove_group(layer_name)
        elif action == ButtonActions.UPDATE:
            self._add_remove_table(ButtonActions.DEL, update=False)
            self._add_remove_table(ButtonActions.ADD, update=False)

        if update:
            self.table_model.sort()
            self._update_slider()

    def _get_all_data(self, metric_name):

        layer_names = self.table_model.group_items
        layer_features = []
        for layer_name in layer_names:
            layer_features.append(
                self.napari_viewer.layers[layer_name].features[metric_name]
            )
        return pd.concat(layer_features, ignore_index=True)

    def _update_slider(self):
        for col_idx, label in enumerate(self.table_model.label_dict):
            if label in self.read_only:
                continue

            metric_name, _ = self.trim_suffix(label)
            labels_data = self._get_all_data(metric_name)

            if label in ("boxsize",):
                if metric_name in self.metric_dict:
                    viewer = self.metric_dict[metric_name]
                else:
                    layout = QHBoxLayout()
                    layout.addWidget(QLabel(label))
                    viewer = EditView(
                        col_idx,
                        QRegularExpressionValidator(
                            QRegularExpression("[0-9]*")
                        ),
                    )
                    layout.addWidget(viewer)
                    self.metric_area.addLayout(layout)
                    viewer.value_changed.connect(self.table_widget.update_elements)  # type: ignore
                    self.metric_dict[label] = viewer
                viewer.set_value(int(labels_data.mean()))
                continue

            if metric_name in self.metric_dict:
                viewer = self.metric_dict[metric_name]
            else:
                viewer = HistogramMinMaxView(metric_name, self)
                self.metric_area.addWidget(viewer)
                viewer.value_changed.connect(self.table_widget.update_elements)  # type: ignore
                self.metric_dict[metric_name] = viewer

            viewer.set_data(labels_data)

            if label.endswith("_min"):
                viewer.set_col_min(col_idx)
            elif label.endswith("_max"):
                viewer.set_col_max(col_idx)
            else:
                assert False, label


class HistogramMinMaxView(QWidget):
    value_changed = Signal(float, int)

    def __init__(self, label_name, parent=None):
        super().__init__(parent)

        self.setLayout(QVBoxLayout())
        self.col_min = -1
        self.col_max = -1
        self.step_size = 1000

        self.canvas = FigureCanvas()
        self.canvas.setMaximumHeight(100)
        self.axis = self.canvas.figure.subplots()
        self.axis.get_yaxis().set_visible(False)
        self.axis.set_position([0.01, 0.25, 0.98, 0.73])
        self.line_min = self.axis.axvline(0, color="k")
        self.line_max = self.axis.axvline(0, color="orange")

        self.slider_min = SliderView(
            QRegularExpressionValidator(
                QRegularExpression(r"-?[0-9]*\.[0-9]*")
            ),
            self,
        )
        self.slider_min.value_changed.connect(self.value_changed.emit)
        self.slider_min.value_changed.connect(
            lambda x, _, is_max=False: self.adjust_line(x, is_max)
        )
        self.slider_max = SliderView(
            QRegularExpressionValidator(
                QRegularExpression(r"-?[0-9]*\.[0-9]*")
            ),
            self,
        )
        self.slider_max.value_changed.connect(self.value_changed.emit)
        self.slider_max.value_changed.connect(
            lambda x, _, is_max=True: self.adjust_line(x, is_max)
        )

        layout = QFormLayout()
        layout.addRow(f"{label_name} min", self.slider_min)
        layout.addRow(f"{label_name} max", self.slider_max)

        self.layout().addLayout(layout)
        self.layout().addWidget(self.canvas, stretch=0)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def adjust_line(self, value, is_max):
        if is_max:
            self.line_max.set_data([value, value], [0, 1])
        else:
            self.line_min.set_data([value, value], [0, 1])
        self.canvas.draw_idle()

    def set_data(self, label_data):
        val_min = _get_min_floor(label_data.min())
        val_max = _get_max_floor(label_data.max())
        self.slider_min.set_range(val_min, val_max)
        self.slider_max.set_range(val_min, val_max)
        self.slider_min.set_value(val_min)
        self.slider_max.set_value(val_max)
        self.line_min.set_data([val_min, val_min], [0, 1])
        self.line_max.set_data([val_max, val_max], [0, 1])

        self.axis.clear()
        self.axis.hist(label_data, 100)
        self.axis.add_artist(self.line_min)
        self.axis.add_artist(self.line_max)
        self.canvas.draw_idle()

    def set_col_min(self, col_min):
        self.slider_min.set_col(col_min)

    def set_col_max(self, col_max):
        self.slider_max.set_col(col_max)


class EditView(QWidget):
    value_changed = Signal(float, int)

    def __init__(self, col_idx, validator=None, parent=None):
        super().__init__(parent)
        self.col_idx = col_idx
        self.setLayout(QHBoxLayout())
        self.edit = QLineEdit(self)
        if validator is not None:
            self.edit.setValidator(validator)
        self.edit.returnPressed.connect(self._emit_signal)
        self.layout().addWidget(self.edit, stretch=1)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def _emit_signal(self):
        value = float(self.edit.text())
        self.value_changed.emit(value, self.col_idx)

    def set_value(self, value):
        self.edit.setText(str(value))

    def set_range(self, *_):
        pass


class SliderView(QWidget):
    value_changed = Signal(float, int)

    def __init__(self, validator=None, parent=None):
        super().__init__(parent)
        self.col_idx = -1
        self.setLayout(QHBoxLayout())
        self.step_size = 1000

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.sliderMoved.connect(self.mouse_move)
        self.slider.setRange(
            self.step_size * self.slider.minimum(),
            self.step_size * self.slider.maximum(),
        )
        self.label = QLineEdit(str(self.slider.value() / self.step_size), self)
        if validator is not None:
            self.label.setValidator(validator)
        self.label.returnPressed.connect(self.set_value)

        self.layout().addWidget(self.slider, stretch=1)
        self.layout().addWidget(self.label)

        self.layout().setContentsMargins(0, 0, 0, 0)

    def mouse_move(self, value):
        self.label.setText(str(value / self.step_size))
        self.value_changed.emit(value / self.step_size, self.col_idx)

    def set_value(self, value=None):
        emit_signal = True if value is None else False
        value = value if value is not None else float(self.label.text())
        value = int(self.step_size * value)

        self.slider.setValue(value)
        if emit_signal:
            self.slider.sliderMoved.emit(value)
        else:
            # Otherwise handeld by the sliderMoved event
            self.label.setText(str(value / self.step_size))

    def set_range(self, val_min, val_max):
        self.slider.setRange(
            int(self.step_size * val_min) - 1,
            int(self.step_size * val_max) + 1,
        )

    def set_col(self, col):
        self.col_idx = col

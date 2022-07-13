import enum
import typing

import napari.layers
import numpy as np
from qtpy.QtCore import QModelIndex, Qt, Signal, Slot
from qtpy.QtGui import QDoubleValidator, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
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

        self.dataChanged.connect(self.change_children)

    @Slot("QModelIndex", "QModelIndex", "QVector<int>")
    def change_children(self, top_idx, _, idx):
        if not idx:
            return

        if self.label_dict_rev[top_idx.column()] in self.read_only:
            return

        if top_idx.parent().row() == -1:
            root_item = self.invisibleRootItem().child(top_idx.row(), 0)
            value = (
                self.invisibleRootItem()
                .child(top_idx.row(), top_idx.column())
                .text()
            )
            for grandchild_idx in range(root_item.rowCount()):
                grandchild_item = root_item.child(
                    grandchild_idx, top_idx.column()
                )
                grandchild_item.setText(str(value))

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

    def set_value(self, parent_idx, row_idx, col_name, value):
        root_element = self.invisibleRootItem()
        if parent_idx == -1:
            child_item = root_element
        else:
            child_item = root_element.child(parent_idx, 0)
        child_item.child(row_idx, self.label_dict[col_name]).setText(
            str(value)
        )

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
        parents = {entry[1] for entry in rows_candidates if entry[0] == -1}
        rows = [entry for entry in rows_candidates if entry[0] not in parents]

        for parent_idx, row_idx in rows:
            if parent_idx == -1:
                parent_item = self.model.invisibleRootItem()
            else:
                parent_item = self.model.item(parent_idx, 0)

            child_item = parent_item.child(row_idx, col_idx)
            child_item.setText(str(value))


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
        ]
        self.ignore_idx = [
            "boxsize",
        ] + self.read_only

        self.napari_viewer.layers.events.inserted.connect(self.reset_choices)
        self.napari_viewer.layers.events.removed.connect(self.reset_choices)

        self.layer_input = QComboBox(self)
        self.reset_choices(None)

        self.table_model = GroupModel(self.read_only, self)
        self.table_model.dataChanged.connect(self._update_view)
        self.table_widget = GroupView(self.table_model, self)
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
        self.layout().addWidget(self.table_widget)
        self.layout().addLayout(self.metric_area)  # type: ignore

    @Slot("QModelIndex", "QModelIndex", "QVector<int>")
    def _update_view(self, top_idx, _, idx):
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

        if layer.data.shape[1] == 3:
            mask_dimension = layer.data[:, 0] == row_idx
        elif layer.data.shape[1] == 2:
            mask_dimension = np.ones(layer.data.shape[0])
        else:
            assert False, layer

        if is_min_max:
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
            mask_metric = (min_val <= layer.features[metric_name]) & (
                layer.features[metric_name] <= max_val
            )

            layer.shown[mask_dimension & mask_metric] = True
            layer.shown[mask_dimension & ~mask_metric] = False

            layer.metadata[row_idx][col_name] = layer_val
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
                len(mask_dimension & mask_metric),
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
            features_copy["identifier"] = name or layer.data[:, 0]
        elif layer.data.shape[1] == 2:
            features_copy["identifier"] = name or 0
        else:
            assert False, layer.data

        features_copy["shown"] = layer.shown
        for identifier, ident_df in features_copy.groupby(
            "identifier", sort=False
        ):
            cur_name = name or layer.metadata[identifier]["path"]
            output_list.append(self._prepare_columns(ident_df, cur_name))

        return output_list

    @staticmethod
    def _get_min_floor(vals):
        return np.round(np.floor(np.min(vals) * 1000) / 1000, 3)

    @staticmethod
    def _get_max_floor(vals):
        return np.round(np.ceil(np.max(vals) * 1000) / 1000, 3)

    def _prepare_columns(self, features, name) -> dict:
        output_dict = {}
        output_dict["name"] = name
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
                self._get_min_floor(features[col_name])
            )
            output_dict[f"{col_name}_max"] = str(
                self._get_max_floor(features[col_name])
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

    def _get_min_max(self, label_name):
        metric_name = self.trim_suffix(label_name)[0]

        layer_names = self.table_model.group_items
        cur_minimum = np.inf
        cur_maximum = -np.inf
        for layer_name in layer_names:
            cur_minimum = np.minimum(
                np.min(
                    self.napari_viewer.layers[layer_name].features[metric_name]
                ),
                cur_minimum,
            )
            cur_maximum = np.maximum(
                np.max(
                    self.napari_viewer.layers[layer_name].features[metric_name]
                ),
                cur_maximum,
            )
        return cur_minimum, cur_maximum

    def _update_slider(self):
        for col_idx, label in enumerate(self.table_model.label_dict):
            if label in self.read_only:
                continue
            if label in self.metric_dict:
                viewer = self.metric_dict[label]
            else:
                if label in ("boxsize",):
                    viewer = EditView(label, col_idx)
                else:
                    viewer = SliderView(label, col_idx)
                viewer.value_changed.connect(self.table_widget.update_elements)
                self.metric_dict[label] = viewer
                self.metric_area.addWidget(viewer)
            min_val, max_val = self._get_min_max(label)
            viewer.set_range(min_val, max_val)

            if label.endswith("_min"):
                viewer.set_value(self._get_min_floor(min_val))
            elif label.endswith("_max"):
                viewer.set_value(self._get_max_floor(max_val))
            elif label in ("boxsize",):
                viewer.set_value(max_val)
            else:
                assert False, label


class EditView(QWidget):
    value_changed = Signal(float, int)

    def __init__(self, text, col_idx, parent=None):
        super().__init__(parent)
        self.col_idx = col_idx
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(QLabel(text, self))
        self.edit = QLineEdit(self)
        self.edit.setValidator(QDoubleValidator())
        self.edit.returnPressed.connect(self._emit_signal)
        self.layout().addWidget(self.edit, stretch=1)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def _emit_signal(self):
        value = float(self.edit.text())
        self.value_changed.emit(value, self.col_idx)

    def set_value(self, value):
        self.edit.setText(str(value))
        self._emit_signal()

    def set_range(self, *_):
        pass


class SliderView(QWidget):
    value_changed = Signal(float, int)

    def __init__(self, text, col_idx, parent=None):
        super().__init__(parent)
        self.col_idx = col_idx
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(QLabel(text, self))
        self.step_size = 1000

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.sliderMoved.connect(self.mouse_move)
        self.slider.setRange(
            self.step_size * self.slider.minimum(),
            self.step_size * self.slider.maximum(),
        )
        self.label = QLineEdit(str(self.slider.value() / self.step_size), self)
        self.label.setValidator(QDoubleValidator())
        self.label.returnPressed.connect(self.set_value)

        self.layout().addWidget(self.slider, stretch=1)
        self.layout().addWidget(self.label)

        self.layout().setContentsMargins(0, 0, 0, 0)

    def mouse_move(self, value):
        self.label.setText(str(value / self.step_size))
        self.value_changed.emit(value / self.step_size, self.col_idx)

    def set_value(self, value=None):
        if value in (np.inf, -np.inf):
            return
        value = value if value is not None else float(self.label.text())
        value = int(self.step_size * value)

        self.slider.setValue(value)
        self.slider.sliderMoved.emit(value)

    def set_range(self, val_min, val_max):
        if val_min in (np.inf, -np.inf):
            return
        if val_max in (np.inf, -np.inf):
            return
        self.slider.setRange(
            int(self.step_size * val_min) - 1,
            int(self.step_size * val_max) + 1,
        )

import enum
import typing

import napari.layers
import numpy as np
from qtpy.QtCore import QModelIndex, Slot
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.group_items = {}
        self.label_dict = {}
        self.default_labels = [""]
        self._update_labels(self.default_labels)

    def _update_labels(self, columns):
        self.label_dict = {}
        i_label = -1
        for i_label in range(self.columnCount()):
            label = self.horizontalHeaderItem(i_label).text()
            self.label_dict[label] = i_label

        for i_label, new_label in enumerate(columns, i_label + 1):
            if new_label not in self.label_dict:
                self.label_dict[new_label] = i_label

        self.setHorizontalHeaderLabels(self.label_dict)

    def sort(self):
        self.invisibleRootItem().sortChildren(1)

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
                    str(columns[cur_label]) if cur_label in columns else "-"
                )
                col_item.setEditable(True)
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
        self.setModel(model)
        self.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)

    @Slot(QModelIndex)
    def on_clicked(self, index):
        if not index.parent().isValid() and index.column() == 0:
            self.setExpanded(index, not self.isExpanded(index))


class SelectMetricWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.napari_viewer = napari_viewer
        self.metrics: dict[str, typing.Any] = {}
        self.prev_points: list[str] = []

        self.napari_viewer.layers.events.inserted.connect(self.reset_choices)
        self.napari_viewer.layers.events.removed.connect(self.reset_choices)

        self.layer_input = QComboBox(self)
        self.reset_choices(None)

        self.table_model = GroupModel(self)
        table_widget = GroupView(self.table_model, self)
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
        self.layout().addWidget(table_widget)
        self.layout().addLayout(self.metric_area)  # type: ignore

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
    def _prepare_columns(features, name) -> dict:
        ignore_idx = ("boxsize", "identifier", "shown", "n_selected")
        output_dict = {}
        output_dict["name"] = name
        output_dict["n_boxes"] = len(features)
        output_dict["n_selected"] = np.count_nonzero(features["shown"])
        output_dict["boxsize"] = (
            10 if "boxsize" not in features else np.mean(features["boxsize"])
        )
        for col_name in features.columns:
            if col_name in ignore_idx:
                continue

            output_dict[f"{col_name}_min"] = np.min(features[col_name])
            output_dict[f"{col_name}_max"] = np.max(features[col_name])
        return output_dict

    def _add_remove_table(self, action: "ButtonActions"):
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
            self._add_remove_table(ButtonActions.DEL)
            self._add_remove_table(ButtonActions.ADD)

        self.table_model.sort()

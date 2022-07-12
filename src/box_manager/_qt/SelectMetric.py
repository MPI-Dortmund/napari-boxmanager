import enum
import typing

import napari.layers
from qtpy.QtCore import QModelIndex, Slot
from qtpy.QtGui import QColor, QIcon, QStandardItem, QStandardItemModel
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


class GroupModel(QStandardItemModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.group_items = {}
        self.label_dict = {}
        self._update_labels(["", "name"])

    def _update_labels(self, columns):
        old_columns = []
        i_label = -1
        for i_label in range(self.columnCount()):
            label = self.horizontalHeaderItem(i_label).text()
            old_columns.append(label)
            self.label_dict[label] = i_label

        for i_label, new_label in enumerate(columns, i_label + 1):
            old_columns.append(new_label)
            self.label_dict[new_label] = i_label

        self.setHorizontalHeaderLabels(old_columns)

    def add_group(self, group_name, columns) -> bool:
        if group_name in self.group_items:
            return False

        self._update_labels(columns)

        item_root = QStandardItem()
        item_root.setEditable(False)
        item = QStandardItem(group_name)
        item.setEditable(False)
        root_element = self.invisibleRootItem()
        row_idx = root_element.rowCount()

        for col_idx, col_item in enumerate((item_root, item)):
            root_element.setChild(row_idx, col_idx, col_item)
            root_element.setEditable(False)
        for col_idx in range(self.columnCount()):
            col_item = root_element.child(row_idx, col_idx)
            if col_item is None:
                col_item = QStandardItem()
                root_element.setChild(row_idx, col_idx, col_item)
        self.group_items[group_name] = item_root
        return True

    def remove_group(self, group_name):
        if group_name not in self.group_items:
            return

        index = self.indexFromItem(self.group_items[group_name])
        self.takeRow(index.row())
        del self.group_items[group_name]

    def append_element_to_group(self, group_name, texts):
        group_item = self.group_items[group_name]
        j = group_item.rowCount()
        item_icon = QStandardItem()
        item_icon.setEditable(False)
        item_icon.setIcon(QIcon("game.png"))
        item_icon.setBackground(QColor("#0D1225"))
        group_item.setChild(j, 0, item_icon)
        for i, text in enumerate(texts):
            item = QStandardItem(text)
            item.setEditable(False)
            item.setBackground(QColor("#0D1225"))
            item.setForeground(QColor("#F2F2F2"))
            group_item.setChild(j, i + 1, item)


class GroupDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._plus_icon = QIcon("plus.png")
        self._minus_icon = QIcon("minus.png")

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if not index.parent().isValid():
            is_open = bool(option.state & QStyle.State_Open)
            option.features |= QStyleOptionViewItem.HasDecoration
            option.icon = self._minus_icon if is_open else self._plus_icon


class GroupView(QTreeView):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setIndentation(0)
        self.setExpandsOnDoubleClick(False)
        self.clicked.connect(self.on_clicked)
        delegate = GroupDelegate(self)
        self.setItemDelegateForColumn(0, delegate)
        self.setModel(model)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
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

        self.layer_input = QComboBox(self)
        self.napari_viewer.layers.events.inserted.connect(self.reset_choices)
        self.napari_viewer.layers.events.removed.connect(self.reset_choices)
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
        btn_update.clicked.connect(lambda _: self._update_table())
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

    def _add_remove_table(self, action: "ButtonActions"):
        layer_name = self.layer_input.currentText()
        if not layer_name.strip():
            return None

        if action == ButtonActions.ADD:
            layer = self.napari_viewer.layers[layer_name]
            if self.table_model.add_group(
                layer_name, list(layer.features.columns)
            ):
                pass
        elif action == ButtonActions.DEL:
            self.table_model.remove_group(layer_name)
        else:
            assert False

        self._update_table()

    def _update_table(self):
        pass

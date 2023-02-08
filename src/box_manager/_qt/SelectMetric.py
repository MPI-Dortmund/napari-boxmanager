import enum
import itertools
import os
import pathlib
import typing

import napari.layers
import numpy as np
import pandas as pd
import tqdm
from matplotlib.backends.backend_qt5agg import FigureCanvas
from napari.utils.notifications import show_info
from qtpy.QtCore import (
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QRegularExpression,
    Qt,
    Signal,
    Slot,
)
from qtpy.QtGui import (
    QRegularExpressionValidator,
    QStandardItem,
    QStandardItemModel,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QSlider,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from .._utils import general

if typing.TYPE_CHECKING:
    import napari


ICON_DIR = pathlib.Path(os.path.dirname(__file__), "_icons")


# def debug(func):
#     def inner(*args, **kwargs):
#         print(func.__name__)
#         return func(*args, **kwargs)
#     return inner


def check_equal(layer, compare_data):
    if isinstance(layer, napari.layers.Points):
        return np.array_equal(layer.data, compare_data)
    elif isinstance(layer, napari.layers.Shapes):
        all_same = all(
            [
                np.array_equal(dat, ent)
                for dat, ent in itertools.zip_longest(
                    layer.data, compare_data, fillvalue=np.array([])
                )
            ]
        )
        return all_same
    else:
        assert False, (layer, type(layer))


def get_current_size(layer):
    if isinstance(layer, napari.layers.Points):
        return np.atleast_1d([layer.current_size])
    elif isinstance(layer, napari.layers.Shapes):
        return np.atleast_1d([layer.current_edge_width])
    else:
        assert False, (layer, type(layer))


def get_size(layer):
    if isinstance(layer, napari.layers.Points):
        return layer.size
    elif isinstance(layer, napari.layers.Shapes):
        return np.atleast_1d(layer.edge_width)
    else:
        assert False, (layer, type(layer))


def set_size(layer, mask, size):

    if isinstance(layer, napari.layers.Points):
        currently_selected = layer.selected_data.intersection(
            set(range(len(layer.size)))
        )
        layer.selected_data = currently_selected
        layer.size[mask] = size
        layer.current_size = size
        # TODO: Eventually remove after potential event fix: https://github.com/napari/napari/pull/4951
        layer.events.size()
    elif isinstance(layer, napari.layers.Shapes):
        layer.current_edge_width = size
        edge_width = np.atleast_1d(layer.edge_width)
        edge_width[mask] = size
        layer.edge_width = edge_width.tolist()
        layer.events.edge_width()
    else:
        assert False, (layer, type(layer))


class DimensionAxis(enum.Enum):
    Z = 0
    Y = 1
    X = 2


class ButtonActions(enum.Enum):
    ADD = 0
    DEL = 1
    UPDATE = 2


class GroupModel(QStandardItemModel):
    checkbox_updated = Signal(str, int, str, object)

    def __init__(self, read_only, check_box, parent=None):
        super().__init__(parent)
        self.read_only = read_only
        self.check_box = check_box
        self.group_items = {}
        self.label_dict = {}
        self.label_dict_rev = {}
        self.default_labels = [""]
        self._update_labels(self.default_labels)

    def rename_group(self, current_layers, new_name):
        old_name = [
            key for key in self.group_items if key not in current_layers
        ]
        assert len(old_name) == 1, (old_name, new_name, current_layers)
        old_name = old_name[0]

        self.group_items[new_name] = self.group_items.pop(old_name)
        prev_status = self.blockSignals(True)
        root_idx = self.group_items[new_name].index()
        self.set_value(
            root_idx.parent().row(), root_idx.row(), "name", new_name
        )
        self.blockSignals(prev_status)
        self.layoutChanged.emit()
        return old_name

    def update_model(self, layer_dict, value, col_idx):

        prev_status = self.blockSignals(True)
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
        self.blockSignals(prev_status)
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

    def _update_label_dict(self):
        self.label_dict = {}
        self.label_dict_rev = {}
        for i_label in range(self.columnCount()):
            label = self.horizontalHeaderItem(i_label).text()
            self.label_dict[label] = i_label
            self.label_dict_rev[i_label] = label

        root_item = self.invisibleRootItem()
        update = False
        prev_signal = self.blockSignals(True)
        for row_idx in range(root_item.rowCount()):
            for col_idx in range(1, self.columnCount()):
                parent_item = root_item.child(row_idx, col_idx)
                if parent_item is None:
                    update = True
                    root_item.setChild(row_idx, col_idx, QStandardItem("-"))
        self.blockSignals(prev_signal)
        if update:
            self.layoutChanged.emit()

    def _update_labels(self, columns):
        new_columns = []
        for i_label in range(self.columnCount()):
            label = self.horizontalHeaderItem(i_label).text()
            new_columns.append(label)

        for new_label in columns:
            if new_label not in self.label_dict:
                new_columns.append(new_label)
        self.setHorizontalHeaderLabels(new_columns)
        self._update_label_dict()

    def remove_labels(self, columns):
        if not columns:
            return

        col_idx = []
        for col in columns:
            idx = self.label_dict[col]
            col_idx.append(idx)

        for idx in reversed(sorted(col_idx)):
            self.takeColumn(idx)
        self._update_label_dict()

    def sort_children(self, group_name, label):
        root_item = self.group_items[group_name]

        root_item.sortChildren(self.label_dict[label])

    def find_index(self, parent_name: str, slice_val: str):
        root_item = self.group_items[parent_name]
        parent_idx = self.indexFromItem(root_item).row()

        def binary_search(parent_idx, search_val, low, high):
            if low > high:
                return None

            row_idx = (high + low) // 2
            try:
                cur_slice = int(self.get_value(parent_idx, row_idx, "slice"))
            except AttributeError:
                return None

            if search_val == cur_slice:
                return row_idx
            elif search_val > cur_slice:
                return binary_search(parent_idx, search_val, row_idx + 1, high)
            else:
                return binary_search(parent_idx, search_val, low, row_idx - 1)

        low = 0
        high = root_item.rowCount()
        row_idx = binary_search(parent_idx, int(slice_val), low, high)

        return parent_idx, row_idx

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

    def get_checkstates(self, parent_idx, rows_idx, col_name):
        return [
            self.get_checkstate(parent_idx, row, col_name) for row in rows_idx
        ]

    def get_checkstate(self, parent_idx, row_idx, col_name):
        root_element = self.invisibleRootItem()
        if parent_idx == -1:
            child_item = root_element
        else:
            child_item = root_element.child(parent_idx, 0)
        return (
            child_item.child(row_idx, self.label_dict[col_name]).checkState()
            == Qt.Checked
        )

    def set_checkstate(self, parent_idx, row_idx, col_name, value):
        root_element = self.invisibleRootItem()
        if parent_idx == -1:
            child_item = root_element
        else:
            child_item = root_element.child(parent_idx, 0)
        child_item.child(row_idx, self.label_dict[col_name]).setCheckState(
            Qt.Checked if value else Qt.Unchecked
        )

    def get_values(self, parent_idx, rows_idx, col_name):
        return [self.get_value(parent_idx, row, col_name) for row in rows_idx]

    def get_value(self, parent_idx, row_idx, col_name):
        root_element = self.invisibleRootItem()
        if parent_idx == -1:
            child_item = root_element
        else:
            child_item = root_element.child(parent_idx, 0)
        child = child_item.child(row_idx, self.label_dict[col_name])
        if child is None:
            return child
        else:
            return child.text()

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
        combo_items = []
        for col_idx in range(self.columnCount()):
            cur_label = self.horizontalHeaderItem(col_idx).text()
            if col_idx == 0:
                col_item = first_item or QStandardItem()
                col_item.setEditable(False)
            else:
                text = columns[cur_label] if cur_label in columns else "-"
                col_item = QStandardItem(text)
                if isinstance(text, bool):
                    combo_items.append((col_item, cur_label))
                    col_item.setEditable(True)
                    col_item.setCheckable(True)
                    col_item.setCheckState(
                        Qt.Checked if text else Qt.Unchecked
                    )
                else:
                    col_item.setEditable(
                        cur_label not in self.read_only and text != "-"
                    )
            root_element.setChild(row_idx, col_idx, col_item)
        for combo_item, col_name in combo_items:
            parent_idx = combo_item.index().parent().row()
            idx = combo_item.index().row()
            layer_name = self.get_value(-1, parent_idx, "name")
            slice_idx = int(self.get_value(parent_idx, idx, "slice"))
            self.checkbox_updated.emit(layer_name, slice_idx, col_name, None)

    def append_element_to_group(self, group_name, columns):
        group_item = self.group_items[group_name]
        row_idx = group_item.rowCount()
        item_icon = QStandardItem()
        item_icon.setEditable(False)
        group_item.setChild(row_idx, 0, item_icon)

        self.append_to_row(group_item, columns, row_idx)
        return row_idx


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
    checkbox_updated = Signal(str, int, str, object)

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

        self.model.dataChanged.connect(self.update_by_edit)
        self.model.checkbox_updated.connect(self.checkbox_updated.emit)
        self.parent_only = False

    @Slot(int)
    def set_parent_only(self, value):
        self.parent_only = value

    def get_row_selection(self):
        prev_selection = {
            self.model.get_value(-1, entry[1], "name")
            if entry[0] == -1
            else self.model.get_value(-1, entry[0], "name")
            for entry in self.get_row_candidates(False)
        }
        return prev_selection

    def get_expansion_state(self):
        root_item = self.model.invisibleRootItem()
        rows = root_item.rowCount()
        is_expanded = {}
        for row_idx in range(rows):
            idx = self.model.indexFromItem(root_item.child(row_idx, 0))
            expansion_state = self.isExpanded(idx)
            name = self.model.get_value(-1, row_idx, "name")
            is_expanded[name] = expansion_state

        return is_expanded

    def restore_expansion(self, expansion_dict):
        root_item = self.model.invisibleRootItem()
        rows = root_item.rowCount()
        for row_idx in range(rows):
            idx = self.model.indexFromItem(root_item.child(row_idx, 0))
            name = self.model.get_value(-1, row_idx, "name")
            try:
                value = expansion_dict[name]
            except KeyError:
                value = False
            self.setExpanded(idx, value)

    def sort(self, columns):
        model = self.model
        root_item = model.invisibleRootItem()
        prev_status = self.blockSignals(True)

        prev_selection = self.get_row_selection()
        prev_expansion = self.get_expansion_state()
        for new_idx, row_name in enumerate(reversed(columns)):
            old_idx = -1
            for row_idx in reversed(range(root_item.rowCount())):
                cur_item = root_item.child(row_idx, model.label_dict["name"])
                if row_name == cur_item.text():
                    old_idx = row_idx
                    break
            assert old_idx >= 0
            if new_idx == old_idx:
                continue
            row = model.takeRow(old_idx)
            model.insertRow(new_idx, row)

        self.blockSignals(prev_status)

        model.layoutChanged.emit()
        self.restore_selection(prev_selection)
        self.restore_expansion(prev_expansion)

    @Slot(QModelIndex)
    def on_clicked(self, index):
        if not index.parent().isValid() and index.column() == 0:
            self.setExpanded(index, not self.isExpanded(index))

    def select_first(self):
        self.setCurrentIndex(self.model.index(0, 0))

    def update_by_edit(self, idx, _, role):
        if not role:
            return

        parent_idx = idx.parent().row()
        row_idx = idx.row()
        col_idx = idx.column()
        col_name = self.model.label_dict_rev[col_idx]
        if col_name in self.model.check_box:
            layer_name = self.model.get_value(-1, parent_idx, "name")
            value = self.model.get_checkstate(parent_idx, row_idx, col_name)
            slice_idx = int(self.model.get_value(parent_idx, row_idx, "slice"))
            self.checkbox_updated.emit(layer_name, slice_idx, col_name, value)
            return
        elif col_name in self.model.read_only:
            return

        value = self.model.get_value(parent_idx, row_idx, col_name)
        self.update_elements(value, col_idx)

    def restore_selection(self, prev_selection):

        columns = self.model.columnCount() - 1
        selection = QItemSelection()
        flag = QItemSelectionModel.Select
        for name in prev_selection:
            for idx in range(self.model.invisibleRootItem().rowCount()):
                if self.model.get_value(-1, idx, "name") == name:
                    row_idx = idx
                    break
            else:
                continue

            start = self.model.index(row_idx, 0)
            end = self.model.index(row_idx, columns)
            if selection.indexes():
                selection.merge(QItemSelection(start, end), flag)
            else:
                selection.select(start, end)
        self.selectionModel().clear()
        self.selectionModel().select(selection, flag)

    def get_row_candidates(self, parent_only=None):
        if parent_only is None:
            parent_only = self.parent_only
        if parent_only:
            return {
                (-1, entry.parent().row())
                if entry.parent().row() != -1
                else (entry.parent().row(), entry.row())
                for entry in self.selectedIndexes()
            }
        else:
            return {
                (entry.parent().row(), entry.row())
                for entry in self.selectedIndexes()
            }

    def get_rows(self, rows_candidates, col_idx):
        parents = {entry[1] for entry in rows_candidates if entry[0] == -1}

        layer_dict = {}
        for parent_idx, row_idx in rows_candidates:
            if parent_idx in parents:
                continue
            elif parent_idx == -1:
                parent_item = self.model.invisibleRootItem()
            else:
                parent_item = self.model.item(parent_idx, 0)
            if parent_item.child(row_idx, col_idx).text() not in ("-",):
                layer_dict.setdefault(parent_idx, []).append(row_idx)
        return layer_dict

    def get_all_rows(self, layer_dict):
        output_dict = {}
        for parent_idx, rows_idx in layer_dict.items():
            if parent_idx == -1:
                for row in rows_idx:
                    root_item = self.model.invisibleRootItem().child(row, 0)
                    rows = list(range(root_item.rowCount()))
                    output_dict[row] = rows
            else:
                output_dict[parent_idx] = rows_idx
        return output_dict

    @Slot(float, int)
    def update_elements(self, value, col_idx):
        rows_candidates = self.get_row_candidates()
        if not rows_candidates:
            return

        layer_dict = self.get_rows(rows_candidates, col_idx)

        update_dict = self.model.update_model(layer_dict, value, col_idx)
        layer_dict = self.get_all_rows(update_dict)

        col_name = self.model.label_dict_rev[col_idx]
        self.elementsUpdated.emit(layer_dict, col_name)


class SelectMetricWidget(QWidget):
    sig_update_hist = Signal(object)

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.napari_viewer = napari_viewer
        self.metrics: dict[str, typing.Any] = {}
        self.metric_dict: dict = {}
        self.prev_valid_layers = {}
        self._plugin_view_update = False
        self._cur_slice_dim = self.napari_viewer.dims.order[0]

        self.loadable_layers = (napari.layers.Points, napari.layers.Shapes)
        self.check_box = [
            "write",
        ]
        self.read_only = [
            "",
            "identifier",
            "shown",
            "selected",
            "boxes",
            "name",
            "slice",
        ] + self.check_box
        self.ignore_idx = set(
            [
                "boxsize",
            ]
            + self.read_only
        )

        self.napari_viewer.layers.events.reordered.connect(self._order_table)
        self.napari_viewer.layers.events.inserted.connect(self._handle_insert)
        self.napari_viewer.layers.events.removed.connect(self._handle_remove)
        self.napari_viewer.dims.events.order.connect(self._update_sync)
        self.napari_viewer.events.theme.connect(self._set_color)

        self.table_model = GroupModel(self.read_only, self.check_box, self)
        self.table_widget = GroupView(self.table_model, self)
        self.table_widget.elementsUpdated.connect(self._update_view)
        self.table_widget.checkbox_updated.connect(self._update_check_state)
        self.table_widget.selectionModel().selectionChanged.connect(
            self.update_hist
        )
        self.metric_area = QVBoxLayout()

        self.option_area = QHBoxLayout()
        self.global_checkbox = QCheckBox(
            "Apply on layers, not on slices", self
        )
        self.option_area.addWidget(self.global_checkbox)
        self.global_checkbox.stateChanged.connect(
            self.table_widget.set_parent_only
        )
        self.global_checkbox.setChecked(True)
        self.global_checkbox.stateChanged.connect(lambda _: self.update_hist())

        self.settings_area = QHBoxLayout()
        self.hide_dim = QComboBox(self)
        self.hide_dim.currentTextChanged.connect(self.update_hist)
        self.hide_dim.addItems(
            [
                "Selected",
                "All (Opacity 0.5)",
                "All (Opacity 0.2)",
                "All (Opacity 0.1)",
                "All",
            ]
        )

        self.show_mode = QComboBox(self)
        self.show_mode.addItems(
            [
                "All",
                "Occupied",
            ]
        )
        self.show_mode.currentTextChanged.connect(self._update_sync)
        self._show_mode = self.show_mode.currentText()

        self.settings_area.addWidget(QLabel("Show:", self))
        self.settings_area.addWidget(self.hide_dim, stretch=1)
        self.settings_area.addWidget(QLabel("Slices:", self))
        self.settings_area.addWidget(self.show_mode, stretch=1)

        self.setLayout(QVBoxLayout())
        self.layout().addLayout(self.settings_area, stretch=0)  # type: ignore
        self.layout().addWidget(self.table_widget, stretch=1)
        self.layout().addLayout(self.option_area, stretch=0)  # type: ignore
        self.layout().addLayout(self.metric_area, stretch=0)  # type: ignore

        self._sync_table(select_first=True)
        self._set_color()

    @Slot(object)
    def _handle_remove(self, event):
        layer = event.value
        if not isinstance(layer, self.loadable_layers):
            return

        self._add_remove_table(layer, ButtonActions.DEL)
        del self.prev_valid_layers[layer.name]

    @Slot(object)
    def _handle_insert(self, event):
        layer = event.value
        if not isinstance(layer, self.loadable_layers):
            return

        if "ignore_idx" in layer.metadata:
            self.ignore_idx = self.ignore_idx | set(
                layer.metadata["ignore_idx"]
            )

        try:
            # TODO: Remove try/except after https://github.com/napari/napari/pull/5028
            if layer.source.parent is not None:
                layer.metadata["set_lock"] = False
                layer.metadata["do_activate_on_insert"] = True
        except AttributeError:
            pass

        prev_expansion = self.table_widget.get_expansion_state()
        self._add_remove_table(layer, ButtonActions.ADD)
        if "set_lock" in layer.metadata and layer.metadata["set_lock"]:
            layer.editable = False

        layer.events.set_data.connect(self._update_on_data)
        layer.events.editable.connect(self._update_editable)
        layer.events.name.connect(self._update_name)
        layer.events.opacity.connect(self._update_opacity)
        layer.events.visible.connect(self._update_visible)
        self.prev_valid_layers[layer.name] = [layer, layer.data]

        if "do_activate_on_insert" in layer.metadata:
            self.table_widget.selectionModel().selectionChanged.disconnect(
                self.update_hist
            )
            self.table_widget.restore_selection({layer.name})
            self.table_widget.restore_expansion(prev_expansion)
            self.table_widget.selectionModel().selectionChanged.connect(
                self.update_hist
            )
            self.table_widget.selectionModel().selectionChanged.emit(
                QItemSelection(), QItemSelection()
            )
            del layer.metadata["do_activate_on_insert"]

        self._update_slider()
        self._order_table()

    @Slot(str, int, str, object)
    def _update_check_state(self, layer_name, slice_idx, attr_name, value):
        try:
            old_val = self.napari_viewer.layers[
                layer_name
            ].metadata.setdefault(slice_idx, {})[attr_name]
        except KeyError:
            old_val = None

        if value is None and old_val is not None:
            value = old_val
        self.napari_viewer.layers[layer_name].metadata[slice_idx][
            attr_name
        ] = value

    def _set_color(self):
        if self.napari_viewer.theme == "dark":
            icon = pathlib.Path(ICON_DIR, "checkmark_white.png")
            self.table_widget.setStyleSheet(
                f"""
                QAbstractItemView::indicator {{
                    border: 1px solid white;
                }}

                QAbstractItemView::indicator:checked {{
                    image: url({icon})
                }}
                """
            )
        else:
            icon = pathlib.Path(ICON_DIR, "checkmark_black.png")
            self.table_widget.setStyleSheet(
                f"""
                QAbstractItemView::indicator {{
                    border: 1px solid black;
                }}

                QAbstractItemView::indicator:checked {{
                    image: url({icon})
                }}
                """
            )

    def _update_sync(self, *_):
        self._show_mode = self.show_mode.currentText()
        self._cur_slice_dim = self.napari_viewer.dims.order[0]
        self.table_widget.selectionModel().selectionChanged.disconnect(
            self.update_hist
        )
        prev_expansion = self.table_widget.get_expansion_state()
        prev_selection = self._clear_table()
        self._sync_table(do_selection=False)
        self.table_widget.restore_selection(prev_selection)
        self.table_widget.restore_expansion(prev_expansion)
        self.table_widget.selectionModel().selectionChanged.connect(
            self.update_hist
        )
        self.table_widget.selectionModel().selectionChanged.emit(
            QItemSelection(), QItemSelection()
        )

    @Slot(dict, str)
    def _update_view(self, layer_dict, col_name):
        prev_plugin_view_update = self._plugin_view_update
        self._plugin_view_update = True
        metric_name, is_min_max = self.trim_suffix(col_name)

        for parent_idx, rows_idx in layer_dict.items():
            layer_name = self.table_model.get_value(-1, parent_idx, "name")
            layer = self.napari_viewer.layers[layer_name]  # type: ignore
            do_update = False

            slice_idx = list(
                map(
                    int,
                    self.table_model.get_values(parent_idx, rows_idx, "slice"),
                )
            )
            if rows_idx:
                layer_vals = float(
                    self.table_model.get_value(
                        parent_idx, rows_idx[0], col_name
                    )
                )
            else:
                layer_vals = float(
                    self.table_model.get_value(-1, parent_idx, col_name)
                )

            for z_slice in slice_idx:
                layer.metadata.setdefault(z_slice, {})[col_name] = layer_vals

            if layer.ndim == 3:
                mask_dimension = np.isin(
                    np.round(
                        general.get_identifier(layer, self._cur_slice_dim), 0
                    ),
                    slice_idx,
                )
            elif layer.ndim == 2:
                mask_dimension = np.ones(len(layer.data), dtype=bool)
            else:
                assert False, layer

            if is_min_max:
                old_shown = layer.shown.copy()
                mask_metric = np.ones(mask_dimension.shape, dtype=bool)
                for metric_name in layer.features.columns:
                    if metric_name in self.ignore_idx:
                        continue
                    min_val = min(
                        float(entry)
                        for entry in self.table_model.get_values(
                            parent_idx, rows_idx, f"{metric_name}_min"
                        )
                        if entry.replace(".", "", 1)
                        .removeprefix("-")
                        .isdigit()
                    )
                    max_val = max(
                        float(entry)
                        for entry in self.table_model.get_values(
                            parent_idx, rows_idx, f"{metric_name}_max"
                        )
                        if entry.replace(".", "", 1)
                        .removeprefix("-")
                        .isdigit()
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
                        if layer.ndim == 3:
                            slice_mask = (
                                np.round(
                                    general.get_identifier(
                                        layer, self._cur_slice_dim
                                    ),
                                    0,
                                )
                                == slice_idx[idx]
                            )
                        elif layer.ndim == 2:
                            slice_mask = np.ones(len(layer.data), dtype=bool)
                        else:
                            assert False, layer
                        self.table_model.set_value(
                            parent_idx,
                            row,
                            "selected",
                            np.count_nonzero(
                                mask_dimension & mask_metric & slice_mask
                            ),
                        )
                        self.table_model.set_value(
                            parent_idx,
                            row,
                            "boxes",
                            np.count_nonzero(mask_dimension & slice_mask),
                        )

                    self.table_model.set_value(
                        -1,
                        parent_idx,
                        "selected",
                        np.count_nonzero(layer.shown),
                    )
                    self.table_model.set_value(
                        -1, parent_idx, "boxes", len(layer.shown)
                    )
                    do_update = True
            elif metric_name == "boxsize":
                do_update = True
                set_size(layer, mask_dimension, layer_vals)
            else:
                assert False

            if do_update:
                layer.refresh()
        self._plugin_view_update = prev_plugin_view_update

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

    @Slot(object)
    def _order_table(self, event=None, do_selection=True):
        valid_names: list[str] = [
            entry.name
            for entry in self.napari_viewer.layers
            if isinstance(entry, self.loadable_layers)
        ]

        # if do_selection:
        #    prev_selection = self.table_widget.get_row_selection()
        #    self.table_widget.selectionModel().selectionChanged.disconnect(
        #        self.update_hist
        #    )
        self.table_widget.sort(valid_names)
        # if do_selection:
        #    self.table_widget.restore_selection(prev_selection)
        #    self.table_widget.selectionModel().selectionChanged.connect(
        #        self.update_hist
        #    )
        #    self.table_widget.selectionModel().selectionChanged.emit(
        #        QItemSelection(), QItemSelection()
        #    )

    @Slot(object)
    def _sync_table(
        self, event=None, *, select_first=False, do_selection=True
    ):
        valid_layers: list[napari.layers.Layer] = [
            entry
            for entry in self.napari_viewer.layers
            if isinstance(entry, self.loadable_layers)
        ]

        if (
            event is not None
            and event.type == "inserted"
            and len(valid_layers) == 1
        ):
            select_first = True

        for layer in valid_layers:
            if "ignore_idx" in layer.metadata:
                self.ignore_idx = self.ignore_idx | set(
                    layer.metadata["ignore_idx"]
                )

        prev_layers = [entry for entry, _ in self.prev_valid_layers.values()]
        self.prev_valid_layers = {}
        if sorted(valid_layers, key=lambda x: x.name) != sorted(
            prev_layers, key=lambda x: x.name
        ):
            for layer in tqdm.tqdm(set(prev_layers + valid_layers)):
                if layer in valid_layers:
                    self._add_remove_table(layer, ButtonActions.ADD)
                    if (
                        "set_lock" in layer.metadata
                        and layer.metadata["set_lock"]
                    ):
                        layer.editable = False

                    layer.events.set_data.disconnect(self._update_on_data)
                    layer.events.set_data.connect(self._update_on_data)

                    layer.events.editable.disconnect(self._update_editable)
                    layer.events.editable.connect(self._update_editable)

                    layer.events.name.disconnect(self._update_name)
                    layer.events.name.connect(self._update_name)

                    layer.events.opacity.disconnect(self._update_opacity)
                    layer.events.opacity.connect(self._update_opacity)

                    layer.events.visible.disconnect(self._update_visible)
                    layer.events.visible.connect(self._update_visible)

                    self.prev_valid_layers[layer.name] = [layer, layer.data]
                else:
                    self._add_remove_table(layer, ButtonActions.DEL)
            self._order_table(do_selection=do_selection)
            self._update_slider()
            if select_first:
                self.table_widget.select_first()

    def _clear_table(self):
        prev_selection = self.table_widget.get_row_selection()
        for layer, _ in self.prev_valid_layers.values():
            self._add_remove_table(layer, ButtonActions.DEL)
        self.prev_valid_layers = {}
        return prev_selection

    def _update_opacity(self, event):
        layer = event.source
        layer.metadata["prev_opacity"] = layer.opacity

    def _update_visible(self, event):
        layer = event.source
        layer.metadata["prev_visible"] = layer.visible

    def _update_name(self, event):
        old_name = self.table_model.rename_group(
            [x[0].name for x in self.prev_valid_layers.values()],
            event.source.name,
        )
        self.prev_valid_layers[event.source.name] = self.prev_valid_layers.pop(
            old_name
        )

    @Slot(object)
    def _update_on_data(self, event):
        if not self._plugin_view_update:
            layer = event.source
            try:
                is_creating = layer._is_creating
            except AttributeError:
                is_creating = False

            if is_creating:
                return

            if not check_equal(layer, self.prev_valid_layers[layer.name][1]):
                old_data = self.prev_valid_layers[layer.name][1]

                if self.napari_viewer.dims.ndim == 3:
                    set_old = {tuple(row.ravel().tolist()) for row in old_data}
                    set_new = {
                        tuple(row.ravel().tolist()) for row in layer.data
                    }
                    indices_old = {
                        row[self.napari_viewer.dims.order[0]]
                        for row in set_old - set_new
                    }
                    indices_new = {
                        row[self.napari_viewer.dims.order[0]]
                        for row in set_new - set_old
                    }
                    current_slices = list(indices_new | indices_old)
                    if indices_new:
                        for idx in range(layer.features.shape[1]):
                            layer.features.iloc[-1, idx] = (
                                layer.features.iloc[:-1, idx]
                                .sort_values()
                                .iloc[layer.features.shape[0] // 2]
                            )
                elif self.napari_viewer.dims.ndim == 2:
                    current_slices = [0]
                else:
                    assert False, layer.data

                self.prev_valid_layers[layer.name][1] = layer.data
                for current_slice in current_slices:
                    self._update_table(layer, int(np.round(current_slice, 0)))
                self.update_hist(change_selection=False)

    def _update_editable(self, event):
        layer = event.source
        if (
            "set_lock" in layer.metadata
            and layer.metadata["set_lock"]
            and layer.editable
        ):
            layer.editable = False

    def _prepare_entries(self, layer, *, name=None) -> list:
        output_list = []
        features_copy = layer.features.copy()
        if layer.ndim == 3:
            features_copy["identifier"] = (
                ""
                if name is not None
                else np.round(
                    general.get_identifier(layer, self._cur_slice_dim), 0
                ).astype(int)
            )
        elif layer.ndim == 2:
            features_copy["identifier"] = "" if name is not None else 0
        else:
            assert False, layer.data

        range_list = [
            entry for entry in layer.metadata if isinstance(entry, int)
        ]
        full_range = np.arange(
            *self.napari_viewer.dims.range[0], dtype=int
        ).tolist()
        if range_list:
            if "ignore_idx" in layer.metadata:
                ignore_idx = layer.metadata.pop("ignore_idx")
                label_data = pd.DataFrame(layer.metadata).loc[:, range_list].T
                layer.metadata["ignore_idx"] = ignore_idx
            else:
                label_data = pd.DataFrame(layer.metadata).loc[:, range_list].T
        else:
            label_data = pd.DataFrame()
        range_list.extend(full_range)

        if name is None and self._show_mode == "All":
            if layer.ndim == 3:
                loop_var = sorted(list(set(range_list)))
            elif layer.ndim == 2:
                loop_var = [0]
            else:
                assert False, layer.data
        else:
            idents = np.unique(features_copy["identifier"]).tolist()
            write_slices = [
                key
                for key, value in layer.metadata.items()
                if isinstance(value, dict)
                and "write" in value
                and value["write"]
            ]
            try:
                loop_var = sorted(list(set(idents + write_slices)))
            except TypeError:
                loop_var = idents

        try:
            features_copy["shown"] = layer.shown
        except AttributeError:
            # Shape layers do not have the shown option, fake it!
            features_copy["shown"] = np.ones(len(layer.data))
            layer.shown = features_copy["shown"]
        slice_dict = {
            e1: e2
            for e1, e2 in features_copy.groupby("identifier", sort=False)
        }

        try:
            max_slice = max(loop_var)
        except ValueError:
            max_slice = 0
        for identifier in loop_var:
            try:
                ident_df = slice_dict[identifier]
            except KeyError:
                ident_df = pd.DataFrame(columns=features_copy.columns)
            try:
                cur_name = name or layer.metadata[identifier]["name"]
            except KeyError:
                cur_name = "Manual"
            output_list.append(
                self._prepare_columns(
                    pd.DataFrame(get_size(layer), dtype=float),
                    ident_df,
                    cur_name,
                    identifier,
                    label_data,
                    name is not None,
                    max_slice,
                )
            )

        # Case: No points available
        if not output_list and name is not None:
            identifier = "" if name is not None else 0
            try:
                cur_name = name or layer.metadata[identifier]["name"]
            except KeyError:
                cur_name = "Manual"
            features = pd.DataFrame(columns=["shown"])
            output_list.append(
                self._prepare_columns(
                    pd.DataFrame(get_size(layer), dtype=float),
                    features,
                    cur_name,
                    identifier,
                    label_data,
                    name is not None,
                    0,
                )
            )

        return output_list

    def _prepare_columns(
        self,
        size,
        features,
        name,
        slice_idx,
        label_data,
        is_main_group,
        max_slice,
    ) -> dict:
        output_dict = {}
        output_dict["write"] = ""
        output_dict["name"] = name
        if isinstance(slice_idx, int):
            output_dict["slice"] = f"{slice_idx:0{len(str(max_slice))}d}"
        else:
            output_dict["slice"] = str(slice_idx)
        output_dict["boxes"] = str(len(features))
        output_dict["selected"] = str(np.count_nonzero(features["shown"]))

        output_dict["boxsize"] = (
            "0" if size.empty else str(int(size.mean().mean()))
        )
        if (
            self.napari_viewer.dims.order[0] == 0
            and self.napari_viewer.dims.ndim == 3
        ):
            if is_main_group:
                output_dict["write"] = "-"
            else:
                try:
                    write_val = label_data.loc[slice_idx, "write"]
                    if write_val is not None and not np.isnan(write_val):
                        output_dict["write"] = write_val
                    else:
                        output_dict["write"] = not features.empty
                except KeyError:
                    output_dict["write"] = not features.empty

            for col_name in features.columns:
                if col_name in self.ignore_idx:
                    continue

                label_min = f"{col_name}_min"
                label_max = f"{col_name}_max"

                if (
                    slice_idx in label_data.index
                    and label_min in label_data.columns
                ):
                    val = label_data.loc[slice_idx, label_min]
                else:
                    val = general.get_min_floor(label_data[label_min])
                    if not np.all(val == label_data[label_min].dropna()):
                        val = general.get_min_floor(features[col_name])
                if not np.isnan(val):
                    output_dict[label_min] = str(val)
                else:
                    output_dict[label_min] = "-"

                if (
                    slice_idx in label_data.index
                    and label_max in label_data.columns
                ):
                    val = label_data.loc[slice_idx, label_max]
                else:
                    val = general.get_max_floor(label_data[label_max])
                    if not np.all(val == label_data[label_max].dropna()):
                        val = general.get_max_floor(features[col_name])
                if not np.isnan(val):
                    output_dict[label_max] = str(val)
                else:
                    output_dict[label_max] = "-"

        else:
            output_dict["write"] = "-"
            for col_name in features.columns:
                if col_name in self.ignore_idx:
                    continue

                output_dict[f"{col_name}_min"] = "-"
                output_dict[f"{col_name}_max"] = "-"
        return output_dict

    def _add_remove_table(self, layer, action: "ButtonActions"):
        layer_name = layer.name

        if action == ButtonActions.ADD:
            if self.table_model.add_group(
                layer_name, self._prepare_entries(layer, name=layer_name)[0]
            ):
                entries = self._prepare_entries(layer)
                for entry in entries:
                    self.table_model.append_element_to_group(layer_name, entry)
                self.table_model.sort_children(layer_name, "slice")
        elif action == ButtonActions.DEL:
            self.table_model.remove_group(layer_name)
        else:
            assert False, action

    def _update_table(self, layer: napari.layers.Layer, current_slice):
        prev_plugin_view_update = self._plugin_view_update
        self._plugin_view_update = True
        layer_name = layer.name
        if self.napari_viewer.dims.ndim == 3:
            try:
                mask = (
                    np.round(
                        layer.data[:, self.napari_viewer.dims.order[0]], 0
                    )
                    == current_slice
                )
            except TypeError:
                layer_data = np.array([row[0] for row in layer.data])
                if layer_data.size:
                    mask = (
                        np.round(
                            layer_data[:, self.napari_viewer.dims.order[0]], 0
                        )
                        == current_slice
                    )
                else:
                    mask = np.array(layer_data, dtype=bool)
            boxes = np.count_nonzero(mask)
            try:
                selected = np.count_nonzero(layer.shown[mask])
            except IndexError:
                selected = boxes
        elif self.napari_viewer.dims.ndim == 2:
            current_slice = 0
            boxes = len(layer.data)
            if boxes == len(layer.shown):
                selected = np.count_nonzero(layer.shown)
            else:
                selected = boxes
        else:
            assert False, layer.data

        parent_idx, child_idx = self.table_model.find_index(
            layer_name, str(current_slice)
        )
        do_sort = False
        if child_idx is None:
            range_list = [
                entry for entry in layer.metadata if isinstance(entry, int)
            ]
            full_range = np.arange(
                *self.napari_viewer.dims.range[0], dtype=int
            ).tolist()
            if range_list:
                if "ignore_idx" in layer.metadata:
                    ignore_idx = layer.metadata.pop("ignore_idx")
                    label_data = (
                        pd.DataFrame(layer.metadata).loc[:, range_list].T
                    )
                    layer.metadata["ignore_idx"] = ignore_idx
                else:
                    label_data = (
                        pd.DataFrame(layer.metadata).loc[:, range_list].T
                    )
            else:
                label_data = pd.DataFrame()
            range_list.extend(full_range)
            try:
                name = layer.metadata[current_slice]["name"]
            except KeyError:
                name = "Manual"
            new_col_entry = self._prepare_columns(
                pd.DataFrame(get_size(layer), dtype=float),
                pd.DataFrame(columns=list(layer.features.columns) + ["shown"]),
                name,
                current_slice,
                label_data,
                False,
            )
            child_idx = self.table_model.append_element_to_group(
                layer_name, new_col_entry
            )
            do_sort = True

        self.table_model.set_value(parent_idx, child_idx, "boxes", boxes)
        self.table_model.set_value(parent_idx, child_idx, "selected", selected)

        write_val = layer.metadata.setdefault(current_slice, {}).setdefault(
            "write", None
        )
        check_value = write_val if write_val is not None else bool(selected)
        if check_value != self.table_model.get_checkstate(
            parent_idx, child_idx, "write"
        ):
            self.table_model.set_checkstate(
                parent_idx, child_idx, "write", check_value
            )
            # Do not change the checkstate
            layer.metadata[current_slice]["write"] = write_val

        if len(layer.shown) == len(layer.data):
            self.table_model.set_value(
                -1,
                parent_idx,
                "selected",
                np.count_nonzero(layer.shown),
            )
            self.table_model.set_value(
                -1, parent_idx, "boxes", len(layer.shown)
            )
        else:
            self.table_model.set_value(
                -1,
                parent_idx,
                "selected",
                len(layer.data),
            )
            self.table_model.set_value(
                -1, parent_idx, "boxes", len(layer.data)
            )
        if do_sort:
            self.table_widget.selectionModel().selectionChanged.disconnect(
                self.update_hist
            )
            self.table_model.sort_children(layer_name, "slice")
            self.table_widget.selectionModel().selectionChanged.connect(
                self.update_hist
            )
        self._plugin_view_update = prev_plugin_view_update
        layer.refresh()

    def _get_all_data(self, metric_name, layer_mask=None):

        layer_names = (
            self.table_model.group_items if layer_mask is None else layer_mask
        )
        layer_features = []
        for layer_name in layer_names:
            try:
                layer = self.napari_viewer.layers[layer_name]
            except KeyError:
                # Layer has been deleted
                continue

            if layer_mask is None:
                mask = np.ones(len(layer.data), dtype=bool)
            else:
                mask = layer_mask[layer_name]

            try:
                layer_features.append(layer.features.loc[mask, metric_name])
            except (KeyError, TypeError):
                if metric_name == "boxsize":
                    layer_features.append(
                        pd.DataFrame(get_size(layer)).loc[mask, :].mean(axis=1)
                    )
                elif metric_name == "current_boxsize":
                    # Only called when layer is empty
                    layer_features.append(
                        pd.DataFrame(get_current_size(layer))
                    )
        return pd.concat(layer_features, ignore_index=True)

    def _update_slider(self):
        invalid_labels = []
        for col_idx, label in enumerate(self.table_model.label_dict):
            if label in self.read_only:
                continue

            metric_name, _ = self.trim_suffix(label)
            try:
                labels_data = self._get_all_data(metric_name)
            except ValueError:
                invalid_labels.append(label)
                if metric_name in self.metric_dict:
                    self.metric_dict[metric_name].setParent(None)
                    self.metric_dict[metric_name].deleteLater()
                    del self.metric_dict[metric_name]
                continue

            if label in ("boxsize",):
                if metric_name in self.metric_dict:
                    viewer = self.metric_dict[metric_name]
                else:
                    viewer = EditView(
                        label,
                        col_idx,
                        QRegularExpressionValidator(
                            QRegularExpression("[0-9]*")
                        ),
                    )
                    self.metric_area.addWidget(viewer)
                    viewer.value_changed.connect(self.table_widget.update_elements)  # type: ignore
                    self.metric_dict[label] = viewer
                try:
                    viewer.set_value(int(labels_data.mean()))
                except ValueError:
                    viewer.set_value(0)
                continue

            if metric_name in self.metric_dict:
                viewer = self.metric_dict[metric_name]
            else:
                viewer = HistogramMinMaxView(metric_name, self)
                self.metric_area.addWidget(viewer)
                viewer.value_changed.connect(self.table_widget.update_elements)  # type: ignore
                self.sig_update_hist.connect(viewer.set_data)
                self.metric_dict[metric_name] = viewer

            if label.endswith("_min"):
                viewer.set_col_min(col_idx)
            elif label.endswith("_max"):
                viewer.set_col_max(col_idx)
            else:
                assert False, label
        self.table_model.remove_labels(invalid_labels)
        self.update_hist()

    def update_hist(self, *_, change_selection=True):
        rows_candidates_navigate = self.table_widget.get_row_candidates(False)
        rows_candidates = self.table_widget.get_row_candidates(
            self.global_checkbox.isChecked()
        )
        if not rows_candidates:
            for layer, _ in self.prev_valid_layers.values():
                if "prev_opacity" in layer.metadata:
                    layer.events.opacity.disconnect(self._update_opacity)
                    layer.opacity = layer.metadata["prev_opacity"]
                    layer.events.opacity.connect(self._update_opacity)
                if "prev_visible" in layer.metadata:
                    layer.events.visible.disconnect(self._update_visible)
                    layer.visible = layer.metadata["prev_visible"]
                    layer.events.visible.connect(self._update_visible)

            # Set all to 0 if nothing is selected
            metric_done = []
            if "boxsize" in self.metric_dict:
                self.metric_dict["boxsize"].setVisible(False)

            for label in self.table_model.label_dict:
                if label in self.ignore_idx:
                    continue

                metric_name, _ = self.trim_suffix(label)
                if (
                    metric_name in metric_done
                    or metric_name not in self.metric_dict
                ):
                    continue
                labels_data = pd.Series([0], dtype=int)
                self.metric_dict[metric_name].setVisible(False)
                self.metric_dict[metric_name].set_data(labels_data)
                metric_done.append(metric_name)
            return

        layer_dict = self.table_widget.get_all_rows(
            self.table_widget.get_rows(rows_candidates, 0)
        )

        layer_mask = {}
        min_max_vals = {}
        valid_layers = []
        for parent_idx, rows_idx in layer_dict.items():
            layer_name = self.table_model.get_value(-1, parent_idx, "name")
            layer = [entry for entry in self.napari_viewer.layers if entry.name == layer_name][0]  # type: ignore
            valid_layers.append(layer)
            slice_idx = list(
                map(
                    int,
                    self.table_model.get_values(parent_idx, rows_idx, "slice"),
                )
            )

            if layer.ndim == 3:
                mask = np.isin(
                    np.round(
                        general.get_identifier(layer, self._cur_slice_dim), 0
                    ),
                    slice_idx,
                )
            elif layer.ndim == 2:
                mask = np.ones(len(layer.data), dtype=bool)
            else:
                assert False, layer
            layer_mask[layer_name] = mask

            for label in self.table_model.label_dict:
                if label in self.ignore_idx:
                    continue
                vals = [
                    entry
                    for entry in self.table_model.get_values(
                        parent_idx, rows_idx, label
                    )
                    if entry not in ("-",) and entry is not None
                ]
                min_max_vals.setdefault(label, []).extend(vals)

        for layer, _ in self.prev_valid_layers.values():
            if "prev_opacity" not in layer.metadata:
                layer.metadata["prev_opacity"] = layer.opacity
            if "prev_visible" not in layer.metadata:
                layer.metadata["prev_visible"] = layer.visible
            layer.events.opacity.disconnect(self._update_opacity)
            layer.events.visible.disconnect(self._update_visible)
            if self.hide_dim.currentText() == "Selected":
                if layer in valid_layers:
                    layer.visible = True
                    layer.opacity = layer.metadata["prev_opacity"]
                else:
                    layer.visible = False
            elif self.hide_dim.currentText().startswith("All (Opacity"):
                if layer in valid_layers:
                    layer.opacity = 1
                    layer.visible = True
                else:
                    layer.opacity = float(
                        self.hide_dim.currentText()
                        .removeprefix("All (Opacity ")
                        .removesuffix(")")
                    )
                    layer.visible = layer.metadata["prev_visible"]
            elif self.hide_dim.currentText() == "All":
                layer.opacity = layer.metadata["prev_opacity"]
                layer.visible = layer.metadata["prev_visible"]
            else:
                assert False, self.hide_dim.currentText()
            layer.events.opacity.connect(self._update_opacity)
            layer.events.visible.connect(self._update_visible)

        self.napari_viewer.layers.selection.clear()
        valid_images = []
        for layer in valid_layers:
            self.napari_viewer.layers.selection.add(layer)
            try:
                image_name = layer.metadata["layer_name"]
            except KeyError:
                show_info(
                    f"Layer {layer.name} does not have an 'layer_name' entry."
                )
            else:
                valid_images.append(image_name)

        for layer in self.napari_viewer.layers:
            if not isinstance(layer, napari.layers.Image):
                continue
            elif layer.name not in valid_images:
                layer.visible = False
            else:
                layer.visible = True

        metric_done = []
        if "boxsize" in self.metric_dict:
            labels_data = self._get_all_data("boxsize", layer_mask)
            if labels_data.empty:
                labels_data = self._get_all_data("current_boxsize", layer_mask)

            if np.all(labels_data == labels_data[0]):
                self.metric_dict["boxsize"].set_value(int(labels_data[0]))
            else:
                self.metric_dict["boxsize"].set_value(-1)
            self.metric_dict["boxsize"].setVisible(True)

        for label in self.table_model.label_dict:
            if label in self.ignore_idx or (
                self.napari_viewer.dims.order[0] != 0
                and self.napari_viewer.dims.ndim == 3
            ):
                continue

            metric_name, _ = self.trim_suffix(label)
            if (
                metric_name in metric_done
                or metric_name not in self.metric_dict
            ):
                continue

            try:
                labels_data = self._get_all_data(metric_name, layer_mask)
            except ValueError:
                labels_data = pd.Series([0], dtype=int)
                self.metric_dict[metric_name].setVisible(False)
            else:
                self.metric_dict[metric_name].setVisible(True)

            if labels_data.empty:
                labels_data = pd.Series([0], dtype=int)
                self.metric_dict[metric_name].setVisible(False)
                continue

            min_val = None
            max_val = None
            if len(set(min_max_vals[f"{metric_name}_min"])) == 1:
                min_val = float(min_max_vals[f"{metric_name}_min"][0])
            if len(set(min_max_vals[f"{metric_name}_max"])) == 1:
                max_val = float(min_max_vals[f"{metric_name}_max"][0])
            self.metric_dict[metric_name].set_data(
                labels_data, min_val, max_val
            )

            metric_done.append(metric_name)

        if (
            change_selection
            and len(rows_candidates_navigate) == 1
            and list(rows_candidates_navigate)[0][0] != -1
        ):
            rows = self.table_widget.get_rows(rows_candidates_navigate, 0)
            cur_selection = list(self.table_widget.get_all_rows(rows).items())[
                0
            ]
            slice_idx = list(
                map(
                    int,
                    self.table_model.get_values(
                        cur_selection[0], cur_selection[1], "slice"
                    ),
                )
            )
            self.napari_viewer.dims.set_point(
                [self._cur_slice_dim], (slice_idx[0],)
            )


class HistogramMinMaxView(QWidget):
    value_changed = Signal(float, int)

    def __init__(self, label_name, parent=None):
        super().__init__(parent)

        self.setLayout(QVBoxLayout())
        self.col_min = -1
        self.col_max = -1
        _modes = ["Separate", "Zoom"]
        self._mode = _modes[0]
        self._label_data = None
        self._central_data = None

        self.canvas = FigureCanvas()
        self.canvas.setMaximumHeight(100)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        axis = self.canvas.figure.subplots(1, 3, sharey=True)
        self.axis_list = []
        for idx in range(3):

            line_min = axis[idx].axvline(0, color="k")
            line_max = axis[idx].axvline(0, color="orange")
            self.axis_list.append(
                {
                    "axis": axis[idx],
                    "line_min": line_min,
                    "line_max": line_max,
                }
            )

        self.slider_min = SliderView(
            QRegularExpressionValidator(
                QRegularExpression(r"-?[0-9]*\.?[0-9]*")
            ),
            self,
        )
        self.slider_min.value_changed.connect(self._handle_value_changed)
        self.slider_max = SliderView(
            QRegularExpressionValidator(
                QRegularExpression(r"-?[0-9]*\.?[0-9]*")
            ),
            self,
        )
        self.slider_max.value_changed.connect(self._handle_value_changed)

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addRow(f"{label_name} min", self.slider_min)
        layout.addRow(f"{label_name} max", self.slider_max)

        mode = QComboBox(self)
        mode.addItems(_modes)
        mode.currentTextChanged.connect(self._change_mode)

        self.layout().addLayout(layout)
        self.layout().addWidget(mode)
        self.layout().addWidget(self.canvas, stretch=0)
        self.layout().setContentsMargins(0, 0, 0, 0)

    @Slot(object)
    def _on_canvas_click(self, event):
        if event.button not in (1, 3):
            return
        value = event.xdata
        if value is None:
            return

        if event.button == 3:
            slider = self.slider_max
        elif event.button == 1:
            slider = self.slider_min
        else:
            assert False, event

        slider.set_value(value, emit_signal=True)

    @Slot(object, float, int)
    def _handle_value_changed(self, slider, value, col_idx):
        is_max = slider == self.slider_max
        if is_max:
            val = np.maximum(value, self.slider_min.value())
        else:
            val = np.minimum(value, self.slider_max.value())

        self.adjust_line(val, is_max)
        slider.set_value(val)
        self.value_changed.emit(val, col_idx)

    @Slot(str)
    def _change_mode(self, value):
        self._mode = value
        val_min, val_max = self._get_min_max()
        self.set_data(cur_val_min=val_min, cur_val_max=val_max)

    def _get_min_max(self):
        return (
            self.slider_min.value(),
            self.slider_max.value(),
        )

    def adjust_line(self, value, is_max):
        for entry in self.axis_list:
            if is_max:
                line = entry["line_max"]
            else:
                line = entry["line_min"]
            line.set_visible(True)
            line.set_data([value, value], [0, 1])

        if self._mode == "Separate":
            lower_limit, upper_limit = self.axis_list[1]["axis"].get_xlim()
            for idx, entry in enumerate(self.axis_list):
                if is_max:
                    line = entry["line_max"]
                else:
                    line = entry["line_min"]
                if lower_limit <= value <= upper_limit and idx not in (1,):
                    line.set_visible(False)

        elif self._mode == "Zoom":
            axdict = self.axis_list[0]

            data_min = np.min(self._central_data)
            data_max = np.max(self._central_data)

            upper_lim = axdict["line_max"].get_data()[0][0]
            lower_lim = axdict["line_min"].get_data()[0][0]

            if data_min - 0.01 <= upper_lim <= data_max + 0.01:
                upper_lim = np.maximum(upper_lim, data_max)

            if data_min - 0.01 <= lower_lim <= data_max + 0.01:
                lower_lim = np.minimum(lower_lim, data_min)

            margin = (upper_lim - lower_lim) * 0.05
            axdict["axis"].set_xlim(lower_lim - margin, upper_lim + margin)

            data = axdict["axis"].get_children()[0].get_xy()
            mask = (lower_lim <= data[:, 0]) & (data[:, 0] <= upper_lim)
            y_lim = np.max(data[mask, 1]) or 1

            axdict["axis"].set_ylim(0, y_lim * 1.05)

            for tick in axdict["axis"].get_yticklabels():
                tick.set_rotation(90)
                tick.set_verticalalignment("top")
                tick.set_horizontalalignment("right")

        self.canvas.draw_idle()

    def set_data(self, label_data=None, cur_val_min=None, cur_val_max=None):
        if label_data is None:
            label_data = self._label_data
        else:
            self._label_data = label_data
        val_min = general.get_min_floor(label_data.min())
        val_max = general.get_max_floor(label_data.max())
        val_min = val_min if not np.isnan(val_min) else 0
        val_max = val_max if not np.isnan(val_max) else 0
        self.slider_min.set_range(val_min, val_max)
        self.slider_max.set_range(val_min, val_max)

        if cur_val_min is None:
            cur_val_min = val_min
        if cur_val_max is None:
            cur_val_max = val_max

        self.slider_min.set_value(cur_val_min)
        self.slider_max.set_value(cur_val_max)

        if self._mode == "Separate":
            outlier = 0.05
            quantile_upper = np.quantile(label_data, 1 - outlier / 2)
            quantile_lower = np.quantile(label_data, outlier / 2)
            data_tmp = label_data[
                (quantile_lower <= label_data) & (label_data <= quantile_upper)
            ]
            if data_tmp.empty:
                data_tmp = label_data
            median = np.median(data_tmp)
            val = np.maximum(
                np.abs(np.max(data_tmp) - median),
                np.abs(np.min(data_tmp) - median),
            )

            quantile_upper += val / 2
            quantile_lower -= val / 2

            data_lower = label_data[label_data < quantile_lower]
            data_center = label_data[
                (quantile_lower <= label_data) & (label_data <= quantile_upper)
            ]
            data_upper = label_data[label_data > quantile_upper]
            data_list = [data_lower, data_center, data_upper]
            n_data = len([entry for entry in data_list if not entry.empty])
            do_idx = [
                idx for idx, entry in enumerate(data_list) if not entry.empty
            ]

            axis_idx = -1
            cum_width = 0
            for idx, entry in enumerate(self.axis_list):
                entry["axis"].autoscale()
                if data_list[idx].empty:
                    entry["axis"].set_position([0, 0, 0, 0])
                    continue
                axis_idx += 1

                entry["axis"].get_yaxis().set_visible(False)
                entry["axis"].clear()
                entry["axis"].hist(data_list[idx], 100, histtype="step")
                entry["axis"].ticklabel_format(useOffset=False, style="plain")

                if n_data == 1:
                    entry["axis"].spines["left"].set_visible(True)
                    entry["axis"].spines["right"].set_visible(True)
                elif axis_idx == 0:
                    entry["axis"].spines["left"].set_visible(True)
                    entry["axis"].spines["right"].set_visible(False)
                elif axis_idx == n_data - 1:
                    entry["axis"].spines["left"].set_visible(False)
                    entry["axis"].spines["right"].set_visible(True)
                elif 0 < axis_idx < n_data - 1:
                    entry["axis"].spines["right"].set_visible(False)
                    entry["axis"].spines["left"].set_visible(False)
                else:
                    assert False, (axis_idx, n_data)

                if n_data in (1, 2) or idx == 1:
                    n_ticks = 3
                elif n_data == 3:
                    n_ticks = 2
                else:
                    assert False, n_data

                ticks = np.round(
                    np.linspace(
                        np.min(data_list[idx]), np.max(data_list[idx]), n_ticks
                    ),
                    3,
                )
                if np.all(ticks == ticks[0]):
                    ticks[0] -= 1
                    ticks[-1] += 1
                    margin = 1.2
                else:
                    margin = (
                        np.max(data_list[idx]) - np.min(data_list[idx])
                    ) * 0.05

                entry["axis"].set_xticks(ticks)
                if n_data != 1:
                    if idx == 0:
                        min_val = np.min(data_list[idx]) - max(margin, 0.002)
                        max_val = min(
                            np.min(data_list[1]),
                            np.min(data_list[idx]) + max(margin, 0.002),
                        )
                    elif idx == 1:
                        if 0 in do_idx:
                            tmp_margin = 0
                        else:
                            tmp_margin = max(margin, 0.002)
                        min_val = np.min(data_list[idx]) - tmp_margin
                        if 2 in do_idx:
                            tmp_margin = 0
                        else:
                            tmp_margin = max(margin, 0.002)
                        max_val = np.max(data_list[idx]) + tmp_margin
                    elif idx == 2:
                        min_val = max(
                            np.max(data_list[1]),
                            np.min(data_list[idx]) - max(margin, 0.002),
                        )
                        max_val = np.max(data_list[idx]) + max(margin, 0.002)
                    else:
                        assert False, idx
                else:
                    min_val = np.min(data_list[idx]) - max(margin, 0.002)
                    max_val = np.max(data_list[idx]) + max(margin, 0.002)
                entry["axis"].set_xlim(min_val, max_val)

                if n_data != 1:
                    if n_data == 2 or idx == 1:
                        label = ["left", "center", "right"]
                    elif n_data == 3:
                        label = ["left", "right"]
                    else:
                        assert False, n_data

                    for orient, tick in zip(
                        label, entry["axis"].get_xticklabels()
                    ):
                        tick.set_rotation(-12)
                        tick.set_verticalalignment("top")
                        tick.set_horizontalalignment(orient)

                if idx == 1:
                    new_step_size = int(np.abs(ticks[-1] - ticks[0]) / 20)
                    self.slider_min.setSingleStep(new_step_size)
                    self.slider_max.setSingleStep(new_step_size)

                if n_data != 1:
                    height = 0.35
                else:
                    height = 0.25

                if n_data == 1:
                    space = 0
                    width = 0.98
                elif n_data == 2:
                    space = 0.98 - 0.3 - 0.65
                    if idx in (0, 2):
                        width = 0.3
                    elif idx == 1:
                        width = 0.65
                    else:
                        assert False, (n_data, axis_idx)
                elif n_data == 3:
                    space = (0.98 - 0.3 - 0.65) / 2
                    if idx in (0, 2):
                        width = 0.15
                    elif idx == 1:
                        width = 0.65
                    else:
                        assert False, (n_data, axis_idx)
                else:
                    assert False, (n_data, axis_idx)

                entry["axis"].add_artist(entry["line_min"])
                entry["axis"].add_artist(entry["line_max"])
                entry["axis"].set_position(
                    [0.01 + cum_width, height, width, 1 - height - 0.02]
                )
                cum_width += width + space

            self.adjust_line(cur_val_min, False)
            self.adjust_line(cur_val_max, True)

        elif self._mode == "Zoom":
            outlier = 0.05
            quantile_upper = np.quantile(label_data, 1 - outlier / 2)
            quantile_lower = np.quantile(label_data, outlier / 2)
            data_tmp = label_data[
                (quantile_lower <= label_data) & (label_data <= quantile_upper)
            ]
            if data_tmp.empty:
                data_tmp = label_data
            median = np.median(data_tmp)
            val = np.maximum(
                np.abs(np.max(data_tmp) - median),
                np.abs(np.min(data_tmp) - median),
            )

            quantile_upper += val / 2
            quantile_lower -= val / 2
            self._central_data = label_data[
                (quantile_lower <= label_data) & (label_data <= quantile_upper)
            ]
            for idx, entry in enumerate(self.axis_list):
                if idx != 0:
                    entry["axis"].set_position([0, 0, 0, 0])
                    continue

            axdict = self.axis_list[0]
            axdict["axis"].get_yaxis().set_visible(True)

            _, bins = np.histogram(self._central_data, 100)
            bin_width = bins[1] - bins[0]
            total_range = (np.max(label_data) - np.min(label_data)) / bin_width
            bins = np.arange(np.ceil(total_range) + 1) * bin_width + np.min(
                label_data
            )

            height = 0.25
            start = 0.04
            axdict["axis"].clear()
            axdict["axis"].hist(label_data, bins, histtype="step")
            axdict["axis"].spines["left"].set_visible(True)
            axdict["axis"].spines["right"].set_visible(True)
            axdict["axis"].add_artist(axdict["line_min"])
            axdict["axis"].add_artist(axdict["line_max"])
            axdict["axis"].set_position(
                [start, height, 1 - start - 0.01, 1 - height - 0.02]
            )
            self.adjust_line(cur_val_min, is_max=False)
            self.adjust_line(cur_val_max, is_max=True)

        else:
            assert False, self._mode

    def set_col_min(self, col_min):
        self.slider_min.set_col(col_min)

    def set_col_max(self, col_max):
        self.slider_max.set_col(col_max)


class EditView(QWidget):
    value_changed = Signal(float, int)

    def __init__(self, label, col_idx, validator=None, parent=None):
        super().__init__(parent)
        self.col_idx = col_idx
        self.setLayout(QHBoxLayout())
        self.edit = QLineEdit(self)
        if validator is not None:
            self.edit.setValidator(validator)
        self.edit.editingFinished.connect(self._emit_signal)
        self.layout().addWidget(QLabel(label))
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
    value_changed = Signal(object, float, int)

    def __init__(self, validator=None, parent=None):
        super().__init__(parent)
        self.col_idx = -1
        self.setLayout(QHBoxLayout())
        self.step_size = 1000

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.valueChanged.connect(self.mouse_move)
        self.slider.setRange(
            self.step_size * self.slider.minimum(),
            self.step_size * self.slider.maximum(),
        )
        self.label = QLineEdit(str(self.slider.value() / self.step_size), self)
        if validator is not None:
            self.label.setValidator(validator)
        self.label.editingFinished.connect(self.set_value)

        self.layout().addWidget(self.slider, stretch=1)
        self.layout().addWidget(self.label)

        self.layout().setContentsMargins(0, 0, 0, 0)

    def mouse_move(self, value):
        self.label.setText(str(value / self.step_size))
        self.value_changed.emit(self, value / self.step_size, self.col_idx)

    def set_value(self, value=None, emit_signal=None):
        if emit_signal is None:
            emit_signal = True if value is None else False
        value = value if value is not None else float(self.label.text())
        value = int(self.step_size * value)

        self.slider.valueChanged.disconnect(self.mouse_move)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self.mouse_move)
        if emit_signal:
            self.slider.valueChanged.emit(value)
        else:
            # Otherwise handeld by the sliderMoved event
            self.label.setText(str(value / self.step_size))

    def set_range(self, val_min, val_max):
        try:
            self.slider.valueChanged.disconnect(self.mouse_move)
        except TypeError:
            pass
        self.slider.setRange(
            int(self.step_size * val_min) - 1,
            int(self.step_size * val_max) + 1,
        )
        self.slider.valueChanged.connect(self.mouse_move)

    def value(self):
        return self.slider.value() / self.step_size

    def set_col(self, col):
        self.col_idx = col

    def setSingleStep(self, value):
        self.slider.setSingleStep(self.step_size * value)

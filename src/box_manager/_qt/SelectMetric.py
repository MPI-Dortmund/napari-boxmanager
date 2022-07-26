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


class DimensionAxis(enum.Enum):
    Z = 0
    Y = 1
    X = 2


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

        row_items = []
        for row_idx in reversed(range(root_item.rowCount())):
            text = root_item.child(row_idx, self.label_dict[label]).text()
            try:
                val = float(text)
            except ValueError:
                val = text
            row_items.append((val, root_item.takeRow(row_idx)))

        for row_idx, (_, row_items) in enumerate(sorted(row_items)):
            for col_idx, item in enumerate(row_items):
                root_item.setChild(row_idx, col_idx, item)

    def sort(self, columns):
        row_dict = {}
        root_item = self.invisibleRootItem()
        prev_status = self.blockSignals(True)
        for row_idx in reversed(range(root_item.rowCount())):
            name = root_item.child(row_idx, self.label_dict["name"]).text()
            row_dict[name] = root_item.takeRow(row_idx)

        row_idx = 0
        for col_name in reversed(columns):
            if col_name in row_dict:
                for col_idx, item in enumerate(row_dict[col_name]):
                    root_item.setChild(row_idx, col_idx, item)
                row_idx += 1
        self.blockSignals(prev_status)

        self.layoutChanged.emit()

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

        self.model.dataChanged.connect(self.update_by_edit)

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
        if col_name in self.model.read_only:
            return

        value = self.model.get_value(parent_idx, row_idx, col_name)
        self.update_elements(value, col_idx)

    def get_row_candidates(self):
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
        self._cur_slice_dim = DimensionAxis.Z.value

        self.loadable_layers = (napari.layers.Points,)
        self.read_only = [
            "",
            "identifier",
            "shown",
            "selected",
            "boxes",
            "name",
            "slice",
        ]
        self.ignore_idx = [
            "boxsize",
        ] + self.read_only

        self.napari_viewer.layers.events.reordered.connect(self._order_table)
        self.napari_viewer.layers.events.inserted.connect(self._sync_table)
        self.napari_viewer.layers.events.removed.connect(self._sync_table)
        self.napari_viewer.dims.events.order.connect(self._set_current_slice)

        self.table_model = GroupModel(self.read_only, self)
        self.table_widget = GroupView(self.table_model, self)
        self.table_widget.elementsUpdated.connect(self._update_view)
        self.table_widget.selectionModel().selectionChanged.connect(
            self.update_hist
        )
        self.metric_area = QVBoxLayout()

        self.settings_area = QHBoxLayout()
        self.hide_dim = QComboBox(self)
        self.hide_dim.currentTextChanged.connect(self.update_hist)
        self.hide_dim.addItems(["Show only", "Enhance", "Nothing"])

        self.settings_area.addWidget(QLabel("Highlight:", self))
        self.settings_area.addWidget(self.hide_dim)

        self.setLayout(QVBoxLayout())
        self.layout().addLayout(self.settings_area, stretch=0)  # type: ignore
        self.layout().addWidget(self.table_widget, stretch=1)
        self.layout().addLayout(self.metric_area, stretch=0)  # type: ignore

        self._sync_table(select_first=True)

    def _set_current_slice(self, event):
        self._cur_slice_dim = event.source.order[0]
        self._clear_table()
        self._sync_table(select_first=True)

    @Slot(dict, str)
    def _update_view(self, layer_dict, col_name):
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
            layer_vals = float(
                self.table_model.get_value(parent_idx, rows_idx[0], col_name)
            )

            if layer.data.shape[1] == 3:
                mask_dimension = np.isin(
                    np.round(layer.data[:, self._cur_slice_dim], 0), slice_idx
                )
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
                            slice_mask = (
                                np.round(layer.data[:, self._cur_slice_dim], 0)
                                == slice_idx[idx]
                            )
                        elif layer.data.shape[1] == 2:
                            slice_mask = np.ones(
                                layer.data.shape[0], dtype=bool
                            )
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
                layer.size[mask_dimension] = layer_vals
            else:
                assert False

            if do_update:
                layer.refresh()
        self._plugin_view_update = False

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
    def _order_table(self, event=None):
        valid_names: list[str] = [
            entry.name
            for entry in self.napari_viewer.layers
            if isinstance(entry, self.loadable_layers)
        ]
        self.table_model.sort(valid_names)

    @Slot(object)
    def _sync_table(self, event=None, *, select_first=False):
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

        prev_layers = [entry for entry, _ in self.prev_valid_layers.values()]
        self.prev_valid_layers = {}
        if sorted(valid_layers, key=lambda x: x.name) != sorted(
            prev_layers, key=lambda x: x.name
        ):
            for layer in prev_layers + valid_layers:
                if layer in valid_layers:
                    self._add_remove_table(layer, ButtonActions.ADD)
                    if not layer.features.empty:
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
            self._order_table()
            self._update_slider()
            if select_first:
                self.table_widget.select_first()

    def _clear_table(self):
        for layer, _ in self.prev_valid_layers.values():
            self._add_remove_table(layer, ButtonActions.DEL)
        self.prev_valid_layers = {}

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

    def _update_on_data(self, event):
        if not self._plugin_view_update:
            layer = event.source
            if not np.array_equal(
                layer.data, self.prev_valid_layers[layer.name][1]
            ):
                self.prev_valid_layers[layer.name][1] = layer.data
                self._add_remove_table(layer, ButtonActions.UPDATE)

    def _update_editable(self, event):
        layer = event.source
        if not layer.features.empty and layer.editable:
            layer.editable = False

    def _prepare_entries(self, layer, name=None) -> list:
        output_list = []
        features_copy = layer.features.copy()
        if layer.data.shape[1] == 3:
            features_copy["identifier"] = (
                ""
                if name is not None
                else np.round(layer.data[:, self._cur_slice_dim], 0).astype(
                    int
                )
            )
        elif layer.data.shape[1] == 2:
            features_copy["identifier"] = "" if name is not None else 0
        else:
            assert False, layer.data

        features_copy["shown"] = layer.shown
        for identifier, ident_df in features_copy.groupby(
            "identifier", sort=False
        ):
            try:
                cur_name = name or layer.metadata[identifier]["name"]
            except KeyError:
                cur_name = "Manual"
            output_list.append(
                self._prepare_columns(
                    pd.DataFrame(layer.size, dtype=float),
                    ident_df,
                    cur_name,
                    identifier,
                )
            )

        # Case: No points available
        if not output_list:
            identifier = 0
            try:
                cur_name = name or layer.metadata[identifier]["name"]
            except KeyError:
                cur_name = "Manual"
            features = pd.DataFrame(columns=["shown"])
            output_list.append(
                self._prepare_columns(
                    pd.DataFrame(layer.size, dtype=float),
                    features,
                    cur_name,
                    identifier,
                )
            )

        return output_list

    def _prepare_columns(self, size, features, name, slice_idx) -> dict:
        output_dict = {}
        output_dict["name"] = name
        output_dict["slice"] = str(slice_idx)
        output_dict["boxes"] = str(len(features))
        output_dict["selected"] = str(np.count_nonzero(features["shown"]))
        output_dict["boxsize"] = (
            "0" if size.empty else str(int(size.mean().mean()))
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

    def _add_remove_table(self, layer, action: "ButtonActions"):
        layer_name = layer.name

        if action == ButtonActions.ADD:
            if self.table_model.add_group(
                layer_name, self._prepare_entries(layer, layer_name)[0]
            ):
                entries = self._prepare_entries(layer)
                for entry in entries:
                    self.table_model.append_element_to_group(layer_name, entry)
                self.table_model.sort_children(layer_name, "slice")
        elif action == ButtonActions.DEL:
            self.table_model.remove_group(layer_name)
        elif action == ButtonActions.UPDATE:
            self._add_remove_table(layer, ButtonActions.DEL)
            self._add_remove_table(layer, ButtonActions.ADD)
            self._order_table()

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
                        pd.DataFrame(layer.size).loc[mask, :].mean()
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

    def update_hist(self, *_):
        rows_candidates = self.table_widget.get_row_candidates()
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
        slice_indices = []
        for parent_idx, rows_idx in layer_dict.items():
            layer_name = self.table_model.get_value(-1, parent_idx, "name")
            layer = self.napari_viewer.layers[layer_name]  # type: ignore
            valid_layers.append(layer)
            slice_idx = list(
                map(
                    int,
                    self.table_model.get_values(parent_idx, rows_idx, "slice"),
                )
            )
            slice_indices.extend(slice_idx)

            if layer.data.shape[1] == 3:
                mask = np.isin(
                    np.round(layer.data[:, self._cur_slice_dim], 0), slice_idx
                )
            elif layer.data.shape[1] == 2:
                mask = np.ones(layer.data.shape[0], dtype=bool)
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
                    if entry not in ("-",)
                ]
                min_max_vals.setdefault(label, []).extend(vals)

        for layer, _ in self.prev_valid_layers.values():
            if "prev_opacity" not in layer.metadata:
                layer.metadata["prev_opacity"] = layer.opacity
            if "prev_visible" not in layer.metadata:
                layer.metadata["prev_visible"] = layer.visible
            layer.events.opacity.disconnect(self._update_opacity)
            layer.events.visible.disconnect(self._update_visible)
            if self.hide_dim.currentText() == "Show only":
                if layer in valid_layers:
                    layer.visible = True
                    layer.opacity = layer.metadata["prev_opacity"]
                else:
                    layer.visible = False
            elif self.hide_dim.currentText() == "Enhance":
                if layer in valid_layers:
                    layer.opacity = 1
                    layer.visible = True
                else:
                    layer.opacity = 0.05
                    layer.visible = layer.metadata["prev_visible"]
            elif self.hide_dim.currentText() == "Nothing":
                layer.opacity = layer.metadata["prev_opacity"]
                layer.visible = layer.metadata["prev_visible"]
            else:
                assert False, self.hide_dim.currentText()
            layer.events.opacity.connect(self._update_opacity)
            layer.events.visible.connect(self._update_visible)

        metric_done = []
        self.metric_dict["boxsize"].setVisible(True)
        for label in self.table_model.label_dict:
            if label in self.ignore_idx:
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

        if len(set(slice_indices)) == 1:
            self.napari_viewer.dims.set_point(
                [self._cur_slice_dim], (slice_indices[0],)
            )


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
                QRegularExpression(r"-?[0-9]*\.?[0-9]*")
            ),
            self,
        )
        self.slider_min.value_changed.connect(self.value_changed.emit)
        self.slider_min.value_changed.connect(
            lambda x, _, is_max=False: self.adjust_line(x, is_max)
        )
        self.slider_max = SliderView(
            QRegularExpressionValidator(
                QRegularExpression(r"-?[0-9]*\.?[0-9]*")
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

    def set_data(self, label_data, cur_val_min=None, cur_val_max=None):
        val_min = _get_min_floor(label_data.min())
        val_max = _get_max_floor(label_data.max())
        self.slider_min.set_range(val_min, val_max)
        self.slider_max.set_range(val_min, val_max)

        if cur_val_min is None:
            cur_val_min = val_min
        if cur_val_max is None:
            cur_val_max = val_max

        self.slider_min.set_value(cur_val_min)
        self.slider_max.set_value(cur_val_max)
        self.line_min.set_data([cur_val_min, cur_val_min], [0, 1])
        self.line_max.set_data([cur_val_max, cur_val_max], [0, 1])

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

    def __init__(self, label, col_idx, validator=None, parent=None):
        super().__init__(parent)
        self.col_idx = col_idx
        self.setLayout(QHBoxLayout())
        self.edit = QLineEdit(self)
        if validator is not None:
            self.edit.setValidator(validator)
        self.edit.returnPressed.connect(self._emit_signal)
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
    value_changed = Signal(float, int)

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

        self.slider.valueChanged.disconnect(self.mouse_move)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self.mouse_move)
        if emit_signal:
            self.slider.valueChanged.emit(value)
        else:
            # Otherwise handeld by the sliderMoved event
            self.label.setText(str(value / self.step_size))

    def set_range(self, val_min, val_max):
        self.slider.valueChanged.disconnect(self.mouse_move)
        self.slider.setRange(
            int(self.step_size * val_min) - 1,
            int(self.step_size * val_max) + 1,
        )
        self.slider.valueChanged.connect(self.mouse_move)

    def set_col(self, col):
        self.col_idx = col

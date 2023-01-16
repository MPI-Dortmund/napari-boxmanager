import typing

import napari.layers
import numpy as np
from napari.utils.notifications import show_info
from qtpy.QtCore import QRegularExpression, Slot
from qtpy.QtGui import QRegularExpressionValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from .._utils import filters

if typing.TYPE_CHECKING:
    import napari


class FilterImageWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.napari_viewer = napari_viewer
        self._layer = QComboBox(self)
        self._lp_filter_resolution = QLineEdit("20", self)
        self._lp_filter_resolution.setToolTip(
            "Low-Pass filter value in angstrom. Set to 0 to disable low-pass filtering."
        )
        self._hp_filter_resolution = QLineEdit("0", self)
        self._hp_filter_resolution.setToolTip(
            "High-Pass filter value in angstrom. Set to 0 to disable high-pass filtering."
        )
        self._pixel_size = QLineEdit("-1", self)
        self._hp_filter_resolution.setToolTip("Pixel size of the image.")
        self._filter_2d = QCheckBox(self)
        self._hp_filter_resolution.setToolTip(
            "If the layer is 3D, filter slice-by-slice in z direction rather than the 3D volume."
        )
        self._show_mask = QCheckBox(self)
        self._hp_filter_resolution.setToolTip(
            "Mainly for debugging. Visualizes the filter mask."
        )
        self._run_btn = QPushButton("Run", self)

        float_validator = QRegularExpressionValidator(
            QRegularExpression(r"[0-9]*\.?[0-9]*")
        )
        self._lp_filter_resolution.setValidator(float_validator)
        self._hp_filter_resolution.setValidator(float_validator)
        self._pixel_size.setValidator(float_validator)

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.setLayout(layout)
        self.layout().addRow("Image layer:", self._layer)
        self.layout().addRow("LP resolution [A]:", self._lp_filter_resolution)
        self.layout().addRow("HP resolution [A]:", self._hp_filter_resolution)
        self.layout().addRow("Pixel size [A/px]:", self._pixel_size)
        self.layout().addRow("Live filter 2D slices", self._filter_2d)
        self.layout().addRow("Show mask", self._show_mask)
        self.layout().addRow("", self._run_btn)

        self._layer.currentTextChanged.connect(self._update_pixel_size)
        self._run_btn.clicked.connect(self._run)
        self.napari_viewer.layers.events.inserted.connect(self._update_combo)
        self.napari_viewer.layers.events.removed.connect(self._update_combo)

        self._update_combo()

    @Slot()
    def _run(self):
        for attr in [
            "pixel_size",
            "lp_filter_resolution",
            "hp_filter_resolution",
        ]:
            try:
                if getattr(self, attr) < 0:
                    show_info(f"Error: {attr} needs to be larger than 0")
                    return None
            except ValueError:
                show_info(f"Error: {attr} needs to be provided")
                return None

        if not self._layer.currentText():
            show_info("No layer selected")
            return None

        kwargs = {
            "lp_filter_resolution_ang": self.lp_filter_resolution,
            "hp_filter_resolution_ang": self.hp_filter_resolution,
            "pixel_size": self.pixel_size,
        }
        filtered_image, mask = self.handle_filter(
            self.layer,
            self.filter_2d,
            **kwargs,
        )
        if filtered_image is None:
            return None

        # self.layer.visible = False
        if self.show_mask:
            self.napari_viewer.add_image(
                mask,
                name=f"MASK LP {int(self.lp_filter_resolution)} HP {int(self.hp_filter_resolution)} AP {self.pixel_size} - {self.layer.name}",
            )

        image = napari.layers.Image(
            filtered_image,
            name=f"LP {int(self.lp_filter_resolution)} HP {int(self.hp_filter_resolution)} AP {self.pixel_size} LIVE {self.filter_2d}  - {self.layer.name}",
            metadata=self.layer.metadata,
        )
        image.contrast_limits_range = [
            filtered_image.min(),
            filtered_image.max(),
        ]
        image.contrast_limits = [-3, 3]

        self.napari_viewer.layers.insert(
            self.napari_viewer.layers.index(self.layer) + 1, image
        )

        if self.filter_2d:
            connect2 = self.napari_viewer.dims.events.current_step.connect(
                lambda *x, new_layer=image, old_layer=self.layer, filter_kwargs=kwargs: self._filter_layer(
                    new_layer=new_layer, old_layer=old_layer, **filter_kwargs
                )
            )
            connect1 = image.events.visible.connect(
                lambda *x, new_layer=image, old_layer=self.layer, filter_kwargs=kwargs: self._filter_layer(
                    new_layer=new_layer, old_layer=old_layer, **filter_kwargs
                )
            )
            image.connect1 = connect1
            image.connect2 = connect2

    def _filter_layer(self, new_layer, old_layer, **filter_kwargs):
        if (
            new_layer not in self.napari_viewer.layers
            or old_layer not in self.napari_viewer.layers
        ):
            new_layer.events.disconnect(new_layer.connect1)
            self.napari_viewer.dims.events.current_step.disconnect(
                new_layer.connect2
            )
            return

        if new_layer.visible:
            filtered_image, _ = self.handle_filter(
                old_layer,
                True,
                **filter_kwargs,
            )
            new_layer.data[...] = filtered_image
            new_layer.contrast_limits_range = [
                filtered_image.min(),
                filtered_image.max(),
            ]
            new_layer.contrast_limits = [
                -3,
                3,
            ]
            with new_layer.events.visible.blocker():
                new_layer.visible = True

    def handle_filter(self, layer, filter_2d, **kwargs):
        if filter_2d:
            slice_axis = self.napari_viewer.dims.order[0]
            slice_index = self.napari_viewer.dims.current_step[slice_axis]
            slc = [slice(None)] * len(layer.data.shape)
            slc[slice_axis] = slice_index
            data = layer.data[tuple(slc)]
        else:
            data = layer.data
        try:
            filtered_image, mask = filters.bandpass_filter(
                data,
                log=show_info,
                **kwargs,
            )
            filtered_image = (
                filtered_image - np.mean(filtered_image)
            ) / np.std(filtered_image)
        except TypeError:
            return None, None

        return filtered_image, mask

    @Slot(str)
    def _update_pixel_size(self, layer_name):
        try:
            layer = self.napari_viewer.layers[layer_name]
        except KeyError:
            return

        try:
            pixel_spacing = layer.metadata["pixel_spacing"]
        except KeyError:
            self.pixel_size = -1
        else:
            self.pixel_size = pixel_spacing

        try:
            is_2d_stacked = layer.metadata["is_2d_stack"]
        except KeyError:
            self.filter_2d = False
            self._filter_2d.setEnabled(False)
        else:
            self.filter_2d = is_2d_stacked
            self._filter_2d.setEnabled(is_2d_stacked)

    @Slot(object)
    def _update_combo(self, *_):
        layer_names = sorted(
            entry.name
            for entry in self.napari_viewer.layers
            if isinstance(entry, napari.layers.Image)
        )
        current_text = self._layer.currentText()

        self._layer.currentTextChanged.disconnect(self._update_pixel_size)
        self._layer.clear()
        self._layer.addItems(layer_names)
        self._layer.setCurrentText(current_text)
        self._layer.currentTextChanged.connect(self._update_pixel_size)
        self._layer.currentTextChanged.emit(self._layer.currentText())

    @property
    def layer(self):
        return self.napari_viewer.layers[self._layer.currentText()]

    @property
    def show_mask(self):
        return self._show_mask.isChecked()

    @property
    def lp_filter_resolution(self):
        return float(self._lp_filter_resolution.text())

    @property
    def hp_filter_resolution(self):
        return float(self._hp_filter_resolution.text())

    @property
    def pixel_size(self):
        return float(self._pixel_size.text())

    @pixel_size.setter
    def pixel_size(self, value):
        self._pixel_size.setText(str(value))

    @property
    def filter_2d(self):
        return self._filter_2d.isChecked()

    @filter_2d.setter
    def filter_2d(self, value):
        return self._filter_2d.setChecked(value)

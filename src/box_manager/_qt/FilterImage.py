import typing

import napari.layers
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
        self._lp_filter_resolution = QLineEdit("30", self)
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

        self.setLayout(QFormLayout())
        self.layout().addRow("Label:", self._layer)
        self.layout().addRow("lp resolution / A:", self._lp_filter_resolution)
        self.layout().addRow("hp resolution / A:", self._hp_filter_resolution)
        self.layout().addRow("pixel size / A/px:", self._pixel_size)
        self.layout().addRow("Filter 2d slices", self._filter_2d)
        self.layout().addRow("show mask", self._show_mask)
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

        try:
            filtered_image, mask = filters.bandpass_filter(
                self.layer.data,
                self.lp_filter_resolution,
                self.hp_filter_resolution,
                self.pixel_size,
                self.filter_2d,
                log=show_info,
            )
        except TypeError:
            return None

        self.layer.visible = False
        if self.show_mask:
            self.napari_viewer.add_image(
                mask,
                name=f"MASK LP {int(self.lp_filter_resolution)} HP {int(self.hp_filter_resolution)} - {self.layer.name}",
            )

        self.napari_viewer.add_image(
            filtered_image,
            name=f"LP {int(self.lp_filter_resolution)} HP {int(self.hp_filter_resolution)} - {self.layer.name}",
            metadata=self.layer.metadata,
        )

    @Slot(str)
    def _update_pixel_size(self, layer_name):
        try:
            layer = self.napari_viewer.layers[layer_name]
        except KeyError:
            return

        try:
            pixel_spacing = layer.metadata["pixel_spacing"]
        except KeyError:
            pass
        else:
            self.pixel_size = pixel_spacing

        try:
            is_3d = layer.metadata["is_3d"]
        except KeyError:
            pass
        else:
            self.filter_2d = not is_3d

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
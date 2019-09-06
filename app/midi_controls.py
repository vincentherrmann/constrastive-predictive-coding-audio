import qtpy.QtWidgets
import qtpy.QtCore
import qtpy.QtGui


class AbstractMidiSlider(qtpy.QtWidgets.QWidget):
    def __init__(self, label, parent=None, min=0., max=1., default=None, step_size=0.01, command=None):
        super().__init__(parent=parent)
        self.layout = qtpy.QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.label = qtpy.QtWidgets.QLabel(label)
        self.label.setAlignment(qtpy.QtCore.Qt.AlignHCenter)

        self.min = min
        self.max = max
        self.step_size = step_size

        if default is None:
            default = min
        self.default = default

        self.number = qtpy.QtWidgets.QLineEdit(str(self.default))
        self.number.setFixedWidth(40)
        self.number.setAlignment(qtpy.QtCore.Qt.AlignLeft)
        self.number.setValidator(qtpy.QtGui.QDoubleValidator(decimals=4))
        self.number.editingFinished.connect(self.text_changed)

        self.command = command
        self._midi_controller = None
        self._controller_number = 0

    def value_changed(self, *args):
        self.number.setText(str(self.value()))
        if self._midi_controller is not None:
            midi_value = int(127 * self.normalized_value())
            self._midi_controller.send_control_message(self._controller_number, midi_value)
        if self.command is not None:
            self.command()

    def text_changed(self, *args):
        value = float(self.number.text())
        self.set_value(value)

    def set_value(self, value):
        if self.command is not None:
            self.command()

    def set_midi_value(self, midi_value):
        value = (midi_value / 127) * (self.max - self.min) + self.min
        self.set_value(value)

    def normalized_value(self):
        return (self.value() - self.min) / (self.max - self.min)

    def value(self):
        raise NotImplementedError


class MidiSlider(AbstractMidiSlider):
    def __init__(self, label, parent=None, min=0., max=1., default=None, step_size=0.01, command=None, length=100):
        super().__init__(label, parent=parent, min=min, max=max, default=default, step_size=step_size, command=command)

        self.slider = qtpy.QtWidgets.QSlider()
        self.slider.setMinimum(self.min / step_size)
        self.slider.setMaximum(self.max / step_size)
        self.slider.setSingleStep(1)
        self.slider.setValue(int(self.default / self.step_size))
        self.slider.valueChanged.connect(self.value_changed)
        self.slider.setMinimumHeight(length)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider, alignment=qtpy.QtCore.Qt.AlignHCenter)
        self.layout.addWidget(self.number, alignment=qtpy.QtCore.Qt.AlignHCenter)

    def set_value(self, value):
        val = int(value / self.step_size)
        self.slider.setValue(val)
        super().set_value(val)

    def value(self):
        return self.slider.value() * self.step_size


class MidiDial(AbstractMidiSlider):
    def __init__(self, label, parent=None, min=0., max=1., default=None, step_size=0.01, command=None, length=100):
        super().__init__(label, parent=parent, min=min, max=max, default=default, step_size=step_size, command=command)

        self.dial = qtpy.QtWidgets.QDial()
        self.dial.setMinimum(self.min / step_size)
        self.dial.setMaximum(self.max / step_size)
        self.dial.setSingleStep(1)
        self.dial.setValue(int(self.default / self.step_size))
        self.dial.valueChanged.connect(self.value_changed)
        self.dial.setFixedSize(50, 50)
        self.setFixedHeight(100)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.dial, alignment=qtpy.QtCore.Qt.AlignHCenter)
        self.layout.addWidget(self.number, alignment=qtpy.QtCore.Qt.AlignHCenter)

    def set_value(self, value):
        val = int(value / self.step_size)
        self.dial.setValue(val)
        super().set_value(val)

    def value(self):
        return self.dial.value() * self.step_size


class MidiButton(qtpy.QtWidgets.QPushButton):
    def __init__(self, label, parent=None, command=None, width=60):
        super().__init__(parent, text=label)
        self.command = command
        self._midi_controller = None
        self._controller_number = 0
        self.setFixedWidth(width)
        self.pressed.connect(self.value_changed)

    def value_changed(self, *args):
        if self._midi_controller is not None:
            self._midi_controller.send_control_message(self._controller_number, 0)
        if self.command is not None:
            self.command()

    def set_midi_value(self, midi_value):
        if self._midi_controller is not None:
            self._midi_controller.send_control_message(self._controller_number, 0)
        if self.command is not None:
            self.command()


class MidiSwitch(qtpy.QtWidgets.QCheckBox):
    def __init__(self, label, parent=None, command=None, width=60):
        super().__init__(parent, text=label)
        self.command = command
        self._midi_controller = None
        self._controller_number = 0
        self.setFixedWidth(width)
        self.stateChanged.connect(self.value_changed)

    def value_changed(self, *args):
        if self._midi_controller is not None:
            val = 127 if self.value() else 0
            self._midi_controller.send_control_message(self._controller_number, val)
        if self.command is not None:
            self.command()

    def set_midi_value(self, midi_value):
        self.setCheckState(midi_value > 0)
        if self.command is not None:
            self.command()

    def value(self):
        return self.isChecked()


class MidiListSelect(qtpy.QtWidgets.QListWidget):
    def __init__(self, label, parent=None, command=None, items=[], size=(80, 40)):
        self.label = label
        super().__init__(parent)
        self.command = command
        for item in items:
            self.addItem(qtpy.QtWidgets.QListWidgetItem(item))

        self.num_items = len(items)
        self.itemChanged.connect(self.value_changed)
        self.setMinimumSize(size[0], size[1])
        self.setFixedWidth(size[0])
        self.setCurrentRow(0)

    def value_changed(self, *args):
        if self._midi_controller is not None:
            midi_value = int(127 * self.value() * self.num_items)
            self._midi_controller.send_control_message(self._controller_number, midi_value)
        if self.command is not None:
            self.command()

    def set_midi_value(self, midi_value):
        row = int(self.num_items * midi_value / 127)
        self.set_value(row)

    def set_value(self, value):
        self.setCurrentRow(value)
        if self.command is not None:
            self.command()

    def value(self):
        return self.currentRow()


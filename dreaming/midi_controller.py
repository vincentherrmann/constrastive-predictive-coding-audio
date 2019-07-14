import mido
from mido import Message
from threading import Thread, Lock
import tkinter as tk
import time

midi_controller_mapping = {
    # fader
    'learning rate': 81,
    'mix original': 82,
    'channel': 83,
    'channel region': 84,
    'pitch': 85,
    'pitch region': 86,
    'time': 87,
    'time region': 88,
    # rotary encoder
    'max agg': 1,
    'original soundclips': 2,
    'activation selection': 3,
    'time jitter': 4,
    'activation loss': 5,
    'eq lows': 6,
    'eq mids': 7,
    'eq highs': 8,
    # switches
    'keep targets': 75
}


class MidiController:
    def __init__(self, port, control_mapping):
        self.inport = mido.open_input(port)
        self.outport = mido.open_output(port)

        self.lock = Lock()
        self.message_log = {}

        self.receive_thread = Thread(name='midi receive', target=self.receive_messages)
        self.receive_thread.daemon = True
        self.receive_thread.start()

        self.set_values_thread = Thread(name='midi set', target=self.set_values)
        self.set_values_thread.daemon = True
        self.set_values_thread.start()

        self.control_mapping = control_mapping
        if control_mapping is not None:
            for key, control in control_mapping.items():
                control._midi_controller = self
                control._controller_number = key
                control.value_changed()

    def receive_messages(self):
        while True:
            msg = self.inport.receive()
            with self.lock:
                self.message_log[msg.control] = msg.value

    def set_values(self):
        while True:
            self.lock.acquire()
            for control, value in list(self.message_log.items()):
                del self.message_log[control]
                self.lock.release()
                try:
                    target = self.control_mapping[control]
                except KeyError:
                    print("control", control, "has no target")
                target.set_value(value)
                self.lock.acquire()
            self.lock.release()

    def send_control_message(self, control, value, channel=0):
        self.outport.send(Message('control_change', channel=channel, control=control, value=value))


class MidiSlider(tk.Frame):
    def __init__(self, parent, label, min=0., max=1., default=None, resolution=0.01, length=200, command=None):
        super().__init__(parent)

        if default is None:
            default = min
        self.name = label
        self.label = tk.Label(self, text=label).grid(row=0, column=0)
        self.var = tk.DoubleVar(value=default)
        self.scale = tk.Scale(self, variable=self.var, from_=max, to=min, resolution=resolution, length=length,
                              command=self.value_changed)
        self.scale.grid(row=1, column=0)
        self.min = min
        self.range = max-min

        self.command = command
        self._midi_controller = None
        self._controller_number = 0

    def value_changed(self, *args):
        if self._midi_controller is not None:
            midi_value = int(127 * (self.get_value() - self.min) / self.range)
            self._midi_controller.send_control_message(self._controller_number, midi_value)
        if self.command is not None:
            self.command()

    def set_value(self, midi_value):
        value = (midi_value / 127) * self.range + self.min
        self.var.set(value)
        if self.command is not None:
            self.command()

    def get_value(self):
        return self.var.get()


class MidiListbox(tk.Frame):
    def __init__(self, parent, label, elements=[], default_index=0, width=20, height=10):
        super().__init__(parent)

        self.name = label
        self.label = tk.Label(self, text=label).grid(row=0, column=0)
        self.listbox = tk.Listbox(self, width=width, height=height)
        for element in elements:
            self.listbox.insert(tk.END, element)
        self.listbox.activate(default_index)
        self.listbox.select_set(default_index)
        self.listbox.grid(row=1, column=0)

        self._midi_controller = None
        self._controller_number = 0

    def value_changed(self):
        if self._midi_controller is not None:
            midi_value = int(128 * self.get_value() / self.listbox.size())
            self._midi_controller.send_control_message(self._controller_number, midi_value)

    def set_value(self, midi_value):
        idx = int((midi_value / 128) * self.listbox.size())
        self.selection_clear()
        self.listbox.activate(idx)
        self.listbox.select_set(idx)
        self.listbox.see(idx)

    def get_value(self):
        try:
            return self.listbox.curselection()[0]
        except:
            return 0


class MidiSwitch(tk.Frame):
    def __init__(self, parent, label, default=0, command=None):
        super().__init__(parent)

        self.name = label
        self.var = tk.IntVar(value=default)
        self.checkbutton = tk.Checkbutton(self, variable=self.var, text=label, command=self.value_changed)
        self.checkbutton.grid(row=0, column=0)

        self.command = command
        self._midi_controller = None
        self._controller_number = 0

    def value_changed(self, *args):
        if self._midi_controller is not None:
            midi_value = 127 * self.get_value()
            self._midi_controller.send_control_message(self._controller_number, midi_value)
        if self.command is not None:
            self.command()

    def set_value(self, midi_value):
        value = midi_value // 127
        self.var.set(value)
        if self.command is not None:
            self.command()

    def get_value(self):
        return self.var.get()



if __name__ == '__main__':
    controller = MidiController('BCF2000')
    for i in range(127):
        controller.send_control_message(82, i)
        time.sleep(0.1)

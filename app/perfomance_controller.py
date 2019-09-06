from app.control_app import ControlApp
from app.control_utilities import ModelActivations
from dreaming.midi_controller import MidiController
from dreaming.streaming import *

import numpy as np
import subprocess
import pickle


class PerformanceController:
    def __init__(self, model_path, activation_shapes_path):
        self.model_path = model_path
        self.activation_shapes_path = activation_shapes_path

    def setup_servers(self):
        self.viz_server = SocketDataExchangeServer(port=8001,
                                                   host='127.0.0.1',
                                                   stream_automatically=True)

    def wait_for_clients(self):
        if not self.viz_server(b'abc', timeout=None):
            raise Exception("No connection established")
        else:
            print("viz client online")

    def start_visualization(self):
        subprocess.Popen(['python', 'visualization_script.py'])

    def start_control_app(self):
        try:
            midi_controller = MidiController('BCF2000')
        except IOError:
            print("midi controller not found")
            midi_controller = None
        activations = ModelActivations(self.activation_shapes_path,
                                       ignore_time_dimension=True,
                                       remove_results=True)
        self.control_app = ControlApp(model_activations=activations,
                                      midi_controller=midi_controller,
                                      midi_mapping_function=self.midi_mapping,
                                      viz_communicator=self.viz_server)

    def change_target(self):
        selected_layer = self.control_app.layer_names[self.control_app.layer_selection_list.value()]
        activation_selection_dict = {
            'layer': selected_layer,
            'channel': self.control_app.target_channel.value(),
            'channel_region': self.control_app.target_channel_region.value(),
            'pitch': self.control_app.target_pitch.value(),
            'pitch_region': self.control_app.target_pitch_region.value(),
            'time': self.control_app.target_time.value(),
            'time_region': self.control_app.target_time_region.value(),
            'keep_selection': self.control_app.layer_selection_toggle.value()
        }
        selection_pickle = pickle.dumps(activation_selection_dict)
        if self.viz_server is not None:
            self.viz_server.set_new_data(selection_pickle)
        self.control_app.model_activations.select_activations(activation_selection_dict)
        self.control_app.target_visualizer.focus = self.control_app.model_activations.focus
        self.control_app.target_visualizer.update()

    @staticmethod
    def midi_mapping(c):
        mapping = {
            # fader
            81: c.learning_rate_slider,
            82: c.original_clip_mix,
            83: c.target_channel,
            84: c.target_channel_region,
            85: c.target_pitch,
            86: c.target_pitch_region,
            87: c.target_time,
            88: c.target_time_region,
            # dials
            2: c.original_clip_list,
            3: c.layer_selection_list,
            4: c.time_jitter_dial,
            5: c.activation_loss_dial,
            6: c.lows_dial,
            7: c.mids_dial,
            8: c.highs_dial,
            # switches
            75: c.layer_selection_toggle,
            # buttons
            66: c.original_clip_button,
            67: c.layer_selection_button
        }
        return mapping

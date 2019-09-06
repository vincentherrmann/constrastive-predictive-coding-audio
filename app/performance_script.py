from app.control_app import ControlApp
from app.control_utilities import ModelActivations
from dreaming.midi_controller import MidiController
from dreaming.streaming import *

import numpy as np
import multiprocessing as mp
import subprocess


def define_midi_control_mapping(c):
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


control_viz_client = SocketDataExchangeServer(port=8001,
                                              host='127.0.0.1',
                                              stream_automatically=True)

subprocess.Popen(['python', 'visualization_script.py'])

if not control_viz_client.check_receive(b'abc', timeout=None):
    raise Exception("No connection established")
else:
    print("test bytes received")

path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/activation_shapes.pickle'
activations = ModelActivations(path, ignore_time_dimension=True, remove_results=True)
try:
    midi_controller = MidiController('BCF2000')
except IOError:
    print("midi controller not found")
    midi_controller = None
control_app = ControlApp(model_activations=activations,
                         midi_controller=midi_controller,
                         midi_mapping_function=define_midi_control_mapping,
                         viz_communicator=control_viz_client)
# app.run()
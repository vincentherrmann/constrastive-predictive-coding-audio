import pickle
import torch
import numpy as np
from collections import OrderedDict

activation_selection_dict = {
    'layer': 'scalogram',
    'channel': None,
    'channel_region': None,
    'pitch': None,
    'pitch_region': None,
    'time': None,
    'time_region': None,
    'keep_selection': False
}

class ModelActivations:
    def __init__(self, shapes_path, ignore_time_dimension=False, remove_results=False):
        with open(shapes_path, 'rb') as handle:
            self.shapes = pickle.load(handle)

        if remove_results:
            del self.shapes['z_code']
            del self.shapes['c_code']
            del self.shapes['prediction']

        self.ignore_time_dimensions = ignore_time_dimension
        self.num_activations = 0
        self.layer_starts = OrderedDict()
        for k, v in self.shapes.items():
            if self.ignore_time_dimensions and len(v) > 1:
                v = v[:-1]
                self.shapes[k] = v
            self.layer_starts[k] = self.num_activations
            self.num_activations += np.prod(v)

        self.focus = np.zeros(self.num_activations, dtype=np.bool)

    def select_activations(self, sel=activation_selection_dict):
        if not sel['keep_selection']:
            self.focus = np.zeros(self.num_activations, dtype=np.bool)
        shape = self.shapes[sel['layer']]
        focus = np.zeros(shape, np.bool)

        channel_dim, pitch_dim, time_dim = None, None, None
        if len(shape) == 3:
            channel_dim = 0
            pitch_dim = 1
            time_dim = 2
        elif self.ignore_time_dimensions:
            if len(shape) == 2:
                channel_dim = 0
                pitch_dim = 1
            else:
                channel_dim = 0
        else:
            if len(shape) == 2:
                channel_dim = 0
                time_dim = 1
            else:
                time_dim = 0

        if pitch_dim is not None:
            pitch = sel['pitch']
            pitch_region = sel['pitch_region']
            if pitch is None:
                pitch = 0.
            if pitch_region is None:
                pitch_region = 1.
            num_pitch = shape[pitch_dim]
            pitch_start = int(pitch * (1 - pitch_region) * num_pitch)
            pitch_end = int(pitch_start + pitch_region * num_pitch + 1)

        if channel_dim is not None:
            channel = sel['channel']
            channel_region = sel['channel_region']
            if channel is None:
                channel=0.
            if channel_region is None:
                channel_region = 1.
            num_channels = shape[channel_dim]
            channel_start = int(channel * (1 - channel_region) * num_channels)
            channel_end = int(channel_start + channel_region * num_channels + 1)

        if time_dim is not None:
            time = sel['time']
            time_region = sel['time_region']
            if time is None:
                time=0.
            if time_region is None:
                time_region = 1.
            num_time = shape[time_dim]
            time_start = int(time * (1 - time_region) * num_time)
            time_end = int(time_start + time_region * num_time + 1)

        if len(shape) == 3:
            focus[channel_start:channel_end, pitch_start:pitch_end, time_start:time_end] = True
        elif self.ignore_time_dimensions:
            if len(shape) == 2:
                focus[channel_start:channel_end, pitch_start:pitch_end] = True
            else:
                focus[channel_start:channel_end] = True
        else:
            if len(shape) == 2:
                focus[channel_start:channel_end, time_start:time_end] = True
            else:
                focus[time_start:time_end] = True

        l = np.prod(shape)
        o = self.layer_starts[sel['layer']]
        self.focus[o:o+l] += focus.flatten()


if __name__ == '__main__':
    path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/activation_shapes.pickle'
    activations = ModelActivations(path, ignore_time_dimension=True, remove_results=True)
    activations.select_activations('scalogram', pitch=0.2, pitch_region=0.5)
    pass
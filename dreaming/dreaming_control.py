import tkinter as tk
import pickle
import collections
import numpy as np
import pyaudio
import threading
import multiprocessing as mp
import time
import math
import cv2
import glob
import os.path
from PIL import Image, ImageTk, ImageChops
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from colorcet import fire
from collections import OrderedDict

from dreaming.streaming import SocketDataExchangeClient
from dreaming.visualization_functions import *
from dreaming.dreaming_functions import *
from dreaming.audio_output import AudioLoop
from dreaming.midi_controller import *

activations = [
    "prediction",
    "scalogram",
    "scalogram_block_0_main_conv_2",
    "scalogram_block_1_main_conv_1",
    "scalogram_block_0_main_conv_1",
    "scalogram_block_1_main_conv_2",
    "scalogram_block_2_main_conv_1",
    "scalogram_block_2_main_conv_2",  																												 # length: 64000/1024 = 62.5
    "scalogram_block_3_main_conv_1",
    "scalogram_block_3_main_conv_2",
    "ar_block_0",
    "ar_block_1",
    "ar_block_2",
    "ar_block_3",
    "ar_block_4",
    "ar_block_5",
]

defaults_dict = {
    'port': 2222,
    'host': '127.0.0.1',
    'audio_device': 'Soundflower (2ch)',
    #'audio_device': 'Saffire',
    #'audio_device': 'Built-in Output',
    #'audio_device': 'BW-FYE2  R',
    #'audio_device': 'MADIface USB (23628221)',
    #'audio_device': 'default',
    'lr': -3.,
    'time_jitter': 0.1,
    'noise_loss': 0.,
    'activation_loss': 0.,
    'time_masking': 0.1,
    'pitch_masking': 0.1,
    'batch_size': 8,
    'default_sound_clip': 12,
    'max_agg': 5.,
    'num_frames': 32,
    'sync_timing': 2.8,
    'activation_names': activations,
    'audio_clip': 'silence.wav',
    'selected_layer_name': 'scalogram_block_0_main_conv_1',
    'activation_statistics': '../data_statistics_snapshots_model_2019-05-20_run_0_100000.pickle',
    'layout': '/Users/vincentherrmann/Documents/Projekte/Immersions/visualization/layouts/e25_version_3/e25_positions_interp_100.npy',
    'hues': '/Users/vincentherrmann/Documents/Projekte/Immersions/visualization/layouts/e25_version_3/e25_positions_interp_100_hues.p',
    'connections': '/Users/vincentherrmann/Documents/Projekte/Immersions/visualization/layouts/e25_version_3/e25_positions_interp_100_edges_weighted.p'
}

control_dict = {
    'selected_regions': [],
    'activation_loss': 0,
    'lr': defaults_dict['lr'],
    'time_jitter': defaults_dict['time_jitter'],
    'noise_loss': defaults_dict['noise_loss'],
    'time_masking': defaults_dict['time_masking'],
    'pitch_masking': defaults_dict['pitch_masking'],
    'batch_size': defaults_dict['batch_size'],
    'mix_o': 0.,
    'eq_bands': None
}

range_dict = {
    "scalogram": [10, 500],
    "scalogram_block_0_main_conv_1": (5, 250),
    "scalogram_block_0_main_conv_2": (5, 250),
    "scalogram_block_1_main_conv_1": (2, 125),
    "scalogram_block_1_main_conv_2": (2, 125),
    "scalogram_block_2_main_conv_1": (1, 63),
    "scalogram_block_2_main_conv_2": (1, 63),
    "scalogram_block_3_main_conv_1": (0, 63),
    "scalogram_block_3_main_conv_2": (0, 63),
}

audio_clips = OrderedDict(
    [(os.path.basename(path), path) for path in sorted(glob.glob('loops_16khz/*.wav'))]
)

audio_clip_names = list(audio_clips.keys())


class DreamingControlApp:
    def __init__(self, defaults=defaults_dict):
        self.defaults = defaults
        self.sample_rate = 16000
        self.num_eq_bands = 257
        self.num_frames = 100
        self.band_factors = None
        self.next_loop_start_time = 0.
        self.current_time = 0.
        self.black_screen = False
        self.audio_clip = defaults['audio_clip']
        self.select_layer_name = defaults['selected_layer_name']

        self.root = tk.Tk()
        self.root.title('Immersions')

        with open(defaults['activation_statistics'], 'rb') as handle:
            self.activation_statistics = pickle.load(handle)

        self.layout = torch.from_numpy(np.load(defaults['layout']))

        self.edge = 0.02

        self.x_min = self.layout[:, :, 0].min().item()
        self.x_max = self.layout[:, :, 0].max().item()
        self.y_min = self.layout[:, :, 1].min().item()
        self.y_max = self.layout[:, :, 1].max().item()
        self.x_edge = (self.x_max - self.x_min) * self.edge
        self.y_edge = (self.y_max - self.y_min) * self.edge

        self.activation_names = defaults['activation_names']

        # streaming frame
        self.streaming_frame = tk.Frame(self.root)
        self.streaming_frame.pack()

        tk.Label(self.streaming_frame, text='port:').grid(row=0, column=0)
        self.port_entry = tk.Entry(self.streaming_frame)
        self.port_entry.insert(0, str(defaults['port']))
        self.port_entry.grid(row=0, column=1)

        tk.Label(self.streaming_frame, text='host:').grid(row=1, column=0)
        self.host_entry = tk.Entry(self.streaming_frame)
        self.host_entry.insert(0, str(defaults['host']))
        self.host_entry.grid(row=1, column=1)

        tk.Button(self.streaming_frame, text=' start stream ').grid(row=2, column=1)
        self.start_audio_button = tk.Button(self.streaming_frame, text=' start_audio ', command=self.start_stop_audio)
        self.start_audio_button.grid(row=2, column=0)

        self.start_audio_button = tk.Button(self.streaming_frame, text=' start performance ', command=self.start_performance)
        self.start_audio_button.grid(row=3, column=0)

        self.end_performance_button = tk.Button(self.streaming_frame, text=' end performance ',
                                            command=self.end_performance)
        self.end_performance_button.grid(row=4, column=0)

        # dreaming frame
        # dreaming frame
        self.dreaming_frame = tk.Frame(self.root)
        self.dreaming_frame.pack()

        current_column = 0

        self.lr_slider = MidiSlider(self.dreaming_frame, 'learning rate', default=defaults['lr'],
                                    min=-5., max=2., resolution=0.1, length=200., command=self.send_control_dict)
        self.lr_slider.grid(row=0, column=current_column)

        current_column += 1

        self.time_jitter_slider = MidiSlider(self.dreaming_frame, 'time jitter', default=defaults['time_jitter'],
                                    min=0., max=4., resolution=0.1, length=200., command=self.send_control_dict)
        self.time_jitter_slider.grid(row=0, column=current_column)

        current_column += 1

        self.noise_loss_slider = MidiSlider(self.dreaming_frame, 'noise loss', default=defaults['noise_loss'],
                                             min=0., max=1., resolution=0.01, length=200.,
                                             command=self.send_control_dict)
        self.noise_loss_slider.grid(row=0, column=current_column)

        current_column += 1

        self.activation_loss_slider = MidiSlider(self.dreaming_frame, 'activation loss', default=defaults['activation_loss'],
                                             min=0., max=50., resolution=0.1, length=200.,
                                             command=self.send_control_dict)
        self.activation_loss_slider.grid(row=0, column=current_column)

        current_column += 1

        self.time_masking_slider = MidiSlider(self.dreaming_frame, 'time masking', default=defaults['time_masking'],
                                             min=0., max=1., resolution=0.01, length=200.,
                                             command=self.send_control_dict)
        self.time_masking_slider.grid(row=0, column=current_column)

        current_column += 1

        self.pitch_masking_slider = MidiSlider(self.dreaming_frame, 'pitch masking', default=defaults['pitch_masking'],
                                             min=0., max=1., resolution=0.01, length=200.,
                                             command=self.send_control_dict)
        self.pitch_masking_slider.grid(row=0, column=current_column)

        current_column += 1

        self.batch_size_slider = MidiSlider(self.dreaming_frame, 'batch_size', default=defaults['batch_size'],
                                             min=0., max=64., resolution=1., length=200.,
                                             command=self.send_control_dict)
        self.batch_size_slider.grid(row=0, column=current_column)

        current_column += 1

        self.sound_selection_frame = tk.Frame(self.dreaming_frame)
        self.soundclip_box = MidiListbox(self.sound_selection_frame, 'original soundclips', elements=audio_clips.keys(),
                                         default_index=defaults['default_sound_clip'], width=20, height=11)
        self.soundclip_box.grid(row=0, column=0)

        self.soundclip_button = MidiButton(self.sound_selection_frame, 'select clip', command=self.set_audio_clip)
        self.soundclip_button.grid(row=1, column=0)
        self.sound_selection_frame.grid(row=0, column=current_column)

        current_column += 1

        self.mix_original_slider = MidiSlider(self.dreaming_frame, 'mix original',
                                             min=0., max=1., resolution=0.01, length=200.,
                                             command=self.send_control_dict)
        self.mix_original_slider.grid(row=0, column=current_column)

        current_column += 1

        self.eq_lows_slider = MidiSlider(self.dreaming_frame, 'eq lows', default=0.,
                                              min=-1., max=1., resolution=0.01, length=200.,
                                              command=self.update_equalizer)
        self.eq_lows_slider.grid(row=0, column=current_column)

        current_column += 1

        self.eq_mids_slider = MidiSlider(self.dreaming_frame, 'eq mids', default=0.,
                                         min=-1., max=1., resolution=0.01, length=200.,
                                         command=self.update_equalizer)
        self.eq_mids_slider.grid(row=0, column=current_column)

        current_column += 1

        self.eq_highs_slider = MidiSlider(self.dreaming_frame, 'eq highs', default=0.,
                                         min=-1., max=1., resolution=0.01, length=200.,
                                         command=self.update_equalizer)
        self.eq_highs_slider.grid(row=0, column=current_column)

        current_column += 1

        self.max_agg_slider = MidiSlider(self.dreaming_frame, 'max agg', default=defaults['max_agg'],
                                         min=0., max=50., resolution=0.01, length=200.)
        self.max_agg_slider.grid(row=0, column=current_column)


        self.region_frame = tk.Frame(self.root)
        self.region_frame.pack()

        current_column = 0

        self.activation_selection_frame = tk.Frame(self.region_frame)
        self.activations_box = MidiListbox(self.activation_selection_frame, 'activation selection', elements=self.activation_names,
                                           default_index=3, width=30, height=10)
        self.activations_box.grid(row=0, column=0)

        self.keep_targets_switch = MidiSwitch(self.activation_selection_frame, 'keep targets')
        self.keep_targets_switch.grid(row=1, column=0)

        self.select_target_button = MidiButton(self.activation_selection_frame, 'select target', command=self.target_change)
        self.select_target_button.grid(row=2, column=0)

        self.activation_selection_frame.grid(row=0, column=current_column)

        current_column += 1

        self.channel_slider = MidiSlider(self.region_frame, 'channel',
                                         min=0., max=1., resolution=0.01, length=200.)
        self.channel_slider.grid(row=0, column=current_column)

        current_column += 1

        self.channel_region_slider = MidiSlider(self.region_frame, 'channel region',
                                         min=0., max=1., resolution=0.01, length=200.)
        self.channel_region_slider.grid(row=0, column=current_column)

        current_column += 1

        self.pitch_slider = MidiSlider(self.region_frame, 'pitch',
                                         min=0., max=1., resolution=0.01, length=200.)
        self.pitch_slider.grid(row=0, column=current_column)

        current_column += 1

        self.pitch_region_slider = MidiSlider(self.region_frame, 'pitch region',
                                         min=0., max=1., resolution=0.01, length=200.)
        self.pitch_region_slider.grid(row=0, column=current_column)

        current_column += 1

        self.time_slider = MidiSlider(self.region_frame, 'time',
                                         min=0., max=1., resolution=0.01, length=200.)
        self.time_slider.grid(row=0, column=current_column)

        current_column += 1

        self.time_region_slider = MidiSlider(self.region_frame, 'time region',
                                         min=0., max=1., default=1., resolution=0.01, length=200.)
        self.time_region_slider.grid(row=0, column=current_column)

        # visualization frame
        self.viz_frame = tk.Frame(self.root)
        self.viz_frame.pack()

        # fig = Figure()
        # fig.set_size_inches(3, 3)
        # ax = fig.add_subplot(111)
        # self.scal_plot = ax.imshow(np.zeros([256, 629]), origin='lower', aspect='auto')
        # ax.axis('off')
        # self.scal_canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        # self.scal_canvas.show()
        # self.scal_canvas.get_tk_widget().grid(row=0, column=0)
        #
        # fig = Figure()
        # fig.set_size_inches(3, 3)
        # ax = fig.add_subplot(111)
        # self.scal_grad_plot = ax.imshow(np.zeros([256, 629]), origin='lower', aspect='auto')
        # self.scal_grad_plot.set_data(np.random.randn(256, 629))
        # ax.axis('off')
        # self.scal_grad_canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        # self.scal_grad_canvas.show()
        # self.scal_grad_canvas.get_tk_widget().grid(row=0, column=1)

        fig = Figure()
        fig.set_size_inches(3, 3)
        self.loss_ax = fig.add_subplot(111)
        self.loss_plot = self.loss_ax.plot(np.random.rand(100))[0]
        self.loss_plot_canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.loss_plot_canvas.show()
        self.loss_plot_canvas.get_tk_widget().grid(row=0, column=2)

        fig = Figure()
        fig.set_size_inches(3, 3)
        self.eq_ax = fig.add_subplot(111)
        self.eq_ax.set_ylim([0.85, 1.15])
        self.eq_plot = self.eq_ax.semilogx(np.ones(256))[0]
        self.eq_plot_canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.eq_plot_canvas.show()
        self.eq_plot_canvas.get_tk_widget().grid(row=0, column=3)

        self.region_label = tk.Label(self.viz_frame)
        self.region_label.grid(row=0, column=4)

        self.region_canvas = ds.Canvas(plot_width=300, plot_height=300,
                                       x_range=(self.x_min - self.x_edge, self.x_max + self.x_edge),
                                       y_range=(self.y_min - self.y_edge, self.y_max + self.y_edge),
                                       x_axis_type='linear', y_axis_type='linear')
        self.region_img = None
        self.selected_regions = []

        self.blank_activations = collections.OrderedDict()
        for key, value in self.activation_statistics.items():
            if key in ['c_code', 'z_code', 'prediction']:
                continue
            if len(value['element_mean'].shape) == 2:
                self.blank_activations[key] = 0. * value['element_mean'][:, 0:1].repeat(1, self.num_frames)
            else:
                self.blank_activations[key] = 0. * value['element_mean'][:, :, 0:1].repeat(1, 1, self.num_frames)

        self.activation_dict = None
        self.communicator = SocketDataExchangeClient(port=int(self.port_entry.get()),
                                                     host=self.host_entry.get(),
                                                     stream_automatically=True)

        self.pyaudio = pyaudio.PyAudio()
        audio_device_index = None
        for i in range(self.pyaudio.get_device_count()):
            if self.pyaudio.get_device_info_by_index(i)['name'] == defaults['audio_device']:
                audio_device_index = i
                print("use audio device", self.pyaudio.get_device_info_by_index(i)['name'])
                break
        if audio_device_index is None:
            audio_device_index = self.pyaudio.get_default_output_device_info()['index']
            print("did not find audio device", defaults['audio_device'], "using default device",
                  self.pyaudio.get_device_info_by_index(i)['name'], "instead")
        self.audio_loop = AudioLoop(np.zeros(64000, dtype=np.int16), sample_rate=self.sample_rate)
        self.audio_stream = self.pyaudio.open(format=self.pyaudio.get_format_from_width(2, unsigned=False),
                                              channels=1,
                                              rate=self.sample_rate,
                                              output=True,
                                              stream_callback=self.audio_loop.callback,
                                              frames_per_buffer=1024,
                                              output_device_index=audio_device_index)

        # load dummy data
        with open('dummy_calculated_data.pickle', 'rb') as handle:
            data = pickle.load(handle)
        data['losses'] = np.zeros(100)
        #data['audio'] *= 0
        #data['audio'][32000:32100] += 30000
        self.activation_dict = data['activations']
        try:
            del self.activation_dict['c_code']
            del self.activation_dict['z_code']
            del self.activation_dict['prediction']
        except:
            pass
        for key, value in self.activation_dict.items():
            self.activation_dict[key] = value * 0.
        self.audio_loop.set_new_signal(data['audio'] * 0)
        self.draw_scalogram(data['scalogram'].numpy(), data['scalogram_grad'].numpy(), losses=[0.] * 100)

        # self.streaming_thread = threading.Thread(target=self.streaming_worker)
        # self.streaming_thread.daemon = True
        # self.stream = True
        # self.streaming_thread.start()

        # visualization window
        self.main_viz = MainVisualization(self, 600, 600)

        self.draw_target_lock = threading.Lock()
        self.new_target_img = False
        self.draw_target_thread = Thread(name='draw target', target=self.draw_target_viz)
        self.draw_target_thread.daemon = True
        self.draw_target_thread.start()

        self.time_offset = 0

        # MIDI
        all_children = self.all_tk_children(self.dreaming_frame)
        midi_controls = [w for w in all_children if hasattr(w, 'name')]
        mappable_controls = [c for c in midi_controls if c.name in midi_controller_mapping.keys()]
        mapping = {int(midi_controller_mapping[control.name]): control for control in mappable_controls}
        try:
            self.midi_controller = MidiController('BCF2000', mapping)
        except OSError as e:
            print(e)

        self.audio_stream.start_stream()
        #self.set_new_data(data, 0)
        #self.root.after(0, self.main_viz_callback)
        self.root.mainloop()

    def start_performance(self, *args):
        print("start performance")
        self.root.after(0, self.main_viz_callback)
        self.time_offset = self.audio_loop.get_next_loop_start_time() - self.audio_stream.get_time()
        self.time_offset = self.time_offset % self.main_viz.loop_length
        print("time offset:", self.time_offset)

    def end_performance(self, *args):
        self.black_screen = True

    def set_audio_clip(self, *args):
        self.audio_clip = audio_clip_names[self.soundclip_box.get_value()]
        self.send_control_dict()

    def send_control_dict(self, *args):
        # try:
        #     activation = self.activation_names[self.activations_box.curselection()[0]]
        # except:
        #     activation = None

        control_dict = {
            'selected_regions': self.selected_regions,
            'selected_clip': self.audio_clip,
            'lr': self.lr_slider.get_value(),
            'time_jitter': self.time_jitter_slider.get_value(),
            'noise_loss': self.noise_loss_slider.get_value(),
            'activation_loss': self.activation_loss_slider.get_value(),
            'time_masking': self.time_masking_slider.get_value(),
            'pitch_masking': self.pitch_masking_slider.get_value(),
            'batch_size': self.batch_size_slider.get_value(),
            'mix_o': self.mix_original_slider.get_value(),
            'eq_bands': self.band_factors
        }

        data = pickle.dumps(control_dict)
        self.communicator.set_new_data(data)

    def start_stop_audio(self):
        if self.audio_stream.is_active():
            self.audio_stream.stop_stream()
            self.start_audio_button.config(text=' start audio ')
        else:
            self.audio_stream.start_stream()
            self.start_audio_button.config(text=' stop audio ')

    def set_new_data(self, data, time_buffer=0.2):
        tik = time.time()
        self.activation_dict = data['activations']
        try:
            del self.activation_dict['c_code']
            del self.activation_dict['z_code']
            del self.activation_dict['prediction']
        except:
            pass
        for key, value in self.activation_dict.items():
            self.activation_dict[key] = value.type(torch.float)
        print("process activation dict time:", time.time() - tik)
        tik = time.time()
        self.audio_loop.set_new_signal(data['audio'], earliest_switch_time=0)#self.current_time + time_buffer)
        print("set audio signal time:", time.time() - tik)
        tik = time.time()
        self.main_viz.process_data_dict['activation_dict'] = self.activation_dict
        print("set activation dict time:", time.time() - tik)
        tik = time.time()
        self.draw_scalogram(data['scalogram'].numpy(), data['scalogram_grad'].numpy(), data['losses'])
        print("draw scalograms time:", time.time() - tik)

    def main_viz_callback(self):
        tik = time.time()
        self.current_time = self.audio_stream.get_time()
        if self.communicator.new_data_available:
            data = pickle.loads(self.communicator.get_received_data())
            #print("load data time:", time.time() - tik)
            tik = time.time()
            self.set_new_data(data)
            #print("set data time:", time.time() - tik)

        tik = time.time()
        loop_position = self.next_loop_start_time - self.current_time
        if loop_position < 0. or loop_position > self.main_viz.loop_length:
            self.next_loop_start_time = self.audio_loop.get_next_loop_start_time()
            #print("next start time:", self.next_loop_start_time)
            self.main_viz.process_data_dict['start_time'] = self.next_loop_start_time
        self.main_viz.process_data_dict['max_agg'] = self.max_agg_slider.get_value()
        self.main_viz.process_data_dict['time_jitter'] = self.time_jitter_slider.get_value()
        #print("set process data dict time:", time.time() - tik)
        tik = time.time()
        #self.main_viz.process_data_dict['target_size'] = min(self.main_viz.viz_window.winfo_width(),
        #                                                      self.main_viz.viz_window.winfo_height())
        self.main_viz.draw(self.current_time)
        #print("main viz draw time:", time.time() - tik)
        tik = time.time()

        with self.draw_target_lock:
            if self.region_img is not None:
                self.region_label.imgtk = self.region_img
                self.region_label.configure(image=self.region_img)
                self.region_img = None
                self.new_target_img = False
        #print("set new target time:", time.time() - tik)

    def target_change(self, *args):
        for value in self.blank_activations.values():
            value *= 0.

        this_region = {
            'layer': self.activation_names[self.activations_box.get_value()],
            'channel': self.channel_slider.get_value(),
            'channel region': self.channel_region_slider.get_value(),
            'pitch': self.pitch_slider.get_value(),
            'pitch region': self.pitch_region_slider.get_value(),
            'time': self.time_slider.get_value(),
            'time region': self.time_region_slider.get_value()
        }

        with self.draw_target_lock:
            self.new_target_img = True
            if self.keep_targets_switch.get_value() > 0:
                self.selected_regions.append(this_region)
            else:
                self.selected_regions = [this_region]

        self.send_control_dict()

    def draw_target_viz(self):
        while True:
            if not self.new_target_img:
                time.sleep(0.1)
                continue

            for region in self.selected_regions:
                if region['layer'] == 'prediction':
                    continue
                layer = self.blank_activations[region['layer']]
                layer = layer.unsqueeze(0)
                slice = select_activation_slice(layer,
                                                channel=region['channel'], channel_region=region['channel region'],
                                                pitch=region['pitch'], pitch_region=region['pitch region'],
                                                time=region['time'], time_region=region['time region'])
                slice += 1.

            target_activations = self.blank_activations.copy()
            for key, value in target_activations.items():
                last_dim = len(value.shape) - 1
                perm = [last_dim] + list(range(last_dim))
                target_activations[key] = value.permute(*perm).contiguous()

            focused_activations = flatten_activations(target_activations, exclude_first_dimension=True)
            self.main_viz.process_data_dict['focused_activations'] = focused_activations

            activations = torch.clamp(focused_activations.max(0)[0], 0., 1.)

            selected_positions = self.layout[0][activations > 0., :]
            if selected_positions.shape[0] > 0:
                target_position = torch.mean(selected_positions, dim=0)
                target_size = (torch.max(selected_positions[:, 0]) - torch.min(selected_positions[:, 0])) * \
                              (torch.max(selected_positions[:, 1]) - torch.min(selected_positions[:, 1]))
                self.main_viz.new_target((target_position[0].item(), target_position[1].item()), target_size.item())

            plot = activation_plot(self.layout[0], values=activations.detach().cpu().numpy(), canvas=self.region_canvas,
                                   spread=0, min_agg=0., max_agg=1., alpha=255)
            plot = tf.set_background(plot, 'black')
            img = plot.to_pil()
            imgtk = ImageTk.PhotoImage(image=img)
            with self.draw_target_lock:
                self.region_img = imgtk
                self.new_target_img = True
            time.sleep(0.1)

    def draw_scalogram(self, scalogram, scalogram_grad, losses=None):
        # self.scal_plot.set_data(scalogram)
        # self.scal_plot.set_clim(vmin=scalogram.min(), vmax=scalogram.max())
        # self.scal_canvas.draw()
        #
        # self.scal_grad_plot.set_data(scalogram_grad)
        # self.scal_grad_plot.set_clim(vmin=scalogram_grad.min(), vmax=scalogram_grad.max())
        # self.scal_grad_canvas.draw()

        if losses is not None:
            losses = np.array(losses)
            self.loss_plot.set_ydata(losses)
            self.loss_ax.set_ylim([losses.min() - 0.1, losses.max() + 0.1])
            self.loss_plot_canvas.draw()

    def update_equalizer(self, *args):
        band_factors, freqs = eq_bands(levels=[self.eq_lows_slider.get_value(),
                                               self.eq_mids_slider.get_value(),
                                               self.eq_highs_slider.get_value()],
                                       freqs=[0.1, 0.5, 1.], sizes=[0.2, 0.5, 0.5],
                                       fft_bands=self.num_eq_bands, sample_rate=self.sample_rate)

        self.eq_plot.set_xdata(freqs)
        self.eq_plot.set_ydata(band_factors)
        self.eq_ax.set_xlim([freqs[0], freqs[-1]])
        self.eq_plot_canvas.draw()
        self.band_factors = band_factors
        self.send_control_dict()

    @staticmethod
    def all_tk_children(wid):
        _list = wid.winfo_children()

        for item in _list:
            if item.winfo_children():
                _list.extend(item.winfo_children())

        return _list


class MainVisualization:
    def __init__(self, app, width=600, height=600):
        self.app = app

        self.width = width
        self.height = height
        self.layout = self.app.layout
        self.range_dict = range_dict
        self.current_frame = 0
        self.loop_length = self.app.audio_loop.signal_length / self.app.sample_rate
        self.start_time = 0.
        self.sync_timing = defaults_dict['sync_timing']

        self.rgb_yiq_transform = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]],
                                          dtype=np.float32)
        self.yiq_rgb_transform = np.array([[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.107, 1.705]],
                                          dtype=np.float32)

        self.transform_start_state = np.array([0., 0., 1., 1., 0., 0., 0.])
        self.transform_current_state = self.transform_start_state
        self.transform_target = self.transform_start_state
        self.transform_start_time = 0.
        self.transition_time = 4.

        with open(app.defaults['connections'], 'rb') as handle:
            self.connections = pickle.load(handle)
        with open(app.defaults['hues'], 'rb') as handle:
            self.hues = pickle.load(handle)
        self.num_precalc_frames = len(self.connections)
        self.num_frames = defaults_dict['num_frames']
        self.only_calculate_new_frames = True

        # UI
        self.viz_window = tk.Toplevel(self.app.root)
        self.viz_window.title('Visualization')
        self.viz_window.configure(background='black')
        self.viz_window.geometry(str(self.width) + 'x' + str(self.height))
        #self.viz_window.attributes("-fullscreen")

        # w, h = self.viz_window.winfo_screenwidth(), self.viz_window.winfo_screenheight()
        # self.viz_window.overrideredirect(1)
        # self.viz_window.geometry("%dx%d+0+0" % (w, h))

        #self.viz_window_size = (self.width, self.height)
        self.target_size = min(self.width, self.height)

        self.viz_window_frame = tk.Frame(self.viz_window)
        self.viz_window_frame.pack()

        self.viz_window_label = tk.Label(self.viz_window_frame,
                                         borderwidth=0,
                                         highlightthickness=0,
                                         bg="black")
        self.viz_window_label.pack()

        self.viz_window_canvas = ds.Canvas(plot_width=self.width, plot_height=self.height,
                                           x_range=(self.app.x_min - self.app.x_edge, self.app.x_max + self.app.x_edge),
                                           y_range=(self.app.y_min - self.app.y_edge, self.app.y_max + self.app.y_edge),
                                           x_axis_type='linear', y_axis_type='linear')

        imgtk = ImageTk.PhotoImage(image=Image.new('RGB', (self.width, self.height), color='black'))
        self.viz_window_label.imgtk = imgtk
        self.viz_window_label.configure(image=imgtk)

        self.num_processes = 2
        self.process_manager = mp.Manager()
        input_dict = {
            'visualize': True,
            'start_time': 0.,
            'activation_dict': self.app.activation_dict,
            'focused_activations': None,
            'max_agg': 1.,
            'time_jitter': self.app.time_jitter_slider.get_value(),
            'target_size': min(self.width, self.height),
            'transform_start_time': 0.,
            'transform_start_state': np.array([0., 0., 1., 1., 0., 0., 0.]),
            'transform_target': np.array([0., 0., 1., 1., 0., 0., 0.])
        }
        self.process_data_dict = self.process_manager.dict(input_dict)
        #self.image_list = self.process_manager.list([None] * self.num_frames)
        self.image_queue = mp.Queue()
        self.frame_list = [None] * self.num_frames
        self.processes = []
        for i in range(self.num_processes):
            p = mp.Process(name='main_viz_worker' + str(i),
                           target=self.viz_process_worker, args=(i,
                                                                 self.process_data_dict,
                                                                 self.image_queue))
            p.daemon = True
            p.start()
            self.processes.append(p)

        self.frame_list_lock = threading.Lock()
        self.queue_loading_thread = Thread(name='queue loading', target=self.load_frame_list)
        self.queue_loading_thread.daemon = True
        self.queue_loading_thread.start()

    def new_target(self, position, size):
        position_x = (position[0] - self.app.x_min) / (self.app.x_max - self.app.x_min) * self.width
        position_y = (self.app.y_max - position[1]) / (self.app.y_max - self.app.y_min) * self.height
        scale_factor = 0.1 / (size + 0.15) + 0.9

        target_size = self.target_size

        center_x = 0.5 * (self.width * 0.5) + 0.5 * position_x
        center_y = 0.5 * (self.height * 0.5) + 0.5 * position_y
        scale_x = scale_factor * target_size / self.width
        scale_y = scale_factor * target_size / self.height
        shift_x = target_size * 0.5
        shift_y = target_size * 0.5

        start_state = self.transform_current_state

        theta = math.atan2((position_x - self.width * 0.5), (position_y - self.height * 0.5))
        current_theta = (start_state[6] + np.pi) % (2 * np.pi) - np.pi

        if abs(theta - current_theta) > abs(theta - (current_theta + np.pi * 2)):
            current_theta += np.pi * 2
            start_state[6] = current_theta
            start_state[6] = current_theta
        elif abs(theta - current_theta) > abs(theta - (current_theta - np.pi * 2)):
            current_theta -= np.pi * 2
            start_state[6] = current_theta
            start_state[6] = current_theta

        self.transform_start_time = time.time()
        self.transform_start_state = start_state
        self.transform_target = np.array([center_x, center_y, scale_x, scale_y, shift_x, shift_y, theta])

    @staticmethod
    def calc_affine_parameters_pil(p):
        # input: numpy array with
        #   0         1         2        3        4        5        6
        # center_x, center_y, scale_x, scale_y, shift_x, shift_y, theta
        cos_th = math.cos(p[6])
        sin_th = math.sin(p[6])

        a = cos_th / p[2]
        b = sin_th / p[2]
        c = p[0] - p[4] * a - p[5] * b
        d = -sin_th / p[3]
        e = cos_th / p[3]
        f = p[1] - p[4] * d - p[5] * e

        return [a, b, c, d, e, f]

    @staticmethod
    def calc_affine_parameters_cv2(p):
        # input: numpy array with
        #   0         1         2        3        4        5        6
        # center_x, center_y, scale_x, scale_y, shift_x, shift_y, theta
        cos_th = math.cos(p[6])
        sin_th = math.sin(p[6])

        a = cos_th * p[2]
        b = sin_th * p[2]
        c = p[4] - p[0] * a - p[1] * b
        d = -sin_th * p[3]
        e = cos_th * p[3]
        f = p[5] - p[0] * d - p[1] * e

        return np.float32([[a, b, c], [d, e, f]])

    def load_frame_list(self):
        while True:
            frame_number, img = self.image_queue.get()
            with self.frame_list_lock:
                self.frame_list[frame_number] = img
                #self.frame_list.append((frame_time, img))
                #self.frame_list.sort(key=lambda t: t[0])
            #if len(self.frame_list) > 100:
            #print("length of frame list:", len(self.frame_list))

    def draw(self, current_time):
        #current_time -= 0.5

        self.target_size = min(self.viz_window.winfo_width(), self.viz_window.winfo_height())
        #print("target size", self.target_size)
        #print("current time", current_time)

        tik = time.time()

        # with self.frame_list_lock:
        #     if len(self.frame_list) == 0:
        #         self.viz_window_label.after(50, self.app.main_viz_callback)
        #         print("skip frame")
        #         return
        #
        #     frame_time, img = self.frame_list[0]
        #     #print("frame list length:", len(self.frame_list))
        #
        #     while frame_time < current_time:
        #         if len(self.frame_list) == 0:
        #             self.viz_window_label.after(50, self.app.main_viz_callback)
        #             print("skip frame")
        #             return
        #         else:
        #             frame_time, img = self.frame_list.pop(0)
        #
        # while self.app.audio_stream.get_time() < frame_time:
        #     time.sleep(0.02)

        frame_number = int(((current_time + self.app.time_offset + self.sync_timing) % self.loop_length) / self.loop_length * self.num_frames)
        #print("frame number:", frame_number)
        with self.frame_list_lock:
            img = self.frame_list[frame_number]

        if img is None:
            self.viz_window_label.after(100, self.app.main_viz_callback)
            return

        loading_duration = time.time() - tik
        # if loading_duration > 0.02:
        #     print("loading image duration:", loading_duration)

        tik = time.time()
        img = img.reshape(self.width, self.height, 4)
        img = img[::-1, ::1, :]

        if self.app.black_screen:
            img *= 0

        target_size = min(self.viz_window.winfo_width(), self.viz_window.winfo_height())

        transition_position = self.interpolation_function((time.time() - self.transform_start_time) / self.transition_time)
        self.transform_current_state = self.transform_start_state * (1 - transition_position) + \
                                       self.transform_target * transition_position
        affine_parameters = self.calc_affine_parameters_cv2(self.transform_current_state)

        img = cv2.warpAffine(img, affine_parameters,
                             (target_size, target_size))
        img = Image.fromarray(img)

        transform_duration = time.time() - tik
        # if transform_duration > 0.01:
        #     print("transform duration", transform_duration)

        tik = time.time()
        #print("frame time:", frame_time)
        imgtk = ImageTk.PhotoImage(image=img)
        self.viz_window_label.imgtk = imgtk
        imgtk_duration = time.time() - tik

        tik = time.time()
        self.viz_window_label.configure(image=imgtk)
        update_duration = time.time() - tik
        if loading_duration + transform_duration + update_duration > 0.1:
            print("")
            print("loading", loading_duration)
            print("transform", transform_duration)
            print("imgtk", imgtk_duration)
            print("update", update_duration)
        self.viz_window_label.after(50, self.app.main_viz_callback)

    def viz_process_worker(self, number, input_dict, image_queue):
        activation_dict = input_dict['activation_dict']
        # transform_target = input_dict['transform_target']
        # transform_start_state = input_dict['transform_target']
        hue_shift = 0.
        #target_size = input_dict['target_size']
        last_start_time = 0
        i = 1000
        tik = time.time()
        while input_dict['visualize']:
            frame_number = i * self.num_processes + number
            loop_position = frame_number / self.num_frames
            precalc_frame = int(loop_position * self.num_precalc_frames)
            current_absolute_time = last_start_time + self.loop_length * loop_position  #TODO: time is wrong!!!
            if loop_position >= 1.:
                calc_duration = time.time() - tik
                print("calc duration:", calc_duration)
                if calc_duration > 4.:
                    print("viz calculation too slow!")
                if self.only_calculate_new_frames:
                    while True:
                        if calc_duration > (self.loop_length-0.1):
                            break
                        calc_duration = time.time() - tik
                        time.sleep(0.05)
                i = 0

                tik = time.time()
                last_start_time = input_dict['start_time']
                try:
                    activation_dict = input_dict['activation_dict'].copy()
                except:
                    i += 1
                    continue
                for key, (start, length) in self.range_dict.items():
                    activation_dict[key] = activation_dict[key][:, :, start:start + length]
                # transform_start_state = transform_target
                # transform_target = input_dict['transform_target']
                hue_shift = math.log2(input_dict['time_jitter'] + 1.) / 3.
                continue
            current_activations = {}
            for key, value in activation_dict.items():
                current_activations[key] = interpolate_position(value, loop_position)

            timestep_layout = self.layout[precalc_frame]

            edges_colormap = ['#000000', '#2f4858', '#33658a', '#86bbd8', '#9dd9d2', '#79bcb8']
            edges_img = tf.shade(self.connections[precalc_frame], cmap=edges_colormap)
            edges_img = tf.set_background(edges_img, 'black')
            rgb_edges = edges_img.data.view(dtype='uint8').reshape(-1, 4)[:, :3].astype(np.float32) / 255.

            hues_img = self.hues[precalc_frame]

            if input_dict['focused_activations'] is None:
                rgb_focus = 0.
            else:
                current_focused_activations = input_dict['focused_activations'][precalc_frame]
                focused_positions = timestep_layout[current_focused_activations > 0.]
                focus_plot = activation_plot(focused_positions, canvas=self.viz_window_canvas, spread=4, alpha=255,
                                             min_agg=0., max_agg=1., colormap=[(0, 0, 0), (0, 255, 255)])
                focus_plot = tf.set_background(focus_plot, 'black')
                rgb_focus = focus_plot.data.view(dtype='uint8').reshape(-1, 4)[:, :3].astype(np.float32) / 255.

            flat_activations = flatten_activations(current_activations) #** 2
            flat_activations[flat_activations != flat_activations] = 0  # set nans to 0

            max_agg = input_dict['max_agg']
            max_activation = flat_activations.mean().item() / max_agg * 3.

            # if max_activation <= 0:
            #     image_queue.put((frame_number, None))
            #     print("all activations are 0.")
            #     i += 1
            #     continue

            plot = activation_plot(timestep_layout, values=flat_activations.detach().cpu().numpy(),
                                   canvas=self.viz_window_canvas, spread=1, alpha=100, min_agg=0,
                                   max_agg=max_agg,
                                   colormap=[(0, 0, 0), (255, 255, 255)])

            plot = tf.set_background(plot, 'black')

            rgb_plot = plot.data.view(dtype='uint8').reshape(-1, 4)[:, :3].astype(np.float32) / 255.
            rgb_hue = hues_img.data.view(dtype='uint8').reshape(-1, 4)[:, :3].astype(np.float32) / 255.
            rgb_plot *= rgb_hue
            rgb_plot = (1 - (1 - rgb_edges * max_activation) * (1 - rgb_plot)) * (1. + 2. * rgb_focus)

            yiq_plot = np.matmul(rgb_plot, self.rgb_yiq_transform)

            # color_transform
            # hue_shift = 0.
            u = math.cos(hue_shift)
            w = math.sin(hue_shift)
            s = 1.0  # saturation
            v = 1.  # value
            color_transform = np.array([[v, 0, 0], [0, v * s * u, -v * s * w], [0, v * s * w, v * s * u]],
                                       dtype=np.float32)
            yiq_plot = np.matmul(yiq_plot, color_transform)

            rgb_plot = np.matmul(yiq_plot, self.yiq_rgb_transform)

            rgb_plot = (np.clip(rgb_plot, 0., 1.) * 255.).astype(np.uint8)
            rgb_plot = np.concatenate([rgb_plot, np.full((rgb_plot.shape[0], 1), 255, dtype=np.uint8)], axis=1)
            #rgb_plot = rgb_plot.view('uint32').reshape(plot.data.shape[0], plot.data.shape[1])
            #plot.data = rgb_plot
            #img = plot
            #img = plot.to_pil()
            transform_state = self.transform_state(current_absolute_time, input_dict['transform_start_time'],
                                                   input_dict['transform_start_state'], input_dict['transform_target'])

            # affine_parameters = self.calc_affine_parameters_pil(transform_state)

            # target_size = input_dict['target_size']
            # img = img.transform((target_size, target_size), Image.AFFINE, affine_parameters, resample=Image.BILINEAR)

            #print("write frame", frame_number)
            #print("frame number:", frame_number)
            #print("absolute time:", current_absolute_time)
            #if image_queue.qsize() < 50:
            image_queue.put((frame_number, rgb_plot))
            #output_images[frame_number] = img
            i += 1

    def transform_state(self, time, start_time, start_state, target_state):
        transition_position = (time - start_time) / self.transition_time
        transition_position = self.interpolation_function(transition_position)
        transform_state = start_state * (1 - transition_position) + target_state * transition_position
        return transform_state

    @staticmethod
    def interpolation_function(value):
        value = min(max(0., value), 1.)
        return value**2 * (3 - value*2)


if __name__ == '__main__':
    app = DreamingControlApp()



import tkinter as tk
import pickle
import collections
import numpy as np
import pyaudio
import threading
import multiprocessing as mp
import time
from PIL import Image, ImageTk, ImageChops
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from colorcet import fire

from dreaming.streaming import SocketDataExchangeClient
from dreaming.visualization_functions import *
from dreaming.dreaming_functions import *
from dreaming.audio_output import AudioLoop

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
    'lr': -3.,
    'time_jitter': 0.1,
    'noise_loss': 0.1,
    'activation_loss': 0.,
    'time_masking': 0,
    'pitch_masking': 0,
    'batch_size': 32,
    'activation_names': activations,
    'activation_statistics': '../data_statistics_snapshots_model_2019-05-20_run_0_100000.pickle',
    'layout': '/Users/vincentherrmann/Documents/Projekte/Immersions/visualization/layouts/e25_version_3/e25_positions_interp_100.npy',
    'hues': '/Users/vincentherrmann/Documents/Projekte/Immersions/visualization/layouts/e25_version_3/e25_positions_interp_100_hues.p',
    'connections': '/Users/vincentherrmann/Documents/Projekte/Immersions/visualization/layouts/e25_version_3/e25_positions_interp_100_edges_weighted.p'
}

control_dict = {
    'activation': 'scalogram',
    'channel': 0.,
    'channel_region': 1.,
    'pitch': 0.,
    'pitcpch': 0.,
    'pitch_region': 1.,
    'time': 0.,
    'time_region': 1.,
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


class DreamingControlApp:
    def __init__(self, defaults=defaults_dict):
        self.defaults = defaults
        self.sample_rate = 16000
        self.num_eq_bands = 257
        self.num_frames = 100
        self.band_factors = None

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

        # dreaming frame
        self.dreaming_frame = tk.Frame(self.root)
        self.dreaming_frame.pack()

        current_column = 0

        tk.Label(self.dreaming_frame, text='learning rate').grid(row=0, column=current_column)
        self.lr_var = tk.DoubleVar(value=defaults['lr'])
        self.lr_slider = tk.Scale(self.dreaming_frame, variable=self.lr_var,
                                  from_=2., to=-5., resolution=0.1, length=200, command=self.send_control_dict)
        self.lr_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='time jitter').grid(row=0, column=current_column)
        self.time_jitter_var = tk.DoubleVar(value=defaults['time_jitter'])
        self.time_jitter_slider = tk.Scale(self.dreaming_frame, variable=self.time_jitter_var,
                                           from_=4., to=0., resolution=0.01, length=200, command=self.send_control_dict)
        self.time_jitter_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='noise loss').grid(row=0, column=current_column)
        self.noise_loss_var = tk.DoubleVar(value=defaults['noise_loss'])
        self.noise_loss_slider = tk.Scale(self.dreaming_frame, variable=self.noise_loss_var,
                                          from_=1., to=0., resolution=0.01, length=200, command=self.send_control_dict)
        self.noise_loss_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='activation loss').grid(row=0, column=current_column)
        self.activation_loss_var = tk.DoubleVar(value=defaults['activation_loss'])
        self.activation_loss_slider = tk.Scale(self.dreaming_frame, variable=self.activation_loss_var,
                                          from_=50., to=0., resolution=0.1, length=200, command=self.send_control_dict)
        self.activation_loss_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='time masking').grid(row=0, column=current_column)
        self.time_masking_var = tk.DoubleVar(value=defaults['time_masking'])
        self.time_masking_slider = tk.Scale(self.dreaming_frame, variable=self.time_masking_var,
                                            from_=1., to=0., resolution=0.01, length=200, command=self.send_control_dict)
        self.time_masking_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='pitch masking').grid(row=0, column=current_column)
        self.pitch_masking_var = tk.DoubleVar(value=defaults['pitch_masking'])
        self.pitch_masking_slider = tk.Scale(self.dreaming_frame, variable=self.pitch_masking_var,
                                             from_=1., to=0., resolution=0.01, length=200, command=self.send_control_dict)
        self.pitch_masking_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='batch size').grid(row=0, column=current_column)
        self.batch_size_var = tk.DoubleVar(value=defaults['batch_size'])
        self.batch_size_slider = tk.Scale(self.dreaming_frame, variable=self.batch_size_var,
                                          from_=64., to=0., resolution=1., length=200,
                                          command=self.send_control_dict)
        self.batch_size_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='mix original').grid(row=0, column=current_column)
        self.mix_o_var = tk.DoubleVar(value=0)
        self.mix_o_slider = tk.Scale(self.dreaming_frame, variable=self.mix_o_var,
                                     from_=1., to=0., resolution=0.01, length=200, command=self.send_control_dict)
        self.mix_o_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='eq lows').grid(row=0, column=current_column)
        self.eq_lows_level_var = tk.DoubleVar(value=0)
        self.eq_lows_level_slider = tk.Scale(self.dreaming_frame, variable=self.eq_lows_level_var,
                                     from_=1., to=-1., resolution=0.01, length=200, command=self.update_equalizer)
        self.eq_lows_level_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='eq mids').grid(row=0, column=current_column)
        self.eq_mids_level_var = tk.DoubleVar(value=0)
        self.eq_mids_level_slider = tk.Scale(self.dreaming_frame, variable=self.eq_mids_level_var,
                                             from_=1., to=-1., resolution=0.01, length=200,
                                             command=self.update_equalizer)
        self.eq_mids_level_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='eq highs').grid(row=0, column=current_column)
        self.eq_highs_level_var = tk.DoubleVar(value=0)
        self.eq_highs_level_slider = tk.Scale(self.dreaming_frame, variable=self.eq_highs_level_var,
                                             from_=1., to=-1., resolution=0.01, length=200,
                                             command=self.update_equalizer)
        self.eq_highs_level_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='activation selection').grid(row=0, column=current_column)
        self.activations_box = tk.Listbox(self.dreaming_frame)
        for a in self.activation_names:
            self.activations_box.insert(tk.END, a)
        self.activations_box.activate(1)
        self.activations_box.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='channel').grid(row=0, column=current_column)
        self.channel_var = tk.DoubleVar(value=0)
        self.channel_slider = tk.Scale(self.dreaming_frame, variable=self.channel_var,
                                       from_=1., to=0., resolution=0.01, length=200, command=self.target_change)
        self.channel_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='channel region').grid(row=0, column=current_column)
        self.channel_region_var = tk.DoubleVar(value=0)
        self.channel_region_slider = tk.Scale(self.dreaming_frame, variable=self.channel_region_var,
                                       from_=1., to=0., resolution=0.01, length=200, command=self.target_change)
        self.channel_region_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='pitch').grid(row=0, column=current_column)
        self.pitch_var = tk.DoubleVar(value=0)
        self.pitch_slider = tk.Scale(self.dreaming_frame, variable=self.pitch_var,
                                     from_=1., to=0., resolution=0.01, length=200, command=self.target_change)
        self.pitch_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='pitch region').grid(row=0, column=current_column)
        self.pitch_region_var = tk.DoubleVar(value=0)
        self.pitch_region_slider = tk.Scale(self.dreaming_frame, variable=self.pitch_region_var,
                                            from_=1., to=0., resolution=0.01, length=200, command=self.target_change)
        self.pitch_region_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='time').grid(row=0, column=current_column)
        self.time_var = tk.DoubleVar(value=0)
        self.time_slider = tk.Scale(self.dreaming_frame, variable=self.time_var,
                                     from_=1., to=0., resolution=0.01, length=200, command=self.target_change)
        self.time_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='time region').grid(row=0, column=current_column)
        self.time_region_var = tk.DoubleVar(value=1.)
        self.time_region_slider = tk.Scale(self.dreaming_frame, variable=self.time_region_var,
                                            from_=1., to=0., resolution=0.01, length=200, command=self.target_change)
        self.time_region_slider.grid(row=1, column=current_column)

        current_column += 1

        tk.Label(self.dreaming_frame, text='max agg').grid(row=0, column=current_column)
        self.max_agg_var = tk.DoubleVar(value=1.)
        self.max_agg_slider = tk.Scale(self.dreaming_frame, variable=self.max_agg_var,
                                           from_=100., to=0., resolution=0.1, length=200)
        self.max_agg_slider.grid(row=1, column=current_column)

        # visualization frame
        self.viz_frame = tk.Frame(self.root)
        self.viz_frame.pack()

        fig = Figure()
        fig.set_size_inches(4, 4)
        ax = fig.add_subplot(111)
        self.scal_plot = ax.imshow(np.zeros([256, 629]), origin='lower', aspect='auto')
        ax.axis('off')
        self.scal_canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.scal_canvas.show()
        self.scal_canvas.get_tk_widget().grid(row=0, column=0)

        fig = Figure()
        fig.set_size_inches(4, 4)
        ax = fig.add_subplot(111)
        self.scal_grad_plot = ax.imshow(np.zeros([256, 629]), origin='lower', aspect='auto')
        self.scal_grad_plot.set_data(np.random.randn(256, 629))
        ax.axis('off')
        self.scal_grad_canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.scal_grad_canvas.show()
        self.scal_grad_canvas.get_tk_widget().grid(row=0, column=1)

        fig = Figure()
        fig.set_size_inches(4, 4)
        self.loss_ax = fig.add_subplot(111)
        self.loss_plot = self.loss_ax.plot(np.random.rand(100))[0]
        self.loss_plot_canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.loss_plot_canvas.show()
        self.loss_plot_canvas.get_tk_widget().grid(row=0, column=2)

        fig = Figure()
        fig.set_size_inches(4, 4)
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
        self.draw_target_viz()

        self.activation_dict = None
        self.communicator = SocketDataExchangeClient(port=int(self.port_entry.get()),
                                                     host=self.host_entry.get(),
                                                     stream_automatically=True)

        self.pyaudio = pyaudio.PyAudio()
        self.audio_loop = AudioLoop(np.zeros(64000, dtype=np.int16))
        self.audio_stream = self.pyaudio.open(format=self.pyaudio.get_format_from_width(2, unsigned=False),
                                              channels=1,
                                              rate=self.sample_rate,
                                              output=True,
                                              stream_callback=self.audio_loop.callback,
                                              frames_per_buffer=4096)

        # load dummy data
        with open('dummy_calculated_data.pickle', 'rb') as handle:
            data = pickle.load(handle)
        self.activation_dict = data['activations']
        try:
            del self.activation_dict['c_code']
            del self.activation_dict['z_code']
            del self.activation_dict['prediction']
        except:
            pass
        self.audio_loop.set_new_signal(data['audio'])
        self.draw_scalogram(data['scalogram'].numpy(), data['scalogram_grad'].numpy(), losses=[0.] * 100)

        # self.streaming_thread = threading.Thread(target=self.streaming_worker)
        # self.streaming_thread.daemon = True
        # self.stream = True
        # self.streaming_thread.start()

        # visualization window
        self.main_viz = MainVisualization(self, 600, 600)

        self.audio_stream.start_stream()
        self.root.after(0, self.streaming_callback)
        self.root.mainloop()

    def send_control_dict(self, *args):
        try:
            activation = self.activation_names[self.activations_box.curselection()[0]]
        except:
            activation = None

        control_dict = {
            'activation': activation,
            'channel': self.channel_var.get(),
            'channel_region': self.channel_region_var.get(),
            'pitch': self.pitch_var.get(),
            'pitch_region': self.pitch_region_var.get(),
            'time': self.time_var.get(),
            'time_region': self.time_region_var.get(),
            'lr': self.lr_var.get(),
            'time_jitter': self.time_jitter_var.get(),
            'noise_loss': self.noise_loss_var.get(),
            'activation_loss': self.activation_loss_var.get(),
            'time_masking': self.time_masking_var.get(),
            'pitch_masking': self.pitch_masking_var.get(),
            'batch_size': self.batch_size_var.get(),
            'mix_o': self.mix_o_var.get(),
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

    def streaming_callback(self):
        if self.communicator.new_data_available:
            data = pickle.loads(self.communicator.get_received_data())
            self.activation_dict = data['activations']
            try:
                del self.activation_dict['c_code']
                del self.activation_dict['z_code']
                del self.activation_dict['prediction']
            except:
                pass
            self.audio_loop.set_new_signal(data['audio'])
            start_time = time.time() + (self.audio_loop.signal_length - self.audio_loop.position) / self.sample_rate
            loop_length = self.audio_loop.signal_length / self.sample_rate
            self.main_viz.viz_conn.send({
                'activation_dict': self.activation_dict,
                'start_time': start_time,
                'loop_length': loop_length,
                'max_agg': self.max_agg_var.get(),
                'focused_activations': self.focused_activations
            })
            self.draw_scalogram(data['scalogram'].numpy(), data['scalogram_grad'].numpy(), data['losses'])
        self.main_viz.draw()

    def target_change(self, *args):
        self.send_control_dict()
        self.draw_target_viz()

    def draw_target_viz(self):
        indices = self.activations_box.curselection()
        target_activations = collections.OrderedDict()
        for key, value in self.activation_statistics.items():
            if key in ['c_code', 'z_code', 'prediction']:
                continue
            if len(value['element_mean'].shape) == 2:
                target_activations[key] = 0. * value['element_mean'][:, 0:1].repeat(1, self.num_frames)
            else:
                target_activations[key] = 0. * value['element_mean'][:, :, 0:1].repeat(1, 1, self.num_frames)
        for idx in indices:
            key = self.activation_names[idx]
            try:
                value = target_activations[key]
                value = value.unsqueeze(0)
                slice = select_activation_slice(value,
                                                channel=self.channel_var.get(), channel_region=self.channel_region_var.get(),
                                                pitch=self.pitch_var.get(), pitch_region=self.pitch_region_var.get(),
                                                time=self.time_var.get(), time_region=self.time_region_var.get())
                slice += 1.
            except:
                pass

        for key, value in target_activations.items():
            last_dim = len(value.shape) - 1
            perm = [last_dim] + list(range(last_dim))
            target_activations[key] = value.permute(*perm).contiguous()

        self.focused_activations = flatten_activations(target_activations, exclude_first_dimension=True)
        activations = self.focused_activations.max(0)[0]

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
        self.region_label.imgtk = imgtk
        self.region_label.configure(image=imgtk)

    def draw_scalogram(self, scalogram, scalogram_grad, losses=None):
        self.scal_plot.set_data(scalogram)
        self.scal_plot.set_clim(vmin=scalogram.min(), vmax=scalogram.max())
        self.scal_canvas.draw()

        self.scal_grad_plot.set_data(scalogram_grad)
        self.scal_grad_plot.set_clim(vmin=scalogram_grad.min(), vmax=scalogram_grad.max())
        self.scal_grad_canvas.draw()

        if losses is not None:
            losses = np.array(losses)
            self.loss_plot.set_ydata(losses)
            self.loss_ax.set_ylim([losses.min(), losses.max()])
            self.loss_plot_canvas.draw()

    def update_equalizer(self, *args):
        band_factors, freqs = eq_bands(levels = [self.eq_lows_level_var.get(),
                                                 self.eq_mids_level_var.get(),
                                                 self.eq_highs_level_var.get()],
                                       freqs=[0.1, 0.5, 1.], sizes=[0.2, 0.5, 0.5],
                                       fft_bands=self.num_eq_bands, sample_rate=self.sample_rate)

        self.eq_plot.set_xdata(freqs)
        self.eq_plot.set_ydata(band_factors)
        self.eq_ax.set_xlim([freqs[0], freqs[-1]])
        self.eq_plot_canvas.draw()
        self.band_factors = band_factors
        self.send_control_dict()


class MainVisualization:
    def __init__(self, app, width=600, height=600):
        self.app = app

        self.width = width
        self.height = height

        self.rgb_yiq_transform = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]],
                                          dtype=np.float32)
        self.yiq_rgb_transform = np.array([[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.107, 1.705]],
                                          dtype=np.float32)

        self.transform_start_state = np.array([0., 0., 1., 1., 0., 0., 0.])
        self.transform_current_state = self.transform_start_state
        self.transform_target = self.transform_start_state
        self.transform_start_time = time.time()
        self.transition_time = 4.

        with open(app.defaults['connections'], 'rb') as handle:
            self.connections = pickle.load(handle)
        with open(app.defaults['hues'], 'rb') as handle:
            self.hues = pickle.load(handle)
        self.num_frames = len(self.connections)

        # UI
        self.viz_window = tk.Toplevel(self.app.root)
        self.viz_window.title('Visualization')
        self.viz_window.configure(background='black')
        self.viz_window.geometry(str(self.width) + 'x' + str(self.height))

        self.viz_window_size = (self.width, self.height)

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

        # Background Process
        self.viz_conn, conn = mp.Pipe(duplex=True)
        self.visualization_process = mp.Process(target=self.viz_worker, args=(conn,))
        self.visualization_process.daemon = True
        self.visualize = True
        self.viz_conn.send({'activation_dict': None})
        self.visualization_process.start()
        self.main_viz_tik = time.time()
        audio_loop = self.app.audio_loop
        start_time = time.time() + (audio_loop.signal_length - audio_loop.position) / self.app.sample_rate
        loop_length = audio_loop.signal_length / self.app.sample_rate

    def new_target(self, position, size):
        position_x = (position[0] - self.app.x_min) / (self.app.x_max - self.app.x_min) * self.width
        position_y = (self.app.y_max - position[1]) / (self.app.y_max - self.app.y_min) * self.height
        scale_factor = 0.1 / (size + 0.15) + 0.9

        target_size = min(self.viz_window.winfo_width(), self.viz_window.winfo_height())

        center_x = 0.5 * (self.width * 0.5) + 0.5 * position_x
        center_y = 0.5 * (self.height * 0.5) + 0.5 * position_y
        scale_x = scale_factor * target_size / self.width
        scale_y = scale_factor * target_size / self.height
        shift_x = target_size * 0.5
        shift_y = target_size * 0.5

        theta = math.atan2((position_x - self.width * 0.5), (position_y - self.height * 0.5))
        current_theta = (self.transform_current_state[6] + np.pi) % (2 * np.pi) - np.pi

        if abs(theta - current_theta) > abs(theta - (current_theta + np.pi * 2)):
            current_theta += np.pi * 2
            self.transform_current_state[6] = current_theta
            self.transform_start_state[6] = current_theta
        elif abs(theta - current_theta) > abs(theta - (current_theta - np.pi * 2)):
            current_theta -= np.pi * 2
            self.transform_current_state[6] = current_theta
            self.transform_start_state[6] = current_theta

        self.transform_target = np.array([center_x, center_y, scale_x, scale_y, shift_x, shift_y, theta])
        self.transform_start_state = self.transform_current_state
        self.transform_start_time = time.time()

    @staticmethod
    def calc_affine_parameters(p):
        # input: numpy array with
        #   0         1         2        3        4        5        6
        # center_x, center_y, scale_x, scale_y, shift_x, shift_y, theta
        cos_th = math.cos(-p[6])
        sin_th = math.sin(-p[6])

        a = cos_th / p[2]
        b = sin_th / p[2]
        c = p[0] - p[4] * a - p[5] * b
        d = -sin_th / p[3]
        e = cos_th / p[3]
        f = p[1] - p[4] * d - p[5] * e

        return [a, b, c, d, e, f]

    def draw(self):
        img = self.viz_conn.recv()
        if img is None:
            self.viz_window_label.after(50, self.app.streaming_callback)
            return
        target_size = min(self.viz_window.winfo_width(), self.viz_window.winfo_height())

        transition_position = self.interpolation_function((time.time() - self.transform_start_time) / self.transition_time)
        self.transform_current_state = self.transform_start_state * (1 - transition_position) + \
                                       self.transform_target * transition_position
        affine_parameters = self.calc_affine_parameters(self.transform_current_state)

        img = img.transform((target_size, target_size), Image.AFFINE, affine_parameters, resample=Image.BILINEAR)

        #img = img.resize((target_size, target_size), resample=Image.BICUBIC)
        imgtk = ImageTk.PhotoImage(image=img)
        self.viz_window_label.imgtk = imgtk
        self.viz_window_label.configure(image=imgtk)
        self.viz_window_label.after(50, self.app.streaming_callback)

    def viz_worker(self, conn):
        tik = time.time()
        while self.visualize:
            if conn.poll(0):
                print("new activations")
                received_data = conn.recv()
                activation_dict = received_data['activation_dict']
            if activation_dict is None:
                conn.send(None)
                time.sleep(0.1)
                continue
            loop_length = received_data['loop_length']
            time_position = ((tik - received_data['start_time']) % loop_length) / loop_length
            time_frame = int(time_position * self.num_frames)
            current_activations = activation_dict.copy()
            for key, (start, length) in range_dict.items():
                current_activations[key] = current_activations[key][:, :, start:start + length]
            for key, value in current_activations.items():
                current_activations[key] = interpolate_position(value, time_position)
            timestep_layout = self.app.layout[time_frame]

            edges_colormap = ['#000000', '#2f4858', '#33658a', '#86bbd8', '#9dd9d2', '#79bcb8']
            edges_img = tf.shade(self.connections[time_frame], cmap=edges_colormap)
            edges_img = tf.set_background(edges_img, 'black')
            rgb_edges = edges_img.data.view(dtype='uint8').reshape(-1, 4)[:, :3].astype(np.float32) / 255.

            hues_img = self.hues[time_frame]

            current_focused_activations = received_data['focused_activations'][time_frame]
            focused_positions = timestep_layout[current_focused_activations > 0.]
            focus_plot = activation_plot(focused_positions, canvas=self.viz_window_canvas, spread=4, alpha=255,
                                         min_agg=0., max_agg=1., colormap=[(0, 0, 0), (0, 255, 255)])
            focus_plot = tf.set_background(focus_plot, 'black')

            flat_activations = flatten_activations(current_activations) ** 2
            flat_activations[flat_activations != flat_activations] = 0 # set nans to 0

            max_activation = flat_activations.mean().item() * 0.0

            plot = activation_plot(timestep_layout, values=flat_activations.detach().cpu().numpy(),
                                   canvas=self.viz_window_canvas, spread=1, alpha=100, min_agg=0,
                                   max_agg=received_data['max_agg'],
                                   colormap=[(0, 0, 0), (255, 255, 255)])

            # TEST!!!
            #plot = focus_plot



            #edges_img = ds.transfer_functions.Image(edges_img.data, coords=plot.coords)
            plot = tf.set_background(plot, 'black')

            rgb_focus = focus_plot.data.view(dtype='uint8').reshape(-1, 4)[:, :3].astype(np.float32) / 255.
            rgb_plot = plot.data.view(dtype='uint8').reshape(-1, 4)[:, :3].astype(np.float32) / 255.
            rgb_hue = hues_img.data.view(dtype='uint8').reshape(-1, 4)[:, :3].astype(np.float32) / 255.
            rgb_plot *= rgb_hue
            rgb_plot = 1 - (1 - rgb_edges*max_activation) * (1 - rgb_plot) + 0.2 * rgb_focus

            yiq_plot = np.matmul(rgb_plot, self.rgb_yiq_transform)

            # color_transform
            # hue_shift = time_position * np.pi * 2.
            hue_shift = 0.
            u = math.cos(hue_shift)
            w = math.sin(hue_shift)
            s = 1.0  # saturation
            v = 1.  # value
            color_transform = np.array([[v, 0, 0], [0, v*s*u, -v*s*w], [0, v*s*w, v*s*u]], dtype=np.float32)
            yiq_plot = np.matmul(yiq_plot, color_transform)

            rgb_plot = np.matmul(yiq_plot, self.yiq_rgb_transform)

            rgb_plot = (np.clip(rgb_plot, 0., 1.) * 255.).astype(np.uint8)
            rgb_plot = np.concatenate([rgb_plot, np.full((rgb_plot.shape[0], 1), 255, dtype=np.uint8)], axis=1)
            rgb_plot = rgb_plot.view('uint32').reshape(plot.data.shape[0], plot.data.shape[1])
            plot.data = rgb_plot

            img = plot.to_pil()
            #img = ImageChops.multiply(img, hues_img.to_pil())
            conn.send(img)
            # self.viz_window_label.imgtk = imgtk
            # self.viz_window_label.configure(image=imgtk)
            tok = time.time()
            #print("main viz interval:", tok - tik)
            tik = tok

    @staticmethod
    def interpolation_function(value):
        value = min(max(0., value), 1.)
        return value**2 * (3 - value*2)


if __name__ == '__main__':
    app = DreamingControlApp()



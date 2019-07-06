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
    'time_masking': 0,
    'pitch_masking': 0,
    'batch_size': 32,
    'activation_names': activations,
    'activation_statistics': '../data_statistics_snapshots_model_2019-05-20_run_0_100000.pickle',
    'layout': '/Users/vincentherrmann/Documents/Projekte/Immersions/visualization/layouts/e25_version_2/e25_positions_interp_100.npy',
    'hues': '/Users/vincentherrmann/Documents/Projekte/Immersions/visualization/layouts/e25_version_2/e25_positions_interp_100_hues.p',
    'connections': '/Users/vincentherrmann/Documents/Projekte/Immersions/visualization/layouts/e25_version_2/e25_positions_interp_100_edges_weighted.p'
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
    'batch_size': defaults_dict['batch_size']
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
        self.sample_rate = 16000

        self.rgb_yiq_transform = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]], dtype=np.float32)
        self.yiq_rgb_transform = np.array([[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.107, 1.705]], dtype=np.float32)

        self.root = tk.Tk()
        self.root.title('Immersions')

        with open(defaults['activation_statistics'], 'rb') as handle:
            self.activation_statistics = pickle.load(handle)

        self.layout = torch.from_numpy(np.load(defaults['layout']))
        with open(defaults['connections'], 'rb') as handle:
            self.connections = pickle.load(handle)
        with open(defaults['hues'], 'rb') as handle:
            self.hues = pickle.load(handle)
        self.num_frames = len(self.connections)

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

        tk.Label(self.dreaming_frame, text='learning rate').grid(row=0, column=0)
        self.lr_var = tk.DoubleVar(value=defaults['lr'])
        self.lr_slider = tk.Scale(self.dreaming_frame, variable=self.lr_var,
                                  from_=2., to=-5., resolution=0.1, length=200, command=self.send_control_dict)
        self.lr_slider.grid(row=1, column=0)

        tk.Label(self.dreaming_frame, text='time jitter').grid(row=0, column=1)
        self.time_jitter_var = tk.DoubleVar(value=defaults['time_jitter'])
        self.time_jitter_slider = tk.Scale(self.dreaming_frame, variable=self.time_jitter_var,
                                           from_=4., to=0., resolution=0.01, length=200, command=self.send_control_dict)
        self.time_jitter_slider.grid(row=1, column=1)

        tk.Label(self.dreaming_frame, text='noise loss').grid(row=0, column=2)
        self.noise_loss_var = tk.DoubleVar(value=defaults['noise_loss'])
        self.noise_loss_slider = tk.Scale(self.dreaming_frame, variable=self.noise_loss_var,
                                          from_=1., to=0., resolution=0.01, length=200, command=self.send_control_dict)
        self.noise_loss_slider.grid(row=1, column=2)

        tk.Label(self.dreaming_frame, text='time masking').grid(row=0, column=3)
        self.time_masking_var = tk.DoubleVar(value=defaults['time_masking'])
        self.time_masking_slider = tk.Scale(self.dreaming_frame, variable=self.time_masking_var,
                                            from_=1., to=0., resolution=0.01, length=200, command=self.send_control_dict)
        self.time_masking_slider.grid(row=1, column=3)

        tk.Label(self.dreaming_frame, text='pitch masking').grid(row=0, column=4)
        self.pitch_masking_var = tk.DoubleVar(value=defaults['pitch_masking'])
        self.pitch_masking_slider = tk.Scale(self.dreaming_frame, variable=self.pitch_masking_var,
                                             from_=1., to=0., resolution=0.01, length=200, command=self.send_control_dict)
        self.pitch_masking_slider.grid(row=1, column=4)

        tk.Label(self.dreaming_frame, text='batch size').grid(row=0, column=5)
        self.batch_size_var = tk.DoubleVar(value=defaults['batch_size'])
        self.batch_size_slider = tk.Scale(self.dreaming_frame, variable=self.batch_size_var,
                                          from_=64., to=0., resolution=1., length=200,
                                          command=self.send_control_dict)
        self.batch_size_slider.grid(row=1, column=5)

        tk.Label(self.dreaming_frame, text='activation selection').grid(row=0, column=6)
        self.activations_box = tk.Listbox(self.dreaming_frame)
        for a in self.activation_names:
            self.activations_box.insert(tk.END, a)
        self.activations_box.activate(1)
        self.activations_box.grid(row=1, column=6)

        tk.Label(self.dreaming_frame, text='channel').grid(row=0, column=7)
        self.channel_var = tk.DoubleVar(value=0)
        self.channel_slider = tk.Scale(self.dreaming_frame, variable=self.channel_var,
                                       from_=1., to=0., resolution=0.01, length=200, command=self.draw_target_viz)
        self.channel_slider.grid(row=1, column=7)

        tk.Label(self.dreaming_frame, text='channel region').grid(row=0, column=8)
        self.channel_region_var = tk.DoubleVar(value=0)
        self.channel_region_slider = tk.Scale(self.dreaming_frame, variable=self.channel_region_var,
                                       from_=1., to=0., resolution=0.01, length=200, command=self.draw_target_viz)
        self.channel_region_slider.grid(row=1, column=8)

        tk.Label(self.dreaming_frame, text='pitch').grid(row=0, column=9)
        self.pitch_var = tk.DoubleVar(value=0)
        self.pitch_slider = tk.Scale(self.dreaming_frame, variable=self.pitch_var,
                                     from_=1., to=0., resolution=0.01, length=200, command=self.draw_target_viz)
        self.pitch_slider.grid(row=1, column=9)

        tk.Label(self.dreaming_frame, text='pitch region').grid(row=0, column=10)
        self.pitch_region_var = tk.DoubleVar(value=0)
        self.pitch_region_slider = tk.Scale(self.dreaming_frame, variable=self.pitch_region_var,
                                            from_=1., to=0., resolution=0.01, length=200, command=self.draw_target_viz)
        self.pitch_region_slider.grid(row=1, column=10)

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

        self.region_label = tk.Label(self.viz_frame)
        self.region_label.grid(row=0, column=2)

        self.region_canvas = ds.Canvas(plot_width=300, plot_height=300,
                                       x_range=(-7, 7), y_range=(-7, 7),
                                       x_axis_type='linear', y_axis_type='linear')

        # visualization window
        self.viz_window = tk.Toplevel(self.root)
        self.viz_window.title('Visualization')
        self.viz_window.configure(background='black')
        self.viz_window.geometry('600x600')

        self.viz_window_size = (600, 600)

        self.viz_window_frame = tk.Frame(self.viz_window)
        self.viz_window_frame.pack()

        self.viz_window_label = tk.Label(self.viz_window_frame, borderwidth=0, highlightthickness=0,
                                         bg="black")
        self.viz_window_label.pack()
        #self.viz_window_label.place(x=0, y=0, width=self.viz_window_size[0], height=self.viz_window_size[1])
        #self.viz_window_label.pack(fill=tk.BOTH, expand=tk.YES)
        #self.viz_window.bind('<Configure>', self.resize_viz_window)

        self.viz_window_canvas = ds.Canvas(plot_width=600, plot_height=600,
                                           x_range=(-8, 8), y_range=(-8, 8),
                                           x_axis_type='linear', y_axis_type='linear')

        plot = activation_plot(self.layout[0], values=np.ones(self.layout.shape[1]),
                               canvas=self.viz_window_canvas)
        img = plot.to_pil()
        imgtk = ImageTk.PhotoImage(image=img)
        self.viz_window_label.imgtk = imgtk
        self.viz_window_label.configure(image=imgtk)

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
        del self.activation_dict['c_code']
        del self.activation_dict['z_code']
        del self.activation_dict['prediction']
        self.audio_loop.set_new_signal(data['audio'])
        self.draw_scalogram(data['scalogram'].numpy(), data['scalogram_grad'].numpy())

        self.viz_conn, conn = mp.Pipe(duplex=True)
        #conn=None
        self.visualization_process = mp.Process(target=self.main_viz_worker, args=(conn,))
        self.visualization_process.daemon = True
        self.visualize = True
        self.visualization_process.start()
        self.main_viz_tik = time.time()
        start_time = time.time() + (self.audio_loop.signal_length - self.audio_loop.position) / self.sample_rate
        loop_length = self.audio_loop.signal_length / self.sample_rate
        self.viz_conn.send((self.activation_dict, start_time, loop_length))

        # self.streaming_thread = threading.Thread(target=self.streaming_worker)
        # self.streaming_thread.daemon = True
        # self.stream = True
        # self.streaming_thread.start()

        self.audio_stream.start_stream()
        self.root.after(0, self.streaming_callback)
        self.root.mainloop()

    def send_control_dict(self, value=0):
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
            'time': 0.,
            'time_region': 1.,
            'lr': self.lr_var.get(),
            'time_jitter': self.time_jitter_var.get(),
            'noise_loss': self.noise_loss_var.get(),
            'time_masking': self.time_masking_var.get(),
            'pitch_masking': self.pitch_masking_var.get(),
            'batch_size': self.batch_size_var.get()
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

    def streaming_worker(self):
        while True:
            if self.communicator.new_data_available:
                data = pickle.loads(self.communicator.get_received_data())
                self.activation_dict = data['activations']
                del self.activation_dict['c_code']
                del self.activation_dict['z_code']
                del self.activation_dict['prediction']
                self.audio_loop.set_new_signal(data['audio'])
                self.draw_scalogram(data['scalogram'].numpy(), data['scalogram_grad'].numpy())
            self.draw_main_viz()

    def streaming_callback(self):
        if self.communicator.new_data_available:
            data = pickle.loads(self.communicator.get_received_data())
            self.activation_dict = data['activations']
            del self.activation_dict['c_code']
            del self.activation_dict['z_code']
            del self.activation_dict['prediction']
            self.audio_loop.set_new_signal(data['audio'])
            self.draw_scalogram(data['scalogram'].numpy(), data['scalogram_grad'].numpy())
        self.draw_main_viz()

    def draw_main_viz(self):
        img = self.viz_conn.recv()
        target_size = min(self.viz_window.winfo_width(), self.viz_window.winfo_height())
        img = img.resize((target_size, target_size), resample=Image.BICUBIC)
        imgtk = ImageTk.PhotoImage(image=img)
        self.viz_window_label.imgtk = imgtk
        self.viz_window_label.configure(image=imgtk)
        self.viz_window_label.after(50, self.streaming_callback)

    def main_viz_worker(self, conn):
        #activation_dict = None
        activation_dict, start_time, loop_length = conn.recv()
        tik = time.time()
        while self.visualize:
            if conn.poll(0):
                activation_dict, start_time, loop_length = conn.recv()
            if activation_dict is None:
                time.sleep(0.1)
                continue
            time_position = ((tik - start_time) % loop_length) / loop_length #self.audio_loop.position / self.audio_loop.signal_length
            time_frame = int(time_position * self.num_frames)
            current_activations = activation_dict.copy()
            for key, (start, length) in range_dict.items():
                current_activations[key] = current_activations[key][:, :, start:start + length]
            for key, value in current_activations.items():
                current_activations[key] = interpolate_position(value, time_position)
            #timestep_layout = interpolate_position(self.layout, time_position, dim=0)
            timestep_layout = self.layout[time_frame]
            #colormap = [(0, 0, 0), (80, 80, 80), (100, 100, 100), (110, 110, 110)]
            colormap = ['#000000', '#2f4858', '#33658a', '#86bbd8', '#9dd9d2', '#79bcb8']
            #colormap = [(0, 0, 0), (0, 0, 50), (10, 80, 10), (100, 20, 20)]
            #colormap = fire
            edges_img = tf.shade(self.connections[time_frame], cmap=colormap)
            hues_img = self.hues[time_frame]
            flat_activations = flatten_activations(current_activations)
            plot = activation_plot(timestep_layout, values=flat_activations.detach().cpu().numpy(),
                                   canvas=self.viz_window_canvas, spread=1, alpha=100, min_agg=-2, max_agg=2,
                                   colormap=[(0, 0, 0), (255, 255, 255)])
            plot = tf.set_background(tf.stack(edges_img, plot, how="over"), 'black')
            rgb_plot = plot.data.view(dtype='uint8').reshape(-1, 4)[:, :3].astype(np.float32) / 255.
            rgb_hue = hues_img.data.view(dtype='uint8').reshape(-1, 4)[:, :3].astype(np.float32) / 255.
            rgb_plot *= rgb_hue

            yiq_plot = np.matmul(rgb_plot, self.rgb_yiq_transform)

            # color_transform
            hue_shift = time_position * np.pi * 2.
            u = math.cos(hue_shift)
            w = math.sin(hue_shift)
            s = 0.5  # saturation
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
            print("main viz interval:", tok - tik)
            tik = tok

    def resize_viz_window(self, event):
        self.viz_window_size = (event.width, event.height)

    def draw_target_viz(self, value=0):
        self.send_control_dict()
        indices = self.activations_box.curselection()
        target_activations = collections.OrderedDict()
        for key, value in self.activation_statistics.items():
            if key in ['c_code', 'z_code', 'prediction']:
                continue
            if len(value['element_mean'].shape) == 2:
                target_activations[key] = 0. * value['element_mean'][:, 0]
            else:
                target_activations[key] = 0. * value['element_mean'][:, :, 0]
        for idx in indices:
            key = self.activation_names[idx]
            value = target_activations[key]
            value = value.unsqueeze(0)
            slice = select_activation_slice(value.unsqueeze(len(value.shape)),
                                            channel=self.channel_var.get(), channel_region=self.channel_region_var.get(),
                                            pitch=self.pitch_var.get(), pitch_region=self.pitch_region_var.get())
            slice += 1.

        activations = flatten_activations(target_activations)
        plot = activation_plot(self.layout, values=activations.detach().cpu().numpy(), canvas=self.region_canvas,
                               spread=1, min_agg=0., max_agg=1.)
        img = plot.to_pil()
        imgtk = ImageTk.PhotoImage(image=img)
        self.region_label.imgtk = imgtk
        self.region_label.configure(image=imgtk)

    def draw_scalogram(self, scalogram, scalogram_grad):
        self.scal_plot.set_data(scalogram)
        self.scal_plot.set_clim(vmin=scalogram.min(), vmax=scalogram.max())
        self.scal_canvas.draw()
        self.scal_grad_plot.set_data(scalogram_grad)
        self.scal_grad_plot.set_clim(vmin=scalogram_grad.min(), vmax=scalogram_grad.max())
        self.scal_grad_canvas.draw()


class MainVisualization:
    def __init__(self, app, width=600, height=600):
        self.app = app

        self.width = width
        self.height = height

        self.rgb_yiq_transform = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]],
                                          dtype=np.float32)
        self.yiq_rgb_transform = np.array([[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.107, 1.705]],
                                          dtype=np.float32)

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
                                           x_range=(-8, 8), y_range=(-8, 8),
                                           x_axis_type='linear', y_axis_type='linear')

        imgtk = ImageTk.PhotoImage(image=Image.new('RGB', (self.width, self.height), color='black'))
        self.viz_window_label.imgtk = imgtk
        self.viz_window_label.configure(image=imgtk)

        # Background Process
        self.viz_conn, conn = mp.Pipe(duplex=True)
        self.visualization_process = mp.Process(target=self.viz_worker, args=(conn,))
        self.visualization_process.daemon = True
        self.visualize = True
        self.visualization_process.start()
        self.main_viz_tik = time.time()
        audio_loop = self.app.audio_loop
        start_time = time.time() + (audio_loop.signal_length - audio_loop.position) / self.app.sample_rate
        loop_length = audio_loop.signal_length / self.app.sample_rate
        self.viz_conn.send((self.app.activation_dict, start_time, loop_length))

    def draw_main_viz(self):
        img = self.viz_conn.recv()
        target_size = min(self.viz_window.winfo_width(), self.viz_window.winfo_height())
        img = img.resize((target_size, target_size), resample=Image.BICUBIC)
        imgtk = ImageTk.PhotoImage(image=img)
        self.viz_window_label.imgtk = imgtk
        self.viz_window_label.configure(image=imgtk)
        self.viz_window_label.after(50, self.streaming_callback)

    def viz_worker(self, conn):
        activation_dict, start_time, loop_length = conn.recv()
        tik = time.time()
        while self.visualize:
            if conn.poll(0):
                activation_dict, start_time, loop_length = conn.recv()
            if activation_dict is None:
                time.sleep(0.1)
                continue
            time_position = ((tik - start_time) % loop_length) / loop_length
            time_frame = int(time_position * self.num_frames)
            current_activations = activation_dict.copy()
            for key, (start, length) in range_dict.items():
                current_activations[key] = current_activations[key][:, :, start:start + length]
            for key, value in current_activations.items():
                current_activations[key] = interpolate_position(value, time_position)
            timestep_layout = self.layout[time_frame]

            edges_colormap = ['#000000', '#2f4858', '#33658a', '#86bbd8', '#9dd9d2', '#79bcb8']
            edges_img = tf.shade(self.connections[time_frame], cmap=edges_colormap)

            hues_img = self.hues[time_frame]

            flat_activations = flatten_activations(current_activations)

            plot = activation_plot(timestep_layout, values=flat_activations.detach().cpu().numpy(),
                                   canvas=self.viz_window_canvas, spread=1, alpha=100, min_agg=-2, max_agg=2,
                                   colormap=[(0, 0, 0), (255, 255, 255)])
            plot = tf.set_background(tf.stack(edges_img, plot, how="over"), 'black')

            rgb_plot = plot.data.view(dtype='uint8').reshape(-1, 4)[:, :3].astype(np.float32) / 255.
            rgb_hue = hues_img.data.view(dtype='uint8').reshape(-1, 4)[:, :3].astype(np.float32) / 255.
            rgb_plot *= rgb_hue

            yiq_plot = np.matmul(rgb_plot, self.rgb_yiq_transform)

            # color_transform
            hue_shift = time_position * np.pi * 2.
            u = math.cos(hue_shift)
            w = math.sin(hue_shift)
            s = 0.5  # saturation
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
            print("main viz interval:", tok - tik)
            tik = tok




if __name__ == '__main__':
    app = DreamingControlApp()



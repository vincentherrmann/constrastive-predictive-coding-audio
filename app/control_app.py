import qtpy.QtGui
import qtpy.QtWidgets
import pyqtgraph as pg
import sys
import numpy as np
import pickle
from app.midi_controls import *
from openGLviz.net_visualizer import Visualizer
from app.control_utilities import ModelActivations
from app.visualization import Visualization
from dreaming.midi_controller import MidiController
import multiprocessing as mp
from vispy import app


class ControlApp:
    def __init__(self, model_activations, midi_controller=None, midi_mapping_function=None, viz_communicator=None):
        self.model_activations = model_activations
        self.layer_names = list(self.model_activations.shapes.keys())
        self.viz_communicator = viz_communicator

        self.app = qtpy.QtWidgets.QApplication(sys.argv)
        self.main_window = qtpy.QtWidgets.QWidget()
        self.main_layout = qtpy.QtWidgets.QGridLayout()

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        img_label = qtpy.QtWidgets.QLabel()
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img = qtpy.QtGui.QImage(img, 200, 200, qtpy.QtGui.QImage.Format_RGB888)

        # clip controls
        self.clip_layout = qtpy.QtWidgets.QGridLayout()
        self.learning_rate_slider = MidiSlider('learning rate', min=-5., max=2., step_size=0.1)
        self.original_clip_list = MidiListSelect('original clip',
                                                 items=['noise', 'silence', 'beat'])
        self.original_clip_button = MidiButton('select')
        self.original_clip_mix = MidiSlider('mix original')
        self.clip_layout.addWidget(self.learning_rate_slider, 0, 0, 2, 1)
        self.clip_layout.addWidget(self.original_clip_list, 0, 1)
        self.clip_layout.addWidget(self.original_clip_button, 1, 1)
        self.clip_layout.addWidget(self.original_clip_mix, 0, 2, 2, 1)

        # eq
        self.eq_layout = qtpy.QtWidgets.QGridLayout()
        # self.eq_label = qtpy.QtWidgets.QLabel()
        # eq_pixmap = qtpy.QtGui.QPixmap(img).scaled(240, 100)
        # self.eq_label.setPixmap(eq_pixmap)
        self.eq_plot = pg.PlotWidget()
        self.eq_plot.setMaximumHeight(100)
        #self.eq_plot.setAntialiasing(True)
        self.eq_plot.plot([0, 1, 2], [0., 1., 0.], pen=pg.mkPen('b', width=2.))
        self.eq_plot.showAxis('bottom', False)
        self.eq_plot.showAxis('left', False)
        self.eq_plot.setRange(yRange=(-1., 1.))
        self.lows_dial = MidiDial('lows')
        self.mids_dial = MidiDial('mids')
        self.highs_dial = MidiDial('highs')
        self.eq_layout.addWidget(self.eq_plot, 0, 0, 1, 3, alignment=qtpy.QtCore.Qt.AlignHCenter)
        self.eq_layout.addWidget(self.lows_dial, 1, 0)
        self.eq_layout.addWidget(self.mids_dial, 1, 1)
        self.eq_layout.addWidget(self.highs_dial, 1, 2)

        self.clip_layout.addLayout(self.eq_layout, 2, 0, 1, 3)
        self.main_layout.addLayout(self.clip_layout, 0, 0)

        # scalograms
        self.scalograms_layout = qtpy.QtWidgets.QVBoxLayout()

        self.scalogram_label = qtpy.QtWidgets.QLabel()
        scalogram_pixmap = qtpy.QtGui.QPixmap(img).scaled(400, 200)
        self.scalogram_label.setPixmap(scalogram_pixmap)
        self.gradient_label = qtpy.QtWidgets.QLabel()
        gradient_pixmap = qtpy.QtGui.QPixmap(img).scaled(400, 200)
        self.gradient_label.setPixmap(gradient_pixmap)
        self.scalograms_layout.addWidget(self.scalogram_label)
        self.scalograms_layout.addWidget(self.gradient_label)

        self.dial_controls_layout = qtpy.QtWidgets.QHBoxLayout()
        self.time_jitter_dial = MidiDial('time jitter', min=0., max=4., step_size=0.1)
        self.time_masking_dial = MidiDial('time masking')
        self.pitch_masking_dial = MidiDial('pitch masking')
        self.activation_loss_dial = MidiDial('activation loss', min=0., max=50., step_size=0.1)
        self.dial_controls_layout.addWidget(self.time_jitter_dial)
        self.dial_controls_layout.addWidget(self.time_masking_dial)
        self.dial_controls_layout.addWidget(self.pitch_masking_dial)
        self.dial_controls_layout.addWidget(self.activation_loss_dial)

        self.scalograms_layout.addLayout(self.dial_controls_layout)
        self.main_layout.addLayout(self.scalograms_layout, 0, 1)

        # target
        node_positions = np.load(
            '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13_2/e32_positions_interp_240.npy').astype(
            np.float32)[0:1]
        self.target_visualizer = Visualizer(node_positions=node_positions, animate=False, translate=False)
        self.target_visualizer.node_colors = np.float32([[0., 0.8, 0.5]])
        #self.model_activations.select_activations('scalogram', pitch=0.2, pitch_region=0.5)
        self.target_visualizer.focus = self.model_activations.focus #node_positions[0, :, 0] > 0.5
        self.target_visualizer_widget = self.target_visualizer.native
        self.target_visualizer_widget.setMinimumWidth(400)
        self.target_visualizer_widget.setMinimumHeight(400)
        self.target_layout = qtpy.QtWidgets.QGridLayout()

        self.layer_selection_layout = qtpy.QtWidgets.QVBoxLayout()
        self.layer_selection_toggle = MidiSwitch('keep selection', width=120)
        self.layer_selection_list = MidiListSelect('layer selection',
                                                   items=self.layer_names, size=(200, 40))
        self.layer_selection_button = MidiButton('select', command=self.change_target)
        self.layer_selection_layout.addWidget(self.layer_selection_toggle)
        self.layer_selection_layout.addWidget(self.layer_selection_list)
        self.layer_selection_layout.addWidget(self.layer_selection_button, alignment=qtpy.QtCore.Qt.AlignHCenter)
        self.target_layout.addLayout(self.layer_selection_layout, 0, 0, 1, 2)
        # self.selection_diagram_label = qtpy.QtWidgets.QLabel()
        # selection_diagram_pixmap = qtpy.QtGui.QPixmap(img).scaled(320, 320, qtpy.QtCore.Qt.KeepAspectRatio)
        # self.selection_diagram_label.setPixmap(selection_diagram_pixmap)
        self.target_layout.addWidget(self.target_visualizer_widget, 0, 2, 1, 4)

        self.target_channel = MidiSlider('channel', length=100)
        self.target_channel_region = MidiSlider('region', length=100)
        self.target_pitch = MidiSlider('pitch', length=100)
        self.target_pitch_region = MidiSlider('region', length=100)
        self.target_time = MidiSlider('time', length=100)
        self.target_time_region = MidiSlider('region', length=100)

        target_slider_row = 1
        current_column = 0
        self.target_layout.addWidget(self.target_channel, target_slider_row, current_column)
        current_column += 1
        self.target_layout.addWidget(self.target_channel_region, target_slider_row, current_column)
        current_column += 1
        self.target_layout.addWidget(self.target_pitch, target_slider_row, current_column)
        current_column += 1
        self.target_layout.addWidget(self.target_pitch_region, target_slider_row, current_column)
        current_column += 1
        self.target_layout.addWidget(self.target_time, target_slider_row, current_column)
        current_column += 1
        self.target_layout.addWidget(self.target_time_region, target_slider_row, current_column)
        current_column += 1

        self.main_layout.addLayout(self.target_layout, 0, 2)

        self.main_window.setLayout(self.main_layout)
        self.main_window.setWindowTitle("control window")
        self.main_window.show()

        if midi_controller is not None and midi_mapping_function is not None:
            self.midi_controller = midi_controller
            mapping = midi_mapping_function(self)
            self.midi_controller.set_control_mapping(mapping)

        self.app.exec_‚()

    def change_target(self):
        selected_layer = self.layer_names[self.layer_selection_list.value()]
        activation_selection_dict = {
            'layer': selected_layer,
            'channel': self.target_channel.value(),
            'channel_region': self.target_channel_region.value(),
            'pitch': self.target_pitch.value(),
            'pitch_region': self.target_pitch_region.value(),
            'time': self.target_time.value(),
            'time_region': self.target_time_region.value(),
            'keep_selection': self.layer_selection_toggle.value()
        }
        selection_pickle = pickle.dumps(activation_selection_dict)
        if self.viz_communicator is not None:
            self.viz_communicator.set_new_data(selection_pickle)
        self.model_activations.select_activations(activation_selection_dict)
        self.target_visualizer.focus = self.model_activations.focus
        self.target_visualizer.update()

if __name__ == '__main__':
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
            #switches
            75: c.layer_selection_toggle,
            #buttons
            66: c.original_clip_button,
            67: c.layer_selection_button
        }
        return mapping

    visualization_process = mp.Process

    path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/activation_shapes.pickle'
    activations = ModelActivations(path, ignore_time_dimension=True, remove_results=True)
    midi_controller = MidiController('BCF2000')
    control_app = ControlApp(model_activations=activations,
                             midi_controller=midi_controller,
                             midi_mapping_function=define_midi_control_mapping)
    #app.run()
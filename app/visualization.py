import numpy as np
import pickle
from openGLviz.net_visualizer import Visualizer
from vispy import gloo, app
import vispy
from threading import Thread
import time
import qtpy.QtWidgets
from matplotlib.colors import hex2color


class Visualization:
    def __init__(self, node_positions, edges_textures, model_activations, control_communicator=None):
        edge_colors = self.hexes2colors(['#000000', '#3f34a0', '#334f9a', '#337294', '#338e8c'])
        # node_colors = hexes2colors(['#3d3cb5', '#2e7c89', '#349d55', '#a9ad3a', '#b8663d', '#b33c4c', '#933165'])
        # node_colors = hexes2colors(['#005cff', '#389d34', '#be4f3f', '#ffdc28', '#00af67', '#ba3ea0', '#00a4ff'])
        node_colors = self.hexes2colors(
            ['#005cff', '#ffdc28', '#005cff', '#ffdc28', '#005cff', '#ffdc28', '#005cff', '#ffdc28'])

        self.control_communicator = control_communicator
        self.model_activations = model_activations

        self.viz = Visualizer(node_positions=node_positions,
                              edge_textures=edges_textures,
                              node_weights=None,
                              draw_callback=self.on_draw_callback)
        self.viz.transition_frames = 240

        self.viz.edges_colors = edge_colors
        self.viz.node_colors = node_colors
        self.viz.scale_factor = 6.

        self.window = qtpy.QtWidgets.QMainWindow()
        self.window.setCentralWidget(self.viz.native)
        self.window.show()
        app.Timer(1 / 60., connect=self.viz.update, start=True)
        app.run()

    def on_draw_callback(self):
        if self.control_communicator.new_data_available:
            selection_dict = pickle.loads(self.control_communicator.get_received_data())
            self.model_activations.select_activations(selection_dict)
            self.viz.focus = self.model_activations.focus

    @staticmethod
    def hexes2colors(h):
        colors = [list(hex2color(c)) for c in h]
        colors = np.float32(colors)
        colors = np.concatenate([colors, np.ones([len(h), 1], dtype=np.float32)], axis=1)
        return colors







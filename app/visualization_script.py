import numpy as np
from app.visualization import Visualization
from dreaming.streaming import *
from app.control_utilities import ModelActivations


print("started visualization script")

viz_control_server = SocketDataExchangeClient(port=8001,
                                              host='127.0.0.1',
                                              stream_automatically=True)

node_positions = np.load(
    '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13_2/e32_positions_interp_240.npy').astype(
    np.float32)

# node_weights = np.load(
#     '/Users/vincentherrmann/Documents/Projekte/Immersions/visualization/layouts/e25_version_3/e25_positions_interp_100_weights.npy')
# node_weights = node_weights.astype(np.float32)
# node_weights = np.sqrt(node_weights)

focus = np.zeros(node_positions.shape[1])
focus[10000:10100] += 1.
focus = focus > 0.

edges_textures = np.load(
    '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13_2/e32_positions_interp_240_edges_weighted.npy') * 1.
edges_textures = edges_textures.astype(np.float32)
edges_textures[edges_textures != edges_textures] = 0.
edges_textures /= edges_textures.max()
edges_textures = np.power(edges_textures, 0.3) * 1.

viz_control_server.set_new_data(b'abc')

path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/activation_shapes.pickle'
activations = ModelActivations(path, ignore_time_dimension=True, remove_results=True)

Visualization(node_positions, edges_textures, model_activations=activations, control_communicator=viz_control_server)
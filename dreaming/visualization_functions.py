import datashader as ds
import numpy as np
import pandas as pd
from colorcet import fire
from datashader import transfer_functions as tf


def activation_plot(positions, values, canvas=None, colormap=None):
    node_data = np.concatenate([positions, values[:, np.newaxis]], axis=1)
    df = pd.DataFrame(data=node_data)
    df.columns = ['x', 'y', 'val']

    if canvas is None:
        canvas = ds.Canvas(plot_width=600, plot_height=600,
                           x_range=(-7, 7), y_range=(-7, 7),
                           x_axis_type='linear', y_axis_type='linear')

    if colormap is None:
        colormap = [(0, 0, 0), (255, 255, 255)]

    node_img = tf.shade(canvas.points(df, 'x', 'y', ds.mean('val')), cmap=colormap, alpha=127)
    image = tf.set_background(tf.dynspread(node_img, threshold=0.9, max_px=5, shape='circle'), 'black')
    return image


def normalize_activations(activation_dict, statistics_dict):
    for key, value in activation_dict.items():
        value -= statistics_dict[key]['element_mean']
        value /= statistics_dict[key]['element_var']
        activation_dict[key] = value
    return activation_dict


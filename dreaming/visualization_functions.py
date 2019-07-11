import datashader as ds
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
from colorcet import fire
from dask import dataframe as dd
from datashader import transfer_functions as tf
from datashader.colors import Hot, inferno, viridis


def activation_plot(positions, values=None, canvas=None, colormap=None, spread=1, min_agg=-10., max_agg=10., alpha=127,
                    reduction=ds.sum):
    if values is None:
        values = torch.ones_like(positions[:, 0])
    node_data = np.concatenate([positions, values[:, np.newaxis]], axis=1)
    df = pd.DataFrame(data=node_data)
    df.columns = ['x', 'y', 'val']

    if canvas is None:
        canvas = ds.Canvas(plot_width=600, plot_height=600,
                           x_range=(-7, 7), y_range=(-7, 7),
                           x_axis_type='linear', y_axis_type='linear')

    if colormap is None:
        colormap = viridis

    aggregate = canvas.points(df, 'x', 'y', reduction('val'))
    # valid_aggregate = aggregate.data[np.logical_not(np.isnan(aggregate.data))]
    # min_val = np.min(valid_aggregate)
    # max_val = np.max(valid_aggregate)
    # mean_val = np.mean(valid_aggregate)
    # var_val = np.var(valid_aggregate)
    #print("aggregation range:", min_val, "-", max_val, "mean:", mean_val, "var:", var_val)
    node_img = tf.shade(aggregate, cmap=colormap, alpha=alpha, span=[min_agg, max_agg], how='linear')
    if spread > 0.:
        node_img = tf.spread(node_img, spread, shape='circle')
    #image = tf.set_background(node_img, 'black')
    return node_img


def normalize_activations(activation_dict, statistics_dict=None, element_wise=True, eps=1e-6):
    for key, value in activation_dict.items():
        if statistics_dict is None:
            dim = len(value.shape) - 1
            if value.shape[dim] > 1:
                mean = torch.mean(value, dim=dim).unsqueeze(dim)
                var = torch.var(value, dim=dim).unsqueeze(dim)
                value = (value - mean) / (var + eps)
        elif element_wise:
            value -= statistics_dict[key]['element_mean']
            value /= statistics_dict[key]['element_var'] + eps
        else:
            value -= statistics_dict[key]['global_mean']
            value /= statistics_dict[key]['global_var'] + eps
        activation_dict[key] = value
    return activation_dict

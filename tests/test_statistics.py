from unittest import TestCase
import torch
import pickle
from matplotlib import pyplot as plt

class TestStatistics(TestCase):
    def test_statistics_dicts(self):
        noise_statistics_path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/noise_statistics_snapshots_model_2019-08-13_run_0_90000.pickle'
        with open(noise_statistics_path, 'rb') as handle:
            noise_statistics = pickle.load(handle)
        plt.imshow(noise_statistics['scalogram']['element_mean'][0], origin='lower')
        plt.colorbar()
        plt.show()
        plt.imshow(noise_statistics['scalogram']['element_var'][0], origin='lower')
        plt.colorbar()
        plt.show()

        data_statistics_path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/data_statistics_snapshots_model_2019-08-13_run_0_90000.pickle'
        with open(data_statistics_path, 'rb') as handle:
            data_statistics = pickle.load(handle)
        plt.imshow(data_statistics['scalogram']['element_mean'][0], origin='lower')
        plt.colorbar()
        plt.show()
        plt.imshow(data_statistics['scalogram']['element_var'][0], origin='lower')
        plt.colorbar()
        plt.show()
        pass
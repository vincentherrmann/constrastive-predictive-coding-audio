from unittest import TestCase
from scalogram_model import PreprocessingModule
from audio_dataset import AudioDataset
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class TestPreprocessingModule(TestCase):
    def test_scalogram_preprocessing(self):
        cqt_default_dict = {'sample_rate': 44100,
                            'fmin': 30,
                            'n_bins': 292,
                            'bins_per_octave': 32,
                            'filter_scale': 0.5,
                            'hop_length': 256,
                            'trainable_cqt': False}

        dataset = '/Volumes/Elements/Datasets/MelodicProgressiveHouse_mp3'
        dataset = AudioDataset(location=dataset,
                               item_length=44100*5)

        #audio_clip = torch.rand([1, 1, 44100*5]) * 1. - 0.5
        audio_clip = dataset[20000][0, :].view(1, 1, -1)
        prep_module = PreprocessingModule(cqt_default_dict, phase=False, offset_zero=True, output_power=2,
                                          pooling=[1, 2])

        x = prep_module(audio_clip)
        x_min = x.min()
        x_max = x.max()
        plt.imshow(x[0, 0], origin='lower')
        plt.show()
        pass

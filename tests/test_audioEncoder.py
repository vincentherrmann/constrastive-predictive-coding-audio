from unittest import TestCase
from audio_model import *

import torch

encoder_test_dict = {'strides': [5, 4, 2, 2, 2],
                     'kernel_sizes': [10, 8, 4, 4, 4],
                     'channel_count': [32, 32, 32, 32, 32],
                     'bias': True}


class TestAudioEncoder(TestCase):
    def setUp(self):
        self.encoder = AudioEncoder({'strides': [5, 4, 2, 2, 2],
                                     'kernel_sizes': [10, 8, 4, 4, 4],
                                     'channel_count': [32, 32, 32, 32, 32],
                                     'bias': False})

    def test_audioEncoder(self):
        assert self.encoder.downsampling_factor == 160

        test_input = torch.randn([7, 1, 4800])  # batch_size, channels, length
        test_output = self.encoder(test_input)

        assert list(test_output.shape) == [7, 32, 28]

    def test_receptive_field(self):
        assert self.encoder.receptive_field == 465

        # set all parameter values to 0.1
        for p in self.encoder.parameters():
            p.requires_grad = False
            p += 0.
            p += 0.1

        # set value inside the receptive field to 1.
        test_input = torch.zeros([7, 1, 2000])
        test_input[:, :, self.encoder.receptive_field-1] += 1.
        test_output = self.encoder(test_input)

        assert test_output[0, 0, 0] != 0.

        # set value outside the receptive field to 1.
        test_input = torch.zeros([7, 1, 2000])
        test_input[:, :, self.encoder.receptive_field] += 1.
        test_output = self.encoder(test_input)

        assert test_output[0, 0, 0] == 0.



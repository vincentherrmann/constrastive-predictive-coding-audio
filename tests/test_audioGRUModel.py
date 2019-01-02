from unittest import TestCase
from audio_model import *

import torch


class TestAudioGRUModel(TestCase):
    def test_audioGRUModel(self):
        model = AudioGRUModel(input_size=32, hidden_size=64)

        test_input = torch.randn([7, 13, 32])  # batch, steps, input_size
        test_output = model(test_input)

        assert list(test_output.shape) == [7, 64]

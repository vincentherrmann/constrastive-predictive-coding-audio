from unittest import TestCase
from audio_model import *
import torch


class TestConvArModel(TestCase):
    def test_convArModel(self):
        model = ConvArModel()
        test_input = torch.rand([7, 512, 118])
        test_output = model(test_input)
        assert list(test_output.shape) == [7, 256]

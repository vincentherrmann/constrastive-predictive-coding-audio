from unittest import TestCase
from audio_model import *
from configs.autoregressive_model_configs import *
import torch


class TestConvArModel(TestCase):
    def test_convArModel(self):
        model = ConvArModel()
        test_input = torch.rand([7, 512, 118])
        test_output = model(test_input)
        assert list(test_output.shape) == [7, 256]

    def test_conbBlockModel(self):
        model = ConvolutionalArModel(args_dict=ar_conv_architecture_2)
        print("num parameters:", num_parameters(model))
        test_input = torch.rand([7, 256, 60])
        output = model(test_input)
        assert False

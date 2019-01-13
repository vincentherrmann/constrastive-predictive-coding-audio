from unittest import TestCase
from audio_model import *
from attention_model import *


class TestPositionalEncoder(TestCase):
    def test_encoding(self):
        encoder = PositionalEncoder(512, 128)
        encoding = encoder.pe
        pass


class TestAttentionModel(TestCase):
    def test_model(self):
        model = AttentionModel(channels=512,
                               output_size=256,
                               num_layers=2,
                               num_heads=8,
                               feedforward_size=512)

        print("parameter count:", num_parameters(model))

        test_input = torch.zeros([7, 512, 128])
        test_output = model(test_input)
        pass


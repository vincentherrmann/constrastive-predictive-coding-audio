from unittest import TestCase
from audio_model import *
from attention_model import *
from matplotlib import pyplot as plt


class TestPositionalEncoder(TestCase):
    def test_encoding(self):
        encoder = PositionalEncoder(256, 64, max_wavelength=5000)
        encoding = encoder.pe

        plt.imshow(encoding.squeeze())
        plt.show()
        pass


class TestAttentionModel(TestCase):
    def test_model(self):
        attention_default_dict = {
            'model': AttentionModel,
            'channels': 512,
            'output_size': 256,
            'num_layers': 6,
            'num_heads': 8,
            'feedforward_size': 2048,
            'sequence_length': 60,
            'dropout': 0.1,
            'encoding_size': 512,
            'ar_code_size': 512
        }

        model = AttentionModel(attention_default_dict)
        print(model)

        print("parameter count:", num_parameters(model))

        test_input = torch.zeros([7, 60, 512])
        test_output = model(test_input)
        pass


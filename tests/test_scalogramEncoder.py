from unittest import TestCase
from scalogram_model import *
from audio_model import *

import torch
import time


class TestScalogramEncoder(TestCase):
    def test_scalogram_encoder(self):
        args_dict = scalogram_encoder_default_dict
        #args_dict['channel_count'] = [1, 8, 8, 16, 16, 32, 32, 64, 64]
        args_dict['filter_scale'] = 0.5
        model = ScalogramEncoder(args_dict)
        print("num parameters:", num_parameters(model))
        print(model)
        item_length = model.receptive_field + (180) * model.downsampling_factor

        test_input = torch.randn([16, 1, item_length])
        tic = time.time()
        test_output = model(test_input)
        toc = time.time()
        print("duration:", toc - tic)
        assert list(test_output.shape) == [16, 512, 180]
        pass

    def test_seperable_scalogram_encoder(self):
        args_dict = scalogram_encoder_default_dict
        #args_dict['channel_count'] = [1, 64, 64, 128, 128, 256, 512]
        args_dict['filter_scale'] = 0.5
        args_dict['batch_norm'] = True
        model = ScalogramSeperableEncoder(args_dict)
        print("num parameters:", num_parameters(model))
        print(model)
        item_length = model.receptive_field + (180) * model.downsampling_factor

        test_input = torch.randn([16, 1, item_length])
        tic = time.time()
        test_output = model(test_input)
        toc = time.time()
        print("duration:", toc - tic)
        assert list(test_output.shape) == [16, 512, 180]
        pass

    def test_seperable_scalogram_encoder_phase(self):
        args_dict = scalogram_encoder_default_dict
        #args_dict['channel_count'] = [1, 64, 64, 128, 128, 256, 512]
        args_dict['filter_scale'] = 0.5
        #args_dict['batch_norm'] = True
        args_dict['phase'] = True
        model = ScalogramSeperableEncoder(args_dict)
        print("num parameters:", num_parameters(model))
        print(model)
        item_length = model.receptive_field + (180) * model.downsampling_factor

        test_input = torch.randn([16, 1, item_length])
        tic = time.time()
        test_output = model(test_input)
        toc = time.time()
        print("duration:", toc - tic)
        assert list(test_output.shape) == [16, 512, 180]
        pass
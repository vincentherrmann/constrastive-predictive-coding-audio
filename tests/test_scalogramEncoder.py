from unittest import TestCase
from scalogram_model import *
from audio_model import *
from audio_dataset import *
from contrastive_estimation_training import *

import torch
import time
import torch.nn as nn

from matplotlib import pyplot as plt
from configs.scalogram_resnet_configs import *


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

    def test_separable_scalogram_encoder(self):
        args_dict = scalogram_encoder_default_dict
        #args_dict['channel_count'] = [1, 64, 64, 128, 128, 256, 512]
        args_dict['filter_scale'] = 0.5
        args_dict['batch_norm'] = True
        args_dict['separable'] = True
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

    def test_strided_scalogram_encoder_phase(self):
        args_dict = scalogram_encoder_stride_dict
        #args_dict['channel_count'] = [1, 64, 64, 128, 128, 256, 512]
        #args_dict['filter_scale'] = 0.5
        #args_dict['batch_norm'] = True
        args_dict['phase'] = True
        args_dict['instance_norm'] = True
        args_dict['separable'] = True
        model = ScalogramEncoder(args_dict)
        print("num parameters:", num_parameters(model))
        print(model)
        item_length = model.receptive_field + (118) * model.downsampling_factor

        test_input = torch.randn([16, 1, item_length])
        tic = time.time()
        test_output = model(test_input)
        toc = time.time()
        print("duration:", toc - tic)
        assert list(test_output.shape) == [16, 512, 118]
        pass

    def test_residual_scalogram_encoder_phase(self):
        args_dict = scalogram_encoder_stride_dict
        #args_dict['channel_count'] = [1, 64, 64, 128, 128, 256, 512]
        #args_dict['filter_scale'] = 0.5
        #args_dict['batch_norm'] = True
        args_dict['phase'] = True
        args_dict['separable'] = True
        model = ScalogramResidualEncoder(args_dict)
        print("num parameters:", num_parameters(model))
        print(model)
        item_length = model.receptive_field + (118) * model.downsampling_factor

        test_input = torch.randn([16, 1, item_length])
        tic = time.time()
        test_output = model(test_input)
        toc = time.time()
        print("duration:", toc - tic)
        assert list(test_output.shape) == [16, 512, 118]

    def test_zero_init_training(self):
        args_dict = scalogram_encoder_default_dict
        args_dict['channel_count'] = [1, 16, 16, 32, 32, 64, 64]
        args_dict['filter_scale'] = 0.5
        args_dict['separable'] = True
        args_dict['lowpass_init'] = 60.
        encoder = ScalogramEncoder(args_dict)
        #encoder_modules = list(encoder.module_list)
        #lowpass_init(encoder_modules[1].weight, 60)
        #lowpass_init(encoder_modules[6].weight, 60)
        #lowpass_init(encoder_modules[11].weight, 60)
        #nn.init.constant_(encoder_modules[1].weight, 0.0)
        #encoder_modules[1].weight.register_hook(lambda grad: grad + torch.randn_like(grad) * 1e-6)
        #nn.init.constant_(encoder_modules[6].weight, 0.0)
        #encoder_modules[6].weight.register_hook(lambda grad: grad + torch.randn_like(grad) * 1e-6)
        #nn.init.constant_(encoder_modules[11].weight, 0.0)
        #encoder_modules[11].weight.register_hook(lambda grad: grad + torch.randn_like(grad) * 1e-6)

        visible_steps = 118
        ar_model = ConvArModel(in_channels=64, conv_channels=64, out_channels=64)
        pc_model = AudioPredictiveCodingModel(encoder, ar_model, enc_size=64, ar_size=64, prediction_steps=16)
        item_length = encoder.receptive_field + (visible_steps + pc_model.prediction_steps) * encoder.downsampling_factor
        dataset = AudioDataset(location='/Users/vincentherrmann/Documents/Projekte/Immersions/MelodicProgressiveHouse_Tracks_test',
                               item_length=item_length)
        visible_length = encoder.receptive_field + (visible_steps - 1) * encoder.downsampling_factor
        prediction_length = encoder.receptive_field + (pc_model.prediction_steps - 1) * encoder.downsampling_factor
        trainer = ContrastiveEstimationTrainer(model=pc_model,
                                                    dataset=dataset,
                                                    visible_length=visible_length,
                                                    prediction_length=prediction_length)
        trainer.train(8)

    def test_resnet(self):
        args_dict = scalogram_encoder_resnet_dict
        args_dict['separable'] = True
        encoder = ScalogramResidualEncoder(args_dict)
        visible_steps = 60
        print(encoder)
        print("receptive field:", encoder.receptive_field)

        # test_run
        #tic = time.time()
        #test_result = encoder(torch.rand(16, 1, encoder.receptive_field + 1*encoder.downsampling_factor))
        #print("encoder time:", time.time() - tic)
#
        #return

        ar_model = ConvArModel(in_channels=256, conv_channels=512, out_channels=256)
        pc_model = AudioPredictiveCodingModel(encoder, ar_model, enc_size=256, ar_size=256, prediction_steps=16, visible_steps=visible_steps)
        item_length = encoder.receptive_field + (
                    visible_steps + pc_model.prediction_steps) * encoder.downsampling_factor
        dataset = AudioDataset(
            location='/Users/vincentherrmann/Documents/Projekte/Immersions/MelodicProgressiveHouse_Tracks_test',
            item_length=item_length)
        visible_length = encoder.receptive_field + (visible_steps - 1) * encoder.downsampling_factor
        prediction_length = encoder.receptive_field + (pc_model.prediction_steps - 1) * encoder.downsampling_factor
        trainer = ContrastiveEstimationTrainer(model=pc_model,
                                               dataset=dataset)
        trainer.train(8, max_steps=1)
        print("finished")

    def test_resnet_architecture(self):
        block_1 = default_encoder_block_dict.copy()     #   1 x 500 x 256
        block_1['in_channels'] = 1                      #  32 x 500 x 256
        block_1['out_channels'] = 32                    #  32 x 500 x 256
        block_1['padding_1'] = 1
        block_1['padding_2'] = 1

        block_2 = default_encoder_block_dict.copy()            #  32 x 500 x 256
        block_2['in_channels'] = 32                     #
        block_2['out_channels'] = 64
        block_2['kernel_size_2'] = (64, 1)
        block_2['padding_1'] = 1
        block_2['stride_1'] = 2
        block_2['top_padding_2'] = 63

        block_3 = default_encoder_block_dict.copy()
        block_3['in_channels'] = 64
        block_3['out_channels'] = 64
        block_3['padding_1'] = 1

        block_4 = default_encoder_block_dict.copy()
        block_4['in_channels'] = 64
        block_4['out_channels'] = 128
        block_4['kernel_size_2'] = (30, 1)
        block_4['padding_1'] = 1
        block_4['stride_1'] = 2

        block_5 = default_encoder_block_dict.copy()
        block_5['in_channels'] = 128
        block_5['out_channels'] = 128
        block_5['padding_1'] = 1
        block_5['padding_2'] = 1

        block_6 = default_encoder_block_dict.copy()
        block_6['in_channels'] = 128
        block_6['out_channels'] = 256
        block_6['kernel_size_2'] = (15, 1)
        block_6['padding_1'] = 1
        block_6['stride_1'] = 2

        block_7 = default_encoder_block_dict.copy()
        block_7['in_channels'] = 256
        block_7['out_channels'] = 256
        block_7['padding_1'] = 1
        block_7['padding_2'] = 0

        args_dict = scalogram_encoder_resnet_dict
        args_dict['blocks'] = [block_1,
                               block_2,
                               block_3,
                               block_4,
                               block_5,
                               block_6,
                               block_7]

        encoder = ScalogramResidualEncoder(args_dict=scalogram_resnet_architecture_1, verbose=2)
        tic = time.time()
        test_result = encoder(torch.rand(1, 1, encoder.receptive_field + 60*encoder.downsampling_factor))
        print("encoder time:", time.time() - tic)






import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from constant_q_transform import *


scalogram_encoder_default_dict = {'kernel_sizes': [(127, 1), (5, 5), (63, 1), (5, 5), (26, 1), (5, 5)],
                                  'top_padding': [126, 0, 0, 0, 0, 0],
                                  'channel_count': [1, 128, 128, 256, 256, 512, 512],
                                  'pooling': [1, 2, 1, 2, 1, 1],
                                  'bias': True,
                                  'sample_rate': 16000,
                                  'fmin': 30,
                                  'n_bins': 256,
                                  'bins_per_octave': 32,
                                  'filter_scale': 1.,
                                  'hop_length': 128,
                                  'trainable_cqt': False}

# 500 x 256 -> (127, 1) with 126 padding
# 500 x 256 -> (5, 5)
# 496 x 252 -> pool
# 248 x 126 -> (63, 1)
# 248 x  64 -> (5, 5)
# 244 x  60 -> pool
# 122 x  30 -> (26, 1)
# 122 x   5 -> (5, 5)
# 118 x   1


class ScalogramEncoder(nn.Module):
    def __init__(self, args_dict=scalogram_encoder_default_dict):
        super().__init__()
        self.num_layers = len(args_dict['kernel_sizes'])

        self.cqt = CQT(sr=args_dict['sample_rate'],
                       fmin=args_dict['fmin'],
                       n_bins=args_dict['n_bins'],
                       bins_per_octave=args_dict['bins_per_octave'],
                       filter_scale=args_dict['filter_scale'],
                       hop_length=args_dict['hop_length'],
                       trainable=args_dict['trainable_cqt'])

        self.module_list = nn.ModuleList()

        for l in range(self.num_layers):
            if args_dict['top_padding'][l] > 0:
                self.module_list.add_module('pad_' + str(l),
                                            nn.ZeroPad2d((0, 0, args_dict['top_padding'][l], 0)))

            self.module_list.add_module('conv_' + str(l),
                                        nn.Conv2d(in_channels=args_dict['channel_count'][l],
                                            out_channels=args_dict['channel_count'][l+1],
                                            kernel_size=args_dict['kernel_sizes'][l],
                                            bias=args_dict['bias']))
            if args_dict['pooling'][l] > 1:
                self.module_list.add_module('pooling_' + str(l),
                                            nn.MaxPool2d(kernel_size=args_dict['pooling'][l]))

            if l < self.num_layers-1:
                self.module_list.add_module('relu_' + str(l),
                                            nn.ReLU())

    def forward(self, x):
        x = self.cqt(x)
        x = abs(x).unsqueeze(1)
        for i, module in enumerate(self.module_list):
            x = module(x)
            #print("shape after module", i, " - ", x.shape)
        return x.squeeze(2)


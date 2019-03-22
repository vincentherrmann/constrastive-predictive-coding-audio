import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
from scipy.special import gamma

from constant_q_transform import *


scalogram_encoder_default_dict = {'kernel_sizes': [(127, 1), (5, 5), (63, 1), (5, 5), (26, 1), (5, 5)],
                                  'top_padding': [126, 0, 0, 0, 0, 0],
                                  'channel_count': [1, 32, 32, 64, 128, 256, 512],
                                  'pooling': [1, 2, 1, 2, 1, 1],
                                  'stride': [1, 1, 1, 1, 1, 1],
                                  'bias': True,
                                  'sample_rate': 16000,
                                  'fmin': 30,
                                  'n_bins': 256,
                                  'bins_per_octave': 32,
                                  'filter_scale': 0.5,
                                  'hop_length': 128,
                                  'trainable_cqt': False,
                                  'batch_norm': False,
                                  'phase': False,
                                  'separable': False,
                                  'lowpass_init': 0.,
                                  'instance_norm': False,
                                  'dropout': 0.}

# 500 x 256 -> (127, 1) with 126 padding
# 500 x 256 -> (5, 5)
# 496 x 252 -> pool
# 248 x 126 -> (63, 1)
# 248 x  64 -> (5, 5)
# 244 x  60 -> pool
# 122 x  30 -> (26, 1)
# 122 x   5 -> (5, 5)
# 118 x   1

scalogram_encoder_stride_dict = scalogram_encoder_default_dict.copy()
scalogram_encoder_stride_dict['kernel_sizes'] = [(5, 5), (64, 1), (5, 5), (32, 1), (5, 5), (26, 1)]
scalogram_encoder_stride_dict['top_padding'] = [0, 63, 0, 0, 0, 0]
scalogram_encoder_stride_dict['pooling'] = [1, 1, 1, 1, 1, 1]
scalogram_encoder_stride_dict['stride'] = [2, 1, 2, 1, 1, 1]
scalogram_encoder_stride_dict['padding'] = [0, 0, 0, 0, 0, 0]

# 500 x 256 -> (5, 5) with stride 2      2  256000
# 248 x 126 -> (64, 1) with 63 padding  32  999936
# 248 x 126 -> (5, 5) with stride 2     32  999936
# 122 x  61 -> (32, 1)                  64  476288
# 122 x  30 -> (5, 5)                  128  468480
# 118 x  26 -> (26, 1)                 256
# 118 x   1                            512

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

        self.phase = args_dict['phase']
        if self.phase:
            args_dict['channel_count'][0] = 2
            self.phase_diff = PhaseDifference(sr=args_dict['sample_rate'],
                                              fmin=args_dict['fmin'],
                                              n_bins=args_dict['n_bins'],
                                              bins_per_octave=args_dict['bins_per_octave'],
                                              hop_length=args_dict['hop_length'])
        else:
            args_dict['channel_count'][0] = 1

        self.module_list = nn.ModuleList()

        for l in range(self.num_layers):
            if args_dict['top_padding'][l] > 0:
                self.module_list.add_module('pad_' + str(l),
                                            nn.ZeroPad2d((0, 0, args_dict['top_padding'][l], 0)))

            if l > 0 and args_dict['separable']:
                self.module_list.add_module('conv_' + str(l),
                                            Conv2dSeperable(in_channels=args_dict['channel_count'][l],
                                                            out_channels=args_dict['channel_count'][l+1],
                                                            kernel_size=args_dict['kernel_sizes'][l],
                                                            bias=args_dict['bias'],
                                                            stride=args_dict['stride'][l]))
            else:
                bias = False
                if args_dict['kernel_sizes'][l][1] > 1:
                    bias = args_dict['bias']
                self.module_list.add_module('conv_' + str(l),
                                            nn.Conv2d(in_channels=args_dict['channel_count'][l],
                                                      out_channels=args_dict['channel_count'][l+1],
                                                      kernel_size=args_dict['kernel_sizes'][l],
                                                      bias=bias,
                                                      stride=args_dict['stride'][l]))

            if args_dict['lowpass_init'] > 0 and args_dict['kernel_sizes'][l][1] == 1:
                lowpass_init(list(self.module_list)[-1].weight, args_dict['lowpass_init'])

            if args_dict['pooling'][l] > 1:
                self.module_list.add_module('pooling_' + str(l),
                                            nn.MaxPool2d(kernel_size=args_dict['pooling'][l]))

            if l < self.num_layers-1:
                self.module_list.add_module('relu_' + str(l),
                                            nn.ReLU())

                if args_dict['dropout'] > 0.:
                    self.module_list.add_module('dropout_' + str(l),
                                                nn.Dropout2d(args_dict['dropout']))

                if args_dict['batch_norm']:
                    self.module_list.add_module('batch_norm_' + str(l),
                                                nn.BatchNorm2d(num_features=args_dict['channel_count'][l+1]))

                if args_dict['instance_norm']:
                    self.module_list.add_module('instance_norm_' + str(l),
                                                nn.InstanceNorm2d(num_features=args_dict['channel_count'][l + 1],
                                                                  affine=True,
                                                                  track_running_stats=True))

        self.receptive_field = self.cqt.conv_kernel_sizes[0]
        s = args_dict['hop_length']
        for i in range(self.num_layers):
            self.receptive_field += (args_dict['kernel_sizes'][i][1] - 1) * s
            s *= args_dict['pooling'][i] * args_dict['stride'][i]


        self.downsampling_factor = args_dict['hop_length'] * np.prod(args_dict['pooling']) * np.prod(args_dict['stride'])

    def forward(self, x):
        x = self.cqt(x)

        if self.phase:
            amp = torch.pow(abs(x[:, :, 1:]), 2)
            amp = torch.log(amp + 1e-9)
            phi = self.phase_diff(angle(x))
            x = torch.stack([amp, phi], dim=1)
        else:
            x = torch.pow(abs(x), 2)
            x = torch.log(x + 1e-9).unsqueeze(1)

        for i, module in enumerate(self.module_list):
            x = module(x)
            #print("shape after module", i, " - ", x.shape)
        return x.squeeze(2)


#scalogram_encoder_resnet_dict = scalogram_encoder_default_dict.copy()
#scalogram_encoder_resnet_dict['channel_count'] = [1, 32, 32, 64, 64,
#                                                  128, 128, 128, 128,
#                                                  256, 256, 256, 256,
#                                                  512, 256]
#scalogram_encoder_resnet_dict['channel_count'] = [1, 16, 16, 32, 32,
#                                                  64, 64, 64, 64,
#                                                  128, 128, 128, 128,
#                                                  256, 256]
#scalogram_encoder_resnet_dict['kernel_sizes'] = [(3, 3), (3, 3), (3, 3), (64, 1),
#                                                 (3, 3), (3, 3), (3, 3), (33, 1),
#                                                 (3, 3), (3, 3), (3, 3), (16, 1),
#                                                 (1, 3), (1, 3)]
#scalogram_encoder_resnet_dict['top_padding'] = [0, 0, 0, 63,
#                                                0, 0, 0, 0,
#                                                0, 0, 0, 0,
#                                                0, 0]
#scalogram_encoder_resnet_dict['padding'] = [1, 1, 1, 0,
#                                            1, 1, 1, 0,
#                                            1, 1, 1, 0,
#                                            0, 0]
#scalogram_encoder_resnet_dict['pooling'] = [1, 1, 1, 1,
#                                            1, 1, 1, 1,
#                                            1, 1, 1, 1,
#                                            1, 1]
#scalogram_encoder_resnet_dict['stride'] =  [2, 1, 1, 1,
#                                            2, 1, 1, 1,
#                                            2, 1, 1, 1,
#                                            1, 1]

scalogram_encoder_resnet_dict = scalogram_encoder_default_dict.copy()
scalogram_encoder_resnet_dict['channel_count'] = [1, 32, 32, 64, 64,
                                                  64, 64, 64, 64,
                                                  128, 128, 128, 128,
                                                  256, 256]
scalogram_encoder_resnet_dict['kernel_sizes'] = [(3, 3), (3, 3), (3, 3), (64, 1),
                                                 (3, 3), (3, 3), (3, 3), (30, 1),
                                                 (3, 3), (3, 3), (3, 3), (15, 1),
                                                 (3, 3), (3, 3)]
scalogram_encoder_resnet_dict['top_padding'] = [0, 0, 0, 63,
                                                0, 0, 0, 0,
                                                0, 0, 0, 0,
                                                0, 0]
scalogram_encoder_resnet_dict['padding'] = [1, 1, 1, 0,
                                            1, 0, 1, 0,
                                            1, 1, 1, 0,
                                            1, 0]
scalogram_encoder_resnet_dict['pooling'] = [1, 1, 1, 1,
                                            1, 1, 1, 1,
                                            1, 1, 1, 1,
                                            1, 1]
scalogram_encoder_resnet_dict['stride'] =  [1, 1, 2, 1,
                                            1, 1, 2, 1,
                                            1, 1, 2, 1,
                                            1, 1]


# 512 x 256 -> (3, 3)   with stride 2           32           18.874.368     12 ms
# 256 x 128 -> (3, 3)                           32          301.989.888    104 ms

# 256 x 128 -> (3, 3)                           64          603.979.776    102
# 256 x 128 -> (64, 1)  with 63 top padding     64        8.589.934.592     33

# 256 x 128 -> (3, 3)   with stride 2           128       1,179,648,000     47
# 128 x  64 -> (3, 3)                                     1,179,648,000     48

# 128 x  64 -> (3, 3)                           128         575,963,136     82
# 128 x  64 -> (33, 1)                                    1,919,877,120     37

# 128 x  32 -> (3, 3)   with stride 2           256         621,674,496     29
#  64 x  16 -> (3, 3)                                       621,674,496     29

#  64 x  16 -> (3, 3)                           256         310,837,248      9
#  64 x  16 -> (16, 1)                                    1,036,124,160      7

#  62 x   1 -> (3, 1)   no padding              256         109,707,264      4
#  60 x   1 -> (3, 1)   no padding                          109,707,264      1


# 500 x 256 -> (3, 3)                           32           73,728,000     12 ms
# 500 x 256 -> (3, 3)                           32        1,179,648,000    104 ms

# 500 x 256 -> (3, 3)   with stride 2           64          589,824,000    102
# 250 x 128 -> (64, 1)  with 63 top padding     64        8,388,608,000     33

# 250 x 128 -> (3, 3)                           64        1,179,648,000     47
# 250 x 128 -> (3, 3)   no padding                        1,179,648,000     48

# 248 x 126 -> (3, 3)   with stride 2           128         575,963,136     82
# 124 x  63 -> (30, 1)  no padding                        1,919,877,120     37

# 124 x  34 -> (3, 3)                           128         621,674,496     29
# 124 x  34 -> (3, 3)                                       621,674,496     29

# 124 x  34 -> (3, 3)   with stride 2           256         310,837,248      9
#  62 x  17 -> (15, 1)  no padding                        1,036,124,160      7

#  62 x   3 -> (3, 3)                           256         109,707,264      4
#  62 x   3 -> (3, 3)   no padding                          109,707,264      1



class ScalogramEncoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_a_size,
                 kernel_b_size,
                 bias=True,
                 pooling=1,
                 stride=2,
                 top_padding=None,
                 padding_1=0,
                 padding_2=0,
                 separable=True,
                 residual=True):
        super().__init__()

        #   +--------------- pooling -- conv_1x1 ----------------+
        #   |                                                    |
        # --+-- conv_a -- pooling -- ReLU -- padding -- conv_b --+--

        conv_module = Conv2dSeperable if separable else nn.Conv2d

        self.main_modules = nn.ModuleList()

        self.main_modules.append(conv_module(in_channels=in_channels,
                                             out_channels=hidden_channels,
                                             kernel_size=kernel_a_size,
                                             bias=bias,
                                             padding=padding_1,
                                             stride=stride))

        if pooling > 1:
            self.main_modules.append(nn.MaxPool2d(kernel_size=pooling))

        self.main_modules.append(nn.ReLU())

        if top_padding is not None:
            self.main_modules.append(nn.ZeroPad2d((0, 0, top_padding, 0)))

        self.main_modules.append(conv_module(in_channels=hidden_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_b_size,
                                             bias=bias,
                                             padding=padding_2))

        self.residual = residual
        if self.residual:
            self.residual_modules = nn.ModuleList()

            stride_pool = stride * pooling
            if stride_pool > 1:
                self.residual_modules.append(nn.MaxPool2d(kernel_size=stride_pool, ceil_mode=True))

            if in_channels != out_channels:
                self.residual_modules.append(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=1,
                                                       bias=False))

    def forward(self, x):
        original_input = x
        for m in self.main_modules:
            x = m(x)
        main = x
        x = original_input
        if self.residual:
            for m in self.residual_modules:
                x = m(x)
            res = x
            r_h = res.shape[2]
            r_w = res.shape[3]
            m_h = main.shape[2]
            m_w = main.shape[3]
            o_h = math.ceil((r_h - m_h) / 2)
            o_w = math.ceil((r_w - m_w) / 2)
            if o_h > 0:
                res = res[:, :, -(o_h + m_h):-o_h, :]
            if o_w > 0:
                res = res[:, :, :, -(o_w + m_w):-o_w]
            main = main + res
        return main


class ScalogramResidualEncoder(nn.Module):
    def __init__(self, args_dict=scalogram_encoder_default_dict):
        super().__init__()

        self.cqt = CQT(sr=args_dict['sample_rate'],
                       fmin=args_dict['fmin'],
                       n_bins=args_dict['n_bins'],
                       bins_per_octave=args_dict['bins_per_octave'],
                       filter_scale=args_dict['filter_scale'],
                       hop_length=args_dict['hop_length'],
                       trainable=args_dict['trainable_cqt'])

        self.phase = args_dict['phase']
        if self.phase:
            args_dict['channel_count'][0] = 2
            self.phase_diff = PhaseDifference(sr=args_dict['sample_rate'],
                                              fmin=args_dict['fmin'],
                                              n_bins=args_dict['n_bins'],
                                              bins_per_octave=args_dict['bins_per_octave'],
                                              hop_length=args_dict['hop_length'])
        else:
            args_dict['channel_count'][0] = 1

        block_count = len(args_dict['kernel_sizes']) // 2
        self.blocks = nn.ModuleList()
        for b in range(block_count):
            self.blocks.append(ScalogramEncoderBlock(in_channels=args_dict['channel_count'][2*b],
                                                     hidden_channels=args_dict['channel_count'][2*b+1],
                                                     out_channels=args_dict['channel_count'][2*b+2],
                                                     kernel_a_size=args_dict['kernel_sizes'][2*b],
                                                     kernel_b_size=args_dict['kernel_sizes'][2*b+1],
                                                     bias=args_dict['bias'],
                                                     pooling=args_dict['pooling'][2*b],
                                                     stride=args_dict['stride'][2*b],
                                                     top_padding=args_dict['top_padding'][2*b+1],
                                                     padding_1=args_dict['padding'][2*b],
                                                     padding_2=args_dict['padding'][2*b+1],
                                                     separable=args_dict['separable']))

        self.receptive_field = self.cqt.conv_kernel_sizes[0]
        s = args_dict['hop_length']

        for i in range(len(args_dict['kernel_sizes'])):
            self.receptive_field += (args_dict['kernel_sizes'][i][1] - 1) * s
            s *= args_dict['pooling'][i] * args_dict['stride'][i]

        self.downsampling_factor = args_dict['hop_length'] * np.prod(args_dict['pooling']) * np.prod(
            args_dict['stride'])

    def forward(self, x):
        x = self.cqt(x)

        if self.phase:
            amp = torch.pow(abs(x[:, :, 1:]), 2)
            amp = torch.log(amp + 1e-9)
            phi = self.phase_diff(angle(x))
            x = torch.stack([amp, phi], dim=1)
        else:
            x = torch.pow(abs(x), 2)
            x = torch.log(x + 1e-9).unsqueeze(1)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks)-1:
                x = F.relu(x)
            #print("shape after module", i, " - ", x.shape)
        return x.squeeze(2)



class Conv2dSeperable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False, groups=in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)

    @property
    def weight(self):
        return self.conv.weight

    def forward(self, x):
        return self.conv_1x1(self.conv(x))


def beta_pdf(a, b, num=128):
  x = np.linspace(0, 1, num)
  beta = (gamma(a)*gamma(b)) / gamma(a+b)
  return (x**(a-1) * (1-x)**(b-1)) / beta


def beta_init(tensor, factor=2):
    in_channels = tensor.shape[1]
    out_channels = tensor.shape[0]
    size = tensor.shape[2]
    with torch.no_grad():
        for i in range(out_channels):
            p = i // 2
            if i % 2 == 0:
                v = beta_pdf(2, factor * (p + 1) + 2, size)
            else:
                v = beta_pdf(factor * p + 2, 2, size)
            v = torch.tensor(v, dtype=tensor.dtype, device=tensor.device).unsqueeze(0).repeat([in_channels, 1])
            v *= np.sqrt(2 / (in_channels * size))
            tensor[i, :, :, 0] = v


def lowpass_init(tensor, factor=10.):
    in_channels = tensor.shape[1]
    out_channels = tensor.shape[0]
    size = tensor.shape[2]
    with torch.no_grad():
        noise = torch.randn_like(tensor).view(in_channels*out_channels, size)
        fft = torch.rfft(noise, 1)
        fft_filter = torch.exp(torch.linspace(0, -factor, size//2 + 1))
        fft *= fft_filter.unsqueeze(1)
        ifft = torch.irfft(fft, 1, normalized=False)[:, :size].view(out_channels, in_channels, size, 1)
        tensor[:] = ifft * np.sqrt(2 / (in_channels * size))




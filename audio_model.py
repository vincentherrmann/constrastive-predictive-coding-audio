import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

encoder_default_dict = {'strides': [5, 4, 2, 2, 2],
                        'kernel_sizes': [10, 8, 4, 4, 4],
                        'channel_count': [512, 512, 512, 512, 512],
                        'bias': True}


class AudioEncoder(nn.Module):
    def __init__(self, args_dict=encoder_default_dict):
        super().__init__()

        self.num_layers = len(args_dict['strides'])
        self.downsampling_factor = np.prod(args_dict['strides'])

        self.receptive_field = args_dict['kernel_sizes'][0]
        s = 1
        for i in range(1, self.num_layers):
            s *= args_dict['strides'][i-1]
            self.receptive_field += (args_dict['kernel_sizes'][i]-1) * s

        self.layers = torch.nn.ModuleList()
        for l in range(self.num_layers):
            in_channels = 1 if l == 0 else args_dict['channel_count'][l-1]
            self.layers.append(nn.Conv1d(in_channels=in_channels,
                                         out_channels=args_dict['channel_count'][l],
                                         kernel_size=args_dict['kernel_sizes'][l],
                                         stride=args_dict['strides'][l],
                                         bias=args_dict['bias']))

    def forward(self, x):
        for l, layer in enumerate(self.layers):
            if l < len(self.layers)-1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
                #x = torch.tanh(layer(x)) #/ layer.out_channels

        return x


class AudioGRUModel(nn.Module):
    """
    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: `True`
        reset_hidden: If `True` the hidden state will be reset to zero with every call

    Inputs: input
        - **input** of shape `(batch, input_size, steps)`: tensor containing input features
    """
    def __init__(self, input_size, hidden_size, bias=True, reset_hidden=True):
        super().__init__()
        self.gruCell = nn.GRUCell(input_size=input_size,
                                  hidden_size=hidden_size,
                                  bias=bias)
        self.hidden = None
        self.reset_hidden = reset_hidden

    def forward(self, input):
        batch, input_size, steps = input.shape

        hidden = None if self.reset_hidden else self.hidden

        for step in range(steps):
            x = input[:, :, step]
            hidden = self.gruCell(x, hidden)

        self.hidden = None if self.reset_hidden else hidden

        return hidden


class ConvolutionalArBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling=1, stride=1, bias=True, residual=False, batch_norm=False):
        super().__init__()
        self.main_modules = nn.ModuleList()
        if pooling > 1:
            self.main_modules.append(nn.MaxPool1d(pooling, ceil_mode=True))
        self.main_modules.append(nn.Conv1d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           bias=bias))
        if batch_norm:
            self.main_modules.append(nn.BatchNorm1d(out_channels))

        self.main_modules.append(nn.ReLU())

        self.residual_modules = None

        self.residual = residual
        if self.residual:
            self.residual_modules = nn.ModuleList()
            if pooling*stride > 1:
                self.residual_modules.append(nn.MaxPool1d(pooling*stride, ceil_mode=True))
            if in_channels != out_channels:
                self.residual_modules.append(nn.Conv1d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=1))

    def forward(self, x):
        original_x = x
        for m in self.main_modules:
            x = m(x)
        main_x = x
        if self.residual:
            x = original_x
            for m in self.residual_modules:
                x = m(x)
            main_x += x[:, :, -main_x.shape[2]:]
        return main_x


class ConvolutionalArModel(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        self.module_list = nn.ModuleList()
        for l in range(len(args_dict['kernel_sizes'])):
            self.module_list.append(ConvolutionalArBlock(in_channels=args_dict['channel_count'][l],
                                                         out_channels=args_dict['channel_count'][l+1],
                                                         kernel_size=args_dict['kernel_sizes'][l],
                                                         stride=args_dict['stride'][l],
                                                         pooling=args_dict['pooling'][l],
                                                         bias=args_dict['bias'],
                                                         batch_norm=args_dict['batch_norm'],
                                                         residual=args_dict['residual']))

        self.encoding_size = args_dict['channel_count'][0]
        self.ar_size = args_dict['channel_count'][-1]

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        return x[:, :, -1]


class AudioPredictiveCodingModel(nn.Module):
    def __init__(self, encoder, autoregressive_model, enc_size, ar_size, visible_steps=100, prediction_steps=12):
        super().__init__()
        self.enc_size = enc_size
        self.ar_size = ar_size
        self.visible_steps = visible_steps
        self.prediction_steps = prediction_steps
        self.encoder = encoder
        self.autoregressive_model = autoregressive_model
        self.prediction_model = nn.Linear(in_features=ar_size, out_features=enc_size*prediction_steps, bias=False)
        #self.group_norm = nn.GroupNorm(num_groups=prediction_steps, num_channels=enc_size*prediction_steps, affine=False)

    @property
    def item_length(self):
        item_length = self.encoder.receptive_field
        item_length += (self.visible_steps + self.prediction_steps) * self.encoder.downsampling_factor
        return item_length

    def forward(self, x):
        z = self.encoder(x)
        targets = z[:, :, -self.prediction_steps:]  # batch, enc_size, step  # .detach()  # TODO should this be detached?
        z = z[:, :, -(self.visible_steps+self.prediction_steps):-self.prediction_steps]
        c = self.autoregressive_model(z)
        if len(c.shape) == 3:
            c = c[:, :, 0]
        predicted_z = self.prediction_model(c)  # batch, step*enc_size
        predicted_z = predicted_z.view(-1, self.prediction_steps, self.enc_size)  # batch, step, enc_size

        return predicted_z, targets, z, c

    def parameter_count(self):
        total_parameters = 0
        for p in self.parameters():
            total_parameters += np.prod(p.shape)
        return total_parameters


def load_to_cpu(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    model.cpu()
    return model


def num_parameters(model):
    total_parameters = 0
    for p in model.parameters():
        total_parameters += np.prod(p.shape)
    return total_parameters

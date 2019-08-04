import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoder(nn.Module):
    def __init__(self, code_size, max_seq_len=128, max_wavelength=10000):
        super().__init__()
        self.code_size = code_size

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pi = np.pi

        pe = torch.zeros([max_seq_len, code_size], requires_grad=False)
        #if torch.cuda.is_available():  # TODO: eliminate cuda check
        #    pe = torch.zeros([max_seq_len, code_size], requires_grad=False).cuda()
        for pos in range(max_seq_len):
            for i in range(0, code_size, 2):
                pe[pos, i] = math.sin(pi * pos / (max_wavelength ** ((2 * i) / code_size)))
                pe[pos, i + 1] = math.cos(pi * pos / (max_wavelength ** ((2 * i) / code_size)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):

        # make code relatively larger
        x *= math.sqrt(self.code_size)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class AttentionModel(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        channels = args_dict['channels']
        self.num_layers = args_dict['num_layers']

        self.positional_encoder = PositionalEncoder(channels, args_dict['sequence_length'])

        encoder_layer = TransformerEncoderLayer(args_dict['channels'],
                                                args_dict['num_heads'],
                                                args_dict['feedforward_size'],
                                                args_dict['dropout'])
        encoder_norm = torch.nn.LayerNorm(args_dict['channels'])
        self.encoder = TransformerEncoder(encoder_layer, args_dict['num_layers'], encoder_norm)

        # self.attentions = nn.ModuleList()
        # self.dropout1 = nn.ModuleList()
        # self.norm1 = nn.ModuleList()
        # self.ff1 = nn.ModuleList()
        # self.ff2 = nn.ModuleList()
        # self.dropout2 = nn.ModuleList()
        # self.norm2 = nn.ModuleList()
        #
        # for _ in range(self.num_layers):
        #     self.attentions.append(MultiheadAttention(args_dict['num_heads'], channels))
        #     self.dropout1.append(nn.Dropout(args_dict['dropout']))
        #     self.norm1.append(nn.LayerNorm(channels))
        #     self.ff1.append(nn.Linear(channels, args_dict['feedforward_size']))
        #     self.ff2.append(nn.Linear(args_dict['feedforward_size'], channels))
        #     self.dropout2.append(nn.Dropout(args_dict['dropout']))
        #     self.norm2.append(nn.LayerNorm(channels))

        self.end_layer = nn.Linear(channels, args_dict['output_size'])

    def forward(self, x):
        x = x.transpose(1, 0)
        x = self.positional_encoder(x)  # sequence, batch, channels

        x = self.encoder(x)

        x = torch.sum(x, dim=0)
        x = self.end_layer(x)
        return x

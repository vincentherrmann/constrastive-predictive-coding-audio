import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(self, code_size, max_seq_len=128):
        super().__init__()
        self.code_size = code_size

        # create constant 'pe' matrix with values dependant on
        # pos and i

        pe = torch.zeros([max_seq_len, code_size], requires_grad=False)
        #if torch.cuda.is_available():  # TODO: eliminate cuda check
        #    pe = torch.zeros([max_seq_len, code_size], requires_grad=False).cuda()
        for pos in range(max_seq_len):
            for i in range(0, code_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / code_size)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / code_size)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # make code relatively larger
        x *= math.sqrt(self.code_size)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, channels):
        super().__init__()

        self.channels = channels
        self.channels_per_head = channels // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(channels, channels, bias=False)
        self.v_linear = nn.Linear(channels, channels, bias=False)
        self.k_linear = nn.Linear(channels, channels, bias=False)
        self.out_linear = nn.Linear(channels, channels)

    def forward(self, q, k, v):
        # batch, steps, channels
        batch_size = q.shape[0]

        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.channels_per_head).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.channels_per_head).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.channels_per_head).transpose(1, 2)

        # q, k, v shape: batch, num_heads, steps, channels_per_head

        lin_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.channels_per_head)
        scores = F.softmax(lin_scores, dim=-1)

        # score shape: batch, num_heads, steps, steps

        attention_output = torch.matmul(scores, v)  # shape: batch, num_heads, steps, channels_per_head

        merged_heads = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.channels)
        output = self.out_linear(merged_heads)
        return output


class AttentionModel(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        channels = args_dict['channels']
        self.num_layers = args_dict['num_layers']

        self.positional_encoder = PositionalEncoder(channels, args_dict['sequence_length'])

        self.attentions = nn.ModuleList()
        self.dropout1 = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.ff1 = nn.ModuleList()
        self.ff2 = nn.ModuleList()
        self.dropout2 = nn.ModuleList()
        self.norm2 = nn.ModuleList()

        for _ in range(self.num_layers):
            self.attentions.append(MultiheadAttention(args_dict['num_heads'], channels))
            self.dropout1.append(nn.Dropout(args_dict['dropout']))
            self.norm1.append(nn.LayerNorm(channels))
            self.ff1.append(nn.Linear(channels, args_dict['feedforward_size']))
            self.ff2.append(nn.Linear(args_dict['feedforward_size'], channels))
            self.dropout2.append(nn.Dropout(args_dict['dropout']))
            self.norm2.append(nn.LayerNorm(channels))

        self.end_layer = nn.Linear(channels * args_dict['sequence_length'], args_dict['output_size'])

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.positional_encoder(x)

        for l in range(self.num_layers):
            a = self.dropout1[l](self.attentions[l](x, x, x))
            x = self.norm1[l](x+a)
            f = self.dropout2[l](self.ff2[l](F.relu(self.ff1[l](x))))
            x = self.norm2[l](x+f)

        x = self.end_layer(F.relu(x).view(x.shape[0], -1))
        return x

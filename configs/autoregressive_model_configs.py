from audio_model import ConvolutionalArModel
from attention_model import AttentionModel
from configs.scalogram_resnet_configs import *

ar_conv_default_dict = {
    'model': ConvolutionalArModel,
    'kernel_sizes': [9, 9, 9],
    'channel_count': [256, 256, 256, 256],
    'stride': [1, 1, 1],
    'pooling': [1, 2, 2],
    'bias': True,
    'batch_norm': False,
    'residual': False,
    'encoding_size': 256,
    'ar_code_size': 256,
    'activation_register': None
}

ar_conv_architecture_1 = ar_conv_default_dict.copy()
ar_conv_architecture_1['channel_count'] = [256, 512, 512, 256]

ar_conv_architecture_2 = ar_conv_default_dict.copy()
ar_conv_architecture_2['kernel_sizes'] = [5, 5, 5, 5, 5, 5]
ar_conv_architecture_2['channel_count'] = [256, 512, 512, 256, 256, 256, 256]
ar_conv_architecture_2['stride'] = [1, 1, 1, 1, 1, 1]
ar_conv_architecture_2['pooling'] = [1, 1, 2, 1, 2, 1]
ar_conv_architecture_2['batch_norm'] = True
ar_conv_architecture_2['residual'] = True

ar_conv_architecture_3 = ar_conv_architecture_2.copy()
ar_conv_architecture_3['channel_count'] = [512, 512, 512, 256, 256, 256, 256]
ar_conv_architecture_3['encoding_size'] = 512


ar_block_default_dict = scalogram_block_default_dict.copy()
ar_block_default_dict['in_channels'] = 256
ar_block_default_dict['out_channels'] = 256
ar_block_default_dict['kernel_size_1'] = (1, 9)
ar_block_default_dict['kernel_size_2'] = (1, 1)
ar_block_default_dict['ceil_pooling'] = True

ar_resnet_default_dict = {'model': ScalogramResidualEncoder,
                          'phase': False,
                          'blocks': [ar_block_default_dict,
                                     ar_block_default_dict,
                                     ar_block_default_dict],
                          'encoding_size': 256,
                          'ar_code_size': 256
                          }

block_0 = ar_block_default_dict.copy()
block_0['out_channels'] = 512
block_0['pooling_1'] = 2
block_0['batch_norm'] = True

block_1 = block_0.copy()
block_1['in_channels'] = 512

block_2 = block_1.copy()
block_2['kernel_size_1'] = (1, 8)

ar_resnet_architecture_1 = ar_resnet_default_dict.copy()
ar_resnet_architecture_1['blocks'] = [block_0, block_1, block_2]
ar_resnet_architecture_1['ar_code_size'] = 512

ar_resnet_architecture_2 = ar_resnet_architecture_1.copy()
ar_resnet_architecture_2['encoding_size'] = 512
ar_resnet_architecture_2['ar_code_size'] = 256
ar_resnet_architecture_2['blocks'][0]['in_channels'] = 512
ar_resnet_architecture_2['blocks'][2]['out_channels'] = 256


attention_default_dict = {
    'model': AttentionModel,
    'channels': 512,
    'output_size': 512,
    'num_layers': 2,
    'num_heads': 8,
    'feedforward_size': 512,
    'sequence_length': 60,
    'dropout': 0.1,
    'encoding_size': 512,
    'ar_code_size': 512
}

attention_architecture_1 = attention_default_dict.copy()
attention_architecture_1['output_size'] = 256
attention_architecture_1['num_layers'] = 3
attention_architecture_1['ar_code_size'] = 256

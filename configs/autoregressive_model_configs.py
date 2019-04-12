from audio_model import ConvolutionalArModel

ar_conv_default_dict = {
    'model': ConvolutionalArModel,
    'kernel_sizes': [9, 9, 9],
    'channel_count': [256, 256, 256, 256],
    'stride': [1, 1, 1],
    'pooling': [1, 2, 2],
    'bias': True,
    'batch_norm': False,
    'residual': False
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

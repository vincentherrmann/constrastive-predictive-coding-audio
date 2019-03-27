from audio_model import ConvolutionalArModel

ar_conv_default_dict = {
    'model': ConvolutionalArModel,
    'kernel_sizes': [9, 9, 9],
    'channel_count': [256, 256, 256, 256],
    'stride': [1, 1, 1],
    'pooling': [1, 2, 2],
    'bias': True
}
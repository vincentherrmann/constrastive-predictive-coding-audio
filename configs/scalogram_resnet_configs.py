from scalogram_model import ScalogramResidualEncoder

scalogram_block_default_dict = {'in_channels': 64,
                                'hidden_channels': None,
                                'out_channels': 64,
                                'kernel_size_1': (3, 3),
                                'kernel_size_2': (3, 3),
                                'top_padding_1': None,
                                'top_padding_2': None,
                                'padding_1': 0,
                                'padding_2': 0,
                                'stride_1': 1,
                                'stride_2': 1,
                                'pooling_1': 1,
                                'pooling_2': 1,
                                'bias': True,
                                'separable': False,
                                'residual': True,
                                'batch_norm': False,
                                'ceil_pooling': False}

scalogram_resnet_default_dict = {'model': ScalogramResidualEncoder,
                                 'phase': True,
                                 'blocks': [scalogram_block_default_dict,
                                            scalogram_block_default_dict,
                                            scalogram_block_default_dict]}

block_3x3 = scalogram_block_default_dict.copy()
block_3x3['padding_1'] = 1
block_3x3['padding_2'] = 1

block_3x3_strided = block_3x3.copy()
block_3x3_strided['stride_1'] = 2

block_3x3_valid = scalogram_block_default_dict.copy()

block_3x3_strided_valid = scalogram_block_default_dict.copy()
block_3x3_strided_valid['stride_1'] = 2


# resnet architecture 1
scalogram_resnet_architecture_1 = scalogram_resnet_default_dict.copy()

# This architecture seems to work, but is quite slow

# activation shape after block 0 : torch.Size([1, 32, 256, 553])
# activation shape after block 1 : torch.Size([1, 64, 128, 277])
# activation shape after block 2 : torch.Size([1, 64, 126, 275])
# activation shape after block 3 : torch.Size([1, 128, 34, 138])
# activation shape after block 4 : torch.Size([1, 128, 34, 138])
# activation shape after block 5 : torch.Size([1, 256, 3, 69])
# activation shape after block 6 : torch.Size([1, 256, 1, 67])

block_0 = block_3x3.copy()
block_0['in_channels']  = 1
block_0['out_channels'] = 32

block_1 = block_3x3_strided.copy()
block_1['in_channels']  = 32
block_1['out_channels'] = 64
block_1['kernel_size_2'] = (64, 1)
block_1['top_padding_2'] = 63
block_1['padding_2'] = 0

block_2 = block_3x3.copy()
block_2['in_channels']  = 64
block_2['out_channels'] = 64
block_2['padding_2'] = 0

block_3 = block_3x3_strided.copy()
block_3['in_channels']  = 64
block_3['out_channels'] = 128
block_3['kernel_size_2'] = (30, 1)
block_3['padding_2'] = 0

block_4 = block_3x3.copy()
block_4['in_channels']  = 128
block_4['out_channels'] = 128

block_5 = block_3x3_strided.copy()
block_5['in_channels']  = 128
block_5['out_channels'] = 256
block_5['kernel_size_2'] = (15, 1)
block_5['padding_2'] = 0

block_6 = block_3x3.copy()
block_6['in_channels']  = 256
block_6['out_channels'] = 256
block_6['padding_2'] = 0

scalogram_resnet_architecture_1['blocks'] = [block_0, block_1,
                                             block_2, block_3,
                                             block_4, block_5,
                                             block_6]


# resnet architecture 2
scalogram_resnet_architecture_2 = scalogram_resnet_default_dict.copy()

block_0 = block_3x3_strided.copy()
block_0['in_channels']  = 1
block_0['out_channels'] = 32
block_0['kernel_size_2'] = (64, 1)
block_0['top_padding_2'] = 63
block_0['padding_2'] = 0

block_1 = block_3x3_strided.copy()
block_1['in_channels']  = 32
block_1['out_channels'] = 64
block_1['kernel_size_2'] = (30, 1)
block_1['padding_2'] = 0

block_2 = block_3x3_strided.copy()
block_2['in_channels']  = 64
block_2['out_channels'] = 128
block_2['kernel_size_2'] = (15, 1)
block_2['padding_2'] = 0

block_3 = block_3x3.copy()
block_3['in_channels']  = 128
block_3['out_channels'] = 256
block_3['padding_2'] = 0

scalogram_resnet_architecture_2['blocks'] = [block_0, block_1,
                                             block_2, block_3]

scalogram_resnet_architecture_2_wo_res = scalogram_resnet_architecture_2.copy()
for block in scalogram_resnet_architecture_2_wo_res['blocks']:
    block['residual'] = False


# resnet architecure 3
scalogram_resnet_architecture_3 = scalogram_resnet_default_dict.copy()

block_0 = block_3x3_strided.copy()
block_0['in_channels']  = 1
block_0['out_channels'] = 32
block_0['padding_1'] = 0
block_0['kernel_size_2'] = (64, 1)
block_0['top_padding_2'] = 63
block_0['padding_2'] = 0

block_1 = block_3x3_strided.copy()
block_1['in_channels']  = 32
block_1['out_channels'] = 64
block_1['padding_1'] = 0
block_1['kernel_size_2'] = (30, 1)
block_1['padding_2'] = 0

block_2 = block_3x3_strided.copy()
block_2['in_channels']  = 64
block_2['out_channels'] = 128
block_2['padding_1'] = 0
block_2['kernel_size_2'] = (15, 1)
block_2['padding_2'] = 0

block_3 = block_3x3.copy()
block_3['in_channels']  = 128
block_3['out_channels'] = 256
block_3['kernel_size_1'] = (2, 2)
block_3['padding_1'] = 0
block_3['kernel_size_2'] = (1, 1)
block_3['padding_2'] = 0

scalogram_resnet_architecture_3['blocks'] = [block_0, block_1,
                                             block_2, block_3]


# resnet architecture 4
scalogram_resnet_architecture_4 = scalogram_resnet_default_dict.copy()

block_0 = block_3x3_strided_valid.copy()
block_0['in_channels']  = 1
block_0['out_channels'] = 32

block_1 = block_3x3_valid.copy()
block_1['in_channels']  = 32
block_1['out_channels'] = 64
block_1['kernel_size_2'] = (64, 1)
block_1['top_padding_2'] = 63

block_2 = block_3x3_strided_valid.copy()
block_2['in_channels']  = 64
block_2['out_channels'] = 128

block_3 = block_3x3_valid.copy()
block_3['in_channels']  = 128
block_3['out_channels'] = 128
block_3['kernel_size_2'] = (20, 1)

block_4 = block_3x3_strided_valid.copy()
block_4['in_channels']  = 128
block_4['out_channels'] = 256

block_5 = block_3x3_valid.copy()
block_5['in_channels']  = 256
block_5['out_channels'] = 256
block_5['kernel_size_2'] = (14, 1)

block_6 = block_3x3_valid.copy()
block_6['in_channels']  = 256
block_6['out_channels'] = 256
block_6['kernel_size_1'] = (1, 3)
block_6['kernel_size_2'] = (1, 3)

scalogram_resnet_architecture_4['blocks'] = [block_0, block_1,
                                             block_2, block_3,
                                             block_4, block_5,
                                             block_6]


# resnet architecure 5
scalogram_resnet_architecture_5 = scalogram_resnet_default_dict.copy()

block_0 = block_3x3_strided_valid.copy()
block_0['in_channels']  = 1
block_0['out_channels'] = 32
block_0['kernel_size_2'] = (64, 1)
block_0['top_padding_2'] = 63

block_1 = block_3x3_strided_valid.copy()
block_1['in_channels']  = 32
block_1['out_channels'] = 64
block_1['kernel_size_2'] = (30, 1)

block_2 = block_3x3_strided_valid.copy()
block_2['in_channels']  = 64
block_2['out_channels'] = 128
block_2['kernel_size_2'] = (15, 1)

block_3 = block_3x3_valid.copy()
block_3['in_channels']  = 128
block_3['out_channels'] = 256
block_3['kernel_size_1'] = (2, 2)
block_3['kernel_size_2'] = (1, 1)

scalogram_resnet_architecture_5['blocks'] = [block_0, block_1,
                                             block_2, block_3]


# resnet architecure 6
scalogram_resnet_architecture_6 = scalogram_resnet_architecture_5.copy()
for block in scalogram_resnet_architecture_6['blocks']:
    block['batch_norm'] = True


scalogram_resnet_architecture_7 = scalogram_resnet_architecture_6.copy()
scalogram_resnet_architecture_7['blocks'][1]['out_channels'] = 128
scalogram_resnet_architecture_7['blocks'][2]['in_channels'] = 128
scalogram_resnet_architecture_7['blocks'][2]['out_channels'] = 256
scalogram_resnet_architecture_7['blocks'][3]['in_channels'] = 256
scalogram_resnet_architecture_7['blocks'][3]['out_channels'] = 512


# classification architecture
block_4 = block_3x3_valid.copy()
block_4['in_channels']  = 256
block_4['out_channels'] = 346
block_4['kernel_size_1'] = (1, 30)
block_4['kernel_size_2'] = (1, 1)

scalogram_resnet_classification_1 = scalogram_resnet_architecture_6.copy()
scalogram_resnet_classification_1['blocks'] = scalogram_resnet_classification_1['blocks'].copy()
scalogram_resnet_classification_1['blocks'][3]['batch_norm'] = False

scalogram_resnet_classification_1['blocks'].append(block_4)


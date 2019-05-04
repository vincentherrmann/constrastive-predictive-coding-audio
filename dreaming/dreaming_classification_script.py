import torch
import argparse
import datetime
import random
from setup_functions import *
from configs.experiment_configs import *
from matplotlib import pyplot as plt

experiment = 'c2'
name = 'snapshots_model_2019-04-09_run_0'

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("using device", dev)
settings = experiments[experiment]

if name is not None:
    settings['snapshot_config']['name'] = name

model, preprocessing_module = setup_classification_model(cqt_params=settings['cqt_config'],
                                                         model_params=settings['model_config'])

settings['snapshot_config']['snapshot_location'] = 'snapshots'
model, snapshot_manager, continue_training_at_step = setup_snapshot_manager(model=model,
                                                                            args_dict=settings['snapshot_config'],
                                                                            try_proceeding=True,
                                                                            load_to_cpu=True)

audio_input = torch.randn(1, 1, model.receptive_field + model.downsampling_factor) * 0.1
scalogram_input = preprocessing_module(audio_input)

input_shape = list(scalogram_input.shape)
width = input_shape[3]
jitter = 10
input_shape[3] = width+jitter

scalogram_input = torch.randn(input_shape, device=dev)

scalogram_input.requires_grad = True
# scalogram shape: 256 253

optimizer = torch.optim.Adam([scalogram_input], lr=2.)

for step in range(1000):
    j = random.randint(0, jitter-1)
    cropped_input = scalogram_input[:, :, :, j:j+width]
    outputs = model(cropped_input)
    loss = -outputs[0, 0, 0]
    if scalogram_input.grad is not None:
        scalogram_input.grad *= 0
    loss.backward()
    optimizer.step()
    print("loss:", loss.item())

    if step % 10 == 0:
        print(scalogram_input[0, 0, 100, 100])
        plt.imshow(scalogram_input[0, 0].squeeze().detach(), aspect='auto')
        plt.show()




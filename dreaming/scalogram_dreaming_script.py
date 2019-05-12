import torch
import argparse
import datetime
import random
import pprint

from setup_functions import *
from configs.experiment_configs import *
from matplotlib import pyplot as plt
from dreaming.dreaming_functions import *

pp = pprint.PrettyPrinter(indent=4)

experiment = 'e18'
name = 'snapshots_model_2019-04-14_run_1_95000'

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("using device", dev)
settings = experiments[experiment]

if name is not None:
    settings['snapshot_config']['name'] = name

model, preprocessing_module = setup_model(cqt_params=settings['cqt_config'],
                                          encoder_params=settings['encoder_config'],
                                          ar_params=settings['ar_model_config'],
                                          visible_steps=settings['training_config']['visible_steps'],
                                          prediction_steps=settings['training_config']['prediction_steps'])

settings['snapshot_config']['snapshot_location'] = 'snapshots'
model, snapshot_manager, continue_training_at_step = setup_snapshot_manager(model=model,
                                                                            args_dict=settings['snapshot_config'],
                                                                            try_proceeding=True,
                                                                            load_to_cpu=(dev == 'cpu'))

pp.pprint(model)

audio_input = torch.randn(1, 1, model.encoder.receptive_field + model.encoder.downsampling_factor * (model.visible_steps + model.prediction_steps)) * 0.1
scalogram_input = preprocessing_module(audio_input)
visible_steps = model.visible_steps

if torch.cuda.device_count() > 1:
    print("using", torch.cuda.device_count(), "GPUs")
    preprocessing_module = torch.nn.DataParallel(preprocessing_module).cuda()
    model = torch.nn.DataParallel(model).cuda()

model.eval()

input_shape = list(scalogram_input.shape)
freq_jitter = 8
time_jitter = 32
input_shape[2] += freq_jitter
input_shape[3] += time_jitter

scalogram_input = torch.randn(input_shape, device=dev)

scalogram_input.requires_grad = True
# scalogram shape: 256 253

optimizer = torch.optim.Adam([scalogram_input], lr=0.3)
jitter_module = Jitter([freq_jitter, time_jitter], dims=[2, 3], jitter_batches=32)

for step in range(1000):
    cropped_input = jitter_module(scalogram_input)
    _, _, z, c = model(cropped_input)

    loss = -torch.mean(c, dim=0)[10]**2
    #loss = -torch.sum(torch.mean(c, dim=0)**2)

    if scalogram_input.grad is not None:
        scalogram_input.grad *= 0
    loss.backward()
    optimizer.step()
    print("loss:", loss.item())

    if step % 10 == 0:
        print(scalogram_input[0, 0, 100, 100])
        plt.imshow(scalogram_input[0, 0].squeeze().cpu().detach(), aspect='auto')
        plt.show()




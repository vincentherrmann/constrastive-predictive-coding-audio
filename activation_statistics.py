import torch
import argparse
import datetime
import random
import pickle
import functools
#import torchaudio
import pprint
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
from setup_functions import *
from configs.experiment_configs import *
from matplotlib import pyplot as plt
from dreaming.dreaming_functions import *
from dreaming.streaming import *
from ml_utilities.pytorch_utilities import *

pp = pprint.PrettyPrinter(indent=4)

experiment = 'e26'
name = 'snapshots_model_2019-05-17_run_0_85000'

try:
    dev = 'cuda:' + str(torch.cuda.current_device())
except:
    dev = 'cpu'
print("using device", dev)
settings = experiments[experiment]

if name is not None:
    settings['snapshot_config']['name'] = name

register = ActivationRegister()

model, preprocessing_module, untraced_model = setup_model(cqt_params=settings['cqt_config'],
                                                          encoder_params=settings['encoder_config'],
                                                          ar_params=settings['ar_model_config'],
                                                          trainer_args=settings['training_config'],
                                                          device=dev,
                                                          activation_register=register)

settings['snapshot_config']['snapshot_location'] = 'snapshots'
loaded_model, snapshot_manager, continue_training_at_step = setup_snapshot_manager(model=model,
                                                                            args_dict=settings['snapshot_config'],
                                                                            try_proceeding=True,
                                                                            load_to_cpu=(dev == 'cpu'))

trainer = setup_ce_trainer(loaded_model,
                               snapshot_manager,
                               item_length=untraced_model.item_length,
                               downsampling_factor=untraced_model.encoder.downsampling_factor,
                               ar_size=untraced_model.ar_size,
                               preprocessing_module=preprocessing_module,
                               dataset_args=settings['dataset_config'],
                               trainer_args=settings['training_config'],
                               dev=dev)

pp.pprint(model)

try:
    input_length = model.item_length
except:
    try:
        input_length = model.module.item_length
    except:
        raise

clip_length = 64000


def activation_statistics(activation_dict, avg_dict=None):
    if avg_dict is None:
        avg_dict = activation_dict.copy()
        for key, value in avg_dict.items():
            avg_dict[key] = {'global_mean': [], 'global_var': [],
                             'element_mean': [], 'element_var': []}
    for key, value in activation_dict.items():
        avg_dict[key]['global_mean'].append(torch.mean(value).cpu().detach())
        avg_dict[key]['global_var'].append(torch.var(value).cpu().detach())
        avg_dict[key]['element_mean'].append(torch.mean(value, dim=0).cpu().detach())
        avg_dict[key]['element_var'].append(torch.var(value, dim=0).cpu().detach())
    return avg_dict


def condense_statistics_dict(avg_dict):
    for key, value in avg_dict.items():
        for inner_key, inner_value in value.items():
            l = len(inner_value)
            sum = functools.reduce(lambda a, b: a+b, inner_value, 0)
            avg_dict[key][inner_key] = sum / l
    return avg_dict


model.eval()

noise_avg_dict = None
for i in range(10):
    audio_input = (torch.rand(64, 1, input_length, device=dev) * 2. - 1.) * 1e-2
    scal = preprocessing_module(audio_input)
    #plt.imshow(scal[0, 0], origin='lower', aspect='auto')
    #plt.colorbar()
    #plt.show()
    output = model(scal)
    noise_avg_dict = activation_statistics(register.activations, noise_avg_dict)
noise_avg_dict = condense_statistics_dict(noise_avg_dict)
pprint.pprint(noise_avg_dict)

with open('noise_statistics_' + name + '.pickle', 'wb') as handle:
    pickle.dump(noise_avg_dict, handle)

data_avg_dict = None
v_dataloader = torch.utils.data.DataLoader(trainer.validation_set,
                                           batch_size=64,
                                           num_workers=4)
for step, batch in enumerate(iter(v_dataloader)):
    if step >= 100:
        break
    batch = batch.to(device=dev).unsqueeze(1)
    if preprocessing_module is not None:
        batch = preprocessing_module(batch)
    output = model(batch)
    data_avg_dict = activation_statistics(register.activations, data_avg_dict)
data_avg_dict = condense_statistics_dict(data_avg_dict)
pprint.pprint(data_avg_dict)

with open('data_statistics_' + name + '.pickle', 'wb') as handle:
    pickle.dump(data_avg_dict, handle)

pass








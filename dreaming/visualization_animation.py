import datashader as ds
import numpy as np
import pandas as pd
from colorcet import fire
from datashader import transfer_functions as tf
import torch
import numpy as np
import pickle
import pprint

from configs.experiment_configs import *
from setup_functions import *
from ml_utilities.pytorch_utilities import *
from dreaming.dreaming_functions import *

with open('../noise_statistics_snapshots_model_2019-05-17_run_0_85000.pickle', 'rb') as handle:
    noise_statistics = pickle.load(handle)

experiment = 'e26'
name = 'snapshots_model_2019-05-17_run_0_85000'

try:
    dev = 'cuda:' + str(torch.cuda.current_device())
except:
    dev = 'cpu'
#dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("using device", dev)
settings = experiments[experiment]

if name is not None:
    settings['snapshot_config']['name'] = name

register = ActivationRegister(batch_filter=0)

model, preprocessing_module, untraced_model = setup_model(cqt_params=settings['cqt_config'],
                                                          encoder_params=settings['encoder_config'],
                                                          ar_params=settings['ar_model_config'],
                                                          trainer_args=settings['training_config'],
                                                          device=dev,
                                                          activation_register=register)

settings['snapshot_config']['snapshot_location'] = '../snapshots'
loaded_model, snapshot_manager, continue_training_at_step = setup_snapshot_manager(model=model,
                                                                            args_dict=settings['snapshot_config'],
                                                                            try_proceeding=True,
                                                                            load_to_cpu=(dev == 'cpu'))

state_dict = loaded_model.state_dict()

if dev == 'cpu':
    loaded_model = loaded_model.module
loaded_model.eval()
#loaded_model.module.encoder.eval()
#loaded_model.module.autoregressive_model.eval()

try:
    input_length = model.item_length
except:
    try:
        input_length = model.module.item_length
    except:
        raise

jitter_loop_module = JitterLoop(output_length=input_length, dim=2, jitter_batches=1, jitter_size=64000)

audio_input, sr = torchaudio.load('base_loop_2_16khz.wav')

jittered_input = jitter_loop_module(audio_input.unsqueeze(0))
scal = preprocessing_module(jittered_input)
output = loaded_model(scal)

pprint.pprint(loaded_model)


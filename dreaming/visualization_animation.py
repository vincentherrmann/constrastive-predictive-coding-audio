import datashader as ds
import numpy as np
import pandas as pd
from colorcet import fire
from datashader import transfer_functions as tf
import torch
import numpy as np
import pickle
import imageio
import pprint
import subprocess
from matplotlib import pyplot as plt
from PIL import Image

from configs.experiment_configs import *
from setup_functions import *
from ml_utilities.pytorch_utilities import *
from dreaming.dreaming_functions import *
from dreaming.visualization_functions import *

with open('../noise_statistics_snapshots_model_2019-05-17_run_0_85000.pickle', 'rb') as handle:
    noise_statistics = pickle.load(handle)

experiment = 'e26'
name = 'snapshots_model_2019-05-20_run_0_100000'
#name = None

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

if dev == 'cpu' and type(loaded_model) is torch.nn.DataParallel:
    loaded_model = loaded_model.module

#loaded_model.module.encoder.eval()
#loaded_model.module.autoregressive_model.eval()

try:
    input_length = loaded_model.item_length
except:
    try:
        input_length = loaded_model.module.item_length
    except:
        raise

model.load_state_dict(loaded_model.state_dict())
loaded_model = model
loaded_model.eval()

audio_clip = 'base_loop_3_16khz.wav'
audio_input, sr = torchaudio.load(audio_clip)
# audio_input *= 0
# audio_input[0, 0] = 1.
# audio_input[0, 32000] = 0.1

# plt.plot(audio_input.squeeze())
# plt.show()

jitter_loop_module = JitterLoop(output_length=input_length,
                                dim=2,
                                jitter_batches=1,
                                jitter_size=1,
                                first_batch_offset=audio_input.shape[1] - model.encoder.receptive_field//2)
jittered_input = jitter_loop_module(audio_input.unsqueeze(0))
scal = preprocessing_module(jittered_input)

# plt.plot(jittered_input.squeeze())
# plt.show()

# plt.imshow(scal[0, 0, :, :], origin='lower')
# plt.show()
output = loaded_model(scal)

activation_dict = register.activations
del activation_dict['c_code']
del activation_dict['z_code']
del activation_dict['prediction']
# 'data_statistics_snapshots_model_2019-05-20_run_0_100000.pickle'
with open('../data_statistics_snapshots_model_2019-05-20_run_0_100000.pickle', 'rb') as handle:
    noise_statistics = pickle.load(handle)
activation_dict = normalize_activations(activation_dict, noise_statistics)

range_dict = {
    "scalogram": [10, 500],
    "scalogram_block_0_main_conv_1": (5, 250),
    "scalogram_block_0_main_conv_2": (5, 250),
    "scalogram_block_1_main_conv_1": (2, 125),
    "scalogram_block_1_main_conv_2": (2, 125),
    "scalogram_block_2_main_conv_1": (1, 63),
    "scalogram_block_2_main_conv_2": (1, 63),
    "scalogram_block_3_main_conv_1": (0, 63),
    "scalogram_block_3_main_conv_2": (0, 63),
}

for key, (start, length) in range_dict.items():
    activation_dict[key] = activation_dict[key][:, :, start:start+length]

# for key, value in activation_dict.items():
#     if len(value.shape) == 2:
#         plt.plot(value[-1, :20].detach())
#     else:
#         plt.plot(value[0, -1, :20].detach())
#     plt.show()

video_name = 'activation_animation_' + name + '.mp4'
video_writer = imageio.get_writer(video_name, fps=24)
images = []
positions = np.load('/Users/vincentherrmann/Documents/Projekte/Immersions/visualization/layouts/layout_e25_3.npy')
for time_position in np.linspace(0., 1., 96, endpoint=False):
    current_activations = activation_dict.copy()
    for key, value in activation_dict.items():
        current_activations[key] = interpolate_position(value, time_position)
    activations = flatten_activations(current_activations)

    plot = activation_plot(positions, values=activations.detach().cpu().numpy())
    image_data = np.frombuffer(plot.data.tobytes(), dtype=np.uint8).reshape(600, 600, 4)
    img = Image.fromarray(image_data, mode='RGBA')
    #img = img.rotate(360*time_position)
    video_writer.append_data(np.asarray(img))
    #images.append(img)

    #img.show()

video_writer.close()

cmd = "ffmpeg -i " + video_name + " -i " + audio_clip + \
      " -c:v copy -c:a aac -strict experimental muxed_" + video_name + " -y"
subprocess.call(cmd, shell=True)
# images[0].save('activation_animation.gif',
#                save_all=True,
#                append_images=images[1:],
#                duration=50,
#                loop=0)
#pprint.pprint(loaded_model)


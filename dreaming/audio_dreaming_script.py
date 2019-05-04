import torch
import argparse
import datetime
import random
import torchaudio
from scipy.io.wavfile import write
from setup_functions import *
from configs.experiment_configs import *
from matplotlib import pyplot as plt
from dreaming.dreaming_functions import *

experiment = 'e18'
name = 'snapshots_model_2019-04-14_run_1_95000'

try:
    dev = 'cuda:' + str(torch.cuda.current_device())
except:
    dev = 'cpu'
#dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("using device", dev)
settings = experiments[experiment]

if name is not None:
    settings['snapshot_config']['name'] = name

model, preprocessing_module = setup_model(cqt_params=settings['cqt_config'],
                                          encoder_params=settings['encoder_config'],
                                          ar_params=settings['ar_model_config'],
                                          visible_steps=settings['training_config']['visible_steps'],
                                          prediction_steps=settings['training_config']['prediction_steps'])

settings['snapshot_config']['snapshot_location'] = '../snapshots'
model, snapshot_manager, continue_training_at_step = setup_snapshot_manager(model=model,
                                                                            args_dict=settings['snapshot_config'],
                                                                            try_proceeding=True,
                                                                            load_to_cpu=(dev == 'cpu'))
target_activations = []


def activation_hook(module, input, output):
    target_activations.append(output)


model.autoregressive_model.module_list[4].register_forward_hook(activation_hook)

time_masking = 100
pitch_masking = 50
time_jitter = 0
input_length = model.encoder.receptive_field
input_length += model.encoder.downsampling_factor * (model.visible_steps + model.prediction_steps)
input_length += time_jitter

clip_length = 64000
audio_input, sr = torchaudio.load('base_loop_3_16khz.wav')
audio_input = audio_input.unsqueeze(0)
audio_input = audio_input.to(dev)
audio_input += torch.rand(1, 1, clip_length, device=dev) * 1e-6
audio_input.requires_grad = True

if torch.cuda.device_count() > 1:
    print("using", torch.cuda.device_count(), "GPUs")
    preprocessing_module = torch.nn.DataParallel(preprocessing_module).cuda()
    model = torch.nn.DataParallel(model).cuda()

model.eval()

optimizer = torch.optim.Adam([audio_input], lr=0.001)
jitter_module = Jitter([time_jitter], dims=[2], jitter_batches=32)
jitter_loop_module = JitterLoop(output_length=input_length, dim=2, jitter_batches=32)

for step in range(1000):
    #norm_audio_input = audio_input / torch.var(audio_input)
    norm_audio_input = audio_input
    jittered_input = jitter_loop_module(norm_audio_input)
    scal = preprocessing_module(jittered_input)

    scal = mask_width_section(scal, time_masking)
    scal = mask_height_section(scal, pitch_masking)

    _, _, z, c = model(scal)

    target = torch.cuda.comm.gather(target_activations, dim=0, destination=torch.cuda.current_device())
    target_activations = []
    #loss = -torch.mean(torch.mean(target, dim=2), dim=0)[0]**2
    loss = -torch.mean(c, dim=0)[15]**2
    #loss = -torch.sum(torch.mean(c, dim=0)**2)

    noise_loss = torch.clamp(scal[:, 0], -20., 100.) + 20.01
    noise_loss = torch.mean(torch.pow(torch.abs(noise_loss), 0.5))
    loss = loss + (2.0 * noise_loss)
    
    if audio_input.grad is not None:
        audio_input.grad *= 0
    loss.backward()

    # normalize gradient
    #grad_stft = torch.stft(audio_input.grad[0], 512)
    #grad_norm = torch.sqrt(torch.sum(grad_stft[:, :, :, 0]**2 + grad_stft[:, :, :, 1]**2, dim=1))
    #grad_stft /= grad_norm.view(1, 1, -1, 1)
    #normalized_grad = istft(grad_stft)
    #padding_length = (normalized_grad.shape[1] - audio_input.grad.shape[2]) // 2
    #audio_input.grad = normalized_grad[:, padding_length:padding_length + audio_input.grad.shape[2]].unsqueeze(0)
    #normalized_grad = spectral_local_response_normalization(audio_input.grad[0], size=11).unsqueeze(0)
    #audio_input.grad = normalized_grad

    optimizer.step()
    print("loss:", loss.item())

    if step % 20 == 0:
        data = audio_input.detach().squeeze().cpu().numpy()
        scaled = np.int16(data / np.max(np.abs(data)) * 32767)
        write('last_output.wav', 16000, scaled)

        plt.subplot(2, 1, 1)
        plt.imshow(scal[0, 0].squeeze().cpu().detach(), aspect='auto', origin='lower')

        stft = torch.stft(audio_input.grad[0], 512)
        spec = torch.log(stft[0, :, :, 0]**2 + stft[0, :, :, 1]**2)
        plt.subplot(2, 1, 2)
        plt.imshow(spec.cpu().numpy(), aspect='auto', origin='lower')
        plt.show()


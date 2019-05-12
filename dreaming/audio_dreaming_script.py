import torch
import argparse
import datetime
import random
#import torchaudio
import pprint
from scipy.io.wavfile import write
from setup_functions import *
from configs.experiment_configs import *
from matplotlib import pyplot as plt
from dreaming.dreaming_functions import *
from dreaming.streaming import *
from ml_utilities.pytorch_utilities import *

pp = pprint.PrettyPrinter(indent=4)

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

model, preprocessing_module, untraced_model = setup_model(cqt_params=settings['cqt_config'],
                                                          encoder_params=settings['encoder_config'],
                                                          ar_params=settings['ar_model_config'],
                                                          trainer_args=settings['training_config'],
                                                          device=dev)

settings['snapshot_config']['snapshot_location'] = '../snapshots'
model, snapshot_manager, continue_training_at_step = setup_snapshot_manager(model=model,
                                                                            args_dict=settings['snapshot_config'],
                                                                            try_proceeding=True,
                                                                            load_to_cpu=(dev == 'cpu'))

pp.pprint(model)

server = LoopStreamServer(port=8765, message_length=128000)
server.start_server()

print("server started")

target_activations = []

def activation_hook(module, input, output):
    target_activations.append(output)


#model.autoregressive_model.module_list[4].register_forward_hook(activation_hook)
#model.encoder.blocks[1].register_forward_hook(activation_hook)

time_masking = 100
pitch_masking = 50
time_jitter = 0
input_length = model.encoder.receptive_field
input_length += model.encoder.downsampling_factor * (model.visible_steps + model.prediction_steps)
input_length += time_jitter

clip_length = 64000
audio_input, sr = torchaudio.load('base_loop_2_16khz.wav')
audio_input = audio_input.unsqueeze(0)
audio_input = audio_input.to(dev)
audio_input += torch.rand(1, 1, clip_length, device=dev) * 1e-6
audio_input.requires_grad = True

model.eval()

#traced_model = torch.jit.trace(model, torch.ones(32, 2, 256, 629, requires_grad=True, device=dev))

if torch.cuda.device_count() > 1:
    print("using", torch.cuda.device_count(), "GPUs")
    preprocessing_module = torch.nn.DataParallel(preprocessing_module).cuda()
    model = torch.nn.DataParallel(model).cuda()
    #traced_model = torch.nn.DataParallel(traced_model).cuda()

optimizer = torch.optim.Adam([audio_input], lr=0.0005)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda s: 1.05**s)
jitter_module = Jitter([time_jitter], dims=[2], jitter_batches=32)
jitter_loop_module = JitterLoop(output_length=input_length, dim=2, jitter_batches=32, jitter_size=64000)

time_meter = AverageMeter()
tic = time.time()

for step in range(1000):
    #norm_audio_input = audio_input / torch.var(audio_input)
    norm_audio_input = audio_input
    jittered_input = jitter_loop_module(norm_audio_input)
    scal = preprocessing_module(jittered_input)

    scal = mask_width_section(scal, time_masking)  # batch, channel, pitch, time
    scal = mask_height_section(scal, pitch_masking)

    _, _, z, c = model(scal)
    #_, _, z_o, c_o = model(scal)

    #target = torch.cuda.comm.gather(target_activations, dim=0, destination=torch.cuda.current_device())
    #target_activations = []
    #loss = -torch.mean(torch.mean(target, dim=2), dim=0)[0]**2  # autoregressive channel for whole loop
    #loss = -torch.mean(torch.mean(target**2, dim=0), dim=0)[5, 0] # pitch, time
    #loss = -torch.mean(c, dim=0)[2]**2
    loss = -torch.mean(torch.mean(c, dim=0)**2)
    #loss_o = -torch.mean(torch.mean(c_o, dim=0)**2)
    #print("original loss:", loss_o)

    noise_loss = torch.clamp(scal[:, 0], -20., 100.) + 20.01
    noise_loss = torch.mean(torch.pow(torch.abs(noise_loss), 0.5))
    loss = loss + (0.1 * noise_loss)
    
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
    scheduler.step()
    toc = time.time()
    time_meter.update(toc-tic)
    tic = toc

    print("loss:", loss.item())
    print("lr:", scheduler.get_lr())
    print("duration:", time_meter.val)

    try:
        signal_data = audio_input.detach().squeeze().cpu().numpy()
        signal_data /= np.max(signal_data)
        signal_data *= 32000.
        server.set_data(signal_data.astype(np.int16).tobytes())
    except:
        raise
        #pass

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

server.stop()


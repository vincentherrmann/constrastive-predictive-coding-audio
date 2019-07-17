import torch
import pprint
import time
import pickle
import os.path
import glob
from matplotlib import pyplot as plt

from configs.experiment_configs import *
from setup_functions import *
from dreaming.streaming import SocketDataExchangeServer
from dreaming.dreaming_functions import *
from contrastive_estimation_training import linear_score_function
try:
    from dreaming.dreaming_control import control_dict
except:
    control_dict = None

score_function = linear_score_function

#port = 2222  # if running on the local machine
port = 8765  # for running on a server, then ssh tunneling is required:
# gcloud compute ssh k80x4-preemptible-2 --project pytorch-wavenet --zone europe-west1-d -- -L 2222:localhost:8765
host = '127.0.0.1'


class DreamingCalculation:
    def __init__(self, port, host, start_control_dict=None):
        self.experiment = 'e26'
        self.name = 'snapshots_model_2019-05-20_run_0_100000'
        self.mean_statistics_path = '../noise_statistics_snapshots_model_2019-05-20_run_0_100000.pickle'
        self.variance_statistics_path = '../data_statistics_snapshots_model_2019-05-20_run_0_100000.pickle'
        self.loss_count = 100
        self.losses = [0.] * 100
        self.port = port
        self.host = host

        with open(self.mean_statistics_path, 'rb') as handle:
            self.mean_statistics = pickle.load(handle)
            for key, value in self.mean_statistics.items():
                self.mean_statistics[key] = value['element_mean']

        with open(self.variance_statistics_path, 'rb') as handle:
            self.variance_statistics = pickle.load(handle)
            for key, value in self.variance_statistics.items():
                self.variance_statistics[key] = value['element_var'] + 0.1 #* 0. + 1.

        try:
            self.dev = 'cuda:' + str(torch.cuda.current_device())
        except:
            self.dev = 'cpu'
        print("using device", self.dev)
        if torch.cuda.device_count() > 1:
            all_devices = [i for i in range(torch.cuda.device_count())]
        else:
            all_devices = None
        self.register = ActivationRegister(devices=all_devices)
        self.input_length = 0

        self.model, self.preprocessing_module, self.activation_normalization = self.load_model(self.experiment,
                                                                                               self.name)
        self.model.eval()

        # audio_input, sr = torchaudio.load('base_loop_2_16khz.wav')
        # audio_input = audio_input.unsqueeze(0) * 0.5
        # audio_input = audio_input.to(self.dev)
        # audio_input += (torch.rand(1, 1, audio_input.shape[2], device=self.dev) * 2. - 1.) * 1e-4

        self.soundclip_dict = OrderedDict(
            [(os.path.basename(path), torchaudio.load(path)[0].unsqueeze(0).to(self.dev))
             for path in sorted(glob.glob('loops_16khz/*.wav'))]
        )

        audio_input = self.soundclip_dict['silence.wav'].clone()
        self.original_input = audio_input.clone()

        #audio_input += torch.rand(1, 1, audio_input.shape[2], device=self.dev) * 1e-4
        audio_input.requires_grad = True
        self.audio_input = audio_input
        self.sr = 16000

        self.optimizer = torch.optim.SGD([self.audio_input], lr=1e-3)
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda s: 1.05 ** s)
        self.jitter_loop_module = JitterLoop(output_length=self.input_length, dim=2, jitter_batches=8, jitter_size=64000,
                                             first_batch_offset=0)
        self.input_extender = JitterLoop(output_length=self.input_length, dim=2, jitter_size=0)
        self.time_masking = Masking2D(size=0, axis='width', value=torch.FloatTensor([-20., 0.]), exclude_first_batch=True)
        self.pitch_masking = Masking2D(size=0, axis='height', value=torch.FloatTensor([-20., 0.]), exclude_first_batch=True)
        self.time_meter = AverageMeter()

        self.communicator = SocketDataExchangeServer(port=self.port,
                                                     host=self.host)
        self.control_dict = start_control_dict
        self.eq_bands = None

    def load_model(self, experiment, name):
        settings = experiments[experiment]
        if name is not None:
            settings['snapshot_config']['name'] = name

        model, preprocessing_module, untraced_model = setup_model(cqt_params=settings['cqt_config'],
                                                                  encoder_params=settings['encoder_config'],
                                                                  ar_params=settings['ar_model_config'],
                                                                  trainer_args=settings['training_config'],
                                                                  device=self.dev,
                                                                  activation_register=self.register)

        settings['snapshot_config']['snapshot_location'] = '../snapshots'
        loaded_model, snapshot_manager, continue_training_at_step = setup_snapshot_manager(model=model,
                                                                                           args_dict=settings[
                                                                                               'snapshot_config'],
                                                                                           try_proceeding=True,
                                                                                           load_to_cpu=(self.dev == 'cpu'))
        if type(model) is torch.nn.DataParallel:
            model = model.module

        if type(loaded_model) is torch.nn.DataParallel:
            loaded_model = loaded_model.module

        if type(preprocessing_module) is torch.nn.DataParallel:
            preprocessing_module = preprocessing_module.module

        model.load_state_dict(loaded_model.state_dict())

        self.input_length = model.item_length
        pprint.pprint(model)

        # if type(loaded_model) is torch.nn.DataParallel:
        #     self.input_length = loaded_model.module.item_length
        # else:
        #     self.input_length = loaded_model.item_length

        normalization_module = ActivationNormalization(self.mean_statistics, self.variance_statistics)

        if torch.cuda.device_count() > 1:
            print("using", torch.cuda.device_count(), "GPUs")
            preprocessing_module = torch.nn.DataParallel(preprocessing_module).cuda()
            normalization_module = torch.nn.DataParallel(normalization_module).cuda()
            model = torch.nn.DataParallel(model).cuda()

        return model, preprocessing_module, normalization_module

    def dreaming_step(self):
        tic = time.time()
        if self.communicator.new_data_available:
            self.control_dict = pickle.loads(self.communicator.get_received_data())
            self.eq_bands = self.control_dict['eq_bands']
            if self.eq_bands is not None:
                self.eq_bands = torch.from_numpy(self.eq_bands).to(self.dev)
            self.original_input = self.soundclip_dict[self.control_dict['selected_clip']]
        if self.control_dict is None:
            time.sleep(0.01)
            return
        lr = 10**self.control_dict['lr']
        if self.control_dict['lr'] < -4.9:
            lr = 0.
        for g in self.optimizer.param_groups:
            g['lr'] = lr

        batch_size = int(self.control_dict['batch_size'])
        self.jitter_loop_module.jitter_size = int(self.sr * self.control_dict['time_jitter'])
        self.jitter_loop_module.jitter_batches = batch_size

        self.audio_input.data += torch.rand_like(self.audio_input.data) * 1e-4
        normalized_audio_input = self.audio_input

        jittered_input = self.jitter_loop_module(normalized_audio_input)
        scal = self.preprocessing_module(jittered_input)
        scal.retain_grad()

        self.time_masking.size = int(self.control_dict['time_masking'] * scal.shape[3])
        self.pitch_masking.size = int(self.control_dict['pitch_masking'] * scal.shape[2])
        masked_scal = self.time_masking(scal)
        masked_scal = self.pitch_masking(masked_scal)

        masked_scal.retain_grad()

        predicted_z, targets, z, c = self.model(masked_scal)

        selected_regions = self.control_dict['selected_regions']
        normalized_activations = self.activation_normalization(self.register.get_activations())

        del normalized_activations['c_code']
        del normalized_activations['z_code']
        del normalized_activations['prediction']

        viz_activations = convert_activation_dict_type(normalized_activations, select_batch=0)

        selected_activations = normalized_activations.copy()
        for key, value in selected_activations.items():
            selected_activations[key] = value * 0.

        loss = 0.

        for region in selected_regions:
            if region['layer'] == 'prediction':
                prediction_steps = predicted_z.shape[1]
                scores = score_function(predicted_z, targets)
                noise_scoring = torch.logsumexp(scores.view(-1, batch_size, prediction_steps),
                                                dim=0)  # target_batch, target_step
                valid_scores = torch.diagonal(scores, dim1=0, dim2=2)  # data_step, target_step, batch
                valid_scores = torch.diagonal(valid_scores, dim1=0, dim2=1)  # batch, step

                prediction_losses = -torch.mean(valid_scores - noise_scoring, dim=1)
                loss += torch.mean(prediction_losses)
                continue
            layer = selected_activations[region['layer']]
            layer = layer.unsqueeze(0)
            slice = select_activation_slice(layer,
                                            channel=region['channel'], channel_region=region['channel region'],
                                            pitch=region['pitch'], pitch_region=region['pitch region'],
                                            time=region['time'], time_region=region['time region'])
            slice += 1.

        flat_selection = flatten_activations(selected_activations)
        flat_activations = flatten_activations(normalized_activations)
        selected_activations = flat_activations[flat_selection > 0.]
        if selected_activations.shape[0] != 0:
            loss += torch.mean(torch.mean(selected_activations, dim=0)**2)

        # ###
        # if selected_activation is None:
        #     return
        # elif selected_activation == "prediction":
        #     prediction_steps = predicted_z.shape[1]
        #     scores = score_function(predicted_z, targets)
        #     noise_scoring = torch.logsumexp(scores.view(-1, batch_size, prediction_steps),
        #                                     dim=0)  # target_batch, target_step
        #     valid_scores = torch.diagonal(scores, dim1=0, dim2=2)  # data_step, target_step, batch
        #     valid_scores = torch.diagonal(valid_scores, dim1=0, dim2=1)  # batch, step
        #
        #     prediction_losses = -torch.mean(valid_scores - noise_scoring, dim=1)
        #     loss = torch.mean(prediction_losses)
        # else:
        #     activations = normalized_activations[selected_activation].to(self.dev)
        #     target = select_activation_slice(activations,
        #                                      channel=self.control_dict['channel'],
        #                                      channel_region=self.control_dict['channel_region'],
        #                                      pitch=self.control_dict['pitch'],
        #                                      pitch_region=self.control_dict['pitch_region'],
        #                                      time=self.control_dict['time'],
        #                                      time_region=self.control_dict['time_region'])
        #     print("target mean:", torch.mean(torch.abs(target)).item(),
        #           "rest mean:", torch.mean(torch.abs(activations)).item())
        #     loss = torch.mean(torch.mean(target, dim=0)**2)
        #     target *= 0.

        activation_energy_loss = torch.mean(torch.abs(flat_activations)**2)
        activation_energy_loss *= self.control_dict['activation_loss']

        noise_loss = torch.clamp(scal[:, 0], -20., 100.) + 20.01
        noise_loss = torch.mean(torch.pow(torch.abs(noise_loss), 0.5))
        loss = -(loss - activation_energy_loss) + (self.control_dict['noise_loss'] * noise_loss)

        self.losses.pop(0)
        self.losses.append(loss.item())

        if self.audio_input.grad is not None:
            self.audio_input.grad *= 0

        if loss != loss: # if loss is nan
            print("nan loss!")
        else:
            loss.backward()

            # normalize gradient
            self.audio_input.grad /= torch.max(self.audio_input.grad).squeeze() + 1e-5

            self.optimizer.step()
        #self.scheduler.step()
        self.time_meter.update(time.time() - tic)

        print("loss:", loss.item())
        #print("max input:", amplitude)
        print("lr:", self.optimizer.param_groups[0]['lr'])
        #print("duration:", self.time_meter.val)

        mix_o = self.control_dict['mix_o']
        self.audio_input.data *= (1 - mix_o)
        self.audio_input.data += mix_o * self.original_input

        amplitude = torch.abs(self.audio_input.max()).item()
        if amplitude > 1.:
            self.audio_input.data /= amplitude

        # if self.control_dict['eq_bands'] is not None:
        #     audio_input_length = self.audio_input.shape[2]
        #     stft = torch.stft(self.audio_input.data[0], n_fft=512, hop_length=128, center=True)
        #     stft_amp = abs(stft)
        #     stft_ang = angle(stft)
        #     stft_amp *= self.eq_bands.view(1, -1, 1)
        #     stft = polar_to_complex(stft_amp, stft_ang)
        #
        #     data = istft(stft, hop_length=128)
        #     offset = max(0, (data.shape[1] - audio_input_length) // 2)
        #     self.audio_input.data = data[:, offset:audio_input_length+offset].unsqueeze(0)

        try:
            input_scal = self.preprocessing_module(self.input_extender(self.audio_input))
            input_grad_scal = self.preprocessing_module(self.input_extender(self.audio_input.grad))

            data_dict = {'activations': viz_activations,
                         'scalogram': input_scal[0, 0].detach().cpu(),
                         'scalogram_grad': input_grad_scal[0, 0].detach().cpu(),
                         'losses': self.losses,
                         'amplitude': amplitude,
                         'step_duration': self.time_meter.val}

            signal_data = self.audio_input.clone().detach().squeeze().cpu().numpy()
            # if amplitude > 1.:
            #     signal_data /= amplitude
            signal_data *= 32000.
            data_dict['audio'] = signal_data.astype(np.int16)

            data = pickle.dumps(data_dict)
            self.communicator.set_new_data(data)
        except:
            raise


if __name__ == '__main__':
    try:
        start_control_dict = control_dict.copy()
        start_control_dict['lr'] = -10.
        start_control_dict['time_masking'] = 0.1
        start_control_dict['pitch_masking'] = 0.1
        start_control_dict['batch_size'] = 5
    except:
        start_control_dict = None

    calc = DreamingCalculation(port=port, host=host, start_control_dict=start_control_dict)
    while True:
        calc.dreaming_step()
    #calc.dreaming_step()
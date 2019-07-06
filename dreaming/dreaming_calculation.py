import torch
import pprint
import time
import pickle
from matplotlib import pyplot as plt

from configs.experiment_configs import *
from setup_functions import *
from dreaming.streaming import SocketDataExchangeServer
from dreaming.dreaming_functions import *
try:
    from dreaming.dreaming_control import control_dict
except:
    control_dict = None

port = 2222  # if running on the local machine
#port = 8765  # for running on a server, then ssh tunneling is required:
# gcloud compute ssh k80x4-preemptible-2 --project pytorch-wavenet --zone europe-west1-d -- -L 2222:localhost:8765
host = '127.0.0.1'

class DreamingCalculation:
    def __init__(self, port, host, start_control_dict=None):
        self.experiment = 'e26'
        self.name = 'snapshots_model_2019-05-20_run_0_100000'
        self.port = port
        self.host = host

        try:
            self.dev = 'cuda:' + str(torch.cuda.current_device())
        except:
            self.dev = 'cpu'
        print("using device", self.dev)
        self.register = ActivationRegister()
        self.input_length = 0

        self.model, self.preprocessing_module = self.load_model(self.experiment,
                                                                self.name)
        self.model.eval()

        audio_input, sr = torchaudio.load('base_loop_2_16khz.wav')
        audio_input = audio_input.unsqueeze(0)
        audio_input = audio_input.to(self.dev)
        audio_input += torch.rand(1, 1, audio_input.shape[2], device=self.dev) * 1e-6
        audio_input.requires_grad = True
        self.audio_input = audio_input
        self.sr = sr

        self.optimizer = torch.optim.SGD([self.audio_input], lr=1e-3)
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda s: 1.05 ** s)
        self.jitter_loop_module = JitterLoop(output_length=self.input_length, dim=2, jitter_batches=32, jitter_size=64000)
        self.input_extender = JitterLoop(output_length=self.input_length, dim=2, jitter_size=0)
        self.time_masking = Masking2D(size=0, axis='width', value=torch.FloatTensor([-20., 0.]))
        self.pitch_masking = Masking2D(size=0, axis='height', value=torch.FloatTensor([-20., 0.]))
        self.time_meter = AverageMeter()

        self.communicator = SocketDataExchangeServer(port=self.port,
                                                     host=self.host)
        self.control_dict = start_control_dict

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

        if torch.cuda.device_count() > 1:
            print("using", torch.cuda.device_count(), "GPUs")
            preprocessing_module = torch.nn.DataParallel(preprocessing_module).cuda()
            model = torch.nn.DataParallel(model).cuda()

        return model, preprocessing_module

    def dreaming_step(self):
        tic = time.time()
        if self.communicator.new_data_available:
            self.control_dict = pickle.loads(self.communicator.get_received_data())
        if self.control_dict is None:
            time.sleep(0.01)
            return
        lr = 10**self.control_dict['lr']
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.jitter_loop_module.jitter_size = int(self.sr * self.control_dict['time_jitter'])
        self.jitter_loop_module.jitter_batches = int(self.control_dict['batch_size'])
        jittered_input = self.jitter_loop_module(self.audio_input)
        scal = self.preprocessing_module(jittered_input)
        scal.retain_grad()

        self.time_masking.size = int(self.control_dict['time_masking'] * scal.shape[3])
        self.pitch_masking.size = int(self.control_dict['pitch_masking'] * scal.shape[2])
        masked_scal = self.time_masking(scal)
        masked_scal = self.pitch_masking(masked_scal)

        masked_scal.retain_grad()

        _, _, z, c = self.model(masked_scal)

        selected_activation = self.control_dict['activation']
        if selected_activation is None:
            return
        activations = self.register.activations[selected_activation].to(self.dev)
        target = select_activation_slice(activations,
                                         channel=self.control_dict['channel'],
                                         channel_region=self.control_dict['channel_region'],
                                         pitch=self.control_dict['pitch'],
                                         pitch_region=self.control_dict['pitch_region'])

        loss = -torch.mean(torch.mean(target, dim=0)**2)
        noise_loss = torch.clamp(scal[:, 0], -20., 100.) + 20.01
        noise_loss = torch.mean(torch.pow(torch.abs(noise_loss), 0.5))
        loss = loss + (self.control_dict['noise_loss'] * noise_loss)

        if self.audio_input.grad is not None:
            self.audio_input.grad *= 0
        loss.backward()

        # normalize gradient
        self.audio_input.grad /= torch.max(self.audio_input.grad).squeeze()

        self.optimizer.step()
        #self.scheduler.step()
        self.time_meter.update(time.time() - tic)

        print("loss:", loss.item())
        print("max input:", self.audio_input.max().item())
        print("lr:", self.optimizer.param_groups[0]['lr'])
        #print("duration:", self.time_meter.val)

        try:
            activations = convert_activation_dict_type(self.register.activations, select_batch=0)

            input_scal = self.preprocessing_module(self.input_extender(self.audio_input))
            input_grad_scal = self.preprocessing_module(self.input_extender(self.audio_input.grad))

            data_dict = {'activations': activations,
                         'scalogram': input_scal[0, 0].detach().cpu(),
                         'scalogram_grad': input_grad_scal[0, 0].detach().cpu()}

            signal_data = self.audio_input.clone().detach().squeeze().cpu().numpy()
            signal_data /= np.max(signal_data)
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
    #while True:
    #    calc.dreaming_step()
    calc.dreaming_step()

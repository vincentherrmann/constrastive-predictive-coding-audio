from unittest import TestCase

import pickle
import time
import multiprocessing as mp

from configs.experiment_configs import *
from setup_functions import *
from dreaming.dreaming_functions import *
from dreaming.visualization_functions import *


class TestDatashaderViz(TestCase):
    def test_viz_performance(self):
        with open('../noise_statistics_snapshots_model_2019-05-17_run_0_85000.pickle', 'rb') as handle:
            noise_statistics = pickle.load(handle)

        experiment = 'e26'
        name = 'snapshots_model_2019-05-20_run_0_100000'

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
                                                                                           args_dict=settings[
                                                                                               'snapshot_config'],
                                                                                           try_proceeding=True,
                                                                                           load_to_cpu=(dev == 'cpu'))

        if dev == 'cpu' and type(loaded_model) is torch.nn.DataParallel:
            loaded_model = loaded_model.module

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

        audio_clip = '../dreaming/base_loop_3_16khz.wav'
        audio_input, sr = torchaudio.load(audio_clip)

        jitter_loop_module = JitterLoop(output_length=input_length,
                                        dim=2,
                                        jitter_batches=1,
                                        jitter_size=1,
                                        first_batch_offset=audio_input.shape[1] - model.encoder.receptive_field // 2)
        jittered_input = jitter_loop_module(audio_input.unsqueeze(0))
        scal = preprocessing_module(jittered_input)

        output = loaded_model(scal)

        activation_dict = register.activations
        del activation_dict['c_code']
        del activation_dict['z_code']
        del activation_dict['prediction']
        # 'data_statistics_snapshots_model_2019-05-20_run_0_100000.pickle'
        with open('../data_statistics_snapshots_model_2019-05-20_run_0_100000.pickle', 'rb') as handle:
            noise_statistics = pickle.load(handle)
        activation_dict = normalize_activations(activation_dict, noise_statistics, element_wise=True)

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
            activation_dict[key] = activation_dict[key][:, :, start:start + length]

        positions = np.load(
            '/Users/vincentherrmann/Documents/Projekte/Immersions/visualization/layouts/layout_e25_3.npy')

        canvas = ds.Canvas(plot_width=400, plot_height=400,
                           x_range=(-7, 7), y_range=(-7, 7),
                           x_axis_type='linear', y_axis_type='linear')

        current_activations = activation_dict.copy()
        for key, value in activation_dict.items():
            current_activations[key] = interpolate_position(value, 0)
        activations = flatten_activations(current_activations)

        for i in range(10):
            tik = time.time()
            plot = activation_plot(positions, values=activations.detach().cpu().numpy(), canvas=canvas)
            print("activation plot time duration:", time.time() - tik)

    def test_multiprocessing(self):
        class TestMultiProc:
            def __init__(self):
                self.foo = 'foo'

                parent_conn, child_conn = mp.Pipe(duplex=True)
                self.p = mp.Process(target=self.worker, args=(child_conn,))
                self.p.daemon = True
                self.p.start()

                parent_conn.send('start')

                for i in range(20):
                    if parent_conn.poll(0):
                        r = parent_conn.recv()
                        print(r)
                    else:
                        time.sleep(0.2)
                        print("no data")

            def worker(self, conn):
                s = conn.recv()
                print(s)
                #while True:
                #    pass
                for i in range(10):
                    print(self.foo)
                    conn.send(i)
                    time.sleep(0.5)

        TestMultiProc()


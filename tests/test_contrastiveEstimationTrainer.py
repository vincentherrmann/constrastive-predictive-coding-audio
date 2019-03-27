from unittest import TestCase
from audio_model import *
from attention_model import *
from contrastive_estimation_training import *

import sys
import time
try:
    from audio_dataset import *
except:
    sys.path.append('/Users/vincentherrmann/Documents/Projekte/Immersions/Data')
    from audio_dataset import *



class TestContrastiveEstimationTrainer(TestCase):
    def setUp(self):
        self.encoder = AudioEncoder({'strides': [5, 4, 2, 2, 2],
                                     'kernel_sizes': [10, 8, 4, 4, 4],
                                     'channel_count': [32, 32, 32, 32, 32],
                                     'bias': True})
        #self.ar_model = AudioGRUModel(input_size=32, hidden_size=48)
        self.ar_model = AttentionModel(channels=32, output_size=48, num_layers=2, seq_length=64)
        self.pc_model = AudioPredictiveCodingModel(encoder=self.encoder,
                                                   autoregressive_model=self.ar_model,
                                                   enc_size=32,
                                                   ar_size=48,
                                                   prediction_steps=12)
        item_length = self.encoder.receptive_field + (64 + 12 - 1) * self.encoder.downsampling_factor
        self.dataset = AudioDataset(location='/Users/vincentherrmann/Documents/Projekte/Immersions/MelodicProgressiveHouse_Tracks_test',
                                    item_length=item_length)

    def test_contrastiveEstimationTraining(self):
        visible_steps = 64
        prediction_steps = self.pc_model.prediction_steps
        visible_length = self.encoder.receptive_field + (visible_steps-1)*self.encoder.downsampling_factor
        prediction_length = self.encoder.receptive_field + (prediction_steps-1)*self.encoder.downsampling_factor

        self.trainer = ContrastiveEstimationTrainer(model=self.pc_model,
                                                    dataset=self.dataset,
                                                    visible_length=visible_length,
                                                    prediction_length=prediction_length)
        tic = time.time()
        self.trainer.train(batch_size=32, max_steps=1000, num_workers=4, lr=1e-3)
        toc = time.time()

        time_per_minibatch = (toc - tic) / 10
        print("time per minibatch:", time_per_minibatch)

    def test_contrastiveEstimationTesting(self):
        visible_steps = 64
        prediction_steps = self.pc_model.prediction_steps
        visible_length = self.encoder.receptive_field + (visible_steps-1)*self.encoder.downsampling_factor
        prediction_length = self.encoder.receptive_field + (prediction_steps-1)*self.encoder.downsampling_factor

        self.trainer = ContrastiveEstimationTrainer(model=self.pc_model,
                                                    dataset=self.dataset,
                                                    validation_set=self.dataset,
                                                    visible_length=visible_length,
                                                    prediction_length=prediction_length)
        self.trainer.score_over_all_timesteps = False
        tic = time.time()
        losses, accuracies, mean_score, mmi_lb = self.trainer.validate(batch_size=16, max_steps=20, num_workers=4)
        toc = time.time()

        print("losses:", losses)
        print("accuracies:", accuracies)

        time_per_minibatch = (toc - tic) / 20
        print("time per minibatch:", time_per_minibatch)
        assert False

    def test_trainingMemoryUsage(self):
        print("parameter count of model:", self.pc_model.parameter_count())

    def test_testTask(self):
        visible_steps = 64
        visible_length = self.encoder.receptive_field + (visible_steps - 1) * self.encoder.downsampling_factor

        test_dataset = AudioTestingDataset(location='/Users/vincentherrmann/Documents/Projekte/Immersions/MelodicProgressiveHouse_Tracks_small_test',
                                           item_length=visible_length)
        self.trainer = ContrastiveEstimationTrainer(model=self.pc_model,
                                                    dataset=self.dataset,
                                                    visible_length=visible_length,
                                                    prediction_length=0,
                                                    test_task_set=test_dataset)
        accuracy = self.trainer.test_task(num_workers=4)
        assert accuracy > 0. and accuracy < 1.




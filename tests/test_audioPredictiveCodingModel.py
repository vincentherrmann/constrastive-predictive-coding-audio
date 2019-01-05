from unittest import TestCase
from audio_model import *

import torch
import gc

encoder_test_dict = {'strides': [5, 4, 2, 2, 2],
                     'kernel_sizes': [10, 8, 4, 4, 4],
                     'channel_count': [32, 32, 32, 32, 32],
                     'bias': True}


class TestAudioPredictiveCodingModel(TestCase):
    def setUp(self):
        self.encoder = AudioEncoder(encoder_test_dict)
        self.ar_model = AudioGRUModel(input_size=32, hidden_size=64)
        self.model = AudioPredictiveCodingModel(encoder=self.encoder,
                                                autoregressive_model=self.ar_model,
                                                enc_size=32,
                                                ar_size=64,
                                                prediction_steps=1)

    def test_audioPredictiveCodingModel(self):
        test_input = torch.randn([7, 1, 4800])  # batch_size, channels, length
        test_output = self.model(test_input)

        assert list(test_output.shape) == [7, 12, 32]

    def test_noiseContrastiveEstimationLoss(self):
        visible_steps = 64
        prediction_steps = self.model.prediction_steps
        item_length = self.encoder.receptive_field + (visible_steps + prediction_steps - 1) * self.encoder.downsampling_factor
        visible_length = self.encoder.receptive_field + (visible_steps - 1) * self.encoder.downsampling_factor
        prediction_length = self.encoder.receptive_field + (prediction_steps - 1) * self.encoder.downsampling_factor

        batch = torch.randn([7, item_length])
        visible_input = batch[:, :visible_length].unsqueeze(1)
        target_input = batch[:, -prediction_length:].unsqueeze(1)
        predictions = self.model(visible_input)
        targets = self.model.encoder(target_input).detach()

        batch_size, steps, _ = predictions.shape

        targets = targets.permute(2, 1, 0)  # step, length, batch
        predictions = predictions.permute(1, 0, 2)  # step, batch, length

        scores = torch.matmul(predictions, targets)#.squeeze()  # step, prediction_batch, target_batch
        scores = torch.exp(scores)
        score_sum = torch.sum(scores, dim=1)  # .detach()  # step, target_batch TODO: should this be detached?
        valid_scores = torch.diagonal(scores, dim1=1, dim2=2)  # step, data_batch
        loss_logits = torch.log(valid_scores / score_sum)  # step, batch

        # calculate prediction accuracy as the proportion of scores that are highest for the correct target
        correct_prediction_template = torch.range(0, batch_size-1, dtype=torch.long).unsqueeze(0).repeat(steps, 1)
        max_score_indices = torch.argmax(scores, dim=1)
        correctly_predicted = torch.eq(correct_prediction_template.type_as(max_score_indices), max_score_indices)
        prediction_accuracy = torch.sum(correctly_predicted, dim=1).type_as(data) / batch_size

        prediction_losses = -torch.sum(loss_logits, dim=1)

        assert list(prediction_losses.shape) == [12]

    def test_memoryConsumption(self):
        self.encoder = AudioEncoder({'strides': [5, 4, 2, 2, 2],
                                     'kernel_sizes': [10, 8, 4, 4, 4],
                                     'channel_count': [512, 512, 512, 512, 512],
                                     'bias': True})
        self.ar_model = AudioGRUModel(input_size=512, hidden_size=256)
        self.model = AudioPredictiveCodingModel(encoder=self.encoder,
                                                autoregressive_model=self.ar_model,
                                                enc_size=512,
                                                ar_size=256,
                                                prediction_steps=12)
        print("parameter count:", self.model.parameter_count())

        total_parameters = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
                    total_parameters += np.prod(obj.size())
            except:
                pass
        print("pure model size", total_parameters)

        batch_size = 32
        visible_steps = 64
        prediction_steps = self.model.prediction_steps
        visible_length = self.encoder.receptive_field + (visible_steps - 1) * self.encoder.downsampling_factor
        prediction_length = self.encoder.receptive_field + (prediction_steps - 1) * self.encoder.downsampling_factor
        visible_input = torch.zeros([batch_size, visible_length]).unsqueeze(1)
        target_input = torch.zeros([batch_size, prediction_length]).unsqueeze(1)

        predictions = self.model(visible_input)
        targets = self.model.encoder(target_input).detach()

        loss = torch.mean(predictions)



        total_parameters = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
                    total_parameters += np.prod(obj.size())
            except:
                pass
        print("backward memory need", total_parameters)

        self.model.zero_grad()
        loss.backward()


        assert False



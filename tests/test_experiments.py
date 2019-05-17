from unittest import TestCase
from scalogram_model import *
from audio_model import *
from configs.experiment_configs import *
from setup_functions import *

import torch
import time
import torch.nn as nn

import pprint

class TestExperiments(TestCase):
    def test_experiment_architecture(self):
        settings = experiments['e25']
        register = ActivationRegister(batch_filter=0)

        dev = 'cpu'
        pc_model, preprocessing_module, untraced_model = setup_model(cqt_params=settings['cqt_config'],
                                                                     encoder_params=settings['encoder_config'],
                                                                     ar_params=settings['ar_model_config'],
                                                                     trainer_args=settings['training_config'],
                                                                     device=dev,
                                                                     activation_register=register)

        dummy_batch = torch.randn(2, 1, untraced_model.item_length)
        if preprocessing_module is not None:
            dummy_batch = preprocessing_module(dummy_batch)
        result = pc_model(dummy_batch)

        for key, value in register.activations.items():
            print(key + ': ', str(value.shape))

        #pprint.pprint(register.activations)


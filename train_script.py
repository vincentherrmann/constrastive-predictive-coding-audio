from audio_dataset import *
from audio_model import *
from contrastive_estimation_training import *
from torch import autograd

dev = 'cpu'

name = "model_2019-01-09_run_0"

encoding_size = 64
ar_code_size = 64
encoder_params = encoder_default_dict
encoder_params["channel_count"] = [encoding_size for _ in range(5)]
encoder = AudioEncoder(encoder_params)

ar_model = AudioGRUModel(input_size=encoding_size, hidden_size=ar_code_size)
pc_model = AudioPredictiveCodingModel(encoder=encoder,
                                      autoregressive_model=ar_model,
                                      enc_size=encoding_size,
                                      ar_size=ar_code_size,
                                      prediction_steps=12)

#pc_model = load_to_cpu('/Users/vincentherrmann/Documents/Projekte/Immersions/snapshots/model_2019-01-07_run_0/immersions-snapshots-error_snapshot.dms')
#encoder = pc_model.encoder
#ar_model = pc_model.autoregressive_model

batch_size = 16
lr = 1e-3

pc_model.to(dev)
print("number of parameters:", pc_model.parameter_count())
print("receptive field:", encoder.receptive_field)

visible_steps = 128
prediction_steps = pc_model.prediction_steps
item_length = encoder.receptive_field + (visible_steps + prediction_steps - 1) * encoder.downsampling_factor
visible_length = encoder.receptive_field + (visible_steps - 1) * encoder.downsampling_factor
prediction_length = encoder.receptive_field + (prediction_steps - 1) * encoder.downsampling_factor

print("item length:", item_length)

dataset = AudioDataset('/Users/vincentherrmann/Documents/Projekte/Immersions/MelodicProgressiveHouse_Tracks_test',
                       item_length=item_length)

validation_set = AudioDataset('/Users/vincentherrmann/Documents/Projekte/Immersions/MelodicProgressiveHouse_Tracks_test',
                              item_length=item_length)
print("dataset length:", len(dataset))

trainer = ContrastiveEstimationTrainer(model=pc_model,
                                       dataset=dataset,
                                       visible_length=visible_length,
                                       prediction_length=prediction_length,
                                       device=dev,
                                       validation_set=validation_set)
#trainer.validate(batch_size=batch_size, num_workers=4, max_steps=50)

with autograd.detect_anomaly():
    trainer.train(batch_size=batch_size, epochs=10, lr=lr, continue_training_at_step=0, num_workers=4)

#[3774, 2889, 7195, 6719, 8825, 6582, 2423, 8199, 1382, 5508, 3253, 7898, 4941, 520, 7968, 3920, 7159, 3794, 2341, 2662, 2915, 1962, 7767, 4618, 1459, 7975, 3257, 5740, 3781, 6502, 5346, 4535, 4401, 3514, 1407, 4974, 8161, 886, 6131, 6692, 7948, 6252, 8127, 289, 2328, 4166, 491, 2179, 1143, 3160, 3549, 6923, 5636, 4634, 7196, 6215, 2674, 7850, 1814, 8716, 1828, 1976, 2831, 5498, 501, 7437, 5177, 3486, 7022, 319, 3161, 5570, 1957, 2756, 4150, 6992, 2711, 1669, 4959, 4421, 7229, 7048, 2223, 8010, 6195, 7025, 1510, 3415, 2964, 6355, 4706, 8050, 3155, 714, 5167, 2367, 8052, 2394, 7277, 5300]
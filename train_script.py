from audio_dataset import *
from audio_model import *
from contrastive_estimation_training import *

dev = 'cpu'

name = "model_2019-01-02_run_0"

encoding_size = 128
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

batch_size = 16
lr = 0.0001

pc_model.to(dev)
print("number of parameters:", pc_model.parameter_count())
print("receptive field:", encoder.receptive_field)

visible_steps = 64
prediction_steps = pc_model.prediction_steps
item_length = encoder.receptive_field + (visible_steps + prediction_steps - 1) * encoder.downsampling_factor
visible_length = encoder.receptive_field + (visible_steps - 1) * encoder.downsampling_factor
prediction_length = encoder.receptive_field + (prediction_steps - 1) * encoder.downsampling_factor

print("item length:", item_length)

dataset = AudioDataset('/Users/vincentherrmann/Documents/Projekte/Immersions/MelodicProgressiveHouse_Tracks_test',
                       item_length=item_length)
print("dataset length:", len(dataset))

trainer = ContrastiveEstimationTrainer(model=pc_model,
                                       dataset=dataset,
                                       visible_length=visible_length,
                                       prediction_length=prediction_length,
                                       device=dev)

trainer.train(batch_size=batch_size, epochs=10, lr=lr, continue_training_at_step=0, num_workers=4)
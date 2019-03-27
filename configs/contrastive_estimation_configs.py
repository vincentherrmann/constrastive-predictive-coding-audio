from torch.optim import *

contrastive_estimation_default_dict = {
    'regularization': 0.01,
    'prediction_noise': 0.,
    'optimizer': Adam,
    'file_batch_size': 1,
    'score_over_all_timesteps': True,
    'log_interval': 20,
    'validation_interval': 1000,
    'snapshot_interval': 5000,
    'train_batch_size': 64,
    'validate_batch_size': 64,
    'max_validation_steps': 300,
    'learning_rate': 1e-4,
    'max_epochs': 100,
    'visible_steps': 60,
    'prediction_steps': 16
}
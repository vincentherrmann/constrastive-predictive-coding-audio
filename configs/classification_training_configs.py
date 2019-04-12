from torch.optim import *

classification_training_default_dict = {
    'optimizer': Adam,
    'log_interval': 20,
    'validation_interval': 1000,
    'snapshot_interval': 5000,
    'train_batch_size': 64,
    'validate_batch_size': 64,
    'max_validation_steps': 200,
    'learning_rate': 1e-4,
    'max_epochs': 100,
    'validation_split': 0.1,
    'unique_length': 16000
}
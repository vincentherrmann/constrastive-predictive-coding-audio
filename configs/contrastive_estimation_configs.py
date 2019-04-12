from torch.optim import *
from contrastive_estimation_training import softplus_score_function, difference_score_function, linear_score_function

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
    'prediction_steps': 16,
    'score_function': softplus_score_function,
    'wasserstein_gradient_penalty': False,
    'gradient_penalty_factor': 10.
}

contrastive_estimation_difference_scoring = contrastive_estimation_default_dict.copy()
contrastive_estimation_difference_scoring['score_function'] = difference_score_function

contrastive_estimation_linear_scoring = contrastive_estimation_default_dict.copy()
contrastive_estimation_linear_scoring['score_function'] = linear_score_function

contrastive_estimation_wasserstein = contrastive_estimation_linear_scoring.copy()
contrastive_estimation_wasserstein['wasserstein_gradient_penalty'] = True
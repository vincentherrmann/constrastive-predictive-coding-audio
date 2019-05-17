from configs import autoregressive_model_configs, \
    contrastive_estimation_configs, \
    cqt_configs, \
    dataset_configs, \
    scalogram_resnet_configs, \
    snapshot_configs, \
    classification_training_configs

import copy

experiment_default_dict = {
    'cqt_config': cqt_configs.cqt_default_dict,
    'encoder_config': scalogram_resnet_configs.scalogram_resnet_architecture_1,
    'ar_model_config': autoregressive_model_configs.ar_conv_default_dict,
    'dataset_config': dataset_configs.melodic_progressive_house_default_dict,
    'snapshot_config': snapshot_configs.snapshot_manager_default_dict,
    'training_config': contrastive_estimation_configs.contrastive_estimation_default_dict
}

classification_default_dict = {
    'cqt_config': cqt_configs.cqt_default_dict,
    'model_config': scalogram_resnet_configs.scalogram_resnet_classification_1,
    'dataset_config': dataset_configs.melodic_progressive_house_single_files,
    'snapshot_config': snapshot_configs.snapshot_manager_default_dict,
    'training_config': classification_training_configs.classification_training_default_dict
}

experiments = {'default': experiment_default_dict}

experiments['e0'] = experiment_default_dict
experiments['e0']['training_config']['learning_rate'] = 1e-3

experiments['e0_local'] = copy.deepcopy(experiments['e0'])
experiments['e0_local']['dataset_config'] = dataset_configs.melodic_progressive_house_local_test

experiments['e1'] = copy.deepcopy(experiments['e0'])
experiments['dataset_config'] = dataset_configs.melodic_progressive_house_single_files

experiments['e2'] = copy.deepcopy(experiments['e1'])
experiments['e2']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_2

experiments['e3'] = copy.deepcopy(experiments['e1'])
experiments['e3']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_2_wo_res

experiments['e4'] = copy.deepcopy(experiments['e1'])
experiments['e4']['training_config'] = contrastive_estimation_configs.contrastive_estimation_difference_scoring

experiments['e5'] = copy.deepcopy(experiments['e4'])
experiments['e5']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_3

experiments['e6'] = copy.deepcopy(experiments['e1'])
experiments['e6']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_3
experiments['e6']['training_config']['regularization'] = 0.

experiments['e7'] = copy.deepcopy(experiments['e1'])
experiments['e7']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_4
experiments['e7']['ar_model_config'] = autoregressive_model_configs.ar_conv_architecture_1

experiments['e8'] = copy.deepcopy(experiments['e1'])
experiments['e8']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_4
experiments['e8']['training_config']['regularization'] = 0.

experiments['e9'] = copy.deepcopy(experiments['e1'])
experiments['e9']['training_config'] = contrastive_estimation_configs.contrastive_estimation_linear_scoring

experiments['e10'] = copy.deepcopy(experiments['e9'])
experiments['e10']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_4
experiments['e10']['ar_model_config'] = autoregressive_model_configs.ar_conv_architecture_1

experiments['e11'] = copy.deepcopy(experiments['e10'])
experiments['e11']['training_config'] = contrastive_estimation_configs.contrastive_estimation_wasserstein
experiments['e11']['training_config']['regularization'] = 0.

experiments['e12'] = copy.deepcopy(experiments['e11'])
experiments['e12']['training_config']['gradient_penalty_factor'] = 1.

experiments['e13'] = copy.deepcopy(experiments['e1'])
experiments['e13']['training_config'] = contrastive_estimation_configs.contrastive_estimation_linear_scoring
experiments['e13']['training_config']['regularization'] = 0.
experiments['e13']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_6

experiments['e14'] = copy.deepcopy(experiments['e13'])
experiments['e14']['training_config'] = contrastive_estimation_configs.contrastive_estimation_wasserstein
experiments['e14']['training_config']['regularization'] = 0.

experiments['e15'] = copy.deepcopy(experiments['e14'])
experiments['e15']['training_config']['train_batch_size'] = 32

experiments['e16'] = copy.deepcopy(experiments['e15'])
experiments['e16']['ar_model_config'] = autoregressive_model_configs.ar_conv_architecture_2

experiments['e17'] = copy.deepcopy(experiments['e16'])
#experiments['e17']['training_config']['sum_over_all_timesteps'] = False
experiments['e17']['training_config']['file_batch_size'] = 8

experiments['e18'] = copy.deepcopy(experiments['e17'])
experiments['e18']['ar_model_config'] = autoregressive_model_configs.ar_resnet_architecture_1

experiments['e19'] = copy.deepcopy(experiments['e18'])
experiments['e19']['training_config']['gradient_penalty_factor'] = 10.

experiments['e20'] = copy.deepcopy(experiments['e19'])
experiments['e20']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_7
experiments['e20']['ar_model_config'] = autoregressive_model_configs.attention_architecture_1

experiments['e21'] = copy.deepcopy(experiments['e18'])
experiments['e21']['ar_model_config'] = autoregressive_model_configs.ar_resnet_architecture_2

experiments['e22'] = copy.deepcopy(experiments['e18'])
experiments['e22']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_7
experiments['e22']['ar_model_config'] = autoregressive_model_configs.ar_conv_architecture_3

experiments['e23'] = copy.deepcopy(experiments['e22'])
experiments['e23']['training_config']['wasserstein_gradient_penalty'] = False

experiments['e24'] = copy.deepcopy(experiments['e23'])
experiments['e24']['training_config']['train_batch_size'] = 64

experiments['e25'] = copy.deepcopy(experiments['e23'])
experiments['e25']['training_config']['score_over_all_timesteps'] = False

experiments['e26'] = copy.deepcopy(experiments['e22'])
experiments['e26']['training_config']['train_batch_size'] = 16
experiments['e26']['training_config']['gradient_penalty_factor'] = 1.


experiments['c1'] = classification_default_dict
experiments['c1']['training_config']['learning_rate'] = 1e-3

experiments['c2'] = copy.deepcopy(experiments['c1'])
experiments['c2']['training_config']['unique_length'] = 64000

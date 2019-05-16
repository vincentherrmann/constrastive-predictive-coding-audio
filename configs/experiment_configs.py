from configs import autoregressive_model_configs, \
    contrastive_estimation_configs, \
    cqt_configs, \
    dataset_configs, \
    scalogram_resnet_configs, \
    snapshot_configs, \
    classification_training_configs

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

experiments['e0'] = experiment_default_dict.copy()
experiments['e0']['training_config']['learning_rate'] = 1e-3

experiments['e0_local'] = experiments['e0'].copy()
experiments['e0_local']['dataset_config'] = dataset_configs.melodic_progressive_house_local_test

experiments['e1'] = experiments['e0'].copy()
experiments['dataset_config'] = dataset_configs.melodic_progressive_house_single_files

experiments['e2'] = experiments['e1'].copy()
experiments['e2']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_2

experiments['e3'] = experiments['e1'].copy()
experiments['e3']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_2_wo_res

experiments['e4'] = experiments['e1'].copy()
experiments['e4']['training_config'] = contrastive_estimation_configs.contrastive_estimation_difference_scoring

experiments['e5'] = experiments['e4'].copy()
experiments['e5']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_3

experiments['e6'] = experiments['e1'].copy()
experiments['e6']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_3
experiments['e6']['training_config']['regularization'] = 0.

experiments['e7'] = experiments['e1'].copy()
experiments['e7']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_4
experiments['e7']['ar_model_config'] = autoregressive_model_configs.ar_conv_architecture_1

experiments['e8'] = experiments['e1'].copy()
experiments['e8']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_4
experiments['e8']['training_config']['regularization'] = 0.

experiments['e9'] = experiments['e1'].copy()
experiments['e9']['training_config'] = contrastive_estimation_configs.contrastive_estimation_linear_scoring

experiments['e10'] = experiments['e9'].copy()
experiments['e10']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_4
experiments['e10']['ar_model_config'] = autoregressive_model_configs.ar_conv_architecture_1

experiments['e11'] = experiments['e10'].copy()
experiments['e11']['training_config'] = contrastive_estimation_configs.contrastive_estimation_wasserstein
experiments['e11']['training_config']['regularization'] = 0.

experiments['e12'] = experiments['e11'].copy()
experiments['e12']['training_config']['gradient_penalty_factor'] = 1.

experiments['e13'] = experiments['e1'].copy()
experiments['e13']['training_config'] = contrastive_estimation_configs.contrastive_estimation_linear_scoring
experiments['e13']['training_config']['regularization'] = 0.
experiments['e13']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_6

experiments['e14'] = experiments['e13'].copy()
experiments['e14']['training_config'] = contrastive_estimation_configs.contrastive_estimation_wasserstein
experiments['e14']['training_config']['regularization'] = 0.

experiments['e15'] = experiments['e14'].copy()
experiments['e15']['training_config']['train_batch_size'] = 32

experiments['e16'] = experiments['e15'].copy()
experiments['e16']['ar_model_config'] = autoregressive_model_configs.ar_conv_architecture_2

experiments['e17'] = experiments['e16'].copy()
experiments['e17']['training_config']['sum_over_all_timesteps'] = False
experiments['e17']['training_config']['file_batch_size'] = 8

experiments['e18'] = experiments['e17'].copy()
experiments['e18']['ar_model_config'] = autoregressive_model_configs.ar_resnet_architecture_1

experiments['e19'] = experiments['e18'].copy()
experiments['e19']['training_config']['gradient_penalty_factor'] = 10.

experiments['e20'] = experiments['e19'].copy()
experiments['e20']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_7
experiments['e20']['ar_model_config'] = autoregressive_model_configs.attention_architecture_1

experiments['e21'] = experiments['e18'].copy()
experiments['e21']['ar_model_config'] = autoregressive_model_configs.ar_resnet_architecture_2

experiments['e22'] = experiments['e18'].copy()
experiments['e22']['encoder_config'] = scalogram_resnet_configs.scalogram_resnet_architecture_7
experiments['e22']['ar_model_config'] = autoregressive_model_configs.ar_conv_architecture_3

experiments['e23'] = experiments['e22'].copy()
experiments['e23']['training_config']['wasserstein_gradient_penalty'] = False

experiments['e24'] = experiments['e23'].copy()
experiments['e24']['training_config']['train_batch_size'] = 64

experiments['e25'] = experiments['e23'].copy()
experiments['e25']['training_config']['score_over_all_timesteps'] = True


experiments['c1'] = classification_default_dict
experiments['c1']['training_config']['learning_rate'] = 1e-3

experiments['c2'] = experiments['c1'].copy()
experiments['c2']['training_config']['unique_length'] = 64000

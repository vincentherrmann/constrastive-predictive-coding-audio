from configs import autoregressive_model_configs, \
    contrastive_estimation_configs, \
    cqt_configs, \
    dataset_configs, \
    scalogram_resnet_configs, \
    snapshot_configs

experiment_default_dict = {
    'cqt_config': cqt_configs.cqt_default_dict,
    'encoder_config': scalogram_resnet_configs.scalogram_resnet_architecture_1,
    'ar_model_config': autoregressive_model_configs.ar_conv_default_dict,
    'dataset_config': dataset_configs.melodic_progressive_house_default_dict,
    'snapshot_config': snapshot_configs.snapshot_manager_default_dict,
    'training_config': contrastive_estimation_configs.contrastive_estimation_default_dict
}

experiments = {'default': experiment_default_dict}

experiments['e0'] = experiment_default_dict.copy()
experiments['e0']['training_config']['learning_rate'] = 1e-3

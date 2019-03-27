import torch
import argparse
import datetime
from setup_functions import *
from configs.experiment_configs import *

parser = argparse.ArgumentParser(description='Contrastive Predictive Coding Training')
parser.add_argument('--experiment', default='default', type=str)
parser.add_argument('--name', default='model_' + datetime.datetime.today().strftime('%Y-%m-%d') + '_run_0', type=str)


def main(experiment='default', name=None):
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("using device", dev)

    settings = experiments[experiment]

    if name is not None:
        settings['snapshot_config']['name'] = name

    pc_model = setup_model(cqt_params=settings['cqt_config'],
                           encoder_params=settings['encoder_config'],
                           ar_params=settings['ar_model_config'],
                           visible_steps=settings['training_config']['visible_steps'],
                           prediction_steps=settings['training_config']['prediction_steps'])

    pc_model, snapshot_manager, continue_training_at_step = setup_snapshot_manager(model=pc_model,
                                                                                   args_dict=settings['snapshot_config'],
                                                                                   try_proceeding=True)

    trainer = setup_trainer(pc_model,
                            snapshot_manager,
                            dataset_args=settings['dataset_config'],
                            trainer_args=settings['training_config'],
                            dev=dev)

    trainer.train(batch_size=settings['training_config']['training_batch_size'],
                  epochs=settings['training_config']['max_epochs'],
                  lr=settings['training_config']['learning_rate'],
                  num_workers=8,
                  continue_training_at_step=continue_training_at_step)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.experiment, args.name)
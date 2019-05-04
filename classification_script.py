import torch
import argparse
import datetime
from setup_functions import *
from configs.experiment_configs import *

#default_experiment = 'default'
default_experiment = 'c2'
run = 'run_2'

parser = argparse.ArgumentParser(description='Contrastive Predictive Coding Training')
parser.add_argument('--experiment', default=default_experiment, type=str)
parser.add_argument('--name', default='classification_model_' + datetime.datetime.today().strftime('%Y-%m-%d') + '_' + run, type=str)


def main(experiment='default', name=None):
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("using device", dev)

    settings = experiments[experiment]

    if name is not None:
        settings['snapshot_config']['name'] = name

    model, preprocessing_module = setup_classification_model(cqt_params=settings['cqt_config'],
                                                             model_params=settings['model_config'])

    model, snapshot_manager, continue_training_at_step = setup_snapshot_manager(model=model,
                                                                                args_dict=settings['snapshot_config'],
                                                                                try_proceeding=True)

    trainer = setup_classification_trainer(model,
                                           snapshot_manager,
                                           preprocessing_module=preprocessing_module,
                                           dataset_args=settings['dataset_config'],
                                           trainer_args=settings['training_config'],
                                           dev=dev)

    print("training set length:", len(trainer.dataset))
    print("validation set length:", len(trainer.validation_set))

    try:
        trainer.logger.writer.add_text('configuration', experiment, 0)
    except:
        print("unable to write expriment to tensorboard")

    trainer.train(batch_size=settings['training_config']['train_batch_size'],
                  epochs=settings['training_config']['max_epochs'],
                  lr=settings['training_config']['learning_rate'],
                  num_workers=8,
                  continue_training_at_step=continue_training_at_step)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.experiment, args.name)

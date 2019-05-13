import torch
import argparse
import datetime
import pprint
from setup_functions import *
from configs.experiment_configs import *

pp = pprint.PrettyPrinter(indent=4)

#default_experiment = 'default'
default_experiment = 'e22'
run = 'run_0'

parser = argparse.ArgumentParser(description='Contrastive Predictive Coding Training')
parser.add_argument('--experiment', default=default_experiment, type=str)
parser.add_argument('--name', default='model_' + datetime.datetime.today().strftime('%Y-%m-%d') + '_' + run, type=str)


def main(experiment='default', name=None):
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("using device", dev)

    settings = experiments[experiment]

    pp.pprint(settings)

    if name is not None:
        settings['snapshot_config']['name'] = name

    settings['training_config']['trace_model'] = False

    pc_model, preprocessing_module, untraced_model = setup_model(cqt_params=settings['cqt_config'],
                                                                 encoder_params=settings['encoder_config'],
                                                                 ar_params=settings['ar_model_config'],
                                                                 trainer_args=settings['training_config'],
                                                                 device=dev)

    pc_model, snapshot_manager, continue_training_at_step = setup_snapshot_manager(model=pc_model,
                                                                                   args_dict=settings['snapshot_config'],
                                                                                   try_proceeding=True)

    trainer = setup_ce_trainer(pc_model,
                               snapshot_manager,
                               item_length=untraced_model.item_length,
                               downsampling_factor=untraced_model.encoder.downsampling_factor,
                               ar_size=untraced_model.ar_size,
                               preprocessing_module=preprocessing_module,
                               dataset_args=settings['dataset_config'],
                               trainer_args=settings['training_config'],
                               dev=dev)

    pp.pprint(pc_model)

    try:
        trainer.logger.writer.add_text('configuration', experiment, 0)
    except:
        print("unable to write expriment to tensorboard")

    trainer.train(batch_size=settings['training_config']['train_batch_size'],
                  epochs=settings['training_config']['max_epochs'],
                  lr=settings['training_config']['learning_rate'],
                  num_workers=4,
                  continue_training_at_step=continue_training_at_step)


if __name__ == '__main__':
    args = parser.parse_args()
    #args.name = "model_2019-05-11_run_0"
    main(args.experiment, args.name)

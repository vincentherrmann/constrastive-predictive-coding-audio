from audio_dataset import *
from audio_model import *
from contrastive_estimation_training import *
from scalogram_model import *
from configs.scalogram_resnet_configs import *
from configs.cqt_configs import *
from configs.autoregressive_model_configs import *
from configs.contrastive_estimation_configs import *
from configs.dataset_configs import *
from configs.snapshot_configs import *
from torch import autograd
from ml_utilities.train_logging import *
from ml_utilities.colab_utilities import *
from ml_utilities.pytorch_utilities import *
import threading


class CPCLogger(TensorboardLogger):
    def __init__(self, *args, trainer, validation_batch_size, max_validation_steps, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_meter = AverageMeter()
        self.trainer = trainer
        self.validation_batch_size = validation_batch_size
        self.max_validation_steps = max_validation_steps

    def log_loss(self, current_step):
        self.writer.add_scalar('loss', self.loss_meter.avg, current_step)
        self.loss_meter.reset()
        self.writer.add_scalar('max_score', self.score_meter.max, current_step)
        self.score_meter.reset()

    def extended_validation_function(self):
        print("start task test")
        task_data, task_labels = self.trainer.calc_test_task_data(batch_size=self.validation_batch_size, num_workers=4)
        task_thread = threading.Thread(target=self.task_function,
                                       args=(task_data, task_labels, self.trainer.training_step),
                                       daemon=False)
        task_thread.start()

        losses, accuracies, mean_score, mmi_lb = self.trainer.validate(batch_size=self.validation_batch_size,
                                                                       num_workers=8,
                                                                       max_steps=self.max_validation_steps)

        self.writer.add_scalar("score mean", mean_score, self.trainer.training_step)
        self.writer.add_scalar("validation mutual information",
                               torch.mean(mmi_lb).item(),
                               self.trainer.training_step)
        for step in range(losses.shape[0]):
            self.writer.add_scalar("validation loss/step " + str(step),
                                    losses[step].item(),
                                    self.trainer.training_step)
            self.writer.add_scalar("validation accuracy/step " + str(step),
                                    accuracies[step].item(),
                                    self.trainer.training_step)
            self.writer.add_scalar("validation mutual information/step " + str(step),
                                    mmi_lb[step].item(),
                                    self.trainer.training_step)
        return torch.mean(losses).item(), torch.mean(accuracies).item()

    def task_function(self, task_data, task_labels, step):
        task_accuracy = self.trainer.test_task(task_data, task_labels)
        self.writer.add_scalar("task accuracy", task_accuracy, step)


def setup_model(cqt_params=cqt_default_dict,
                encoder_params=scalogram_resnet_architecture_1,
                ar_params=ar_conv_default_dict,
                visible_steps=60,
                prediction_steps=16):
    encoder = encoder_params['model'](cqt_dict=cqt_params,
                                      args_dict=encoder_params)
    ar_model = ar_params['model'](args_dict=ar_params)
    pc_model = AudioPredictiveCodingModel(encoder=encoder,
                                          autoregressive_model=ar_model,
                                          enc_size=ar_model.encoding_size,
                                          ar_size=ar_model.ar_code_size,
                                          visible_steps=visible_steps,
                                          prediction_steps=prediction_steps)
    return pc_model


def setup_snapshot_manager(model,
                           args_dict=snapshot_manager_default_dict,
                           try_proceeding=True):
    gcs_manager = GCSManager(args_dict['gcs_project'],
                             args_dict['gcs_bucket'])
    snapshot_manager = SnapshotManager(model,
                                       gcs_manager,
                                       snapshot_location=args_dict['snapshot_location'],
                                       logs_location=args_dict['logs_location'],
                                       gcs_snapshot_location=args_dict['gcs_snapshot_location'],
                                       gcs_logs_location=args_dict['gcs_logs_location'])

    continue_training_at_step = 0
    if try_proceeding:
        try:
            pc_model, newest_snapshot = snapshot_manager.load_latest_snapshot()
            continue_training_at_step = int(newest_snapshot.split('_')[-1])
            print("loaded", newest_snapshot)
            print("continue training at step", continue_training_at_step)
        except:
            print("no previous snapshot found, starting training from scratch")

    return model, snapshot_manager, continue_training_at_step


def setup_trainer(model,
                  snapshot_manager,
                  dataset_args=melodic_progressive_house_default_dict,
                  trainer_args=contrastive_estimation_default_dict,
                  dev='cpu'):
    item_length = model.encoder.receptive_field
    item_length += (model.visible_steps + model.prediction_steps) * model.encoder.downsampling_factor

    training_set = AudioDataset(dataset_args['training_set'],
                                item_length=item_length,
                                unique_length=model.encoder.downsampling_factor * dataset_args['unique_steps'])
    print("training set length:", len(training_set))

    validation_set = AudioDataset(dataset_args['validation_set'],
                                  item_length=item_length,
                                  unique_length=model.prediction_steps * model.encoder.downsampling_factor)
    print("validation set length:", len(validation_set))

    task_set = AudioTestingDataset(dataset_args['task_set'],
                                   item_length=item_length)
    print("task set length:", len(task_set))

    trainer = ContrastiveEstimationTrainer(model=model,
                                           dataset=training_set,
                                           validation_set=validation_set,
                                           test_task_set=task_set,
                                           regularization=trainer_args['regularization'],
                                           prediction_noise=trainer_args['prediction_noise'],
                                           file_batch_size=trainer_args['file_batch_size'],
                                           score_over_all_timesteps=trainer_args['score_over_all_timestamps'],
                                           device=dev)

    logger = CPCLogger(trainer=trainer,
                       log_interval=trainer_args['log_interval'],
                       validation_interval=trainer_args['validation_interval'],
                       snapshot_function=snapshot_manager.make_snapshot,
                       snapshot_interval=trainer_args['snapshot_interval'],
                       background_function=lambda _: snapshot_manager.upload_latest_files(),
                       background_interval=trainer_args['snapshot_interval'])
    logger.validation_function = logger.extended_validation_function
    trainer.logger = logger

    return trainer

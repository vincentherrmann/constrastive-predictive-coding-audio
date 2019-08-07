from audio_dataset import *
from audio_model import *
from contrastive_estimation_training import *
from classification_training import *
from scalogram_model import *
from configs.scalogram_resnet_configs import *
from configs.cqt_configs import *
from configs.autoregressive_model_configs import *
from configs.contrastive_estimation_configs import *
from configs.dataset_configs import *
from configs.snapshot_configs import *
from configs.classification_training_configs import *
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
        if self.trainer.test_task_set is not None:
            print("start task test")
            task_data, task_labels = self.trainer.calc_test_task_data(batch_size=self.validation_batch_size, num_workers=4)
            task_thread = threading.Thread(target=self.task_function,
                                           args=(task_data, task_labels, self.trainer.training_step),
                                           daemon=False)
            task_thread.start()

        losses, accuracies, mean_score, mmi_lb = self.trainer.validate(batch_size=self.validation_batch_size,
                                                                       num_workers=4,
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
                trainer_args=contrastive_estimation_default_dict,
                device=None,
                visible_steps=60,
                prediction_steps=16,
                trace_model=False,
                use_all_GPUs=True,
                activation_register=None):
    encoder_params['activation_register'] = activation_register
    ar_params['activation_register'] = activation_register
    preprocessing_module = PreprocessingModule(cqt_dict=cqt_params,
                                               phase=encoder_params['phase'],
                                               offset_zero=encoder_params['scalogram_offset_zero'],
                                               output_power=encoder_params['scalogram_output_power'],
                                               pooling=encoder_params['scalogram_pooling'],
                                               scaling=encoder_params['scalogram_scaling'])
    encoder = encoder_params['model'](args_dict=encoder_params,
                                      preprocessing_module=preprocessing_module)
    ar_model = ar_params['model'](args_dict=ar_params)
    pc_model = AudioPredictiveCodingModel(encoder=encoder,
                                          autoregressive_model=ar_model,
                                          enc_size=ar_params['encoding_size'],
                                          ar_size=ar_params['ar_code_size'],
                                          visible_steps=trainer_args['visible_steps'],
                                          prediction_steps=trainer_args['prediction_steps'],
                                          activation_register=activation_register)

    untraced_model = pc_model

    item_length = pc_model.item_length

    if trainer_args['trace_model']:
        print('trace model...')
        dummy_batch = torch.randn(2, item_length)
        dummy_batch = dummy_batch.unsqueeze(1)
        if preprocessing_module is not None:
            dummy_batch = preprocessing_module(dummy_batch)
            dummy_batch.requires_grad = True
            # batch.retain_grad()
        pc_model = torch.jit.trace(pc_model, dummy_batch)
        print('...traced')

    if torch.cuda.device_count() > 1 and trainer_args['use_all_GPUs']:
        print("using", torch.cuda.device_count(), "GPUs")
        pc_model = torch.nn.DataParallel(pc_model).cuda()
        preprocessing_module = torch.nn.DataParallel(preprocessing_module).cuda()

    return pc_model, preprocessing_module, untraced_model

    # TODO is it okay to trace after wrapping in DataParallel?

    # create traced model
    print('trace model...')
    dummy_batch = torch.randn(trainer_args['train_batch_size'], item_length)
    dummy_batch = dummy_batch.unsqueeze(1).to(device)
    if preprocessing_module is not None:
        dummy_batch = preprocessing_module(dummy_batch)
        dummy_batch.requires_grad = True
        # batch.retain_grad()
    pc_model = torch.jit.trace(pc_model, dummy_batch)
    print('...traced')
    return pc_model, preprocessing_module


def setup_snapshot_manager(model,
                           args_dict=snapshot_manager_default_dict,
                           try_proceeding=True,
                           load_to_cpu=False):
    try:
        gcs_manager = GCSManager(args_dict['gcs_project'],
                                 args_dict['gcs_bucket'])
    except:
        gcs_manager = None
        print("Unable to setup GCS manager")
    snapshot_manager = SnapshotManager(model,
                                       gcs_manager,
                                       name=args_dict['name'],
                                       snapshot_location=args_dict['snapshot_location'],
                                       logs_location=args_dict['logs_location'],
                                       gcs_snapshot_location=args_dict['gcs_snapshot_location'],
                                       gcs_logs_location=args_dict['gcs_logs_location'],
                                       load_to_cpu=load_to_cpu)

    continue_training_at_step = 0
    if try_proceeding:
        try:
            pc_model, newest_snapshot = snapshot_manager.load_latest_snapshot()
            continue_training_at_step = int(newest_snapshot.split('_')[-1])
            print("loaded", newest_snapshot)
            print("continue training at step", continue_training_at_step)
            model = pc_model
        except:
            print("no previous snapshot found, starting training from scratch")

    return model, snapshot_manager, continue_training_at_step


def setup_classification_model(cqt_params=cqt_default_dict,
                               model_params=scalogram_resnet_classification_1):
    preprocessing_module = PreprocessingModule(cqt_dict=cqt_params,
                                               phase=model_params['phase'])
    model = model_params['model'](args_dict=model_params,
                                  preprocessing_module=preprocessing_module)

    # create traced model
    dummy_batch = torch.random(1, model.item_length)
    dummy_batch = dummy_batch.to(device=model.device).unsqueeze(1)
    if preprocessing_module is not None:
        dummy_batch = preprocessing_module(dummy_batch)
        dummy_batch.requires_grad = True
        # batch.retain_grad()
    model = torch.jit.trace(model, dummy_batch)

    return model, preprocessing_module


def setup_ce_trainer(model,
                     snapshot_manager,
                     item_length,
                     downsampling_factor,
                     ar_size,
                     preprocessing_module=None,
                     dataset_args=melodic_progressive_house_default_dict,
                     trainer_args=contrastive_estimation_default_dict,
                     dev='cpu'):

    training_set = AudioDataset(dataset_args['training_set'],
                                item_length=item_length,
                                unique_length=downsampling_factor * dataset_args['unique_steps'],
                                sampling_rate=dataset_args['sample_rate'])
    print("training set length:", len(training_set))

    validation_set = AudioDataset(dataset_args['validation_set'],
                                  item_length=item_length,
                                  unique_length=trainer_args['prediction_steps'] * downsampling_factor,
                                  sampling_rate=dataset_args['sample_rate'])
    print("validation set length:", len(validation_set))

    if dataset_args['task_set'] is not None:
        task_set = AudioTestingDataset(dataset_args['task_set'],
                                   item_length=item_length)
        print("task set length:", len(task_set))
    else:
        task_set = None

    trainer = ContrastiveEstimationTrainer(model=model,
                                           dataset=training_set,
                                           validation_set=validation_set,
                                           test_task_set=task_set,
                                           optimizer=trainer_args['optimizer'],
                                           regularization=trainer_args['regularization'],
                                           prediction_noise=trainer_args['prediction_noise'],
                                           file_batch_size=trainer_args['file_batch_size'],
                                           score_over_all_timesteps=trainer_args['score_over_all_timesteps'],
                                           score_function=trainer_args['score_function'],
                                           device=dev,
                                           wasserstein_gradient_penalty=trainer_args['wasserstein_gradient_penalty'],
                                           gradient_penalty_factor=trainer_args['gradient_penalty_factor'],
                                           preprocessing=preprocessing_module,
                                           prediction_steps=trainer_args['prediction_steps'],
                                           ar_size=ar_size)

    if trainer_args['wasserstein_gradient_penalty']:
        preprocessing_module.output_requires_grad = True

    logger = CPCLogger(trainer=trainer,
                       log_interval=trainer_args['log_interval'],
                       validation_interval=trainer_args['validation_interval'],
                       validation_batch_size=trainer_args['validate_batch_size'],
                       max_validation_steps=trainer_args['max_validation_steps'],
                       snapshot_function=snapshot_manager.make_snapshot,
                       snapshot_interval=trainer_args['snapshot_interval'],
                       background_function=lambda _: snapshot_manager.upload_latest_files(),
                       background_interval=trainer_args['snapshot_interval'],
                       log_directory=snapshot_manager.current_tb_location)
    logger.validation_function = logger.extended_validation_function
    trainer.logger = logger

    return trainer


def setup_classification_trainer(model,
                                 snapshot_manager,
                                 preprocessing_module=None,
                                 dataset_args=melodic_progressive_house_default_dict,
                                 trainer_args=classification_training_default_dict,
                                 dev='cpu'):
    dataset = AudioTestingDataset(dataset_args['training_set'],
                                  item_length=model.receptive_field + model.downsampling_factor,
                                  unique_length=trainer_args['unique_length'])

    trainer = ClassificationTrainer(model=model,
                                    dataset=dataset,
                                    device=dev,
                                    optimizer=trainer_args['optimizer'],
                                    preprocessing=preprocessing_module,
                                    validation_split=trainer_args['validation_split'])

    logger = TensorboardLogger(log_interval=trainer_args['log_interval'],
                               validation_interval=trainer_args['validation_interval'],
                               snapshot_function=snapshot_manager.make_snapshot,
                               snapshot_interval=trainer_args['snapshot_interval'],
                               background_function=lambda _: snapshot_manager.upload_latest_files(),
                               background_interval=trainer_args['snapshot_interval'],
                               log_directory=snapshot_manager.current_tb_location)

    logger.validation_function = lambda: trainer.validate(batch_size=trainer_args['train_batch_size'],
                                                          num_workers=8,
                                                          max_steps=trainer_args['max_validation_steps'])
    trainer.logger = logger

    return trainer

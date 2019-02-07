import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import random
from audio_model import *
from audio_dataset import *
from sklearn import svm


class ContrastiveEstimationTrainer:
    def __init__(self, model: AudioPredictiveCodingModel, dataset, visible_length, prediction_length, logger=None, device=None,
                 use_all_GPUs=True,
                 regularization=1., validation_set=None, test_task_set=None, prediction_noise=0.01,
                 optimizer=torch.optim.Adam,
                 file_batch_size=1):
        self.model = model
        self.encoder = model.encoder
        self.ar_size = model.ar_size
        self.prediction_steps = self.model.prediction_steps
        self.visible_length = visible_length
        self.prediction_length = prediction_length
        self.dataset = dataset
        self.logger = logger
        self.device = device
        if torch.cuda.device_count() > 1 and use_all_GPUs:
            print("using", torch.cuda.device_count(), "GPUs")
            self.model = torch.nn.DataParallel(model).cuda()
        self.regularization = regularization
        self.validation_set = validation_set
        self.test_task_set = test_task_set
        self.training_step = 0
        self.print_out_scores = False
        self.prediction_noise = prediction_noise
        self.optimizer = optimizer
        self.file_batch_size = file_batch_size

    def train(self,
              batch_size=32,
              epochs=10,
              lr=0.0001,
              continue_training_at_step=0,
              num_workers=1,
              max_steps=None):
        self.model.train()
        optimizer = self.optimizer(self.model.parameters(), lr=lr)
        sampler = FileBatchSampler(index_count_per_file=self.dataset.get_example_count_per_file(),
                                   batch_size=batch_size,
                                   file_batch_size=self.file_batch_size,
                                   drop_last=True)
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_sampler=sampler,
                                                 num_workers=num_workers,
                                                 pin_memory=True)
        self.training_step = continue_training_at_step

        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            for batch in iter(dataloader):
                batch = batch.to(device=self.device)
                #visible_input = batch[:, :self.visible_length].unsqueeze(1)
                #target_input = batch[:, -self.prediction_length:].unsqueeze(1)
                predictions, targets, _, _ = self.model(batch.unsqueeze(1))
                #targets = self.encoder(target_input).detach()  # TODO: should this really be detached? (Probably yes...)

                targets = targets.permute(2, 1, 0)  # step, length, batch
                predictions = predictions.permute(1, 0, 2)  # step, batch, length

                # prediction noise injection
                predictions += torch.randn_like(predictions) * self.prediction_noise

                #scores = torch.sigmoid(torch.matmul(predictions, targets)).squeeze() # step, data_batch, target_batch
                lin_scores = torch.matmul(predictions, targets).squeeze()  # step, data_batch, target_batch
                scores = F.softplus(lin_scores)
                score_sum = torch.sum(scores, dim=1)  # step, target_batch TODO: should this be detached?
                valid_scores = torch.diagonal(scores, dim1=1, dim2=2)  # step, data_batch
                loss_logits = torch.log(valid_scores / score_sum)  # step, batch

                prediction_losses = -torch.mean(loss_logits, dim=1)
                loss = torch.mean(prediction_losses)

                if torch.sum(torch.isnan(loss)).item() > 0.:
                    print("nan loss")
                    print("scores:", scores)
                    print("mean target:", torch.mean(targets).item())
                    print("mean prediction:", torch.mean(predictions).item())
                    print("mean score:", torch.mean(scores).item())
                    print("mean score sum:", torch.mean(score_sum).item())
                    print("ratio:", torch.mean(score_sum).item() / torch.mean(scores).item())
                    return
                elif self.training_step % 20 == 0 and self.print_out_scores:
                    print("mean target:", torch.mean(targets).item())
                    print("mean prediction:", torch.mean(predictions).item())
                    print("mean score:", torch.mean(scores).item())
                    print("mean score sum:", torch.mean(score_sum).item())
                    print("ratio:", torch.mean(score_sum).item() / torch.mean(scores).item())

                loss += self.regularization * torch.mean(torch.mean(lin_scores, dim=1)**2)  # regulate loss
                #loss = torch.clamp(loss, 0, 5)

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                if self.logger is not None:
                    self.logger.loss_meter.update(loss.item())
                    self.logger.score_meter.update(torch.max(scores).item())
                    self.logger.log(self.training_step)
                elif self.training_step % 1 == 0:
                    print("loss at step step " + str(self.training_step) + ":", loss.item())

                self.training_step += 1

                if max_steps is not None and self.training_step >= max_steps:
                    return

    def validate(self, batch_size=64, num_workers=1, max_steps=None):
        if self.validation_set is None:
            print("No validation set")
            return 0, 0
        self.model.eval()
        total_loss = 0

        sampler = FileBatchSampler(index_count_per_file=self.validation_set.get_example_count_per_file(),
                                   batch_size=batch_size,
                                   file_batch_size=self.file_batch_size,
                                   drop_last=True,
                                   seed=0)
        v_dataloader = torch.utils.data.DataLoader(self.validation_set,
                                                   batch_sampler=sampler,
                                                   num_workers=num_workers,
                                                   pin_memory=True)

        total_prediction_losses = torch.zeros(self.prediction_steps, requires_grad=False).to(device=self.device)
        total_accurate_predictions = torch.zeros(self.prediction_steps, requires_grad=False).to(device=self.device)
        prediction_template = torch.arange(0, batch_size, dtype=torch.long).unsqueeze(0)
        prediction_template = prediction_template.repeat(self.prediction_steps, 1)
        total_score = 0

        if max_steps is None:
            max_steps = len(v_dataloader)
        elif max_steps > len(v_dataloader):
            max_steps = len(v_dataloader)

        for step, batch in enumerate(iter(v_dataloader)):
            batch = batch.to(device=self.device)
            predictions, targets, _, _ = self.model(batch.unsqueeze(1))

            targets = targets.permute(2, 1, 0)  # step, length, batch
            predictions = predictions.permute(1, 0, 2)  # step, batch, length

            lin_scores = torch.matmul(predictions, targets).squeeze()  # step, data_batch, target_batch
            scores = F.softplus(lin_scores)
            score_sum = torch.sum(scores, dim=1)  # step, target_batch
            valid_scores = torch.diagonal(scores, dim1=1, dim2=2)  # step, data_batch
            loss_logits = torch.log(valid_scores / score_sum)  # step, batch

            # calculate prediction accuracy as the proportion of scores that are highest for the correct target
            max_score_indices = torch.argmax(scores, dim=1)
            correctly_predicted = torch.eq(prediction_template.type_as(max_score_indices), max_score_indices)
            prediction_accuracy = torch.sum(correctly_predicted, dim=1).type_as(batch) / batch_size

            prediction_losses = -torch.mean(loss_logits, dim=1)
            loss = torch.mean(prediction_losses)

            loss += self.regularization * torch.mean(torch.mean(lin_scores, dim=1) ** 2)  # regulate loss
            #loss += self.regularization * (1 - torch.mean(scores)) ** 2

            total_prediction_losses += prediction_losses.detach()
            total_accurate_predictions += prediction_accuracy.detach()
            total_score += torch.mean(scores).item()

            if step+1 >= max_steps:
                break

        total_prediction_losses /= max_steps
        total_accurate_predictions /= max_steps
        total_score /= max_steps

        self.model.train()
        return total_prediction_losses, total_accurate_predictions, total_score

    def calc_test_task_data(self, batch_size=64, num_workers=1):
        if self.test_task_set is None:
            print("No test task set")
        num_items = len(self.test_task_set)

        self.model.eval()

        # calculate data for test task
        task_data = torch.FloatTensor(num_items, self.ar_size)
        task_labels = torch.LongTensor(num_items)
        task_data.needs_grad = False
        task_labels.needs_grad = False
        t_dataloader = torch.utils.data.DataLoader(self.test_task_set,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   pin_memory=True)
        for step, (batch, labels) in enumerate(iter(t_dataloader)):
            #print("step", step)
            batch = batch.to(device=self.device)
            predictions, targets, z, c = self.model(batch.unsqueeze(1))
            #z = self.model.encoder(batch.unsqueeze(1))
            #c = self.model.autoregressive_model(z)
            task_data[step*batch_size:(step+1)*batch_size, :] = c.detach().cpu()
            task_labels[step*batch_size:(step+1)*batch_size] = labels.detach()


        task_data = task_data.detach().numpy()
        task_labels = task_labels.detach().numpy()
        self.model.train()
        return task_data, task_labels

    def test_task(self, task_data, task_labels, evaluation_ratio=0.2):
        num_items = task_data.shape[0]
        index_list = list(range(num_items))
        random.seed(0)
        random.shuffle(index_list)
        train_indices = index_list[int(num_items*evaluation_ratio):]
        eval_indices = index_list[:int(num_items*evaluation_ratio)]

        classifier = svm.SVC(kernel='rbf')
        print("fit SVM...")
        classifier.fit(task_data[train_indices], task_labels[train_indices])
        predictions = classifier.predict(task_data[eval_indices])
        correct_predictions = np.equal(predictions, task_labels[eval_indices])

        prediction_accuracy = np.sum(correct_predictions) / len(eval_indices)
        print("task accuracy:", prediction_accuracy)
        return prediction_accuracy


class DeterministicSampler(torch.utils.data.Sampler):
    """Samples elements pseudo-randomly, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self.seed = seed

    def __iter__(self):
        o = list(range(len(self.data_source)))
        random.seed(self.seed)
        random.shuffle(o)
        #print(o[:100])
        return iter(o)

    def __len__(self):
        return len(self.data_source)

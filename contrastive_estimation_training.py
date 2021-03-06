import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import random
import time
from audio_model import *
from audio_dataset import *
from sklearn import svm


def softplus_score_function(predicted_z, targets):
    lin_scores = torch.tensordot(predicted_z, targets,
                                 dims=([2], [1]))  # data_batch, data_step, target_batch, target_step
    scores = F.softplus(lin_scores)
    return scores


def linear_score_function(predicted_z, targets):
    lin_scores = torch.tensordot(predicted_z, targets,
                                 dims=([2], [1]))  # data_batch, data_step, target_batch, target_step
    return lin_scores


def difference_score_function(predicted_z, targets):
    # predicted_z: batch, step, encoding_size
    # targets: batch, encoding_size, step
    #  (data_batch,     data_step,  encoding_size, 1,               1)
    # -(1,              1,          encoding_size, target_batch,    target_step)
    diff = predicted_z.unsqueeze(3).unsqueeze(4) - targets.permute(1, 0, 2).unsqueeze(0).unsqueeze(1)
    dist = torch.sum(diff**2, dim=2)
    scores = 1 / dist
    return scores


class ContrastiveEstimationTrainer:
    def __init__(self, model, dataset, logger=None, device=None,
                 regularization=1., validation_set=None, test_task_set=None, prediction_noise=0.01,
                 optimizer=torch.optim.Adam,
                 file_batch_size=1,
                 score_over_all_timesteps=False,
                 score_function=softplus_score_function,
                 wasserstein_gradient_penalty=False,
                 gradient_penalty_factor=10.,
                 preprocessing=None,
                 ar_size=256,
                 prediction_steps=16):
        self.model = model
        self.ar_size = ar_size
        self.prediction_steps = prediction_steps
        self.dataset = dataset
        self.logger = logger
        self.device = device
        #if torch.cuda.device_count() > 1 and use_all_GPUs:
        #    print("using", torch.cuda.device_count(), "GPUs")
        #    self.model = torch.nn.DataParallel(model).cuda()
        self.regularization = regularization
        self.validation_set = validation_set
        self.test_task_set = test_task_set
        self.training_step = 0
        self.print_out_scores = False
        self.prediction_noise = prediction_noise
        self.optimizer = optimizer
        self.file_batch_size = file_batch_size
        self.score_over_all_timesteps = score_over_all_timesteps
        self.score_function = score_function
        self.wasserstein_gradient_penalty = wasserstein_gradient_penalty
        self.gradient_penalty_factor = gradient_penalty_factor
        self.preprocessing = preprocessing
        #if torch.cuda.device_count() > 1 and use_all_GPUs:
        #    self.preprocessing = torch.nn.DataParallel(preprocessing).cuda()
        print("use score function", self.score_function)

    def train(self,
              batch_size=32,
              epochs=10,
              lr=0.0001,
              continue_training_at_step=0,
              num_workers=1,
              max_steps=None,
              profile=False):
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
            with torch.autograd.set_detect_anomaly(True), torch.autograd.profiler.profile(use_cuda=True, enabled=profile) as prof:
                for batch in iter(dataloader):
                    tic = time.time()
                    batch = batch.to(device=self.device).unsqueeze(1)
                    if self.preprocessing is not None:
                        batch = self.preprocessing(batch)
                        batch.requires_grad = True
                        #batch.retain_grad()
                    predicted_z, targets, _, _ = self.model(batch)  # data_batch, data_step, target_batch, target_step

                    scores = self.score_function(predicted_z, targets)

                    if self.score_over_all_timesteps:
                        noise_scoring = torch.logsumexp(scores.view(-1, batch_size, self.prediction_steps),
                                                        dim=0) # target_batch, target_step
                        score_sum = torch.sum(scores.view(-1, batch_size, self.prediction_steps),
                                              dim=0)
                        valid_scores = torch.diagonal(scores, dim1=0, dim2=2)  # data_step, target_step, batch
                        valid_scores = torch.diagonal(valid_scores, dim1=0, dim2=1)  # batch, step
                    else:
                        scores = torch.diagonal(scores, dim1=1, dim2=3).permute([0, 2, 1]).contiguous()  # data_batch, step, target_batch
                        noise_scoring = torch.logsumexp(scores.view(-1, batch_size, self.prediction_steps),
                                                        dim=0)  # target_batch, target_step
                        valid_scores = torch.diagonal(scores, dim1=0, dim2=2).permute([1, 0])  # batch, step

                    prediction_losses = -torch.mean(valid_scores - noise_scoring, dim=1)
                    loss = torch.mean(prediction_losses)

                    if torch.sum(torch.isnan(loss)).item() > 0.:
                        print("nan loss")
                        print("scores:", scores)
                        print("mean target:", torch.mean(targets).item())
                        print("mean prediction:", torch.mean(predicted_z).item())
                        print("mean score:", torch.mean(scores).item())
                        print("mean score sum:", torch.mean(score_sum).item())
                        print("ratio:", torch.mean(score_sum).item() / torch.mean(scores).item())
                        print("returned with nan loss at step", self.training_step)
                        return
                    elif self.training_step % 20 == 0 and self.print_out_scores:
                        print("mean target:", torch.mean(targets).item())
                        print("mean prediction:", torch.mean(predicted_z).item())
                        print("mean score:", torch.mean(scores).item())
                        print("mean score sum:", torch.mean(score_sum).item())
                        print("ratio:", torch.mean(score_sum).item() / torch.mean(scores).item())

                    loss += self.regularization * torch.mean(torch.mean(scores, dim=1)**2)  # regulate loss
                    self.model.zero_grad()

                    if self.wasserstein_gradient_penalty:
                        score_sum = torch.sum(scores)
                        #self.model.zero_grad()
                        #batch.zero_grad()
                        batch_grad = torch.autograd.grad(outputs=score_sum,
                                                         inputs=batch,
                                                         create_graph=True,
                                                         retain_graph=True,
                                                         only_inputs=True)
                        #score_sum.backward(retain_graph=True)
                        gradient_penalty = ((batch_grad[0].norm(2, dim=1) - 1) ** 2).mean()
                        gradient_penalty = gradient_penalty * self.gradient_penalty_factor
                        #gradient_penalty.backward(retain_graph=True)
                    else:
                        gradient_penalty = loss * 0.

                    loss = loss + gradient_penalty
                    loss.backward()
                    optimizer.step()

                    if self.logger is not None:  # and self.training_step > 0:
                        self.logger.loss_meter.update(loss.item())
                        self.logger.score_meter.update(torch.max(scores).item())
                        self.logger.log(self.training_step)
                    elif self.training_step % 1 == 0:
                        print("loss at step step " + str(self.training_step) + ":", loss.item())

                    #print("step", self.training_step, "duration:", time.time() - tic)

                    self.training_step += 1

                    if max_steps is not None and self.training_step >= max_steps:
                        return prof

    def validate(self, batch_size=64, num_workers=1, max_steps=None):
        if self.validation_set is None:
            print("No validation set")
            return 0, 0
        self.model.eval()
        total_loss = 0

        sampler = FileBatchSampler(index_count_per_file=self.validation_set.get_example_count_per_file(),
                                   batch_size=batch_size,
                                   file_batch_size=8,
                                   drop_last=True,
                                   seed=0)
        v_dataloader = torch.utils.data.DataLoader(self.validation_set,
                                                   batch_sampler=sampler,
                                                   num_workers=num_workers,
                                                   pin_memory=False)

        total_prediction_losses = torch.zeros(self.prediction_steps, requires_grad=False).to(device=self.device)
        total_accurate_predictions = torch.zeros(self.prediction_steps, requires_grad=False).to(device=self.device)
        n = batch_size
        if self.score_over_all_timesteps:
            print("sum over time steps")
            n *= self.prediction_steps
        else:
            print("do not sum over time steps")
        prediction_template = torch.arange(0, n, dtype=torch.long).to(device=self.device)
        if self.score_over_all_timesteps:
            prediction_template = prediction_template.view(batch_size, self.prediction_steps)
        else:
            prediction_template = prediction_template.unsqueeze(1).repeat(1, self.prediction_steps)
        #prediction_template_batch = torch.arange(0, batch_size, dtype=torch.long).unsqueeze(0)
        #prediction_template_batch = prediction_template_batch.repeat(self.prediction_steps, 1).to(device=self.device)
        #prediction_template_step = torch.arange(0, self.prediction_steps, dtype=torch.long).unsqueeze(1)
        #prediction_template_step = prediction_template_step.repeat(1, batch_size).to(device=self.device)
        total_score = 0

        if max_steps is None:
            max_steps = len(v_dataloader)
        elif max_steps > len(v_dataloader):
            max_steps = len(v_dataloader)

        for step, batch in enumerate(iter(v_dataloader)):
            tic = time.time()
            batch = batch.to(device=self.device).unsqueeze(1)
            if self.preprocessing is not None:
                batch = self.preprocessing(batch)
            predicted_z, targets, _, _ = self.model(batch)  # data_batch, data_step, target_batch, target_step

            scores = self.score_function(predicted_z, targets)  # data_batch, data_step, target_batch, target_step

            if self.score_over_all_timesteps:
                noise_scoring = torch.logsumexp(scores.view(-1, batch_size, self.prediction_steps),
                                                dim=0)  # target_batch, target_step
                score_sum = torch.sum(scores.view(-1, batch_size, self.prediction_steps),
                                      dim=0)
                valid_scores = torch.diagonal(scores, dim1=0, dim2=2)  # data_step, target_step, batch
                valid_scores = torch.diagonal(valid_scores, dim1=0, dim2=1)  # batch, step
            else:
                scores = torch.diagonal(scores, dim1=1, dim2=3).permute([0, 2, 1]).contiguous()  # data_batch, step, target_batch
                noise_scoring = torch.logsumexp(scores.view(-1, batch_size, self.prediction_steps),
                                                dim=0)  # target_batch, target_step
                valid_scores = torch.diagonal(scores, dim1=0, dim2=2).permute([1, 0])  # batch, step

            prediction_losses = -torch.mean(valid_scores - noise_scoring, dim=0)
            loss = torch.mean(prediction_losses)

            # calculate prediction accuracy as the proportion of scores that are highest for the correct target
            max_score = torch.argmax(scores.view(batch_size, self.prediction_steps, -1), dim=2)  # batch, step
            correctly_predicted = torch.eq(prediction_template, max_score)
            prediction_accuracy = torch.sum(correctly_predicted, dim=0).type_as(batch) / n

            #loss += self.regularization * torch.mean(torch.mean(lin_scores, dim=1) ** 2)  # regulate loss
            #loss += self.regularization * (1 - torch.mean(scores)) ** 2

            total_prediction_losses += prediction_losses.detach()
            total_accurate_predictions += prediction_accuracy.detach()
            total_score += torch.mean(scores).item()

            #print("step", step, "duration:", time.time()-tic)

            if step+1 >= max_steps:
                break

        del v_dataloader

        total_prediction_losses /= max_steps
        total_accurate_predictions /= max_steps
        total_score /= max_steps
        mean_mutual_information_lb = math.log(n) - total_prediction_losses

        self.model.train()
        return total_prediction_losses, total_accurate_predictions, total_score, mean_mutual_information_lb

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
                                                   pin_memory=False)
        for step, (batch, labels) in enumerate(iter(t_dataloader)):
            #print("step", step)
            batch = batch.to(device=self.device).unsqueeze(1)
            if self.preprocessing is not None:
                batch = self.preprocessing(batch)
            predictions, targets, z, c = self.model(batch)
            #z = self.model.encoder(batch.unsqueeze(1))
            #c = self.model.autoregressive_model(z)
            task_data[step*batch_size:(step+1)*batch_size, :] = c.detach().cpu()
            task_labels[step*batch_size:(step+1)*batch_size] = labels.detach()

        del t_dataloader

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

        # create neural net classifier
        classifier_model = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.ar_size, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=len(self.test_task_set.files))
        )

        if torch.cuda.device_count() > 1:
            classifier_model = torch.nn.DataParallel(classifier_model).cuda()

        train_data = torch.from_numpy(task_data[train_indices]).to(self.device)
        train_labels = torch.from_numpy(task_labels[train_indices]).to(self.device)

        eval_data = torch.from_numpy(task_data[eval_indices]).to(self.device)
        eval_labels = torch.from_numpy(task_labels[eval_indices]).to(self.device)

        batch_size = 64
        optimizer = torch.optim.Adam(classifier_model.parameters(), lr=1e-3)

        for epoch in range(10):
            for step in range(0, train_labels.shape[0], batch_size):
                batch_data = train_data[step:step+batch_size]
                batch_label = train_labels[step:step+batch_size]

                model_output = classifier_model(batch_data)
                loss = F.cross_entropy(model_output, batch_label)
                classifier_model.zero_grad()
                loss.backward()
                optimizer.step()

            predictions = torch.argmax(classifier_model(eval_data), dim=1)
            correct_predictions = torch.eq(predictions, eval_labels)
            prediction_accuracy = torch.sum(correct_predictions).item() / len(eval_indices)
            print("task accuracy after epoch", epoch, ":", prediction_accuracy)

        return prediction_accuracy

        # classifier = svm.SVC(kernel='rbf')
        # print("fit SVM...")
        # classifier.fit(task_data[train_indices], task_labels[train_indices])
        # predictions = classifier.predict(task_data[eval_indices])
        # correct_predictions = np.equal(predictions, task_labels[eval_indices])
        #
        # prediction_accuracy = np.sum(correct_predictions) / len(eval_indices)
        # print("task accuracy:", prediction_accuracy)
        # return prediction_accuracy


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


def grad_mean_var(module):
    grad_dict = {}
    for p in module.named_parameters():
        if p[1].grad is not None:
            grad_dict[p[0]] = [torch.mean(p[1].grad).item(), torch.var(p[1].grad).item()]

    return grad_dict


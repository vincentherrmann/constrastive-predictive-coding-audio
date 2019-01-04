import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from audio_model import *


class ContrastiveEstimationTrainer:
    def __init__(self, model: AudioPredictiveCodingModel, dataset, visible_length, prediction_length, logger=None, device=None,
                 regularization=1., validation_set=None):
        self.model = model
        self.visible_length = visible_length
        self.prediction_length = prediction_length
        self.dataset = dataset
        self.logger = logger
        self.device = device
        self.regularization = regularization
        self.validation_set = validation_set
        self.training_step = 0
        self.print_out_scores = False

    def train(self,
              batch_size=32,
              epochs=10,
              lr=0.0001,
              continue_training_at_step=0,
              num_workers=1,
              max_steps=None):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 drop_last=True)
        self.training_step = continue_training_at_step

        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            for batch in iter(dataloader):
                batch = batch.to(device=self.device)
                visible_input = batch[:, :self.visible_length].unsqueeze(1)
                target_input = batch[:, -self.prediction_length:].unsqueeze(1)
                predictions = self.model(visible_input)  # TODO delete factor
                targets = self.model.encoder(target_input).detach()  # TODO: should this really be detached? (Probably yes...)

                targets = targets.permute(2, 1, 0)  # step, length, batch
                predictions = predictions.permute(1, 0, 2)  # step, batch, length

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

                loss += self.regularization * (torch.mean(lin_scores))**2  # regulate loss
                loss = torch.clamp(loss, 0, 5)

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                self.training_step += 1
                if self.logger is not None:
                    self.logger.log(self.training_step, loss.item())
                elif self.training_step % 1 == 0:
                    print("loss at step step " + str(self.training_step) + ":", loss.item())

                if max_steps is not None and self.training_step >= max_steps:
                    return

    def validate(self, batch_size=64, num_workers=1, max_steps=None):
        if self.validation_set is None:
            print("No validation set")
            return 0, 0
        self.model.eval()
        total_loss = 0

        v_dataloader = torch.utils.data.DataLoader(self.validation_set,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 drop_last=True)

        total_prediction_losses = torch.zeros(self.model.prediction_steps, requires_grad=False).to(device=self.device)
        total_accurate_predictions = torch.zeros(self.model.prediction_steps, requires_grad=False).to(device=self.device)
        prediction_template = torch.arange(0, batch_size, dtype=torch.long).unsqueeze(0)
        prediction_template = prediction_template.repeat(self.model.prediction_steps, 1)
        total_score = 0

        if max_steps is None:
            max_steps = len(v_dataloader)

        for step, batch in enumerate(iter(v_dataloader)):
            batch = batch.to(device=self.device)
            visible_input = batch[:, :self.visible_length].unsqueeze(1)
            target_input = batch[:, -self.prediction_length:].unsqueeze(1)
            predictions = self.model(visible_input)
            targets = self.model.encoder(target_input).detach()

            targets = targets.permute(2, 1, 0)  # step, length, batch
            predictions = predictions.permute(1, 0, 2)  # step, batch, length

            scores = torch.matmul(predictions, targets).squeeze()  # step, data_batch, target_batch
            scores = F.softplus(scores)
            score_sum = torch.sum(scores, dim=1)  # step, target_batch
            valid_scores = torch.diagonal(scores, dim1=1, dim2=2)  # step, data_batch
            loss_logits = torch.log(valid_scores / score_sum)  # step, batch

            # calculate prediction accuracy as the proportion of scores that are highest for the correct target
            max_score_indices = torch.argmax(scores, dim=1)
            correctly_predicted = torch.eq(prediction_template.type_as(max_score_indices), max_score_indices)
            prediction_accuracy = torch.sum(correctly_predicted, dim=1).type_as(visible_input) / batch_size

            prediction_losses = -torch.mean(loss_logits, dim=1)
            loss = torch.mean(prediction_losses)

            loss += self.regularization * (1 - torch.mean(scores)) ** 2  # regulate loss

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

import torch.optim
import torch.utils.data
import random
from audio_model import *
from audio_dataset import *

class ClassificationTrainer:
    def __init__(self,
                 model,
                 dataset,
                 logger=None,
                 device=None,
                 use_all_GPUs=True,
                 optimizer=torch.optim.Adam,
                 preprocessing=None,
                 validation_split=None):
        self.model = model
        self.dataset = dataset
        self.logger = logger
        self.device = device
        if torch.cuda.device_count() > 1 and use_all_GPUs:
            print("using", torch.cuda.device_count(), "GPUs")
            self.model = torch.nn.DataParallel(model).cuda()
        self.training_step = 0
        self.print_out_scores = False
        self.optimizer = optimizer
        self.preprocessing = preprocessing
        if torch.cuda.device_count() > 1 and use_all_GPUs:
            self.preprocessing = torch.nn.DataParallel(preprocessing).cuda()

        self.validation_set = None
        if validation_split is not None:
            o = list(range(len(self.dataset)))
            random.seed(0)
            random.shuffle(o)

            eval_size = int(len(dataset) * validation_split)
            eval_indices = o[:eval_size]
            train_indices = o[eval_size:]
            self.validation_set = torch.utils.data.Subset(self.dataset, eval_indices)
            self.dataset = torch.utils.data.Subset(self.dataset, train_indices)

    def train(self,
              batch_size=32,
              epochs=10,
              lr=1e-4,
              continue_training_at_step=0,
              num_workers=1,
              max_steps=None,
              profile=False):
        self.model.train()
        optimizer = self.optimizer(self.model.parameters(), lr=lr)
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 pin_memory=True)
        self.training_step = continue_training_at_step

        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            with torch.autograd.set_detect_anomaly(True), torch.autograd.profiler.profile(use_cuda=True, enabled=profile) as prof:
                for batch, labels in iter(dataloader):
                    batch = batch.to(device=self.device).unsqueeze(1)
                    labels = labels.to(device=self.device)
                    if self.preprocessing is not None:
                        batch = self.preprocessing(batch)

                    prediction = self.model(batch)[:, :, 0]
                    loss = F.cross_entropy(prediction, labels, reduce=True)

                    self.model.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if self.logger is not None:  # and self.training_step > 0:
                        self.logger.loss_meter.update(loss.item())
                        self.logger.log(self.training_step)
                    elif self.training_step % 1 == 0:
                        print("loss at step step " + str(self.training_step) + ":", loss.item())

                    self.training_step += 1

                    if max_steps is not None and self.training_step >= max_steps:
                        return prof

    def validate(self, batch_size=64, num_workers=1, max_steps=None):
        if self.validation_set is None:
            print("No validation set")
            return 0, 0

        self.model.eval()
        total_loss = 0
        total_accuracy = 0

        v_dataloader = torch.utils.data.DataLoader(self.validation_set,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=False,
                                                   pin_memory=True)

        if max_steps is None:
            max_steps = len(v_dataloader)
        elif max_steps > len(v_dataloader):
            max_steps = len(v_dataloader)

        for step, (batch, labels) in enumerate(iter(v_dataloader)):
            batch = batch.to(device=self.device).unsqueeze(1)
            labels = labels.to(device=self.device)
            if self.preprocessing is not None:
                batch = self.preprocessing(batch)

            prediction = self.model(batch)[:, :, 0]
            loss = F.cross_entropy(prediction, labels, reduce=True)

            _, predicted_label = torch.max(prediction, dim=1)
            correctly_predicted = torch.eq(predicted_label, labels)
            prediction_accuracy = torch.sum(correctly_predicted, dim=0).type_as(batch) / batch_size

            total_loss += loss.item()
            total_accuracy += prediction_accuracy.item()

            if step+1 >= max_steps:
                break

        self.model.train()
        total_loss = total_loss / max_steps
        total_accuracy = total_accuracy / max_steps
        return total_loss, total_accuracy

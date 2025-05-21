import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
import os
import sys
import datetime

writer = SummaryWriter(log_dir="runs/test")
import numpy as np


class step_by_step(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.epochs = 0
        self.train_losses = []
        self.val_losses = []
        self.train_step_fn = self._make_train_step()
        self.val_step_fn = self._make_val_step()

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_tensorboard(self, name, folder="runs"):
        suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.writer = SummaryWriter(f"{folder}/{name}_{suffix}")

    def to(self, device):
        try:
            self.device = device
            self.model.to(device)

        except:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"sending to {device} failed, sending to {self.device} instead")
            self.model.to(self.device)

    def _make_train_step(self):
        def train_fn(x, y):
            self.model.train()
            prediction = self.model(x.unsqueeze(1))
            loss = self.loss_fn(prediction, y.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()

        return train_fn

    def _make_val_step(self):
        def val_fn(x, y):
            self.model.eval()
            prediction = self.model(x.unsqueeze(1))
            loss = self.loss_fn(prediction, y.unsqueeze(1))
            return loss.item()

        return val_fn

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _mini_batch(self, validation=False):
        if validation:
            dataloader = self.val_loader
            step_fn = self.val_step_fn
        else:
            dataloader = self.train_loader
            step_fn = self.train_step_fn

        if dataloader is None:
            return None

        mini_batch_losses = []

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

    def train(self, epochs, seed):
        self.set_seed(seed)

        for epoch in range(epochs):
            self.total_epochs += 1

            train_loss = self._mini_batch(validation=False)
            self.train_losses.append(train_loss)

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            if self.writer:
                self.writer.add_scalars(
                    global_step=epoch,
                    main_tag="loss",
                    tag_scalar_dict={
                        "training": train_loss,
                        "validation": validation_loss,
                    },
                )
# the train function is incomplete and leaving in the middle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
import os
import sys
import datetime
import matplotlib.pyplot as plt

writer = SummaryWriter(log_dir="runs/test")


class model_trainer(object):
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
        torch.cuda.manual_seed(seed)

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

        # loss = np.mean(mini_batch_losses)
        loss = sum(mini_batch_losses) / len(mini_batch_losses)
        return loss

    def train(self, epochs, seed=None):
        self.set_seed(seed if seed is not None else 42)

        for epoch in range(epochs):
            self.epochs += 1

            train_loss = self._mini_batch(validation=False)
            self.train_losses.append(train_loss)

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            if self.writer:
                scalars = {"training": train_loss}

                if val_loss is not None:
                    scalars.update({"validation": val_loss})
                self.writer.add_scalars(
                    main_tag="loss", tag_scalar_dict=scalars, global_step=epoch
                )

        if self.writer:
            self.writer.flush()

    def save_checkpoint(self, filename):
        checkpoint = {
            "epoch": self.epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.epochs = checkpoint["epochs"]
        self.model.train()

    def predict(self, x):
        self.model.eval()
        prediction = self.model(x)
        self.model.train()
        return prediction

    def plot_losses(self):
        plt.figure(figsize=(10, 4))
        losses_dict = {
            "train loss": self.train_losses,
            "validation loss": self.val_losses,
        }

        for label, losses in losses_dict.items():
            plt.plot(losses, label=label)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    def add_graph(self):
        if self.train_loader and self.writer:
            x_dummy, y_dummy = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_dummy.to(self.device))

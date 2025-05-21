__all__ = ["_make_train_step", "_make_val_step", "_mini_batch", "plot_losses"]

import numpy as np
import matplotlib.pyplot as plt


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


def plot_losses(loss_dict):
    """
    Plots multiple loss curves on the same graph.

    Args:
        loss_dict (dict): Dictionary where keys are labels (str),
                          and values are lists or arrays of losses.
                          Example: {"Train Loss": [...], "Val Loss": [...]}
    """
    plt.figure(figsize=(8, 5))

    for label, losses in loss_dict.items():
        plt.plot(losses, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

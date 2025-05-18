__all__ = ["make_train_step", "make_validation_step", "mini_batch", "plot_losses"]

import numpy as np
import matplotlib.pyplot as plt


def make_train_step(model, loss_fn, optimizer):
    def train_fn(x, y):
        model.train()
        prediction = model(x.unsqueeze(1))
        loss = loss_fn(prediction, y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    return train_fn


def make_validation_step(model, loss_fn, optimizer):
    def validation_fn(x, y):
        model.eval()
        prediction = model(x.unsqueeze(1))
        loss = loss_fn(prediction, y.unsqueeze(1))
        return loss.item()

    return validation_fn


def mini_batch(device, dataloader, step_fn):
    mini_batch_losses = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

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

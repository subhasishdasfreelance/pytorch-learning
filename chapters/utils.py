__all__ = ["make_train_step", "mini_batch", "plot_loss"]

import numpy as np
import matplotlib.pyplot as plt


def make_train_step(model, loss_fn, optimizer):
    def perform_train_step(x, y):
        model.train()
        prediction = model(x.unsqueeze(1))
        loss = loss_fn(prediction, y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    return perform_train_step


def mini_batch(device, dataloader, step_fn):
    mini_batch_losses = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)

    loss = np.mean(mini_batch_losses)
    return loss


def plot_loss(losses, label="Loss curve"):
    plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

epochs = 100
losses = []

for epoch in range(epochs):
  loss = mini_batch(device=device, dataloader=train_loader, step_fn=train_fn)
  losses.append(loss)

print(model.state_dict())

plot_loss(losses, label="Training Loss")

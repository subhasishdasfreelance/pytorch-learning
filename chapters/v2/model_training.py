epochs = 30
train_losses = []
validation_losses = []

for epoch in range(epochs):
  train_loss = mini_batch(device=device, dataloader=train_loader, step_fn=train_fn)
  train_losses.append(train_loss)
  with torch.no_grad():
    validation_loss = mini_batch(device=device, dataloader=validation_loader, step_fn=validation_fn)
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)

print(model.state_dict())

plot_losses({
  "train loss": train_losses,
  "validation loss": validation_losses
})

writer = SummaryWriter(log_dir='runs/test')
dummy_x, dummy_y = next(iter(train_loader))
writer.add_graph(model, dummy_x.unsqueeze(1).to(device))

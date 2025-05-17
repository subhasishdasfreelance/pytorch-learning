epochs = 100
losses = []

for epoch in range(epochs):
  mini_batch_losses = []

  for x_batch, y_batch in train_loader:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    mini_batch_loss = train_fn(x_batch, y_batch)
    mini_batch_losses.append(mini_batch_loss)

  loss = np.mean(mini_batch_losses)
  losses.append(loss)

print(model.state_dict())

# Assuming losses is a list or 1D tensor of loss values per epoch
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

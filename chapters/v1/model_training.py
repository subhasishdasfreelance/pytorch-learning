epochs = 100
losses = []

for epoch in range(epochs):
  loss = train_fn(x_train, y_train)
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

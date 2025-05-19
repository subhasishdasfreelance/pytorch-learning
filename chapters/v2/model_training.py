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

  writer.add_scalars(global_step=epoch, main_tag="loss", tag_scalar_dict={
    "training": train_loss,
    "validation": validation_loss
  })

writer.close()

print(model.state_dict())

plot_losses({
  "train loss": train_losses,
  "validation loss": validation_losses
})

checkpoint = {
  "model_state_dict": model.state_dict(),
  "optimizer_state_dict": optimizer.state_dict(),
  "train_losses": train_losses,
  "validation_losses": validation_losses,
  "epochs": epochs
}

torch.save(checkpoint, f'model_checkpoint{version}.pth')

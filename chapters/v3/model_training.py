
trainer = model_trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
trainer.set_loaders(train_loader=train_loader, val_loader=val_loader)
trainer.set_tensorboard(name="classy")

trainer.train(epochs=100)
print(trainer.model.state_dict())
trainer.plot_losses()

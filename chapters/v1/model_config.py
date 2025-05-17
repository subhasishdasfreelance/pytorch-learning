device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
lr = 0.1

class Linear_regression_model_class(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(1,1)

  def forward(self, x):
    return self.linear(x)

model = Linear_regression_model_class().to(device)
loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=lr)
train_fn = make_train_step(model, loss_fn, optimizer)

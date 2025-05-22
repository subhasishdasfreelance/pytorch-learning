
lr = 0.1

torch.manual_seed(42)

class Linear_regression_model_class(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(1,1)

  def forward(self, x):
    return self.linear(x)

model = Linear_regression_model_class()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss(reduction="mean")

print(model.state_dict())

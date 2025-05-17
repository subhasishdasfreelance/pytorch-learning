device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

w = torch.randn(1, device=device, requires_grad=True, dtype=torch.float)
b = torch.randn(1, device=device, requires_grad=True, dtype=torch.float)

print("real params", b, w)

x = torch.linspace(0, 1, 100).to(device)
y = (w * x + b).detach()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

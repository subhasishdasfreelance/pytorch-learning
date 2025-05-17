device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

class Custom_dataset(Dataset):
    def __init__(self, features, labels):
        self.x = features
        self.y = labels

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

# w = torch.randn(1, device=device, requires_grad=True, dtype=torch.float)
# b = torch.randn(1, device=device, requires_grad=True, dtype=torch.float)
w = 0.5
b = 1.2

# print("real params", b, w)

# x = torch.linspace(0, 1, 100).to(device)
# this time not sending to gpu as we don't want to load the whole data in gpu memory
x = torch.linspace(0, 1, 100)
# y = (w * x + b).detach()
y = (w * x + b)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

train_data = Custom_dataset(x_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

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

w = 0.5
b = 1.2

x = torch.linspace(0, 1, 100)
y = (w * x + b)

dataset = Custom_dataset(x, y)

n_train = int(len(dataset) * 0.8)
n_val = len(dataset) - n_train

train_data, val_data = random_split(dataset=dataset, lengths=[n_train, n_val])

train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16, shuffle=True)

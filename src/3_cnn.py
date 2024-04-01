import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_datasets = MNIST(root='./data', train=True, download=True, transform=transform)
test_datasets = MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_datasets, batch_size=128, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=128, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1*28*28

        # 4*24*24
        self.conv1 = nn.Conv2d(1, 4, 5)
        # 4*12*12
        self.subsample1 = nn.MaxPool2d(2, 2)
        # 8*8*8
        self.conv2 = nn.Conv2d(4, 8, 5)
        # 8*4*4
        self.subsample2 = nn.MaxPool2d(2, 2)
        # 128
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.subsample1(x)
        x = self.conv2(x)
        x = self.subsample2(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

device = torch.device("cuda" if torch.cuda.is_available() else "")
print('device: ', device)
model.to(device=device)

for epoch in range(10):
    for x, y in train_loader:
        y_pred = model(x.to(device=device))
        loss = criterion(y_pred, y.to(device=device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            y_pred = model(x.to(device=device))
            _, predicated = torch.max(y_pred.data, dim=1)
            total += len(y)
            correct += (predicated == y.to(device=device)).sum().item()

        print('epoch: ', epoch, ' accuracy: ', 100 * correct / total)

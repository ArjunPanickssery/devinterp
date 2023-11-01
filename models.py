import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = t.relu(self.fc1(x))
        x = t.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)  # 16 channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 16, 3)  # 16 channels, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(11 * 11 * 16, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 11 * 11 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(name, model, loader, epochs=10, lr=0.01, max_label=9):
    model.to(DEVICE)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for data, target in tqdm(loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)[:, : max_label + 1]

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        print(f"Epoch {epoch+1}: Loss = {total_loss / num_batches}")

    t.save(
        model.state_dict(),
        f"models/mnist_{name}_model_{max_label}.pth",
    )
    print(f"{name} model saved!")


def get_accuracy(
    model, data_loader, device=t.device("cuda" if t.cuda.is_available() else "cpu")
):
    correct = 0
    total = 0

    with t.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = t.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform= transform)
testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform= transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# model ML

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train loop in pytorch fit the data in to the model
for epoche in range(10):
    running_loss = 0.0
    for inputs, activityels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, activityels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoche {epoche + 1}, Loss: {running_loss/len(trainloader)}")

correct = 0
total = 0
with torch.no_grad():
    for inputs, activityels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += activityels.size(0)
        correct += (predicted == activityels).sum().item()

print(f'Test accuracy: {100* correct / total}')
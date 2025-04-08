import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from itertools import islice

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 20
batch_size = 100
lr = 1e-3

train_data = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net(input_size, hidden_size, num_classes)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28 * 28)
        # labels = labels  # already a tensor

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d], Step [%d/%d], Loss: %.4f"
                  % (epoch + 1, num_epochs, i + 1, len(train_data) // batch_size, loss.item()))

correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, 28 * 28)
    output = net(images)
    _, predicted = torch.max(output, 1)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)

print("Accuracy of the model: %.3f %%" % ((100 * correct) / total))

images, labels = next(iter(test_loader))
for image, label in islice(zip(images, labels), 20):
    image = image.view(-1, 28 * 28)
    _, pred = torch.max(net(image), 1)
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.show()
    print(f"label={label.item()}")
    print(f"pred={pred.item()}")
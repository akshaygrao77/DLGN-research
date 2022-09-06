import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import math
import tqdm

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainset, val_set = torch.utils.data.random_split(trainset, [math.ceil(
    0.9 * len(trainset)), len(trainset) - (math.ceil(0.9 * len(trainset)))])
batch_size = 64
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

validloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


def evaluate_model(dataloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(
                device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct // total


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_g = nn.Conv2d(3, 128, 3, padding=1)
        self.conv2_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_w = nn.Conv2d(3, 128, 3, padding=1)
        self.conv2_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_w = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(128, 10)

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        x_g1 = self.conv1_g(inp)
        x_g2 = self.conv2_g(x_g1)
        x_g3 = self.conv3_g(x_g2)
        x_g4 = self.conv4_g(x_g3)

        g1 = nn.Sigmoid()(4 * x_g1)
        g2 = nn.Sigmoid()(4 * x_g2)
        g3 = nn.Sigmoid()(4 * x_g3)
        g4 = nn.Sigmoid()(4 * x_g4)

        inp_all_ones = torch.ones(inp.size(),
                                  requires_grad=True, device=device)

        x_w1 = self.conv1_w(inp_all_ones) * g1
        x_w2 = self.conv2_w(x_w1) * g2
        x_w3 = self.conv3_w(x_w2) * g3
        x_w4 = self.conv4_w(x_w3) * g4

        x_w5 = self.pool(x_w4)
        x_w5 = torch.flatten(x_w5, 1)
        x_w6 = self.fc1(x_w5)

        return x_w6


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)

for epoch in range(50):  # loop over the dataset multiple times
    print('EPOCH {}:'.format(epoch + 1))
    running_loss = 0.0
    loader = tqdm.tqdm(trainloader, desc='Train data loader')
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(
            device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    valid_acc = evaluate_model(validloader)
    print("Valid_acc: ", valid_acc)

print('Finished Training')

test_acc = evaluate_model(testloader)
print("Test_acc: ", test_acc)

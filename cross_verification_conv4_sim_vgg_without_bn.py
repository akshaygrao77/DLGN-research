import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import math
import tqdm

import time
from external_utils import format_time


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


class Net_sim_VGG_without_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_g = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2_g = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_g = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=1)

        self.conv1_w = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2_w = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_w = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_w = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(128, 10)

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        conv_outs = []
        x_g1 = self.conv1_g(inp)
        g1 = nn.Sigmoid()(4 * x_g1)
        conv_outs.append(x_g1)

        x_g2 = self.conv2_g(x_g1)
        g2 = nn.Sigmoid()(4 * x_g2)
        conv_outs.append(x_g2)

        x_g3 = self.conv3_g(x_g2)
        g3 = nn.Sigmoid()(4 * x_g3)
        conv_outs.append(x_g3)

        x_g4 = self.conv4_g(x_g3)
        g4 = nn.Sigmoid()(4 * x_g4)
        conv_outs.append(x_g4)
        self.linear_conv_outputs = conv_outs

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


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net_sim_VGG_without_BN()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=3e-4)

    best_test_acc = 0
    for epoch in range(32):  # loop over the dataset multiple times
        correct = 0
        total = 0

        running_loss = 0.0
        loader = tqdm.tqdm(trainloader, desc='Training')
        for batch_idx, data in enumerate(loader, 0):
            begin_time = time.time()
            loader.set_description(f"Epoch {epoch+1}")
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

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

            cur_time = time.time()
            step_time = cur_time - begin_time
            loader.set_postfix(train_loss=running_loss/(batch_idx+1),
                               train_acc=100.*correct/total, ratio="{}/{}".format(correct, total), stime=format_time(step_time))

        valid_acc = evaluate_model(validloader)
        print("Valid_acc: ", valid_acc)

        test_acc = evaluate_model(testloader)
        print("Test_acc: ", test_acc)
        if(test_acc > best_test_acc):
            best_test_acc = test_acc
            path = 'root/model/save/cross_verification_conv4_sim_vgg_wo_bn_norm_dir.pt'
            torch.save(
                net, path)
            print("Model saved at:",path)
            print("Current best test accuracy:",best_test_acc)

    print('Finished Training: Best saved model test acc is:', best_test_acc)

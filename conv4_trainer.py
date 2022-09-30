import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import time

from external_utils import format_time
from data_preprocessing import preprocess_dataset_get_data_loader
from structure.dlgn_conv_config_structure import DatasetConfig

from conv4_models import Plain_CONV4_Net


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


if __name__ == '__main__':
    dataset = 'mnist'
    model_arch_type = 'plain_pure_conv4_dnn'

    if(dataset == "cifar10"):
        inp_channel = 3
        print("Training over CIFAR 10")

        cifar10_config = DatasetConfig(
            'cifar10', is_normalize_data=False, valid_split_size=0.1, batch_size=128)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            cifar10_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    elif(dataset == "mnist"):
        inp_channel = 1
        print("Training over MNIST")

        mnist_config = DatasetConfig(
            'mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=128)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            mnist_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if(model_arch_type == 'plain_pure_conv4_dnn'):
        net = Plain_CONV4_Net(inp_channel)

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

        test_acc = evaluate_model(testloader)
        print("Test_acc: ", test_acc)
        if(test_acc > best_test_acc):
            best_test_acc = test_acc
            torch.save(
                net, 'root/model/save/mnist/plain_pure_conv4_dnn_dir.pt')

    print('Finished Training: Best saved model test acc is:', best_test_acc)

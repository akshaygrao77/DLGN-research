import torch
import torch.nn as nn
import torch.optim as optim
import time

from tqdm import tqdm
from external_utils import format_time

from keras.datasets import mnist
import numpy as np
import torch.backends.cudnn as cudnn
from structure.fc_models import DLGN_FC_Network


def get_data_loader(x_data, labels, bs, orig_labels=None):
    merged_data = []
    if(orig_labels is None):
        for i in range(len(x_data)):
            merged_data.append([x_data[i], labels[i]])
    else:
        for i in range(len(x_data)):
            merged_data.append([x_data[i], labels[i], orig_labels[i]])
    dataloader = torch.utils.data.DataLoader(
        merged_data, shuffle=True, batch_size=bs)
    return dataloader


def preprocess_dataset(verbose=1):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    max = np.max(X_train)
    X_train = X_train / max
    X_test = X_test / max
    if(verbose > 2):
        print("After normalizing dataset")
        print("Max value:{}".format(max))
        print("filtered_X_train size:{} filtered_y_train size:{}".format(
            X_train.shape, y_train.shape))
        print("filtered_X_test size:{} y_test size:{}".format(
            X_test.shape, y_test.shape))

    train_data_loader = get_data_loader(
        X_train, y_train, 64)
    test_data_loader = get_data_loader(
        X_test, y_test, 64)

    return train_data_loader, test_data_loader


def evaluate_model(dlgn_fc_model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    correct = 0
    total = 0
    dlgn_fc_model.train(False)
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(
                device, non_blocking=True), labels.to(device, non_blocking=True)
            # calculate outputs by running images through the network
            outputs = dlgn_fc_model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100 * correct // total


def train(dlgn_fc_model, trainloader, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(dlgn_fc_model.parameters(), lr=0.01,
    #                       nesterov=False, momentum=0.9, weight_decay=2.5e-4)
    optimizer = optim.Adam(dlgn_fc_model.parameters(),
                           lr=0.001,)
    step = 0
    for epoch in range(20):
        dlgn_fc_model.train(True)
        # print('EPOCH {}:'.format(epoch + 1))
        running_loss = 0.0
        step_per_batch = 0
        last_time = time.time()
        begin_time = last_time
        correct = 0
        total = 0
        with tqdm(trainloader, unit="batch", desc='Training') as loader:
            for batch_idx, (inputs, labels) in enumerate(loader):
                loader.set_description(f"Epoch {epoch+1}")
                cur_time = time.time()
                step_time = cur_time - last_time
                last_time = cur_time
                tot_time = cur_time - begin_time
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(
                    device, non_blocking=True), labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = dlgn_fc_model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                step += 1
                step_per_batch += 1

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # print statistics
                running_loss += loss.item()
                loader.set_postfix(step=step, train_loss=running_loss/(batch_idx+1),
                                   train_acc=100.*correct/total, ratio="{}/{}".format(correct, total), stime=format_time(step_time), ttime=format_time(tot_time))

        print(f'loss: {running_loss / step_per_batch:.3f}')

        test_acc = evaluate_model(dlgn_fc_model, testloader)
        print("Test_acc: ", test_acc)

    torch.save({
        'model_state_dict': dlgn_fc_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, 'root/model/save/mnist10_dlgn_fc_w_128_d_4.pt')
    torch.save(dlgn_fc_model, 'root/model/save/mnist10_dlgn_fc_w_128_d_4_dir.pt')
    print('Finished Training')


if __name__ == '__main__':
    trainloader, testloader = preprocess_dataset()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nodes_in_each_layer_list = [128] * 4
    dlgn_fc_model = DLGN_FC_Network(nodes_in_each_layer_list)
    dlgn_fc_model.to(device)

    print("Device count:", torch.cuda.device_count())
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_str == 'cuda':
        if(torch.cuda.device_count() > 1):
            print("Parallelizing model")
            dlgn_fc_model = torch.nn.DataParallel(dlgn_fc_model)

        cudnn.benchmark = True

    train(dlgn_fc_model, trainloader, testloader)

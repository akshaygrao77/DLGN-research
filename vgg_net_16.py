import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import math
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

import tensorflow as tf


class FromNumpyDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_x, list_of_y):
        self.list_of_y = list_of_y
        self.list_of_x = list_of_x

    def __len__(self):
        return len(self.list_of_x)

    def __getitem__(self, idx):
        x = self.list_of_x[idx]
        y = self.list_of_y[idx]

        return x, y


std = tf.reshape((0.2023, 0.1994, 0.2010), shape=(1, 1, 3))
mean = tf.reshape((0.4914, 0.4822, 0.4465), shape=(1, 1, 3))


def prep_train_data_with_TF(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
    x = tf.image.random_crop(x, (32, 32, 3))
    x = (x - mean) / std
    return x


def prep_valid_data_with_TF(x):
    x = (x - mean) / std
    return x


def preprocess_dataset_in_tensorflow(dataset, is_train=True):
    dataset_x_np = []
    dataset_y_np = []
    for (x, y) in dataset:
        x_np = x.numpy()
        if(is_train == True):
            x_preprocessed_np = prep_train_data_with_TF(
                x_np.transpose(1, 2, 0)).numpy()
        else:
            x_preprocessed_np = prep_valid_data_with_TF(
                x_np.transpose(1, 2, 0)).numpy()

        final_x_np = x_preprocessed_np.transpose(2, 0, 1)
        dataset_x_np.append(final_x_np)
        dataset_y_np.append(y)

    return dataset_x_np, dataset_y_np


def preprocess_data():
    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainset, val_set = torch.utils.data.random_split(trainset, [math.ceil(
        0.9 * len(trainset)), len(trainset) - (math.ceil(0.9 * len(trainset)))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    print("Preprocessing train set")
    preprocessed_x_train, preprocessed_y_train = preprocess_dataset_in_tensorflow(
        trainset, is_train=True)

    print("Preprocessing validation set")
    preprocessed_x_val, preprocessed_y_val = preprocess_dataset_in_tensorflow(
        val_set, is_train=False)

    print("Preprocessing test set")
    preprocessed_x_test, preprocessed_y_test = preprocess_dataset_in_tensorflow(
        testset, is_train=False)

    preprocessed_trainset = FromNumpyDataset(
        preprocessed_x_train, preprocessed_y_train)

    preprocessed_val_set = FromNumpyDataset(
        preprocessed_x_val, preprocessed_y_val)

    preprocessed_testset = FromNumpyDataset(
        preprocessed_x_test, preprocessed_y_test)

    batch_size = 128
    trainloader = torch.utils.data.DataLoader(preprocessed_trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1, pin_memory=True)

    validloader = torch.utils.data.DataLoader(preprocessed_val_set, batch_size=batch_size,
                                              shuffle=True, num_workers=1, pin_memory=True)

    testloader = torch.utils.data.DataLoader(preprocessed_testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1, pin_memory=True)

    return trainloader, validloader, testloader


def evaluate_model(net, dataloader):
    correct = 0
    total = 0
    net.train(False)
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


class DLGN_VGG_Network(nn.Module):
    def __init__(self):
        super(DLGN_VGG_Network, self).__init__()

        self.linear_conv_net = DLGN_VGG_LinearNetwork()
        print("self.linear_conv_net ", self.linear_conv_net)

        self.weight_conv_net = DLGN_VGG_WeightNetwork()

        print("self.weight_conv_net ", self.weight_conv_net)

    def forward(self, inp, verbose=2):
        linear_conv_outputs, _ = self.linear_conv_net(inp, verbose=verbose)
        self.linear_conv_outputs = linear_conv_outputs

        # for indx in range(len(linear_conv_outputs)):
        #     each_conv_out = linear_conv_outputs[indx]
        #     print("each_conv_out: {} => size {}".format(
        #         indx, each_conv_out.size()))

        self.gating_node_outputs = [None] * len(linear_conv_outputs)
        print("self.gating_node_outputs before", self.gating_node_outputs)
        for indx in range(len(linear_conv_outputs)):
            each_linear_conv_output = linear_conv_outputs[indx]
            self.gating_node_outputs[indx] = nn.Sigmoid()(
                10 * each_linear_conv_output)

        print("self.gating_node_outputs after", self.gating_node_outputs)

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # inp = torch.ones((2,5),dtype=torch.double, requires_grad=True,device=device)
        inp = torch.ones(inp.size(),
                         requires_grad=True, device=device)

        final_layer_out = self.weight_conv_net(
            inp, self.gating_node_outputs, verbose=verbose)
        # self.final_outs = final_outs

        return final_layer_out


def initialize_weights(mod_obj):
    nn.init.kaiming_normal_(mod_obj.weight, mode='fan_in')
    if mod_obj.bias is not None:
        nn.init.constant_(mod_obj.bias, 0)


class DLGN_VGG_LinearNetwork(nn.Module):
    def __init__(self):
        super(DLGN_VGG_LinearNetwork, self).__init__()
        self.conv_64_1 = nn.Conv2d(3, 64, 3, padding=1)
        initialize_weights(self.conv_64_1)
        self.conv_64_2 = nn.Conv2d(64, 64, 3, padding=1)
        initialize_weights(self.conv_64_2)
        self.bn_11 = nn.BatchNorm2d(64)
        self.bn_12 = nn.BatchNorm2d(64)

        self.conv_128_1 = nn.Conv2d(64, 128, 3, padding=1)
        initialize_weights(self.conv_128_1)
        self.conv_128_2 = nn.Conv2d(128, 128, 3, padding=1)
        initialize_weights(self.conv_128_2)
        self.bn_21 = nn.BatchNorm2d(128)
        self.bn_22 = nn.BatchNorm2d(128)

        self.conv_256_1 = nn.Conv2d(128, 256, 3, padding=1)
        initialize_weights(self.conv_256_1)
        self.conv_256_2 = nn.Conv2d(256, 256, 3, padding=1)
        initialize_weights(self.conv_256_2)
        self.conv_256_3 = nn.Conv2d(256, 256, 3, padding=1)
        initialize_weights(self.conv_256_3)
        self.bn_31 = nn.BatchNorm2d(256)
        self.bn_32 = nn.BatchNorm2d(256)
        self.bn_33 = nn.BatchNorm2d(256)

        self.conv_512_1 = nn.Conv2d(256, 512, 3, padding=1)
        initialize_weights(self.conv_512_1)
        self.conv_512_2 = nn.Conv2d(512, 512, 3, padding=1)
        initialize_weights(self.conv_512_2)
        self.conv_512_3 = nn.Conv2d(512, 512, 3, padding=1)
        initialize_weights(self.conv_512_3)
        self.conv_512_4 = nn.Conv2d(512, 512, 3, padding=1)
        initialize_weights(self.conv_512_4)
        self.conv_512_5 = nn.Conv2d(512, 512, 3, padding=1)
        initialize_weights(self.conv_512_5)
        self.conv_512_6 = nn.Conv2d(512, 512, 3, padding=1)
        initialize_weights(self.conv_512_6)

        self.bn_41 = nn.BatchNorm2d(512)
        self.bn_42 = nn.BatchNorm2d(512)
        self.bn_43 = nn.BatchNorm2d(512)
        self.bn_44 = nn.BatchNorm2d(512)
        self.bn_45 = nn.BatchNorm2d(512)
        self.bn_46 = nn.BatchNorm2d(512)

        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, inp, verbose=2):
        conv_outs = []
        # 64 blocks *********************************************

        x_g1 = self.conv_64_1(inp)
        x_g1 = self.bn_11(x_g1)
        conv_outs.append(x_g1)

        x_g2 = self.conv_64_2(x_g1)
        x_g2 = self.bn_12(x_g2)
        conv_outs.append(x_g2)
        x_g2 = self.avg_pool(x_g2)

        # ********************************************************

        # 128 block *********************************************

        x_g3 = self.conv_128_1(x_g2)
        x_g3 = self.bn_21(x_g3)
        conv_outs.append(x_g3)

        x_g4 = self.conv_128_2(x_g3)
        x_g4 = self.bn_22(x_g4)
        conv_outs.append(x_g4)
        x_g4 = self.avg_pool(x_g4)

        # **********************************************************

        # 256 blocks ***********************************************

        x_g5 = self.conv_256_1(x_g4)
        x_g5 = self.bn_31(x_g5)
        conv_outs.append(x_g5)

        x_g6 = self.conv_256_2(x_g5)
        x_g6 = self.bn_32(x_g6)
        conv_outs.append(x_g6)

        x_g7 = self.conv_256_3(x_g6)
        x_g7 = self.bn_33(x_g7)
        conv_outs.append(x_g7)

        x_g7 = self.avg_pool(x_g7)

        # **********************************************************

        # 512 blocks 1 ***************************************************

        x_g8 = self.conv_512_1(x_g7)
        x_g8 = self.bn_41(x_g8)
        conv_outs.append(x_g8)

        x_g9 = self.conv_512_2(x_g8)
        x_g9 = self.bn_42(x_g9)
        conv_outs.append(x_g9)

        x_g10 = self.conv_512_3(x_g9)
        x_g10 = self.bn_43(x_g10)
        conv_outs.append(x_g10)

        x_g10 = self.avg_pool(x_g10)

        # **********************************************************

        # 512 blocks 2 ***************************************************

        x_g11 = self.conv_512_4(x_g10)
        x_g11 = self.bn_44(x_g11)
        conv_outs.append(x_g11)

        x_g12 = self.conv_512_5(x_g11)
        x_g12 = self.bn_45(x_g12)
        conv_outs.append(x_g12)

        x_g13 = self.conv_512_6(x_g12)
        x_g13 = self.bn_46(x_g13)
        conv_outs.append(x_g13)

        # **********************************************************

        return conv_outs, x_g13


class DLGN_VGG_WeightNetwork(nn.Module):
    def __init__(self):
        super(DLGN_VGG_WeightNetwork, self).__init__()

        self.conv_64_1 = nn.Conv2d(3, 64, 3, padding=1)
        initialize_weights(self.conv_64_1)
        self.conv_64_2 = nn.Conv2d(64, 64, 3, padding=1)
        initialize_weights(self.conv_64_2)
        self.bn_11 = nn.BatchNorm2d(64)
        self.bn_12 = nn.BatchNorm2d(64)

        self.conv_128_1 = nn.Conv2d(64, 128, 3, padding=1)
        initialize_weights(self.conv_128_1)
        self.conv_128_2 = nn.Conv2d(128, 128, 3, padding=1)
        initialize_weights(self.conv_128_2)
        self.bn_21 = nn.BatchNorm2d(128)
        self.bn_22 = nn.BatchNorm2d(128)

        self.conv_256_1 = nn.Conv2d(128, 256, 3, padding=1)
        initialize_weights(self.conv_256_1)
        self.conv_256_2 = nn.Conv2d(256, 256, 3, padding=1)
        initialize_weights(self.conv_256_2)
        self.conv_256_3 = nn.Conv2d(256, 256, 3, padding=1)
        initialize_weights(self.conv_256_3)
        self.bn_31 = nn.BatchNorm2d(256)
        self.bn_32 = nn.BatchNorm2d(256)
        self.bn_33 = nn.BatchNorm2d(256)

        self.conv_512_1 = nn.Conv2d(256, 512, 3, padding=1)
        initialize_weights(self.conv_512_1)
        self.conv_512_2 = nn.Conv2d(512, 512, 3, padding=1)
        initialize_weights(self.conv_512_2)
        self.conv_512_3 = nn.Conv2d(512, 512, 3, padding=1)
        initialize_weights(self.conv_512_3)
        self.conv_512_4 = nn.Conv2d(512, 512, 3, padding=1)
        initialize_weights(self.conv_512_4)
        self.conv_512_5 = nn.Conv2d(512, 512, 3, padding=1)
        initialize_weights(self.conv_512_5)
        self.conv_512_6 = nn.Conv2d(512, 512, 3, padding=1)
        initialize_weights(self.conv_512_6)

        self.bn_41 = nn.BatchNorm2d(512)
        self.bn_42 = nn.BatchNorm2d(512)
        self.bn_43 = nn.BatchNorm2d(512)
        self.bn_44 = nn.BatchNorm2d(512)
        self.bn_45 = nn.BatchNorm2d(512)
        self.bn_46 = nn.BatchNorm2d(512)

        self.avg_pool = nn.AvgPool2d(2)

        self.adapt_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_1 = nn.Linear(512, 10)
        initialize_weights(self.fc_1)

    def forward(self, inp, gating_signals, verbose=2):
        indx = 0
        conv_outs = []
        # 64 blocks *********************************************

        x_g1 = self.conv_64_1(inp)
        x_g1 = self.bn_11(x_g1)
        x_g1 = x_g1 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g1)

        x_g2 = self.conv_64_2(x_g1)
        x_g2 = self.bn_12(x_g2)
        x_g2 = x_g2 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g2)
        x_g2 = self.avg_pool(x_g2)

        # ********************************************************

        # 128 block *********************************************

        x_g3 = self.conv_128_1(x_g2)
        x_g3 = self.bn_21(x_g3)
        x_g3 = x_g3 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g3)

        x_g4 = self.conv_128_2(x_g3)
        x_g4 = self.bn_22(x_g4)
        x_g4 = x_g4 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g4)
        x_g4 = self.avg_pool(x_g4)

        # **********************************************************

        # 256 blocks ***********************************************

        x_g5 = self.conv_256_1(x_g4)
        x_g5 = self.bn_31(x_g5)
        x_g5 = x_g5 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g5)

        x_g6 = self.conv_256_2(x_g5)
        x_g6 = self.bn_32(x_g6)
        x_g6 = x_g6 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g6)

        x_g7 = self.conv_256_3(x_g6)
        x_g7 = self.bn_33(x_g7)
        x_g7 = x_g7 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g7)

        x_g7 = self.avg_pool(x_g7)

        # **********************************************************

        # 512 blocks 1 ***************************************************

        x_g8 = self.conv_512_1(x_g7)
        x_g8 = self.bn_41(x_g8)
        x_g8 = x_g8 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g8)

        x_g9 = self.conv_512_2(x_g8)
        x_g9 = self.bn_42(x_g9)
        x_g9 = x_g9 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g9)

        x_g10 = self.conv_512_3(x_g9)
        x_g10 = self.bn_43(x_g10)
        x_g10 = x_g10 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g10)

        x_g10 = self.avg_pool(x_g10)

        # **********************************************************

        # 512 blocks 2 ***************************************************

        x_g11 = self.conv_512_4(x_g10)
        x_g11 = self.bn_44(x_g11)
        x_g11 = x_g11 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g11)

        x_g12 = self.conv_512_5(x_g11)
        x_g12 = self.bn_45(x_g12)
        x_g12 = x_g12 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g12)

        x_g13 = self.conv_512_6(x_g12)
        x_g13 = self.bn_46(x_g13)
        x_g13 = x_g13 * gating_signals[indx]
        indx += 1
        conv_outs.append(x_g13)

        # **********************************************************

        out = self.adapt_pool(x_g13)

        out = torch.flatten(out, 1)

        out = self.fc_1(out)

        return out


def custom_piecewise_lr_decay_scheduler(optimizer, n_iter):
    if n_iter > 48000:
        optimizer.param_groups[0]['lr'] = 0.001
    elif n_iter > 32000:
        optimizer.param_groups[0]['lr'] = 0.01
    elif n_iter > 400:
        optimizer.param_groups[0]['lr'] = 0.1
    else:
        optimizer.param_groups[0]['lr'] = 0.01


def evaluate_model_and_loss(model, data_loader, loss_fn):
    # We don't need gradients on to do reporting
    model.train(False)
    running_vloss = 0.0
    running_vacc = 0.
    avg_vacc = 0.

    vpredictions, vactuals = list(), list()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, vdata in enumerate(data_loader):
        vinputs, vlabels = vdata
        # print("vlabels:",vlabels)
        # print("vinputs:",vinputs)
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)

        voutputs = model(vinputs)
        # print("voutputs:",voutputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss.item()

        voutputs = voutputs.softmax(dim=1).max(1).indices
        vyhat = voutputs.cpu().clone().detach().numpy()
        vactual = vlabels.cpu().numpy()

        vactual = vactual.reshape((len(vactual), 1))
        vyhat = vyhat.reshape((len(vyhat), 1))

        # store
        vpredictions.append(vyhat)
        vactuals.append(vactual)
        # running_vacc += accuracy_score(vlabels, voutputs.numpy())

    vpredictions, vactuals = np.vstack(vpredictions), np.vstack(vactuals)
    avg_vloss = running_vloss / (i + 1)
    avg_vacc = accuracy_score(vactuals, vpredictions)
    return avg_vloss, avg_vacc


def train(net, trainloader, validloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                          nesterov=False, momentum=0.9, weight_decay=2.5e-4)
    step = 0

    # for epoch in range(20):
    while (step < 64000):
        print("step", step)
        print("optimizer", optimizer)
        net.train(True)
        # print('EPOCH {}:'.format(epoch + 1))
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
            step += 1
            custom_piecewise_lr_decay_scheduler(optimizer, step)

            # print statistics
            running_loss += loss.item()
            if step % 352 == 351:    # print every 2000 mini-batches
                print(f'[{step + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                valid_acc = evaluate_model(net, validloader)
                # evaluate_model_and_loss(model)
                print("Valid_acc: ", valid_acc)

    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, 'root/model/save/vggnet_tf_16.pt')
    torch.save(net, 'root/model/save/vggnet_tf_16_dir.pt')
    print('Finished Training')


if __name__ == '__main__':
    trainloader, validloader, testloader = preprocess_data()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = DLGN_VGG_Network()
    net.to(device)

    train(net, trainloader, validloader)

    test_acc = evaluate_model(net, testloader)
    print("Test_acc: ", test_acc)

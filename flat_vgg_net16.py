import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import math
from tqdm import tqdm
import numpy as np
import time

import torch.backends.cudnn as cudnn
from external_utils import format_time

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
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    num_workers = 2

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def evaluate_model(net, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            correct += predicted.eq(labels).sum().item()

    return 100 * correct // total


class DLGN_VGG_Network(nn.Module):
    def __init__(self):
        super(DLGN_VGG_Network, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.g_conv_64_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.initialize_weights(self.g_conv_64_1)
        self.g_conv_64_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.initialize_weights(self.g_conv_64_2)
        self.g_bn_11 = nn.BatchNorm2d(64)
        self.g_bn_12 = nn.BatchNorm2d(64)

        self.g_conv_128_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.initialize_weights(self.g_conv_128_1)
        self.g_conv_128_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.initialize_weights(self.g_conv_128_2)
        self.g_bn_21 = nn.BatchNorm2d(128)
        self.g_bn_22 = nn.BatchNorm2d(128)

        self.g_conv_256_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.initialize_weights(self.g_conv_256_1)
        self.g_conv_256_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.initialize_weights(self.g_conv_256_2)
        self.g_conv_256_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.initialize_weights(self.g_conv_256_3)
        self.g_bn_31 = nn.BatchNorm2d(256)
        self.g_bn_32 = nn.BatchNorm2d(256)
        self.g_bn_33 = nn.BatchNorm2d(256)

        self.g_conv_512_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.initialize_weights(self.g_conv_512_1)
        self.g_conv_512_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.g_conv_512_2)
        self.g_conv_512_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.g_conv_512_3)
        self.g_conv_512_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.g_conv_512_4)
        self.g_conv_512_5 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.g_conv_512_5)
        self.g_conv_512_6 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.g_conv_512_6)

        self.g_bn_41 = nn.BatchNorm2d(512)
        self.g_bn_42 = nn.BatchNorm2d(512)
        self.g_bn_43 = nn.BatchNorm2d(512)
        self.g_bn_44 = nn.BatchNorm2d(512)
        self.g_bn_45 = nn.BatchNorm2d(512)
        self.g_bn_46 = nn.BatchNorm2d(512)

        self.g_avg_pool = nn.AvgPool2d(2)

        self.w_conv_64_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.initialize_weights(self.w_conv_64_1)
        self.w_conv_64_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.initialize_weights(self.w_conv_64_2)
        self.w_bn_11 = nn.BatchNorm2d(64)
        self.w_bn_12 = nn.BatchNorm2d(64)

        self.w_conv_128_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.initialize_weights(self.w_conv_128_1)
        self.w_conv_128_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.initialize_weights(self.w_conv_128_2)
        self.w_bn_21 = nn.BatchNorm2d(128)
        self.w_bn_22 = nn.BatchNorm2d(128)

        self.w_conv_256_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.initialize_weights(self.w_conv_256_1)
        self.w_conv_256_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.initialize_weights(self.w_conv_256_2)
        self.w_conv_256_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.initialize_weights(self.w_conv_256_3)
        self.w_bn_31 = nn.BatchNorm2d(256)
        self.w_bn_32 = nn.BatchNorm2d(256)
        self.w_bn_33 = nn.BatchNorm2d(256)

        self.w_conv_512_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.initialize_weights(self.w_conv_512_1)
        self.w_conv_512_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.w_conv_512_2)
        self.w_conv_512_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.w_conv_512_3)
        self.w_conv_512_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.w_conv_512_4)
        self.w_conv_512_5 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.w_conv_512_5)
        self.w_conv_512_6 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.w_conv_512_6)

        self.w_bn_41 = nn.BatchNorm2d(512)
        self.w_bn_42 = nn.BatchNorm2d(512)
        self.w_bn_43 = nn.BatchNorm2d(512)
        self.w_bn_44 = nn.BatchNorm2d(512)
        self.w_bn_45 = nn.BatchNorm2d(512)
        self.w_bn_46 = nn.BatchNorm2d(512)

        self.w_avg_pool = nn.AvgPool2d(2)

        self.w_adapt_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.w_fc_1 = nn.Linear(512, 10)
        self.initialize_weights(self.w_fc_1)

        self.allones = torch.ones((1, 3, 32, 32),
                                  device=device)

    def initialize_weights(self, mod_obj):
        nn.init.kaiming_normal_(mod_obj.weight, mode='fan_in')
        if mod_obj.bias is not None:
            nn.init.constant_(mod_obj.bias, 0)

    def forward(self, inp, verbose=2):
        beta = 10
        # conv_g_outs = []
        # 64 blocks *********************************************

        x_g = self.g_conv_64_1(inp)

        x_g = self.g_bn_11(x_g)
        g_1 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g1)

        x_g = self.g_conv_64_2(x_g)
        x_g = self.g_bn_12(x_g)
        g_2 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g2)
        x_g = self.g_avg_pool(x_g)

        # ********************************************************

        # 128 block *********************************************

        x_g = self.g_conv_128_1(x_g)
        x_g = self.g_bn_21(x_g)
        g_3 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g3)

        x_g = self.g_conv_128_2(x_g)
        x_g = self.g_bn_22(x_g)
        g_4 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g4)
        x_g = self.g_avg_pool(x_g)

        # **********************************************************

        # 256 blocks ***********************************************

        x_g = self.g_conv_256_1(x_g)
        x_g = self.g_bn_31(x_g)
        g_5 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g5)

        x_g = self.g_conv_256_2(x_g)
        x_g = self.g_bn_32(x_g)
        g_6 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g6)

        x_g = self.g_conv_256_3(x_g)
        x_g = self.g_bn_33(x_g)
        g_7 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g7)

        x_g = self.g_avg_pool(x_g)

        # **********************************************************

        # 512 blocks 1 ***************************************************

        x_g = self.g_conv_512_1(x_g)
        x_g = self.g_bn_41(x_g)
        g_8 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g8)

        x_g = self.g_conv_512_2(x_g)
        x_g = self.g_bn_42(x_g)
        g_9 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g9)

        x_g = self.g_conv_512_3(x_g)
        x_g = self.g_bn_43(x_g)
        g_10 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g10)

        x_g = self.g_avg_pool(x_g)

        # **********************************************************

        # 512 blocks 2 ***************************************************

        x_g = self.g_conv_512_4(x_g)
        x_g = self.g_bn_44(x_g)
        g_11 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g11)

        x_g = self.g_conv_512_5(x_g)
        x_g = self.g_bn_45(x_g)
        g_12 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g12)

        x_g = self.g_conv_512_6(x_g)
        x_g = self.g_bn_46(x_g)
        g_13 = nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g13)

        # **********************************************************
        # self.linear_conv_outputs =  conv_g_outs

        # conv_outs = []
        # 64 blocks *********************************************

        x_w = self.w_conv_64_1(self.allones.cuda())
        x_w = self.w_bn_11(x_w)
        x_w = x_w * g_1
        # conv_outs.append(x_w1)

        x_w = self.w_conv_64_2(x_w)
        x_w = self.w_bn_12(x_w)
        x_w = x_w * g_2

        # conv_outs.append(x_w2)
        x_w = self.w_avg_pool(x_w)

        # ********************************************************

        # 128 block *********************************************

        x_w = self.w_conv_128_1(x_w)
        x_w = self.w_bn_21(x_w)
        x_w = x_w * g_3

        # conv_outs.append(x_w3)

        x_w = self.w_conv_128_2(x_w)
        x_w = self.w_bn_22(x_w)
        x_w = x_w * g_4

        # conv_outs.append(x_w4)
        x_w = self.w_avg_pool(x_w)

        # **********************************************************

        # 256 blocks ***********************************************

        x_w = self.w_conv_256_1(x_w)
        x_w = self.w_bn_31(x_w)
        x_w = x_w * g_5

        # conv_outs.append(x_w5)

        x_w = self.w_conv_256_2(x_w)
        x_w = self.w_bn_32(x_w)
        x_w = x_w * g_6

        # conv_outs.append(x_w6)

        x_w = self.w_conv_256_3(x_w)
        x_w = self.w_bn_33(x_w)
        x_w = x_w * g_7

        # conv_outs.append(x_w7)

        x_w = self.w_avg_pool(x_w)

        # **********************************************************

        # 512 blocks 1 ***************************************************

        x_w = self.w_conv_512_1(x_w)
        x_w = self.w_bn_41(x_w)
        x_w = x_w * g_8

        # conv_outs.append(x_w8)

        x_w = self.w_conv_512_2(x_w)
        x_w = self.w_bn_42(x_w)
        x_w = x_w * g_9

        # conv_outs.append(x_w9)

        x_w = self.w_conv_512_3(x_w)
        x_w = self.w_bn_43(x_w)
        x_w = x_w * g_10

        # conv_outs.append(x_w10)

        x_w = self.w_avg_pool(x_w)

        # **********************************************************

        # 512 blocks 2 ***************************************************

        x_w = self.w_conv_512_4(x_w)
        x_w = self.w_bn_44(x_w)
        x_w = x_w * g_11

        # conv_outs.append(x_w11)

        x_w = self.w_conv_512_5(x_w)
        x_w = self.w_bn_45(x_w)
        x_w = x_w * g_12

        # conv_outs.append(x_w12)

        x_w = self.w_conv_512_6(x_w)
        x_w = self.w_bn_46(x_w)
        x_w = x_w * g_13

        # conv_outs.append(x_w13)

        # **********************************************************

        out = self.w_adapt_pool(x_w)

        out = torch.flatten(out, 1)

        out = self.w_fc_1(out)

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


def train(net, trainloader, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                          nesterov=False, momentum=0.9, weight_decay=2.5e-4)
    step = 0
    # total_steps = 64000
    total_steps = 352
    # for epoch in range(20):
    while (step < total_steps):
        print("step", step)
        print("optimizer", optimizer)
        net.train(True)
        # print('EPOCH {}:'.format(epoch + 1))
        running_loss = 0.0
        step_per_batch = 0
        eps = 0
        last_time = time.time()
        begin_time = last_time
        with tqdm(trainloader, unit="batch", desc='Training') as loader:
            for batch_idx, (inputs, labels) in enumerate(loader):
                loader.set_description(f"Epoch {eps+1}")
                correct = 0
                total = 0
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
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                step += 1
                step_per_batch += 1
                custom_piecewise_lr_decay_scheduler(optimizer, step)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # print statistics
                running_loss += loss.item()
                loader.set_postfix(step=step, train_loss=running_loss/(batch_idx+1),
                                   train_acc=100.*correct/total, ratio="{}/{}".format(correct, total), stime=format_time(step_time), ttime=format_time(tot_time))

        print(f'loss: {running_loss / step_per_batch:.3f}')
        eps += 1

        # valid_acc = evaluate_model(net, validloader)
        # print("Valid_acc: ", valid_acc)
        test_acc = evaluate_model(net, testloader)
        print("Test_acc: ", test_acc)

    # torch.save({
    #     'model_state_dict': net.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': criterion,
    # }, 'root/model/save/vggnet_tf_16.pt')
    # torch.save(net, 'root/model/save/vggnet_tf_16_dir.pt')
    print('Finished Training')


if __name__ == '__main__':
    trainloader, testloader = preprocess_data()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = DLGN_VGG_Network()
    net.to(device)
    print("Device count:", torch.cuda.device_count())
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_str == 'cuda':
        if(torch.cuda.device_count() > 1):
            print("Parallelizing model")
            net = torch.nn.DataParallel(net)
        # net = torch.nn.parallel.DistributedDataParallel(net)

        cudnn.benchmark = True

    train(net, trainloader, testloader)

    # test_acc = evaluate_model(net, testloader)
    # print("Test_acc: ", test_acc)

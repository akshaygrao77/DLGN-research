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
from conv4_models import TorchVision_DLGN
import torchvision.datasets as datasets
import os
from conv4_models import get_model_save_path, get_model_instance_from_dataset
import wandb
from vgg_cifar10_trainer import DLGN_VGG_Network_without_BN

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
        # transforms.Resize(224),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    num_workers = 4

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def evaluate_model(net, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        if isinstance(mod_obj, nn.Conv2d):
            nn.init.kaiming_normal_(
                mod_obj.weight, mode='fan_out', nonlinearity='relu')
            if mod_obj.bias is not None:
                nn.init.constant_(mod_obj.bias, 0)
        elif isinstance(mod_obj, nn.BatchNorm2d):
            nn.init.constant_(mod_obj.weight, 1)
            nn.init.constant_(mod_obj.bias, 0)
        # elif isinstance(mod_obj, nn.Linear):
        #     nn.init.normal_(mod_obj.weight, 0, 0.01)
        #     nn.init.constant_(mod_obj.bias, 0)

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


def train(net, trainloader, testloader, total_steps, criterion, optimizer, final_model_save_path, wand_project_name=None):
    is_log_wandb = not(wand_project_name is None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    step = 0
    best_test_acc = 0
    eps = 0
    # for epoch in range(20):
    while (step < total_steps):
        print("step", step)
        print("optimizer", optimizer)
        net.train(True)
        # print('EPOCH {}:'.format(epoch + 1))
        running_loss = 0.0
        step_per_batch = 0
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
        test_acc = evaluate_model(net, testloader)
        print("Test_acc: ", test_acc)
        train_acc = 100. * correct/total
        if(is_log_wandb):
            wandb.log(
                {"cur_epoch": eps, "train_acc": train_acc, "test_acc": test_acc})
        per_epoch_model_save_path = final_model_save_path.replace(
            "_dir.pt", "")
        if not os.path.exists(per_epoch_model_save_path):
            os.makedirs(per_epoch_model_save_path)
        per_epoch_model_save_path += "/epoch_{}_dir.pt".format(eps)
        if(eps % 7 == 0):
            torch.save({
                'epoch': eps,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': criterion,
            }, per_epoch_model_save_path)
        if(test_acc >= best_test_acc):
            best_test_acc = test_acc

    torch.save({
        'epoch': eps,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': criterion,
    }, final_model_save_path)

    print('Finished Training: Best saved model test acc is:', best_test_acc)
    return best_test_acc


if __name__ == '__main__':
    # cifar10 , imagenet1000
    dataset = "cifar10"
    model_arch_type = "dlgn__vgg16__"
    # wand_project_name = "common_model_init_exps"
    wand_project_name = None

    batch_size = 128
    pretrained = False
    torchseed = 2022
    if(dataset == "cifar10"):
        trainloader, testloader = preprocess_data()
    elif(dataset == "imagenet1000"):
        data_dir = "/home/rbcdsai/ImageNet/"
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True,
            num_workers=4, pin_memory=True)

        testloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # net = DLGN_VGG_Network()
    if(model_arch_type == "dlgn__vgg16__"):
        net = DLGN_VGG_Network_without_BN(num_classes=10)
    # net = get_model_instance_from_dataset(
    #     dataset=dataset, model_arch_type=model_arch_type, num_classes=10, pretrained=pretrained)
    net.to(device)

    print("Device count:", torch.cuda.device_count())
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_str == 'cuda':
        if(torch.cuda.device_count() > 1):
            print("Parallelizing model")
            net = torch.nn.DataParallel(net).cuda()
        # net = torch.nn.parallel.DistributedDataParallel(net)

        cudnn.benchmark = True

    final_model_save_path = get_model_save_path(
        model_arch_type+"_PRET_"+str(pretrained), dataset, torchseed)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                          nesterov=False, momentum=0.9, weight_decay=2.5e-4)
    total_steps = 64000

    is_log_wandb = not(wand_project_name is None)
    if(is_log_wandb):
        wandb_group_name = "DS_"+str(dataset) + \
            "_MT_"+str(model_arch_type)+"_PRET_"+str(pretrained) + \
            "_SEED_"+str(torchseed)
        wandb_run_name = "MT_" + \
            str(model_arch_type)+"/SEED_"+str(torchseed)+"/total_steps_"+str(total_steps)+"/OPT_"+str(optimizer)+"/LOSS_TYPE_" + \
            str(criterion)+"/BS_"+str(batch_size) + \
            "/pretrained"+str(pretrained)
        wandb_run_name = wandb_run_name.replace("/", "")

        wandb_config = dict()
        wandb_config["dataset"] = dataset
        wandb_config["model_arch_type"] = model_arch_type
        wandb_config["total_steps"] = total_steps
        wandb_config["optimizer"] = optimizer
        wandb_config["criterion"] = criterion
        wandb_config["batch_size"] = batch_size
        wandb_config["pretrained"] = pretrained

        wandb.init(
            project=f"{wand_project_name}",
            name=f"{wandb_run_name}",
            group=f"{wandb_group_name}",
            config=wandb_config,
        )

    model_save_folder = final_model_save_path[0:final_model_save_path.rfind(
        "/")+1]
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    best_test_acc = train(net, trainloader, testloader, total_steps, criterion,
                          optimizer, final_model_save_path, wand_project_name)

    if(is_log_wandb):
        wandb.log({"best_test_acc": best_test_acc})
        wandb.finish()

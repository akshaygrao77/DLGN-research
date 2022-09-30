import torch
import torchvision

import math
import numpy as np

from keras.datasets import mnist
from algos.dlgn_conv_preprocess import add_channel_to_image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms


def get_data_loader(x_data, labels, bs, orig_labels=None):
    merged_data = []
    if(orig_labels is None):
        for i in range(len(x_data)):
            merged_data.append([x_data[i], labels[i]])
    else:
        for i in range(len(x_data)):
            merged_data.append([x_data[i], labels[i], orig_labels[i]])
    dataloader = torch.utils.data.DataLoader(
        merged_data, shuffle=False, batch_size=bs)
    return dataloader


def preprocess_dataset_get_data_loader(dataset_config, model_arch_type, verbose=1, dataset_folder='./Datasets/', is_split_validation=True):
    valid_data_loader = None
    if(dataset_config.name == 'cifar10'):
        if(model_arch_type == 'cifar10_vgg_dlgn_16'):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

        elif(model_arch_type == 'cifar10_conv4_dlgn_with_inbuilt_norm'):
            transform = transforms.Compose([
                transforms.ToTensor()])
        elif(model_arch_type == 'plain_pure_conv4_dnn'):
            transform = transforms.Compose([
                transforms.ToTensor()])
        elif(model_arch_type == 'random_cifar10_conv4_dlgn_with_inbuilt_norm'):
            transform = transforms.Compose([
                transforms.ToTensor()])
        elif(model_arch_type == 'random_cifar10_conv4_dlgn_with_bn_with_inbuilt_norm'):
            transform = transforms.Compose([
                transforms.ToTensor()])

        elif(model_arch_type == 'random_cifar10_vgg_dlgn_16_with_inbuilt_norm'):
            transform = transforms.Compose([
                transforms.ToTensor()])

        elif(model_arch_type == 'cifar10_conv4_dlgn_with_bn_with_inbuilt_norm'):
            transform = transforms.Compose([
                transforms.ToTensor()])
        elif(model_arch_type == 'cifar10_conv4_dlgn_with_inbuilt_norm_with_flip_crop'):
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        elif(model_arch_type == 'cifar10_conv4_dlgn_with_bn_with_inbuilt_norm_with_flip_crop'):
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

        elif(model_arch_type == 'cifar10_vgg_dlgn_16_with_inbuilt_norm'):
            transform = transforms.Compose([
                transforms.ToTensor()])

        elif(model_arch_type == 'cifar10_vgg_dlgn_16_with_inbuilt_norm_wo_bn'):
            transform = transforms.Compose([
                transforms.ToTensor()])

        elif(model_arch_type == 'cifar10_conv4_dlgn'):
            transform = transforms.Compose([
                transforms.ToTensor()])

            # transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                          (0.2023, 0.1994, 0.2010)),
            # ])

        elif(model_arch_type == 'random_conv4_dlgn'):
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        elif(model_arch_type == 'random_vggnet_dlgn'):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
        elif(model_arch_type == 'cifar10_conv4_dlgn_sim_vgg_with_bn'):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

            # transform = transforms.Compose([
            #     transforms.ToTensor()])
        elif(model_arch_type == 'cifar10_conv4_dlgn_sim_vgg_wo_bn'):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

            # transform = transforms.Compose([
            #     transforms.ToTensor()])
        elif(model_arch_type == 'random_conv4_dlgn_sim_vgg_wo_bn'):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

            # transform = transforms.Compose([
            #     transforms.ToTensor()])
        elif(model_arch_type == 'random_conv4_dlgn_sim_vgg_with_bn'):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

            # transform = transforms.Compose([
            #     transforms.ToTensor()])

        validloader = None
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        if(is_split_validation):
            trainset, val_set = torch.utils.data.random_split(trainset, [math.ceil(
                0.9 * len(trainset)), len(trainset) - (math.ceil(0.9 * len(trainset)))])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=dataset_config.batch_size,
                                                  shuffle=False, num_workers=2)
        if(is_split_validation):
            validloader = torch.utils.data.DataLoader(val_set, batch_size=dataset_config.batch_size,
                                                      shuffle=False, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=transform)

        testloader = torch.utils.data.DataLoader(testset, batch_size=dataset_config.batch_size,
                                                 shuffle=False, num_workers=2)

        return trainloader, validloader, testloader
    elif(dataset_config.name == 'mnist'):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        if(dataset_config.is_normalize_data == True):
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

        if(not("dlgn_fc" in model_arch_type)):
            X_train = add_channel_to_image(X_train)
            X_test = add_channel_to_image(X_test)
        if(is_split_validation):
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=dataset_config.valid_split_size, random_state=42)

        train_data_loader = get_data_loader(
            X_train, y_train, dataset_config.batch_size)
        if(is_split_validation):
            valid_data_loader = get_data_loader(
                X_valid, y_valid, dataset_config.batch_size)
        test_data_loader = get_data_loader(
            X_test, y_test, dataset_config.batch_size)

        return train_data_loader, valid_data_loader, test_data_loader

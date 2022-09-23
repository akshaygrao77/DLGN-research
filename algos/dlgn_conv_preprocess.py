import torch
import torchvision
import torchvision.transforms as transforms
import math
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np


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


def add_channel_to_image(X):
    out_X = []
    for each_X in X:
        out_X.append(each_X[None, :])
    return out_X


def preprocess_dataset_get_data_loader(dataset_config, verbose=1, dataset_folder='./Datasets/', is_split_validation=True):
    if(dataset_config.name == 'cifar10'):
        transform = transforms.Compose(
            [transforms.ToTensor()])
        validloader = None
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        if(is_split_validation):
            trainset, val_set = torch.utils.data.random_split(trainset, [math.ceil(
                0.9 * len(trainset)), len(trainset) - (math.ceil(0.9 * len(trainset)))])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=dataset_config.batch_size,
                                                  shuffle=True, num_workers=2)
        if(is_split_validation):
            validloader = torch.utils.data.DataLoader(val_set, batch_size=dataset_config.batch_size,
                                                      shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

        testloader = torch.utils.data.DataLoader(testset, batch_size=dataset_config.batch_size,
                                                 shuffle=False, num_workers=2)

        return trainloader, validloader, testloader
    elif(dataset_config.name == 'mnist'):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        # print("X_train[0].type", X_train.dtype)
        # print("X_train[0].shape", X_train.shape)
        # print("y_train[0].type", y_train.dtype)
        # print("y_train[0].shape", y_train.shape)

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

        X_train = add_channel_to_image(X_train)
        X_test = add_channel_to_image(X_test)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=dataset_config.valid_split_size, random_state=42)

        train_data_loader = get_data_loader(
            X_train, y_train, dataset_config.batch_size)
        valid_data_loader = get_data_loader(
            X_valid, y_valid, dataset_config.batch_size)
        test_data_loader = get_data_loader(
            X_test, y_test, dataset_config.batch_size)

        return train_data_loader, valid_data_loader, test_data_loader

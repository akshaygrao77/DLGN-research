import torch
import torchvision

import math
import numpy as np
from tqdm import tqdm

from keras.datasets import mnist, fashion_mnist
from algos.dlgn_conv_preprocess import add_channel_to_image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from structure.generic_structure import CustomSimpleDataset, CustomMergedDataset
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_id + worker_seed)
    random.seed(worker_id - worker_seed)


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


def unwrap_dataloader(data_loader):
    data_loader = tqdm(data_loader, desc='Unwrapping dataloader')
    list_of_x = []
    list_of_y = []
    for _, inp_data in enumerate(data_loader):
        x, y = inp_data
        for each_x in x:
            list_of_x.append(each_x)
        for each_y in y:
            list_of_y.append(each_y)

    return list_of_x, list_of_y


def equalize_two_lists(list1, list2):
    if(len(list1) < len(list2)):
        list2 = list2[0:len(list1)]
    elif(len(list1) > len(list2)):
        list1 = list1[0:len(list2)]
    return list1, list2


def generate_merged_dataset_from_two_loader(data_loader1, data_loader2):
    list_of_x1, list_of_y1 = unwrap_dataloader(data_loader1)
    list_of_x2, list_of_y2 = unwrap_dataloader(data_loader2)

    list_of_x1, list_of_x2 = equalize_two_lists(list_of_x1, list_of_x2)
    list_of_y1, list_of_y2 = equalize_two_lists(list_of_y1, list_of_y2)

    dataset = CustomMergedDataset(
        list_of_x1, list_of_x2, list_of_y1, list_of_y2)
    return dataset


def generate_dataset_from_loader(data_loader):
    data_loader = tqdm(data_loader, desc='Generating dataset from loader')
    list_of_x = []
    list_of_y = []
    for _, inp_data in enumerate(data_loader):
        x, y = inp_data
        for each_x in x:
            list_of_x.append(each_x)
        for each_y in y:
            list_of_y.append(each_y)

    dataset = CustomSimpleDataset(
        list_of_x, list_of_y)
    return dataset


def segregate_input_over_labels(model, data_loader, num_classes):
    print("Segregating predicted labels")
    # We don't need gradients on to do reporting
    model.train(False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_data_list_per_class = [None] * num_classes
    for i in range(num_classes):
        input_data_list_per_class[i] = []

    data_loader = tqdm(data_loader, desc='Processing loader')
    for i, inp_data in enumerate(data_loader):
        input_data, _ = inp_data

        input_data = input_data.to(device)

        outputs = model(input_data)

        outputs = outputs.softmax(dim=1).max(1).indices

        for indx in range(len(outputs)):
            each_out = outputs[indx]
            input_data_list_per_class[each_out].append(input_data[indx])

    return input_data_list_per_class


def true_segregation(data_loader, num_classes):
    input_data_list_per_class = [0] * num_classes
    for i in range(num_classes):
        input_data_list_per_class[i] = []

    data_loader = tqdm(data_loader, desc='Processing original loader')
    for i, inp_data in enumerate(data_loader):
        input_image, labels = inp_data
        for indx in range(len(labels)):
            each_label = labels[indx]
            input_data_list_per_class[each_label].append(input_image[indx])
    return input_data_list_per_class


def print_segregation_info(input_data_list_per_class):
    sum = 0
    for indx in range(len(input_data_list_per_class)):
        each_inp = input_data_list_per_class[indx]
        length = len(each_inp)
        sum += length
        print("Indx {} len:{}".format(indx, length))
    print("Sum", sum)


def segregate_classes(model, trainloader, testloader, num_classes, is_template_image_on_train, is_class_segregation_on_ground_truth):
    input_data_list_per_class = None

    if(is_template_image_on_train):
        train_repdicted_input_data_list_per_class = segregate_input_over_labels(
            model, trainloader, num_classes)

        print("train Model segregation of classes:")
        print_segregation_info(train_repdicted_input_data_list_per_class)

        train_true_input_data_list_per_class = true_segregation(
            trainloader, num_classes)

        print("trainset Ground truth segregation of classes:")
        print_segregation_info(train_true_input_data_list_per_class)
        if(is_class_segregation_on_ground_truth):
            input_data_list_per_class = train_true_input_data_list_per_class
        else:
            input_data_list_per_class = train_repdicted_input_data_list_per_class
    else:
        test_predicted_input_data_list_per_class = segregate_input_over_labels(
            model, testloader, num_classes)

        print("Model segregation of classes:")
        print_segregation_info(test_predicted_input_data_list_per_class)

        test_true_input_data_list_per_class = true_segregation(
            testloader, num_classes)

        print("Ground truth segregation of classes:")
        print_segregation_info(test_true_input_data_list_per_class)

        if(is_class_segregation_on_ground_truth):
            input_data_list_per_class = test_true_input_data_list_per_class
        else:
            input_data_list_per_class = test_predicted_input_data_list_per_class

    return input_data_list_per_class


def preprocess_dataset_get_data_loader(dataset_config, model_arch_type, verbose=1, dataset_folder='./Datasets/', is_split_validation=True):
    valid_data_loader = None
    transform = None
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
        elif(model_arch_type == 'conv4_dlgn'):
            transform = transforms.Compose([
                transforms.ToTensor()])

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
        if(transform is None):
            return None, None, None
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
    elif(dataset_config.name == 'mnist' or dataset_config.name == 'fashion_mnist'):
        if(dataset_config.name == 'mnist'):
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
        elif(dataset_config.name == 'fashion_mnist'):
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
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

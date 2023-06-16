import torch
import torchvision

import math
import numpy as np
from tqdm import tqdm

import torchvision.datasets as datasets

from keras.datasets import mnist, fashion_mnist
from algos.dlgn_conv_preprocess import add_channel_to_image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from structure.generic_structure import CustomSimpleDataset, CustomMergedDataset, CustomSimpleArrayDataset
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_id + worker_seed)
    random.seed(worker_id - worker_seed)


def get_data_loader(x_data, labels, bs, orig_labels=None, transforms=None):
    merged_data = []
    if(orig_labels is None):
        for i in range(len(x_data)):
            merged_data.append([x_data[i], labels[i]])
    else:
        for i in range(len(x_data)):
            merged_data.append([x_data[i], labels[i], orig_labels[i]])
    merged_dataset = CustomSimpleArrayDataset(
        merged_data, transform=transforms)
    dataloader = torch.utils.data.DataLoader(
        merged_dataset, shuffle=False, pin_memory=True, num_workers=4, batch_size=bs)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        if(not is_class_segregation_on_ground_truth):
            train_repdicted_input_data_list_per_class = segregate_input_over_labels(
                model, trainloader, num_classes)

            print("train Model segregation of classes:")
            print_segregation_info(train_repdicted_input_data_list_per_class)
            input_data_list_per_class = train_repdicted_input_data_list_per_class
        else:
            train_true_input_data_list_per_class = true_segregation(
                trainloader, num_classes)

            print("trainset Ground truth segregation of classes:")
            print_segregation_info(train_true_input_data_list_per_class)

            input_data_list_per_class = train_true_input_data_list_per_class

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


def filter_dataset_to_contain_certain_classes(X, Y, list_of_classes):
    list_of_classes.sort()
    value_ind_map = dict()
    for ind in range(len(list_of_classes)):
        each_c = list_of_classes[ind]
        value_ind_map[str(int(each_c))] = ind
    filtered_X = []
    filtered_y = []
    for each_X, each_y in zip(X, Y):
        if(each_y in list_of_classes):
            modified_y = value_ind_map[str(int(each_y))]
            filtered_X.append(each_X)
            filtered_y.append(modified_y)

    filtered_X = np.array(filtered_X, dtype=np.double)
    filtered_y = np.array(filtered_y)
    return filtered_X, filtered_y


def preprocess_dataset_get_data_loader(dataset_config, model_arch_type, verbose=1, dataset_folder='./Datasets/', is_split_validation=True):
    valid_data_loader = None
    if(dataset_config.name == 'cifar10'):
        trainset, val_set, testset = preprocess_dataset_get_dataset(
            dataset_config, model_arch_type, verbose=verbose, dataset_folder=dataset_folder, is_split_validation=is_split_validation)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=dataset_config.batch_size,
                                                  shuffle=False, pin_memory=True, num_workers=4)
        if(is_split_validation):
            valid_data_loader = torch.utils.data.DataLoader(val_set, batch_size=dataset_config.batch_size,
                                                            shuffle=False, pin_memory=True, num_workers=4)

        testloader = torch.utils.data.DataLoader(testset, batch_size=dataset_config.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)

        return trainloader, valid_data_loader, testloader
    elif(dataset_config.name == 'mnist' or dataset_config.name == 'fashion_mnist'):
        filtered_X_train, filtered_y_train, X_valid, y_valid, filtered_X_test, filtered_y_test = preprocess_dataset_get_dataset(
            dataset_config, model_arch_type, verbose=verbose, dataset_folder=dataset_folder, is_split_validation=is_split_validation)

        train_data_loader = get_data_loader(
            filtered_X_train, filtered_y_train, dataset_config.batch_size, transforms=dataset_config.train_transforms)
        if(is_split_validation):
            valid_data_loader = get_data_loader(
                X_valid, y_valid, dataset_config.batch_size, transforms=dataset_config.test_transforms)
        test_data_loader = get_data_loader(
            filtered_X_test, filtered_y_test, dataset_config.batch_size, transforms=dataset_config.test_transforms)

        return train_data_loader, valid_data_loader, test_data_loader
    elif(dataset_config.name == 'imagenet_1000'):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            "/home/rbcdsai/ImageNet/train",
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            "/home/rbcdsai/ImageNet/val",
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=dataset_config.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

        testloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=dataset_config.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

        return trainloader, None, testloader


def preprocess_dataset_get_dataset(dataset_config, model_arch_type, verbose=1, dataset_folder='./Datasets/', is_split_validation=True):
    transform_list = []
    test_transform_list = []
    if(dataset_config.name == 'cifar10'):
        if(model_arch_type == 'cifar10_vgg_dlgn_16'):
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]

        elif(model_arch_type == 'cifar10_conv4_dlgn_with_inbuilt_norm'):
            transform_list = [
                transforms.ToTensor()]
        elif(model_arch_type == 'plain_pure_conv4_dnn'):
            transform_list = [
                transforms.ToTensor()]
        elif(model_arch_type == 'random_cifar10_conv4_dlgn_with_inbuilt_norm'):
            transform_list = [
                transforms.ToTensor()]
        elif(model_arch_type == 'random_cifar10_conv4_dlgn_with_bn_with_inbuilt_norm'):
            transform_list = [
                transforms.ToTensor()]

        elif(model_arch_type == 'random_cifar10_vgg_dlgn_16_with_inbuilt_norm'):
            transform_list = [
                transforms.ToTensor()]

        elif(model_arch_type == 'cifar10_conv4_dlgn_with_bn_with_inbuilt_norm'):
            transform_list = [
                transforms.ToTensor()]
        elif(model_arch_type == 'cifar10_conv4_dlgn_with_inbuilt_norm_with_flip_crop'):
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        elif(model_arch_type == 'cifar10_conv4_dlgn_with_bn_with_inbuilt_norm_with_flip_crop'):
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]

        elif(model_arch_type == 'cifar10_vgg_dlgn_16_with_inbuilt_norm'):
            transform_list = [
                transforms.ToTensor()]

        elif(model_arch_type == 'cifar10_vgg_dlgn_16_with_inbuilt_norm_wo_bn'):
            transform_list = [
                transforms.ToTensor()]

        elif(model_arch_type == 'cifar10_conv4_dlgn'):
            transform_list = [
                transforms.ToTensor()]

            # transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                          (0.2023, 0.1994, 0.2010)),
            # ])
        elif(model_arch_type == 'conv4_dlgn' or model_arch_type == "conv4_deep_gated_net_n16_small" or model_arch_type == "conv4_dlgn_n16_small" or model_arch_type == "plain_pure_conv4_dnn_n16_small"):
            transform_list = [
                transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [
                    0.2023, 0.1994, 0.2010])]

        elif(model_arch_type == 'random_conv4_dlgn'):
            transform_list = [
                transforms.ToTensor()
            ]
        elif(model_arch_type == 'random_vggnet_dlgn'):
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]
        elif(model_arch_type == 'cifar10_conv4_dlgn_sim_vgg_with_bn'):
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]

            # transform = transforms.Compose([
            #     transforms.ToTensor()])
        elif(model_arch_type == 'cifar10_conv4_dlgn_sim_vgg_wo_bn'):
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]

            # transform = transforms.Compose([
            #     transforms.ToTensor()])
        elif(model_arch_type == 'random_conv4_dlgn_sim_vgg_wo_bn'):
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]

            # transform = transforms.Compose([
            #     transforms.ToTensor()])
        elif(model_arch_type == 'random_conv4_dlgn_sim_vgg_with_bn'):
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]
        elif(model_arch_type in ['dlgn__im_conv4_dlgn_pad_k_1_st1_bn_wo_bias__','dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias__','dlgn__conv4_dlgn_pad0_st1_bn__','dlgn__st1_pad0_vgg16_bn__','dlgn__vgg16_bn__', 'dlgn__pad2_vgg16_bn__', 'dlgn__st1_pad2_vgg16_bn_wo_bias__', 'dlgn__st1_pad1_vgg16_bn_wo_bias__', 'dnn__cvgg16_bn__', 'dnn__st1_pad2_vgg16_bn_wo_bias__']):
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            test_transform_list = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            # transform = transforms.Compose([
            #     transforms.ToTensor()])

        val_set = None

        if(len(transform_list) == 0):
            transform = None
        else:
            transform = transforms.Compose(transform_list)

        if(transform is None):
            return None, None, None

        if(len(test_transform_list) == 0):
            test_transform = transform
        else:
            test_transform = transforms.Compose(test_transform_list)

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        if(is_split_validation):
            trainset, val_set = torch.utils.data.random_split(trainset, [math.ceil(
                0.9 * len(trainset)), len(trainset) - (math.ceil(0.9 * len(trainset)))])

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=test_transform)

        return trainset, val_set, testset

    elif(dataset_config.name == 'mnist' or dataset_config.name == 'fashion_mnist'):
        X_valid = None
        y_valid = None
        if(dataset_config.custom_dataset_path is None):
            if(dataset_config.name == 'mnist'):
                (X_train, y_train), (X_test, y_test) = mnist.load_data()
            elif(dataset_config.name == 'fashion_mnist'):
                (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        else:
            with np.load(dataset_config.custom_dataset_path, allow_pickle=True) as f:
                X_train, y_train = f['X_train'], f['y_train']
                X_test, y_test = f['X_test'], f['y_test']

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        if(dataset_config.list_of_classes is None):
            filtered_X_train, filtered_y_train = X_train, y_train
            filtered_X_test, filtered_y_test = X_test, y_test
        else:
            filtered_X_train, filtered_y_train = filter_dataset_to_contain_certain_classes(
                X_train, y_train, dataset_config.list_of_classes)
            filtered_X_test, filtered_y_test = filter_dataset_to_contain_certain_classes(
                X_test, y_test, dataset_config.list_of_classes)
            filtered_X_train = filtered_X_train.astype(np.float32)
            filtered_X_test = filtered_X_test.astype(np.float32)

        if(verbose > 2):
            print("After filtering dataset")
            print("filtered_X_train size:{} filtered_y_train size:{}".format(
                filtered_X_train.shape, filtered_y_train.shape))
            print("filtered_X_test size:{} filtered_y_test size:{}".format(
                filtered_X_test.shape, filtered_y_test.shape))
            print("filtered_y_train[0]", filtered_y_train[0])
            print("filtered_y_train[1]", filtered_y_train[1])

        if(dataset_config.is_normalize_data == True):
            max = np.max(filtered_X_train)
            filtered_X_train = filtered_X_train / max
            filtered_X_test = filtered_X_test / max
            if(verbose > 2):
                print("After normalizing dataset")
                print("Max value:{}".format(max))
                print("filtered_X_train size:{} filtered_y_train size:{}".format(
                    filtered_X_train.shape, filtered_y_train.shape))
                print("filtered_X_test size:{} filtered_y_test size:{}".format(
                    filtered_X_test.shape, filtered_y_test.shape))

        if(not("dlgn_fc" in model_arch_type)):
            filtered_X_train = add_channel_to_image(filtered_X_train)
            filtered_X_test = add_channel_to_image(filtered_X_test)
        if(is_split_validation):
            filtered_X_train, X_valid, filtered_y_train, y_valid = train_test_split(
                filtered_X_train, filtered_y_train, test_size=dataset_config.valid_split_size, random_state=42)

        return filtered_X_train, filtered_y_train, X_valid, y_valid, filtered_X_test, filtered_y_test

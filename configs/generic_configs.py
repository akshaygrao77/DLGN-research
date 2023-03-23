from structure.dlgn_conv_config_structure import DatasetConfig
import json


def get_preprocessing_and_other_configs(dataset, valid_split_size, batch_size=128):
    ret_config = None
    if(dataset == "cifar10"):
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = len(classes)

        ret_config = DatasetConfig(
            'cifar10', is_normalize_data=False, valid_split_size=valid_split_size, batch_size=batch_size)
    elif(dataset == "imagenet_1000"):
        class_idx = json.load(open(
            "/home/akshay/dgx-code/DLGN-research/root/Datasets/imagenet_class_index.json"))
        classes = [class_idx[str(k)][1] for k in range(len(class_idx))]
        num_classes = len(classes)

        ret_config = DatasetConfig(
            'imagenet_1000', is_normalize_data=False, valid_split_size=valid_split_size, batch_size=batch_size)

    elif(dataset == "mnist"):
        classes = [str(i) for i in range(0, 10)]
        num_classes = len(classes)

        ret_config = DatasetConfig(
            'mnist', is_normalize_data=True, valid_split_size=valid_split_size, batch_size=batch_size)

    elif(dataset == "fashion_mnist"):
        classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot')
        num_classes = len(classes)

        ret_config = DatasetConfig(
            'fashion_mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size)

    return classes, num_classes, ret_config

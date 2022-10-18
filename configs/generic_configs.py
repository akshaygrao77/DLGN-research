from structure.dlgn_conv_config_structure import DatasetConfig


def get_preprocessing_and_other_configs(dataset, valid_split_size, batch_size=128):
    ret_config = None
    if(dataset == "cifar10"):
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = len(classes)

        ret_config = DatasetConfig(
            'cifar10', is_normalize_data=False, valid_split_size=valid_split_size, batch_size=batch_size)

    elif(dataset == "mnist"):
        classes = [str(i) for i in range(0, 10)]
        num_classes = len(classes)

        ret_config = DatasetConfig(
            'mnist', is_normalize_data=True, valid_split_size=valid_split_size, batch_size=batch_size)

    return classes, num_classes, ret_config

import torch
import torchvision
import torchvision.transforms as transforms
import math


def preprocess_dataset_get_data_loader(dataset_config, verbose=1, dataset_folder='./Datasets/'):
    if(dataset_config.name == 'cifar10'):
        transform = transforms.Compose(
            [transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainset, val_set = torch.utils.data.random_split(trainset, [math.ceil(0.9 * len(trainset)), len(trainset) -(math.ceil(0.9 * len(trainset)))])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=dataset_config.batch_size,
                                                shuffle=True, num_workers=2)

        validloader = torch.utils.data.DataLoader(val_set, batch_size=dataset_config.batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=dataset_config.batch_size,
                                                shuffle=False, num_workers=2)

    return trainloader,validloader,testloader
    


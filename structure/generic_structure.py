import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.APR import APRecombination


class PerClassDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_images, label):
        self.list_of_images = list_of_images
        self.label = label

    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, idx):
        image = self.list_of_images[idx]

        return image, self.label


class CustomSimpleDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_x, list_of_y):
        self.list_of_x = list_of_x
        self.list_of_y = list_of_y

    def __len__(self):
        return len(self.list_of_x)

    def __getitem__(self, idx):
        x = self.list_of_x[idx]
        y = self.list_of_y[idx]

        return x, y


to_tensor = transforms.ToTensor()
class CustomSimpleArrayDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = self.data_list[idx][0]
        if self.transform:
            x = np.array(x).astype('uint8')
            # Image has to be either MxN or MxNx3
            if(x.shape[0] == 1):
                x = Image.fromarray(x[0])
            else:
                x = Image.fromarray(x.transpose(1, 2, 0))
            if(isinstance(self.transform, transforms.Compose)):
                x = self.transform(x)
            elif(isinstance(self.transform, transforms.RandomApply) and isinstance(self.transform.transforms[0], APRecombination)):
                x, phase_used_flag, amp_used_flag, x_aug, x_orig, orig_img = self.transform(
                    x)
                if(isinstance(x, Image.Image)):
                    x = to_tensor(x)
                    x_aug = to_tensor(x_aug)
                    x_orig = to_tensor(x_orig)
                    orig_img = to_tensor(orig_img)
                y = self.data_list[idx][1]
                return x, y, phase_used_flag, amp_used_flag, x_aug, x_orig, orig_img
        y = self.data_list[idx][1]
        if(len(self.data_list[idx]) == 2):
            return x, y
        z = self.data_list[idx][2]
        return x, y, z


class CustomMergedDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_x1, list_of_x2, list_of_y1, list_of_y2):
        self.list_of_x1 = list_of_x1
        self.list_of_y1 = list_of_y1
        self.list_of_x2 = list_of_x2
        self.list_of_y2 = list_of_y2

    def __len__(self):
        return len(self.list_of_x1)

    def __getitem__(self, idx):
        x1 = self.list_of_x1[idx]
        y1 = self.list_of_y1[idx]
        x2 = self.list_of_x2[idx]
        y2 = self.list_of_y2[idx]

        return x1, x2, y1, y2

import torch


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



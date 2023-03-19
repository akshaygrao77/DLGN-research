import io
import random
import torch
from PIL import Image
import numpy as np
import utils.apr_augmentations as augmentations


class APRecombination(object):
    def __init__(self, img_size=32, aug=None):
        if aug is None:
            augmentations.IMAGE_SIZE = img_size
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = aug.augmentations

    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''

        op = np.random.choice(self.aug_list)
        x = op(x, 3)

        p = random.uniform(0, 1)
        if p > 0.5:
            return x

        x_aug = x.copy()
        op = np.random.choice(self.aug_list)
        x_aug = op(x_aug, 3)

        x = np.array(x).astype(np.uint8)
        x_aug = np.array(x_aug).astype(np.uint8)

        fft_1 = np.fft.fftshift(np.fft.fftn(x))
        fft_2 = np.fft.fftshift(np.fft.fftn(x_aug))

        abs_1, angle_1 = np.abs(fft_1), np.angle(fft_1)
        abs_2, angle_2 = np.abs(fft_2), np.angle(fft_2)

        fft_1 = abs_1*np.exp((1j) * angle_2)
        fft_2 = abs_2*np.exp((1j) * angle_1)

        p = random.uniform(0, 1)

        if p > 0.5:
            x = np.fft.ifftn(np.fft.ifftshift(fft_1))
        else:
            x = np.fft.ifftn(np.fft.ifftshift(fft_2))

        x = x.astype(np.uint8)
        x = Image.fromarray(x)

        return x


def mix_data(x, use_cuda=True, prob=0.6):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    p = random.uniform(0, 1)

    if p > prob:
        return x

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    fft_1 = torch.fft.fftn(x, dim=(1, 2, 3))
    abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)

    fft_2 = torch.fft.fftn(x[index, :], dim=(1, 2, 3))
    abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)

    fft_1 = abs_2*torch.exp((1j) * angle_1)

    mixed_x = torch.fft.ifftn(fft_1, dim=(1, 2, 3)).float()

    return mixed_x

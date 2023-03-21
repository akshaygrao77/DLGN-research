import io
import random
import torch
from PIL import Image
import numpy as np
import utils.apr_augmentations as augmentations


class APRecombination(object):
    def __init__(self, img_size=32, prob_threshold=0.5, aprs_mix_prob=0.6, aug=None):
        self.prob_threshold = prob_threshold
        self.aprs_mix_prob = aprs_mix_prob
        if aug is None:
            augmentations.IMAGE_SIZE = img_size
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = aug.augmentations

    def __call__(self, orig_img):
        '''
        :param orig_img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''
        phase_used_flag = 0
        amp_used_flag = 0
        op = np.random.choice(self.aug_list)
        x = op(orig_img, 3)

        p = random.uniform(0, 1)
        if p > self.prob_threshold:
            return x, phase_used_flag, amp_used_flag, orig_img, orig_img, orig_img

        x_aug = x.copy()
        op = np.random.choice(self.aug_list)
        x_aug = op(x_aug, 3)

        x_orig = np.array(x).astype(np.uint8)
        x_aug = np.array(x_aug).astype(np.uint8)

        fft_x = np.fft.fftshift(np.fft.fftn(x_orig))
        fft_x_aug = np.fft.fftshift(np.fft.fftn(x_aug))

        abs_x, angle_x = np.abs(fft_x), np.angle(fft_x)
        abs_x_aug, angle_x_aug = np.abs(fft_x_aug), np.angle(fft_x_aug)

        fft_phase_x_aug = abs_x*np.exp((1j) * angle_x_aug)
        fft_phase_x = abs_x_aug*np.exp((1j) * angle_x)

        p = random.uniform(0, 1)

        if p > self.aprs_mix_prob:
            phase_used_flag = 1
            amp_used_flag = -1
            x = np.fft.ifftn(np.fft.ifftshift(fft_phase_x_aug))
        else:
            phase_used_flag = -1
            amp_used_flag = 1
            x = np.fft.ifftn(np.fft.ifftshift(fft_phase_x))

        x = x.astype(np.uint8)
        x = Image.fromarray(x)

        return x, phase_used_flag, amp_used_flag, x_aug, x_orig, orig_img


def mix_data(x, use_cuda=True, prob=0.6):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    p = random.uniform(0, 1)

    if p > prob:
        return x, None,None

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
    fft_2 = abs_1*torch.exp((1j) * angle_2)

    mixed_x = torch.fft.ifftn(fft_1, dim=(1, 2, 3)).float()
    mixed_x2 = torch.fft.ifftn(fft_2, dim=(1, 2, 3)).float()

    return mixed_x, index, mixed_x2

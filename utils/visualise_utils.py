import torch
from PIL import Image
import numpy as np
import copy

from configs.dlgn_conv_config import HardRelu

# Activation map is a 1,0 tensor


def calculate_common_among_two_activation_patterns(active_map_one, active_map_two):
    common_active_pixels = 0
    total_pixels = 0
    active_pixels_one = 0
    active_pixels_two = 0
    for act_indx in range(len(active_map_one)):
        sub_activation_map_one = active_map_one[act_indx]
        sub_activation_map_two = active_map_two[act_indx]

        active_pixels_one += torch.count_nonzero(
            sub_activation_map_one).item()
        active_pixels_two += torch.count_nonzero(
            sub_activation_map_two).item()

        total_pixels += torch.numel(sub_activation_map_one)

        common_activations_map = sub_activation_map_one * sub_activation_map_two

        common_active_pixels += torch.count_nonzero(
            common_activations_map).item()

    common_active_percentage = (100. * (common_active_pixels/total_pixels))

    return common_active_percentage, common_active_pixels, total_pixels, active_pixels_one, active_pixels_two


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def recreate_image(im_as_var, unnormalize=True):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [0.4914, 0.4822, 0.4465]
    reverse_std = [1/0.2023, 1/0.1994, 1/0.2010]

    recreated_im = copy.copy(im_as_var.cpu().clone().detach().numpy()[0])
    arr_max = np.amax(recreated_im)
    arr_min = np.amin(recreated_im)
    recreated_im = (recreated_im-arr_min)/(arr_max-arr_min)

    if(unnormalize):
        for c in range(3):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]
    # recreated_im[recreated_im > 1] = 1
    # recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im)
    # recreated_im = recreated_im..transpose(1, 2, 0)
    return recreated_im

def add_lower_dimension_vectors_within_itself(input_tensor):
    init_dim = input_tensor[0]
    for i in range(1, input_tensor.size()[0]):
        t = input_tensor[i]
        init_dim = init_dim + t

    return init_dim


def multiply_lower_dimension_vectors_within_itself(input_tensor, collect_threshold):
    init_dim = input_tensor[0]
    for i in range(1, input_tensor.size()[0]):
        t = input_tensor[i]
        # init_dim = init_dim * t
        init_dim = init_dim + t

    init_dim = HardRelu()(init_dim - 0.50 * input_tensor.size()[0])
    return init_dim

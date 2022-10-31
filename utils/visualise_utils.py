import torch
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import math
import tqdm
import matplotlib.animation as animation
import matplotlib
import os
import gc
from matplotlib import figure

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


def save_images_from_dataloader(dataloader, classes, postfix_folder_for_save='/', save_image_prefix=None):
    loader = tqdm.tqdm(
        dataloader, desc='Saving images from dataloader:'+str(postfix_folder_for_save))
    for batch_idx, data in enumerate(loader, 0):
        images, labels = data

        if(batch_idx % 20 == 0 and save_image_prefix is not None):
            ac_images = recreate_image(
                images[0], unnormalize=False)

            img_save_folder = save_image_prefix + \
                "/"+str(postfix_folder_for_save)
            if not os.path.exists(img_save_folder):
                os.makedirs(img_save_folder)
            save_im_path = img_save_folder+str(postfix_folder_for_save)+'_c' + \
                str(classes[labels[0]])+'_batch_ind_' + \
                str(batch_idx) + '.jpg'

            save_image(ac_images, save_im_path)


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


def construct_normalized_heatmaps_from_data(heatmap_data, title, save_path=None, cmap='viridis'):
    print("Constructing normalized heatmap for:{} and will be saved at:{}".format(
        title, save_path if not None else 'Not saved'))
    list_of_final_heatmap_data = []
    row, col = determine_row_col_from_features(heatmap_data.shape[0])

    ix = 1
    assert row * \
        col == heatmap_data.shape[0], 'All channels of heatmap data is not fit by row,col'
    plt.suptitle(title, fontsize=14)
    a = row * heatmap_data[0].shape[0]
    b = col*heatmap_data[0].shape[1]
    fig, ax_list = plt.subplots(
        row, col, sharex=True, sharey=True, figsize=(b/2, a/2))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for r in tqdm.tqdm(range(row), desc=" Constructing normalized heatmap for {} with num_row:{}".format(title, row)):
        for c in range(col):
            if(col != 1):
                plt_ax = ax_list[r][c]
            else:
                plt_ax = ax_list
            plt_ax.set_xticks([])
            plt_ax.set_yticks([])
            plt_ax.set_title(ix)
            current_heatmap_data = heatmap_data[ix-1, :, :]

            # Shift the range between [0,1]
            arr_max = np.amax(current_heatmap_data)
            arr_min = np.amin(current_heatmap_data)
            current_heatmap_data = (
                current_heatmap_data - arr_min)/(arr_max - arr_min)

            current_heatmap_data *= 255
            current_heatmap_data = np.clip(
                current_heatmap_data, 0, 255).astype('uint8')

            # sns.set(rc={'figure.figsize': current_heatmap_data.shape})
            sns.heatmap(current_heatmap_data, ax=plt_ax,
                        cbar=c == 0,
                        cbar_ax=None if c else cbar_ax, cmap=cmap)
            # plt.imshow(current_heatmap_data, cmap='viridis')
            ix += 1
            list_of_final_heatmap_data.append(current_heatmap_data)

    if(save_path is not None):
        plt.savefig(save_path)

    return list_of_final_heatmap_data


def gallery(array, nrows, ncols):
    nindex, height, width = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width)
              .swapaxes(1, 2)
              .reshape(height*nrows, width*ncols))
    return result


def recreate_np_image(recreated_im, unnormalize=False):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [0.4914, 0.4822, 0.4465]
    reverse_std = [1/0.2023, 1/0.1994, 1/0.2010]

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
    # print("recreated_im shape", recreated_im.shape)
    # print("recreated_im", recreated_im)
    # recreated_im = recreated_im..transpose(1, 2, 0)
    return recreated_im


def generate_plain_image(image_data, save_path):
    # print("image_data", image_data)
    row, col = determine_row_col_from_features(image_data.shape[0])
    reshaped_data = gallery(image_data, row, col)
    # print("reshaped_data", reshaped_data)
    re_image = recreate_np_image(reshaped_data)
    save_image(re_image, save_path)


def generate_list_of_plain_images_from_data(full_heatmap_data, start=None, end=None, save_each_img_path=None):
    if(start is None):
        start = 0
    if(end is None):
        end = full_heatmap_data.shape[0]
    full_heatmap_data = full_heatmap_data[start:end]
    num_frames = full_heatmap_data.shape[0]

    sfolder = save_each_img_path[0:save_each_img_path.rfind("/")+1]
    if not os.path.exists(sfolder):
        os.makedirs(sfolder)
    
    for i in range(num_frames):
        if(i % 50 == 0):
            print("Writing image:",i)
        if(save_each_img_path is not None):
            temp_each_img_path = save_each_img_path.replace(
                "*", str(start+i))
            feature_maps = full_heatmap_data[i]
            generate_plain_image(feature_maps, temp_each_img_path)


def generate_list_of_images_from_data(full_heatmap_data, start, end, title, save_each_img_path=None, cmap='binary'):
    full_heatmap_data = full_heatmap_data[start:end]
    num_frames = full_heatmap_data.shape[0]
    for i in range(num_frames):
        if(save_each_img_path is not None):
            temp_each_img_path = save_each_img_path.replace(
                "*", str(start+i))
            feature_maps = full_heatmap_data[i]
            construct_images_from_feature_maps(
                feature_maps, title, save_path=temp_each_img_path, cmap=cmap)


def generate_video_of_image_from_data(full_heatmap_data, start, end, title, save_path=None, save_each_img_path=None, cmap='viridis'):
    full_heatmap_data = full_heatmap_data[start:end]
    num_frames = full_heatmap_data.shape[0]
    print("num_frames:", num_frames)
    writervideo = animation.FFMpegWriter(fps=1)

    # Shift the range between [0,1]
    arr_max = np.amax(full_heatmap_data)
    arr_min = np.amin(full_heatmap_data)
    full_heatmap_data = (
        full_heatmap_data-arr_min)/(arr_max-arr_min)

    full_heatmap_data *= 255
    full_heatmap_data = np.clip(
        full_heatmap_data, 0, 255).astype('uint8')

    heatmap_data = full_heatmap_data[0]
    row, col = determine_row_col_from_features(heatmap_data.shape[0])

    plt.suptitle(title, fontsize=14)
    a = row * heatmap_data[0].shape[0]
    b = col*heatmap_data[0].shape[1]
    fig, ax_list = plt.subplots(
        row, col, sharex=True, sharey=True, figsize=(b/4, a/4))

    sfolder = save_path[0:save_path.rfind("/")+1]
    if not os.path.exists(sfolder):
        os.makedirs(sfolder)

    sfolder = save_each_img_path[0:save_each_img_path.rfind("/")+1]
    if not os.path.exists(sfolder):
        os.makedirs(sfolder)

    def init_func():
        ix = 1
        init_hm = np.zeros(
            (heatmap_data.shape[0], heatmap_data.shape[1], heatmap_data.shape[2]))

        for r in tqdm.tqdm(range(row), desc=" Constructing Image for ind: init with title :{} with num_row:{}".format(title, row)):
            for c in range(col):
                ind_c = c
                if(col != 1):
                    plt_ax = ax_list[r][ind_c]
                else:
                    plt_ax = ax_list

                current_heatmap_data = init_hm[ix-1, :, :]

                plt_ax.imshow(current_heatmap_data, cmap=cmap)
                ix += 1

    def image_animate(i):
        print("i:", i)
        heatmap_data = full_heatmap_data[i]  # select data range
        ix = 1

        for r in tqdm.tqdm(range(row), desc=" Constructing image for ind: {} with title :{} with num_row:{}".format(i, title, row)):
            for c in range(col):
                ind_c = c
                if(col != 1):
                    plt_ax = ax_list[r][ind_c]
                else:
                    plt_ax = ax_list

                plt_ax.set_title(ix)
                current_heatmap_data = heatmap_data[ix-1, :, :]

                plt_ax.clear()
                plt_ax.imshow(current_heatmap_data, cmap=cmap)

                ix += 1

        if(save_each_img_path is not None):
            temp_each_img_path = save_each_img_path.replace(
                "*", str(start+i))
            print("temp_each_img_path:", temp_each_img_path)
            plt.savefig(temp_each_img_path)

    ani = matplotlib.animation.FuncAnimation(
        fig, image_animate, frames=num_frames, init_func=init_func)
    if(save_path is not None):
        temp_path = save_path.replace("*", "{}_{}".format(start, end))
        print("temp_path:", temp_path)
        ani.save(temp_path, writer=writervideo)
        plt.clf()
        plt.close()


def generate_video_of_heatmap_from_data(full_heatmap_data, title, save_path=None, cmap='viridis'):
    list_of_prev_ax = None
    num_frames = full_heatmap_data.shape[0]
    print("num_frames:", num_frames)
    writervideo = animation.FFMpegWriter(fps=1)
    # writer = Writer(fps=1, bitrate=1800)

    heatmap_data = full_heatmap_data[0]
    row, col = determine_row_col_from_features(heatmap_data.shape[0])

    plt.suptitle(title, fontsize=14)
    a = row * heatmap_data[0].shape[0]
    b = col*heatmap_data[0].shape[1]
    fig, ax_list = plt.subplots(
        row, col, sharex=True, sharey=True, figsize=(b/4, a/4))

    def init_func():
        ix = 1
        init_hm = np.zeros(
            (heatmap_data.shape[0], heatmap_data.shape[1], heatmap_data.shape[2]))
        list_of_final_heatmap_data = []
        for r in tqdm.tqdm(range(row), desc=" Constructing heatmap for ind: init with title :{} with num_row:{}".format(title, row)):
            for c in range(col):
                print("row:{}, col:{}".format(r, c))
                ind_c = c
                if(col != 1):
                    plt_ax = ax_list[r][ind_c]
                else:
                    plt_ax = ax_list

                current_heatmap_data = init_hm[ix-1, :, :]

                prev_ax = sns.heatmap(current_heatmap_data, ax=plt_ax,
                                      cbar=False,
                                      cmap=cmap)
                ix += 1
                list_of_final_heatmap_data.append(current_heatmap_data)

        return list_of_final_heatmap_data

    def heatmap_animate(i, *args):
        # print("*args", *args)
        # list_of_prev_ax = args[0]
        # if(list_of_prev_ax is not None):
        #     for each_ax in list_of_prev_ax:
        #         each_ax.clear()
        #         each_ax.cla()

        # list_of_prev_ax = []

        heatmap_data = full_heatmap_data[i]  # select data range
        ix = 1
        list_of_final_heatmap_data = []
        # for r in range(row):
        #     for c in range(col):
        #         if(col != 1):
        #             ax_list[r][c].cla()
        #         else:
        #             ax_list.cla()
        # row, col = determine_row_col_from_features(heatmap_data.shape[0])
        # print("construct_heatmaps_from_data Num row:{}, Num col:{}".format(row, col))
        # assert row * \
        #     col == heatmap_data.shape[0], 'All channels of heatmap data is not fit by row,col'
        # plt.suptitle(title, fontsize=14)
        # a = row * heatmap_data[0].shape[0]
        # b = col*heatmap_data[0].shape[1]
        # fig, ax_list = plt.subplots(
        #     row, col, sharex=True, sharey=True, figsize=(b/2, a/2))
        for r in tqdm.tqdm(range(row), desc=" Constructing heatmap for ind: {} with title :{} with num_row:{}".format(i, title, row)):
            for c in range(col):
                print("row:{}, col:{}".format(r, c))
                ind_c = c
                if(col != 1):
                    plt_ax = ax_list[r][ind_c]
                else:
                    plt_ax = ax_list

                plt_ax.set_title(ix)
                current_heatmap_data = heatmap_data[ix-1, :, :]

                prev_ax = sns.heatmap(current_heatmap_data, ax=plt_ax,
                                      cbar=False,
                                      cmap=cmap)

                # list_of_prev_ax.append(prev_ax)
                # plt.imshow(current_heatmap_data, cmap='viridis')
                ix += 1
                list_of_final_heatmap_data.append(current_heatmap_data)

        return list_of_final_heatmap_data

        # plt.setp(p.lines,linewidth=7)
    ani = matplotlib.animation.FuncAnimation(
        fig, heatmap_animate, frames=num_frames, fargs=[list_of_prev_ax], init_func=init_func)

    ani.save(save_path, writer=writervideo)


def construct_heatmaps_from_data(heatmap_data, title, save_path=None, cmap='viridis'):
    print("Constructing heatmap for:{} and will be saved at:{}".format(
        title, save_path if not None else 'Not saved'))
    list_of_final_heatmap_data = []
    row, col = determine_row_col_from_features(heatmap_data.shape[0])

    ix = 1
    assert row * \
        col == heatmap_data.shape[0], 'All channels of heatmap data is not fit by row,col'
    plt.suptitle(title, fontsize=14)
    a = row * heatmap_data[0].shape[0]
    b = col*heatmap_data[0].shape[1]
    fig, ax_list = plt.subplots(
        row, col, sharex=True, sharey=True, figsize=(b/5, a/5))
    for r in tqdm.tqdm(range(row), desc=" Constructing heatmap for {} with num_row:{}".format(title, row)):
        for c in range(col):
            ind_c = c
            if(col != 1):
                plt_ax = ax_list[r][ind_c]
            else:
                plt_ax = ax_list

            plt_ax.set_title(ix)
            current_heatmap_data = heatmap_data[ix-1, :, :]

            sns.heatmap(current_heatmap_data, ax=plt_ax,
                        cbar=True,
                        cmap=cmap)
            # plt.imshow(current_heatmap_data, cmap='viridis')
            ix += 1
            list_of_final_heatmap_data.append(current_heatmap_data)

    if(save_path is not None):
        plt.savefig(save_path)
        plt.clf()
        plt.close()

    return list_of_final_heatmap_data


def determine_row_col_from_features(num_features):
    current_row = int(math.sqrt(num_features))
    if(num_features % current_row == 0):
        return current_row, num_features//current_row

    interval = 1
    while(interval < num_features):
        backward = current_row - interval
        if(num_features % backward == 0):
            return backward, num_features//backward
        interval += 1
    return 0, 0


def construct_images_from_feature_maps(feature_maps, title, save_path=None, is_scale=False, is_shift=False, is_normalize_data=False, cmap='binary'):
    matplotlib.use('Agg')
    print("Constructing Images for:{} and will be saved at:{}".format(
        title, save_path if not None else 'Not saved'))
    row, col = determine_row_col_from_features(feature_maps.shape[0])

    ix = 1
    assert row * \
        col == feature_maps.shape[0], 'All channels of feature_maps is not fit by row,col'

    # plt.suptitle(title, fontsize=14)
    a = row * feature_maps[0].shape[0]
    b = col*feature_maps[0].shape[1]
    fig = figure.Figure(figsize=(b/4, a/4))
    fig.suptitle(title, fontsize=20)
    ax_list = fig.subplots(row, col, sharex=True,
                           sharey=True)
    # fig, ax_list = plt.subplots(
    #     row, col, sharex=True, sharey=True, figsize=(b/4, a/4))

    if(is_shift == True):
        # Shift the range between [0,1]
        arr_max = np.amax(feature_maps)
        arr_min = np.amin(feature_maps)
        feature_maps = (
            feature_maps-arr_min)/(arr_max-arr_min)

    if(is_normalize_data):
        feature_maps -= feature_maps.mean(0)
        feature_maps /= feature_maps.std(0)
    if(is_scale):
        feature_maps *= 255
        feature_maps = np.clip(
            feature_maps, 0, 255).astype('uint8')

    for r in tqdm.tqdm(range(row), desc=" Constructing images for {} with num_row:{}".format(title, row)):
        for c in range(col):
            # specify subplot and turn of axis
            if(col != 1):
                plt_ax = ax_list[r][c]
            else:
                plt_ax = ax_list

            plt_ax.set_title(ix)
            current_feature_map = feature_maps[ix-1, :, :]

            art_obj = plt_ax.imshow(current_feature_map, cmap=cmap)
            ix += 1

    if(save_path is not None):
        fig.savefig(save_path)
        fig.clear()
        plt.clf()
        plt.close(fig)
        plt.close('all')
        gc.collect()

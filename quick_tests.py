import torch
import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np
import seaborn as sns
import math
import tqdm
import matplotlib.animation as animation
import matplotlib
from memory_profiler import profile
import gc


def construct_normalized_heatmaps_from_data(heatmap_data, title, save_path=None, cmap='viridis'):
    list_of_final_heatmap_data = []
    row, col = determine_row_col_from_features(heatmap_data.shape[0])
    print("construct_normalized_heatmaps_from_data Num row:{}, Num col:{}".format(row, col))
    ix = 1
    assert row * \
        col == heatmap_data.shape[0], 'All channels of heatmap data is not fit by row,col'
    plt.suptitle(title, fontsize=14)
    a = row * heatmap_data[0].shape[0]
    b = col*heatmap_data[0].shape[1]
    fig, ax_list = plt.subplots(
        row, col, sharex=True, sharey=True, figsize=(b/2, a/2))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for r in range(row):
        for c in range(col):
            print("row:{}, col:{}".format(r, c))
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


# def construct_heatmaps_from_data(heatmap_data, title, save_path=None, cmap='viridis'):
#     list_of_final_heatmap_data = []
#     row, col = determine_row_col_from_features(heatmap_data.shape[0])
#     print("construct_heatmaps_from_data Num row:{}, Num col:{}".format(row, col))
#     ix = 1
#     assert row * \
#         col == heatmap_data.shape[0], 'All channels of heatmap data is not fit by row,col'
#     plt.suptitle(title, fontsize=14)
#     a = row * heatmap_data[0].shape[0]
#     b = col*heatmap_data[0].shape[1]
#     fig, ax_list = plt.subplots(
#         row, (2*col), sharex=True, sharey=True, figsize=(b/2, a/2))
#     for r in range(row):
#         for c in range(col):
#             print("row:{}, col:{}".format(r, c))
#             ind_c = 2*c
#             if(col != 1):
#                 plt_ax = ax_list[r][ind_c]
#                 cbar_ax = ax_list[r][ind_c+1]
#             else:
#                 plt_ax = ax_list[ind_c]
#                 cbar_ax = ax_list[ind_c+1]

#             plt_ax.set_title(ix)
#             current_heatmap_data = heatmap_data[ix-1, :, :]

#             # # Shift the range between [0,1]
#             # arr_max = np.amax(current_heatmap_data)
#             # arr_min = np.amin(current_heatmap_data)
#             # current_heatmap_data = (
#             #     current_heatmap_data - arr_min)/(arr_max - arr_min)

#             # current_heatmap_data *= 255
#             # current_heatmap_data = np.clip(
#             #     current_heatmap_data, 0, 255).astype('uint8')

#             # sns.set(rc={'figure.figsize': current_heatmap_data.shape})
#             sns.heatmap(current_heatmap_data, ax=plt_ax,
#                         cbar=True,
#                         cbar_ax=cbar_ax, cmap=cmap)
#             # plt.imshow(current_heatmap_data, cmap='viridis')
#             ix += 1
#             list_of_final_heatmap_data.append(current_heatmap_data)

#     if(save_path is not None):
#         plt.savefig(save_path)

#     return list_of_final_heatmap_data


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


# @profile
def generate_video_of_image_from_data(full_heatmap_data, title, save_path=None, cmap='viridis'):
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

    ani = matplotlib.animation.FuncAnimation(
        fig, image_animate, frames=num_frames, init_func=init_func)
    if(save_path is not None):
        ani.save(save_path, writer=writervideo)
        plt.clf()
        plt.close()


def generate_video_of_heatmap_from_data(full_heatmap_data, title, save_path=None, cmap='viridis'):
    matplotlib.use('Agg')
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
    list_of_final_heatmap_data = []
    row, col = determine_row_col_from_features(heatmap_data.shape[0])
    print("construct_heatmaps_from_data Num row:{}, Num col:{}".format(row, col))
    ix = 1
    assert row * \
        col == heatmap_data.shape[0], 'All channels of heatmap data is not fit by row,col'
    plt.suptitle(title, fontsize=14)
    a = row * heatmap_data[0].shape[0]
    b = col*heatmap_data[0].shape[1]
    fig, ax_list = plt.subplots(
        row, col, sharex=True, sharey=True, figsize=(b/2, a/2))
    for r in tqdm.tqdm(range(row), desc=" Constructing heatmap for {} with num_row:{}".format(title, row)):
        for c in range(col):
            print("row:{}, col:{}".format(r, c))
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

    return list_of_final_heatmap_data


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


if __name__ == '__main__':
    torch.manual_seed(2022)
    np.random.seed(2022)
    # feature_map_for_video = np.random.randint(
    #     low=20, high=50, size=(5, 64, 28, 28))
    feature_maps1 = np.random.rand(256, 28, 28)
    # feature_maps1 = np.ones(shape=[72, 28, 28])*128
    feature_map_for_video = np.random.rand(5, 64, 28, 28)
    # feature_maps1 = np.random.uniform(low=-50, high=50, size=(72, 28, 28))
    # feature_maps1 = np.random.normal(loc=128, scale=128, size=(72, 28, 28))
    # print("feature_maps1", feature_maps1)
    # feature_maps2 = np.random.randint(low=-1, high=1, size=(128, 28, 28))
    # heatmap_feature_map1 = np.round(np.random.rand(128, 28, 28)*255)
    # feature_maps = np.random.rand(128, 28, 28)
    for _ in range(20):
        list_of_final_plotted_activation_maps = construct_images_from_feature_maps(
            feature_maps1, "Test title", save_path='root/dummy/new_test.jpg', is_shift=False, is_scale=False)
    # # list_of_final_plotted_activation_maps = construct_images_from_feature_maps(
    # #     feature_maps2, "Test title", save_path='root/dummy/titled_final_image_save2')

    # # print("heatmap_feature_map1", heatmap_feature_map1)
    # list_of_final_heatmap_data = construct_heatmaps_from_data(
    #     feature_maps1, "Test title", "root/dummy/01_float_nonstd_heatmap.jpg")
    # list_of_final_heatmap_data = construct_normalized_heatmaps_from_data(
    #     feature_maps1, "Test title", "root/dummy/01_float_std_heatmap_half_size")

    # generate_video_of_heatmap_from_data(
    #     feature_map_for_video, "Test title for video", save_path='root/dummy/01_float_nonstd_heatmap.mp4')
    # generate_video_of_image_from_data(
    #     feature_map_for_video, "Test title for video", save_path='root/dummy/01_float_nonstd_image.mp4')

    # print("list_of_final_heatmap_data", list_of_final_heatmap_data)

    print("Finished execution")

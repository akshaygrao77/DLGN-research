import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import tqdm
import matplotlib.animation as animation
import matplotlib
from memory_profiler import profile


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
def image_animate(i):
    heatmap_data = full_heatmap_data[i]  # select data range
    ix = 1

    for r in tqdm.tqdm(range(row), desc=" Constructing image for ind: {} with title :{} with num_row:{}".format(i, title, row)):
        for c in range(col):
            print("row:{}, col:{}".format(r, c))
            ind_c = c
            if(col != 1):
                plt_ax = ax_list[r][ind_c]
            else:
                plt_ax = ax_list

            plt_ax.set_title(ix)
            current_heatmap_data = heatmap_data[ix-1, :, :]

            plt_ax.clear()
            current_art_obj = plt_ax.imshow(
                current_heatmap_data, cmap=cmap)
            art_obj_list.append(current_art_obj)
            # art_obj_list[ix-1] = current_art_obj

            ix += 1


# @profile
def save_vid():
    ani = matplotlib.animation.FuncAnimation(
        fig, image_animate, frames=num_frames, save_count=0)

    ani.save(save_path, writer=writervideo)


if __name__ == '__main__':
    full_heatmap_data = np.random.randint(
        low=-1, high=2, size=(5, 64, 28, 28))
    title = "Test title for video"
    save_path = "root/dummy/01_float_nonstd_image.mp4"
    cmap = 'viridis'
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

    ix = 1
    init_hm = np.zeros(
        (heatmap_data.shape[0], heatmap_data.shape[1], heatmap_data.shape[2]))
    art_obj_list = []
    for r in tqdm.tqdm(range(row), desc=" Constructing Image for ind: init with title :{} with num_row:{}".format(title, row)):
        for c in range(col):
            print("row:{}, col:{}".format(r, c))
            ind_c = c
            if(col != 1):
                plt_ax = ax_list[r][ind_c]
            else:
                plt_ax = ax_list

            current_heatmap_data = init_hm[ix-1, :, :]

            plt_ax.clear()
            art_obj = plt_ax.imshow(current_heatmap_data, cmap=cmap)
            art_obj_list.append(art_obj)
            ix += 1
    # for i in range(num_frames):
    #     image_animate(i)
    save_vid()
    print("Execution completed!")

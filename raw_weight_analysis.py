
import torch
import os
import numpy as np
from utils.weight_utils import get_gating_layer_weights
from utils.visualise_utils import generate_plain_image


def convert_list_tensor_to_numpy(list_of_tensors):
    cpudevice = torch.device("cpu")
    list_of_ret_np = []
    for i in range(len(list_of_tensors)):
        current_np_tensor = list_of_tensors[i].to(
            cpudevice).numpy()
        list_of_ret_np.append(current_np_tensor)

    return list_of_ret_np


def standarize_list_of_numpy(list_of_np_array):
    arr_min = None
    arr_max = None
    for i in range(len(list_of_np_array)):
        current_np = list_of_np_array[i]
        curr_arr_max = np.amax(current_np)
        curr_arr_min = np.amin(current_np)
        print("*****************Layer:", i)
        print("curr_arr_max", curr_arr_max)
        print("curr_arr_min", curr_arr_min)
        print("Mean of filters", current_np.mean())
        print("Std of filters", current_np.std())
        if(arr_min is None):
            arr_min = curr_arr_min
        else:
            if(arr_min > curr_arr_min):
                arr_min = curr_arr_min
        if(arr_max is None):
            arr_max = curr_arr_max
        else:
            if(arr_max < curr_arr_max):
                arr_max = curr_arr_max

    for i in range(len(list_of_np_array)):
        list_of_np_array[i] = (list_of_np_array[i]-arr_min)/(arr_max-arr_min)

    return list_of_np_array


def run_raw_weight_analysis_on_config(model, root_save_prefix='root/RAW_WEIGHT_ANALYSIS', final_postfix_for_save="",
                                      is_save_graph_visualizations=True):
    if(root_save_prefix is None):
        root_save_prefix = 'root/RAW_WEIGHT_ANALYSIS'
    if(final_postfix_for_save is None):
        final_postfix_for_save = ""
    if(is_save_graph_visualizations):
        save_folder = root_save_prefix + "/" + \
            str(final_postfix_for_save)+"/PlainImages/KernelWeights/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        list_of_weights, list_of_bias = get_gating_layer_weights(model)

        list_of_weights = convert_list_tensor_to_numpy(list_of_weights)
        list_of_weights = standarize_list_of_numpy(list_of_weights)

        for i in range(len(list_of_weights)):
            current_full_img_save_path = save_folder + \
                "weight_plot_n_{}.jpg".format(i)

            current_weight_np = list_of_weights[i]

            print("current_full_img_save_path:", current_full_img_save_path)

            print("list_of_weights size:{}".format(
                current_weight_np.shape))
            # print("list_of_weights", current_weight_np)

            generate_plain_image(current_weight_np, current_full_img_save_path)


def run_generate_raw_weight_analysis(models_base_path, it_start=1, num_iter=None):
    if(num_iter is None):
        num_iter = it_start + 1

    list_of_model_paths = []
    if(models_base_path != None):
        list_of_save_prefixes = []
        list_of_save_postfixes = []

        for i in range(it_start, num_iter):
            each_model_prefix = "aug_conv4_dlgn_iter_{}_dir.pt".format(i)

            list_of_model_paths.append(models_base_path+each_model_prefix)
            list_of_save_prefixes.append(
                str(models_base_path)+"/RAW_WEIGHT_ANALYSIS/")
            list_of_save_postfixes.append("/aug_indx_{}".format(i))

    else:
        list_of_model_paths = [None]
        list_of_save_prefixes = [
            "root/RAW_WEIGHT_ANALYSIS/"]
        list_of_save_postfixes = [None]

    for ind in range(len(list_of_model_paths)):
        each_model_path = list_of_model_paths[ind]
        each_save_prefix = list_of_save_prefixes[ind]
        each_save_postfix = list_of_save_postfixes[ind]
        analysed_model_path = each_model_path

        custom_model = torch.load(each_model_path)
        print(" #*#*#*#*#*#*#*# Generating weights analysis for model path:{} with save prefix :{} and postfix:{}".format(
            each_model_path, each_save_prefix, each_save_postfix))
        with torch.no_grad():
            run_raw_weight_analysis_on_config(custom_model, root_save_prefix=each_save_prefix, final_postfix_for_save=each_save_postfix,
                                              is_save_graph_visualizations=True)


if __name__ == '__main__':

    models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_plain_pure_conv4_dnn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.75/"
    #models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/"

    num_iterations = 1
    start_index = 1
    for current_it_start in range(start_index, num_iterations + 1):
        run_generate_raw_weight_analysis(models_base_path, current_it_start)

    print("Finished execution!!!")

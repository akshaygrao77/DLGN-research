
import torch
import os
import numpy as np
from utils.weight_utils import get_gating_layer_weights
from utils.visualise_utils import generate_list_of_plain_images_from_data, generate_plain_image, generate_plain_image_data, save_image, recreate_image
from structure.conv4_models import get_model_instance_from_dataset
from utils.data_preprocessing import preprocess_dataset_get_data_loader, generate_dataset_from_loader, seed_worker, true_segregation
from structure.generic_structure import PerClassDataset, CustomSimpleDataset
from structure.dlgn_conv_config_structure import DatasetConfig
from tqdm import tqdm
from PIL import Image
from adversarial_attacks_tester import load_or_generate_adv_examples, generate_adversarial_perturbation_from_adv_orig
import torchvision.transforms as T
import cv2
from configs.dlgn_conv_config import HardRelu
import copy
import cv2


def convert_list_tensor_to_numpy(list_of_tensors):
    cpudevice = torch.device("cpu")
    list_of_ret_np = []
    for i in range(len(list_of_tensors)):
        current_np_tensor = list_of_tensors[i].to(
            cpudevice).detach().numpy()
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


def output_params(list_of_weights, root_save_prefix, final_postfix_for_save):
    save_folder = root_save_prefix + "/" + \
        str(final_postfix_for_save)+"/PlainImages/KernelParams/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    txt_save_folder = root_save_prefix + "/" + \
        str(final_postfix_for_save)+"/RAW_TXT/"
    if not os.path.exists(txt_save_folder):
        os.makedirs(txt_save_folder)

    f_outs_DFT_norms = []
    for layer_num in range(len(list_of_weights)):
        current_layer_weights = list_of_weights[layer_num]

        print("Param {} size:{}".format(
            final_postfix_for_save, current_layer_weights.shape))

        current_f_outs_DFT_norm = None
        for filter_ind in range(len(current_layer_weights)):
            current_filter_weights = current_layer_weights[filter_ind]
            each_filter_txt_save_folder = txt_save_folder + "/LAY_NUM_" + \
                str(layer_num)+"/FILT_IND_" + \
                str(filter_ind)
            if not os.path.exists(each_filter_txt_save_folder):
                os.makedirs(each_filter_txt_save_folder)
            with open(each_filter_txt_save_folder + "/raw_filter_weights.txt", "w") as f:
                f.write("\n==============================================\n".join("\n".join(map(str, x))
                        for x in current_filter_weights))
            current_f_channel_norm_dft_outs = None
            for ch_ind in range(len(current_filter_weights)):
                current_filter_chnl_weights = current_filter_weights[ch_ind]
                # raw_dft_out = generate_centralized_DFT(
                #     current_filter_chnl_weights)

                if(isinstance(current_filter_chnl_weights, torch.Tensor)):
                    current_filter_chnl_weights = copy.copy(
                        current_filter_chnl_weights.cpu().clone().detach().numpy())
                pad_factor = 21
                current_filter_chnl_weights = np.pad(current_filter_chnl_weights, (pad_factor*current_filter_chnl_weights.shape[
                                                     0], pad_factor*current_filter_chnl_weights.shape[1]), 'constant', constant_values=(0))
                img_c2 = np.fft.fft2(current_filter_chnl_weights)
                raw_dft_out = np.fft.fftshift(img_c2)

                for_vis_dft_out = np.log(1+np.abs(raw_dft_out))
                std01_vis_dft_out = normalize_in_range_01(
                    for_vis_dft_out)
                std01_vis_dft_out = std01_vis_dft_out[None, :]

                if(current_f_channel_norm_dft_outs is None):
                    current_f_channel_norm_dft_outs = torch.from_numpy(
                        std01_vis_dft_out)
                else:
                    current_f_channel_norm_dft_outs = torch.vstack(
                        (current_f_channel_norm_dft_outs, torch.from_numpy(std01_vis_dft_out)))

            if(current_f_outs_DFT_norm is None):
                current_f_outs_DFT_norm = torch.unsqueeze(
                    current_f_channel_norm_dft_outs, 0)
            else:
                current_f_outs_DFT_norm = torch.vstack(
                    (current_f_outs_DFT_norm, torch.unsqueeze(current_f_channel_norm_dft_outs, 0)))

        f_outs_DFT_norms.append(current_f_outs_DFT_norm)

    for i in range(len(list_of_weights)):
        # current_full_img_save_path = save_folder + \
        # "weight_plot_n_{}.jpg".format(i)

        current_weight_np = list_of_weights[i]

        print("Norm of {} of layer:{} => {}".format(
            final_postfix_for_save, i, np.linalg.norm(current_weight_np)))

        # print("current_full_img_save_path:", current_full_img_save_path)

        print("Param:{} of layer:{} => {}".format(
            final_postfix_for_save, i, current_weight_np))

        # generate_plain_image(current_weight_np, current_full_img_save_path)

    with open(txt_save_folder+"raw_params.txt", "w") as myfile:
        for l_ind in range(len(list_of_weights)):
            curr_layer = list_of_weights[l_ind]
            myfile.write(
                "\n ************************************ Next Layer:{} *********************************** \n".format(l_ind))
            for f_ind in range(len(curr_layer)):
                myfile.write(
                    "\n -------------------- Lay:{} Next Filter:{} ------------------- \n".format(l_ind, f_ind))
                curr_filter = curr_layer[f_ind]
                myfile.write("%s" % curr_filter)

    with open(txt_save_folder+"formatted_raw_params.txt", "w") as f:
        f.write("\n ************************************ Next Layer *********************************** \n".join(
            "\n".join(map(str, generate_plain_image_data(np.squeeze(x)))) for x in list_of_weights))

    is_all_3D_DFTs = True
    ind = 0
    max_row = 0
    max_col = 0
    while(is_all_3D_DFTs and ind < len(f_outs_DFT_norms)):
        temp = np.squeeze(f_outs_DFT_norms[ind].numpy())
        is_all_3D_DFTs = is_all_3D_DFTs and (len(temp.shape) == 3)
        if(temp.shape[-2] > max_row):
            max_row = temp.shape[-2]
        if(temp.shape[-1] > max_col):
            max_col = temp.shape[-1]
        ind = ind + 1

    if(is_all_3D_DFTs):
        # print("max_row", max_row)
        # print("max_col", max_col)
        merged_padded_fouts = None
        for i in range(len(f_outs_DFT_norms)):
            current_layer_DFT = np.squeeze(f_outs_DFT_norms[i].numpy())
            # print("current_layer_DFT shape", current_layer_DFT.shape)
            # print("min_current_layer_DFT", np.min(current_layer_DFT))
            current_layer_DFT_padded = []
            for each_DFT_padded in current_layer_DFT:
                current_filter_layer_DFT_padded = cv2.resize(
                    each_DFT_padded, (max_row, max_row), interpolation=cv2.INTER_CUBIC)
                current_layer_DFT_padded.append(
                    current_filter_layer_DFT_padded)
            current_layer_DFT_padded = np.array(current_layer_DFT_padded)
            # print("current_layer_DFT_padded shape", current_layer_DFT_padded.shape)
            # current_layer_DFT_padded = np.pad(current_layer_DFT, ((0, 0), (0, max_row - current_layer_DFT.shape[
            #     -2]), (0, max_col - current_layer_DFT.shape[-1])), 'constant', constant_values=(0))
            if(merged_padded_fouts is None):
                merged_padded_fouts = torch.from_numpy(
                    current_layer_DFT_padded)
            else:
                merged_padded_fouts = torch.vstack(
                    (merged_padded_fouts, torch.from_numpy(current_layer_DFT_padded)))

        print("merged_padded_fouts shape", merged_padded_fouts.size())

    for i in range(len(list_of_weights)):
        current_layer_DFT = f_outs_DFT_norms[i]
        current_weight_np = list_of_weights[i]

        current_full_img_save_path = save_folder+"/LAY_NUM_"+str(i)+"/" + \
            "filter_params_*.jpg"

        print("current_full_img_save_path:", current_full_img_save_path)
        # rs_data = generate_plain_image_data(current_weight_np)
        # print("Params:"+str(final_postfix_for_save) +
        #           " shape:" + str(current_weight_np.shape))
        # print("rs_data shape:", rs_data.shape)
        generate_list_of_plain_images_from_data(
            current_weight_np, save_each_img_path=current_full_img_save_path, is_standarize=False)

        current_full_img_save_path = save_folder+"/LAY_NUM_"+str(i)+"/" + \
            "DFT_filter_params_*.jpg"
        generate_list_of_plain_images_from_data(
            current_layer_DFT, save_each_img_path=current_full_img_save_path, is_standarize=False)
        current_weight_np = np.squeeze(current_weight_np)
        generate_plain_image(
            current_weight_np, save_folder+"layer_num_"+str(i)+".jpg", is_standarize=False)
        current_layer_DFT = np.squeeze(current_layer_DFT)
        generate_plain_image(
            current_layer_DFT, save_folder+"DFT_lay_num_"+str(i)+".jpg", is_standarize=False)
        if(is_all_3D_DFTs):
            generate_plain_image(
                merged_padded_fouts, save_folder+"merged_DFTs.jpg", is_standarize=False)


def run_raw_weight_analysis_on_config(model, root_save_prefix='root/RAW_WEIGHT_ANALYSIS', final_postfix_for_save="",
                                      is_save_graph_visualizations=True):
    if(root_save_prefix is None):
        root_save_prefix = 'root/RAW_WEIGHT_ANALYSIS'
    if(final_postfix_for_save is None):
        final_postfix_for_save = ""

    list_of_weights, list_of_bias = get_gating_layer_weights(model)

    list_of_weights = convert_list_tensor_to_numpy(list_of_weights)
    list_of_bias = convert_list_tensor_to_numpy(list_of_bias)
    # list_of_weights = standarize_list_of_numpy(list_of_weights)

    if(is_save_graph_visualizations):
        current_final_postfix_for_save = final_postfix_for_save + "WEIGHTS"
        output_params(list_of_weights, root_save_prefix,
                      current_final_postfix_for_save)
        # current_final_postfix_for_save = final_postfix_for_save + "BIASES"
        # output_params(list_of_bias, root_save_prefix,
        #               current_final_postfix_for_save)

    return list_of_weights, list_of_bias


def run_generate_raw_weight_analysis(models_base_path, it_start=1, num_iter=None, list_of_model_paths=[]):
    if(num_iter is None):
        num_iter = it_start + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        if(len(list_of_model_paths) == 0):
            list_of_model_paths = [None]
            list_of_save_prefixes = [
                "root/RAW_WEIGHT_ANALYSIS/MT_"+str(model_arch_type_str)+"/"]
            list_of_save_postfixes = [None]
        else:
            list_of_save_prefixes = []
            list_of_save_postfixes = []
            for each_model_path in list_of_model_paths:
                base_path = each_model_path[0:each_model_path.rfind("/")+1]
                list_of_save_prefixes.append(
                    str(base_path)+"/"+str(model_arch_type_str)+"/RAW_WEIGHT_ANALYSIS/")
                list_of_save_postfixes.append("")

    for ind in range(len(list_of_model_paths)):
        each_model_path = list_of_model_paths[ind]
        each_save_prefix = list_of_save_prefixes[ind]
        each_save_postfix = list_of_save_postfixes[ind]
        analysed_model_path = each_model_path

        print("each_model_path", each_model_path)

        custom_model = get_model_instance_from_dataset(
            dataset, model_arch_type, torch_seed)
        if("masked" in model_arch_type):
            custom_model = get_model_instance_from_dataset(
                dataset, model_arch_type, torch_seed, mask_percentage=mask_percentage)

        if(each_model_path is not None):
            custom_model = get_model_from_path(
                dataset, model_arch_type, each_model_path)
            if("masked" in model_arch_type):
                custom_model = get_model_from_path(
                    dataset, model_arch_type, each_model_path, mask_percentage=mask_percentage)

        print(" #*#*#*#*#*#*#*# Generating weights analysis for model path:{} with save prefix :{} and postfix:{}".format(
            each_model_path, each_save_prefix, each_save_postfix))
        with torch.no_grad():
            run_raw_weight_analysis_on_config(custom_model, root_save_prefix=each_save_prefix, final_postfix_for_save=each_save_postfix,
                                              is_save_graph_visualizations=True)


def get_model_from_path(dataset, model_arch_type, model_path, mask_percentage=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_model = get_model_instance_from_dataset(
        dataset, model_arch_type)
    if("masked" in model_arch_type):
        custom_model = get_model_instance_from_dataset(
            dataset, model_arch_type, mask_percentage=mask_percentage)

    custom_temp_model = torch.load(
        model_path, map_location=device)
    custom_model.load_state_dict(custom_temp_model.state_dict())
    custom_model = custom_model.to(device)
    return custom_model


def get_prefix_for_save(model_path, model_arch_type):
    if('epoch' in model_path or 'aug' in model_path):
        temp = model_path.rfind("/")+1
        base_path = model_path[0:temp]
        epoch_prefix = model_path[temp:model_path.rfind(".pt")]
        save_prefix = str(base_path)+"/"+str(model_arch_type)+"/"+str(epoch_prefix) + \
            "/RAW_WEIGHT_ANALYSIS/"
    else:
        base_path = model_path[0:model_path.rfind("/")+1]
        save_prefix = str(base_path)+"/"+str(model_arch_type) + \
            "/RAW_WEIGHT_ANALYSIS/"
    return save_prefix


def run_generate_diff_raw_weight_analysis(model1_path, model2_path):

    custom1_model = get_model_from_path(dataset, model_arch_type, model1_path)

    custom2_model = get_model_from_path(dataset, model_arch_type, model2_path)

    save1_prefix = get_prefix_for_save(model1_path, model_arch_type_str)
    save2_prefix = get_prefix_for_save(model2_path, model_arch_type_str)

    list_of_weights_diff = []
    list_of_bias_diff = []

    with torch.no_grad():
        list_of_weights1, list_of_bias1 = run_raw_weight_analysis_on_config(custom1_model, root_save_prefix=save1_prefix, final_postfix_for_save="",
                                                                            is_save_graph_visualizations=True)
        list_of_weights2, list_of_bias2 = run_raw_weight_analysis_on_config(custom2_model, root_save_prefix=save2_prefix, final_postfix_for_save="",
                                                                            is_save_graph_visualizations=True)

        for i in range(len(list_of_weights1)):
            weight1 = list_of_weights1[i]
            weight2 = list_of_weights2[i]
            diff_weight = weight1-weight2
            print("diff_weight", diff_weight)
            list_of_weights_diff.append(diff_weight)

            bias1 = list_of_bias1[i]
            bias2 = list_of_bias2[i]
            list_of_bias_diff.append(bias1-bias2)

        diff_save_prefix = save1_prefix + "/MERGE_DIFF/"+save2_prefix
        current_final_postfix_for_save = "WEIGHTS"

        print("diff_save_prefix", diff_save_prefix)
        output_params(list_of_weights_diff, diff_save_prefix,
                      current_final_postfix_for_save)


def generate_seq_filter_outputs_per_image(model, filter_vis_dataset, class_label, c_indx,
                                          per_class_dataset, save_prefix, num_batches_to_visualize, final_postfix_for_save):
    is_vis_ind_conv_filter_out = False
    is_vis_ind_DFT_conv_filter_out = False
    is_vis_ind_hard_relu_filter_out = False
    is_print_ind_raw_filter_out = False
    is_print_ind_std_filter_out = False
    is_vis_ind_conv_out_orig_diff = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_folder = save_prefix + "/DS_" + str(filter_vis_dataset)+"/" + \
        str(final_postfix_for_save)+"/SEQ_FILTER_OUTS/C_"+str(class_label)+"/"

    grid_save_folder = save_prefix + "/DS_" + str(filter_vis_dataset)+"/" + \
        str(final_postfix_for_save) + \
        "/GRID_SEQ_FILTER_OUTS/C_"+str(class_label)+"/"

    per_class_data_loader = torch.utils.data.DataLoader(
        per_class_dataset, batch_size=batch_size, shuffle=False)

    per_class_data_loader = tqdm(
        per_class_data_loader, desc='Outing seq filter outputs for class:'+str(class_label))
    overall_indx_count = 0
    num_b = 0
    model = model.to(device)
    model.train(False)

    for batch_idx, per_class_per_batch_data in enumerate(per_class_data_loader):
        num_b += 1
        c_inputs, _ = per_class_per_batch_data
        c_inputs = c_inputs.to(device)

        model(c_inputs)

        if(isinstance(model, torch.nn.DataParallel)):
            conv_outs = model.module.linear_conv_outputs
        else:
            conv_outs = model.linear_conv_outputs

        hard_relu_active_percentage = np.zeros(len(conv_outs))

        with torch.no_grad():
            for layer_num in range(len(conv_outs)):
                current_layer_conv_output = conv_outs[layer_num]
                current_layer_Hrelu_output = HardRelu()(current_layer_conv_output)
                for each_batch_indx in range(len(current_layer_conv_output)):
                    current_batch_conv_output = current_layer_conv_output[each_batch_indx]
                    current_batch_HRelu_output = current_layer_Hrelu_output[each_batch_indx]

                    total_pixel_points = torch.numel(
                        current_batch_HRelu_output)
                    current_active_pixel = torch.count_nonzero(
                        (current_batch_HRelu_output))
                    hard_relu_active_percentage[layer_num] += (
                        current_active_pixel/total_pixel_points)

                    current_channel_conv_norm_outs = None
                    current_chanl_conv_DFT_norm_outs = None
                    for channel_ind in range(len(current_batch_conv_output)):
                        current_channel_conv_output = current_batch_conv_output[channel_ind]
                        current_channel_conv_output = current_channel_conv_output[None, :]

                        raw_dft_out = generate_centralized_DFT(
                            current_channel_conv_output[0])
                        for_vis_dft_out = np.log(1+np.abs(raw_dft_out))
                        std01_vis_dft_out = normalize_in_range_01(
                            for_vis_dft_out)
                        std01_vis_dft_out = std01_vis_dft_out[None, :]

                        batch_save_folder = save_folder + \
                            "/BTCH_IND_" + str(each_batch_indx)
                        current_save_folder = str(batch_save_folder) + "/LAY_NUM_" + \
                            str(layer_num)+"/"+"/FILT_IND_" + \
                            str(channel_ind) + "/"

                        std01_conv_out_image = normalize_in_range_01(
                            current_channel_conv_output)
                        if(current_channel_conv_norm_outs is None):
                            current_channel_conv_norm_outs = torch.from_numpy(
                                std01_conv_out_image)
                        else:
                            current_channel_conv_norm_outs = torch.vstack(
                                (current_channel_conv_norm_outs, torch.from_numpy(std01_conv_out_image)))

                        if(current_chanl_conv_DFT_norm_outs is None):
                            current_chanl_conv_DFT_norm_outs = torch.from_numpy(
                                std01_vis_dft_out)
                        else:
                            current_chanl_conv_DFT_norm_outs = torch.vstack(
                                (current_chanl_conv_DFT_norm_outs, torch.from_numpy(std01_vis_dft_out)))

                        std_filter_out_image = None
                        if(is_vis_ind_conv_filter_out):
                            if not os.path.exists(current_save_folder):
                                os.makedirs(current_save_folder)
                            current_filt_channel_save_path = current_save_folder + \
                                "filter_out_channel_" + \
                                str(channel_ind)+".jpg"
                            std_filter_out_image = recreate_image(
                                std01_conv_out_image, unnormalize=False, is_standarize_to_01=False)
                            save_image(std_filter_out_image,
                                       current_filt_channel_save_path)

                        if(is_vis_ind_DFT_conv_filter_out):
                            if not os.path.exists(current_save_folder):
                                os.makedirs(current_save_folder)
                            current_filt_channel_save_path = current_save_folder + \
                                "DFT_filter_out_channel_" + \
                                str(channel_ind)+".jpg"
                            std_filter_out_image = recreate_image(
                                std01_vis_dft_out, unnormalize=False, is_standarize_to_01=False)
                            save_image(std_filter_out_image,
                                       current_filt_channel_save_path)

                        if(is_vis_ind_hard_relu_filter_out):
                            if not os.path.exists(current_save_folder):
                                os.makedirs(current_save_folder)

                            current_channel_HRelu_output = current_batch_HRelu_output[channel_ind]
                            current_channel_HRelu_output = current_channel_HRelu_output[None, :]
                            current_filt_HRelu_channel_save_path = current_save_folder + \
                                "HRelu_filter_out_channel_" + \
                                str(channel_ind)+".jpg"
                            std_HRelu_filter_out_image = recreate_image(
                                current_channel_HRelu_output, unnormalize=False)
                            save_image(std_HRelu_filter_out_image,
                                       current_filt_HRelu_channel_save_path)

                        if(is_print_ind_raw_filter_out):
                            if not os.path.exists(current_save_folder):
                                os.makedirs(current_save_folder)
                            raw_txt_save_folder = current_save_folder+"/raw_filter_out.txt"
                            with open(raw_txt_save_folder, "w") as f:
                                f.write("\n".join(
                                    ",".join(map(str, x)) for x in current_channel_conv_output))
                        if(is_print_ind_std_filter_out):
                            if not os.path.exists(current_save_folder):
                                os.makedirs(current_save_folder)
                            if(std_filter_out_image is None):
                                std_filter_out_image = recreate_image(
                                    std01_conv_out_image, unnormalize=False, is_standarize_to_01=False)
                            std_txt_save_folder = current_save_folder+"/std_filter_out.txt"
                            with open(std_txt_save_folder, "w") as f:
                                f.write("\n".join(
                                    ",".join(map(str, x)) for x in std_filter_out_image))

                        if(is_vis_ind_conv_out_orig_diff):
                            if not os.path.exists(current_save_folder):
                                os.makedirs(current_save_folder)
                            orig_image = c_inputs[each_batch_indx]
                            orig_image = orig_image[None, :]

                            diff_fil_channel = current_channel_conv_output - orig_image
                            current_diff_save_path = current_save_folder + \
                                "filter_diff_" + \
                                str(channel_ind)+".jpg"

                            std_diff_fil_channel = recreate_image(
                                diff_fil_channel, unnormalize=False)
                            save_image(std_diff_fil_channel,
                                       current_diff_save_path)

                    gr_batch_save_folder = grid_save_folder + \
                        "/BTCH_IND_" + str(each_batch_indx)
                    gr_current_save_folder = str(gr_batch_save_folder) + "/LAY_NUM_" + \
                        str(layer_num)
                    if not os.path.exists(gr_current_save_folder):
                        os.makedirs(gr_current_save_folder)

                    gr_current_b_fout_save_path = gr_current_save_folder + \
                        "/sq_lay_lev_ind_std_gridded_filter_output.jpg"

                    current_channel_conv_norm_outs = torch.squeeze(
                        current_channel_conv_norm_outs)
                    current_chanl_conv_DFT_norm_outs = torch.squeeze(
                        current_chanl_conv_DFT_norm_outs)
                    # print("current_channel_conv_norm_outs size:",
                    #       current_channel_conv_norm_outs.size())
                    # print("current_batch_conv_output size:",
                    #       current_batch_conv_output.size())
                    generate_plain_image(
                        current_channel_conv_norm_outs, gr_current_b_fout_save_path, is_standarize=False, is_standarize_01=False)

                    gr_current_b_fout_save_path = gr_current_save_folder + \
                        "/sq_lay_lev_ind_std_grid_DFT_filter_output.jpg"

                    generate_plain_image(
                        current_chanl_conv_DFT_norm_outs, gr_current_b_fout_save_path, is_standarize=False, is_standarize_01=False)

                    gr_current_b_fout_save_path = gr_current_save_folder + \
                        "/sq_layer_level_std_gridded_filter_output.jpg"
                    generate_plain_image(
                        current_batch_conv_output, gr_current_b_fout_save_path, is_standarize=False)

                    gr_current_b_fout_save_path = gr_current_save_folder + \
                        "/sq_HRelu_gridded_filter_output.jpg"
                    generate_plain_image(
                        current_batch_HRelu_output, gr_current_b_fout_save_path, is_standarize=False)
                    # print("gr_current_b_fout_save_path",
                    #       gr_current_b_fout_save_path)
            hard_relu_active_percentage = hard_relu_active_percentage / \
                c_inputs.size()[0]
        overall_indx_count += c_inputs.size()[0]
        if(not(num_batches_to_visualize is None) and batch_idx == num_batches_to_visualize - 1):
            break
    hard_relu_active_percentage = hard_relu_active_percentage / \
        num_b

    print("Average hard_relu_active_percentage:", hard_relu_active_percentage)


def generate_per_batch_filter_outs(inp_channel, filter_weights, batch_inputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("filter_weights before size", filter_weights.shape)
    filter_weights = np.expand_dims(filter_weights, axis=1)
    # print("filter_weights after size", filter_weights.shape)
    filter_weights = torch.from_numpy(filter_weights)
    # print("filter_weights size", filter_weights.size())
    conv_obj = torch.nn.Conv2d(inp_channel, filter_weights.size()[
        0], filter_weights.size()[-1], padding=int(filter_weights.size()[-1]//2))
    # print("conv_obj.weight size", conv_obj.weight.size())
    conv_obj.weight = torch.nn.Parameter(filter_weights)
    conv_obj = conv_obj.to(device)
    # print("filter_weights", filter_weights)
    # print("conv_obj.weight size", conv_obj.weight.size())
    # print("conv_obj.weight", conv_obj.weight)
    filter_out = conv_obj(batch_inputs)
    # print("filter_out size", filter_out.size())
    return filter_out


def normalize_in_range_01(img_data):
    if(isinstance(img_data, torch.Tensor)):
        img_data = copy.copy(img_data.cpu().clone().detach().numpy())
    arr_max = np.amax(img_data)
    arr_min = np.amin(img_data)
    norm_im = (img_data-arr_min)/(arr_max-arr_min)
    return norm_im


def generate_centralized_DFT(img_data):
    if(isinstance(img_data, torch.Tensor)):
        img_data = copy.copy(img_data.cpu().clone().detach().numpy())
    img_c2 = np.fft.fft2(img_data)
    img_c3 = np.fft.fftshift(img_c2)

    return img_c3


def generate_filter_outputs_per_image(filter_vis_dataset, inp_channel, class_label, c_indx,
                                      per_class_dataset, list_of_weights, save_prefix, num_batches_to_visualize, final_postfix_for_save, scheme_type_tag="FILTER_OUTS"):
    is_vis_ind_original = True
    is_vis_ind_filter_out = False
    is_vis_ind_filter_out_DFT = False
    is_print_ind_std_filter_out = False
    is_print_ind_raw_filter_out = False
    is_vis_ind_filter_out_diff_orig = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_folder = save_prefix + "/DS_" + str(filter_vis_dataset)+"/" + \
        str(final_postfix_for_save)+"/" + \
        str(scheme_type_tag)+"/C_"+str(class_label)+"/"

    grid_save_folder = save_prefix + "/DS_" + str(filter_vis_dataset)+"/" + \
        str(final_postfix_for_save)+"/GRID_" + \
        str(scheme_type_tag)+"/C_"+str(class_label)+"/"

    per_class_data_loader = torch.utils.data.DataLoader(
        per_class_dataset, batch_size=batch_size, shuffle=False)

    per_class_data_loader = tqdm(
        per_class_data_loader, desc='Outing filter outputs for class:'+str(class_label))
    overall_indx_count = 0
    for batch_idx, per_class_per_batch_data in enumerate(per_class_data_loader):
        c_inputs, _ = per_class_per_batch_data
        c_inputs = c_inputs.to(device)

        for layer_num in range(len(list_of_weights)):
            current_layer_weights = list_of_weights[layer_num]

            current_layer_outputs = None
            current_layer_norm_outputs = None
            current_layer_DFT_norm_outputs = None
            for filter_ind in range(len(current_layer_weights)):
                current_filter_weights = current_layer_weights[filter_ind]

                print("Layer_num:{} shape:{} =>Filter ind:{}=> shape:{}".format(
                    layer_num, current_layer_weights.shape, filter_ind, current_filter_weights.shape))

                f_outs = generate_per_batch_filter_outs(
                    inp_channel, current_filter_weights, c_inputs)

                if(current_layer_outputs is None):
                    current_layer_outputs = torch.unsqueeze(f_outs, 0)
                else:
                    current_layer_outputs = torch.vstack(
                        (current_layer_outputs, torch.unsqueeze(f_outs, 0)))

                current_f_outs_norm = None
                current_f_outs_DFT_norm = None
                for fil_ind in range(len(f_outs)):
                    each_filter_outs = f_outs[fil_ind]
                    batch_save_folder = save_folder + \
                        "/BTCH_IND_" + str(fil_ind)
                    current_save_folder = str(batch_save_folder) + "/LAY_NUM_" + \
                        str(layer_num)+"/FILT_IND_"+str(filter_ind) + "/"

                    orig_image = c_inputs[fil_ind]
                    orig_image = orig_image[None, :]

                    if(is_vis_ind_original):
                        if not os.path.exists(batch_save_folder):
                            os.makedirs(batch_save_folder)
                        current_original_save_path = batch_save_folder+"/original_img.jpg"
                        std_orig_image = recreate_image(
                            orig_image, unnormalize=False)

                        save_image(std_orig_image, current_original_save_path)

                        raw_dft_out = generate_centralized_DFT(
                            orig_image[0])
                        for_vis_dft_out = np.log(1+np.abs(raw_dft_out))
                        for_vis_dft_out = for_vis_dft_out[None, :]

                        current_original_save_path = batch_save_folder+"/DFT_original_img.jpg"
                        std_orig_dft_image = recreate_image(
                            for_vis_dft_out, unnormalize=False)

                        save_image(std_orig_dft_image,
                                   current_original_save_path)

                    current_f_channel_norm_outs = None
                    current_f_channel_norm_dft_outs = None
                    for each_fil_channel_indx in range(len(each_filter_outs)):
                        each_fil_channel = each_filter_outs[each_fil_channel_indx]
                        each_fil_channel = each_fil_channel[None, :]

                        raw_dft_out = generate_centralized_DFT(
                            each_fil_channel[0])
                        for_vis_dft_out = np.log(1+np.abs(raw_dft_out))
                        std01_vis_dft_out = normalize_in_range_01(
                            for_vis_dft_out)
                        std01_vis_dft_out = std01_vis_dft_out[None, :]

                        if(current_f_channel_norm_dft_outs is None):
                            current_f_channel_norm_dft_outs = torch.from_numpy(
                                std01_vis_dft_out)
                        else:
                            current_f_channel_norm_dft_outs = torch.vstack(
                                (current_f_channel_norm_dft_outs, torch.from_numpy(std01_vis_dft_out)))

                        std01_filter_out_image = normalize_in_range_01(
                            each_fil_channel)
                        if(current_f_channel_norm_outs is None):
                            current_f_channel_norm_outs = torch.from_numpy(
                                std01_filter_out_image)
                        else:
                            current_f_channel_norm_outs = torch.vstack(
                                (current_f_channel_norm_outs, torch.from_numpy(std01_filter_out_image)))

                        std_filter_out_image = None
                        if(is_vis_ind_filter_out):
                            if not os.path.exists(current_save_folder):
                                os.makedirs(current_save_folder)
                            current_filt_channel_save_path = current_save_folder + \
                                "filter_out_channel_" + \
                                str(each_fil_channel_indx)+".jpg"
                            std_filter_out_image = recreate_image(
                                std01_filter_out_image, unnormalize=False, is_standarize_to_01=False)
                            save_image(std_filter_out_image,
                                       current_filt_channel_save_path)

                        if(is_vis_ind_filter_out_DFT):
                            if not os.path.exists(current_save_folder):
                                os.makedirs(current_save_folder)
                            current_dft_filt_channel_save_path = current_save_folder + \
                                "DFT_filter_out_channel_" + \
                                str(each_fil_channel_indx)+".jpg"
                            std_filter_out_dft_image = recreate_image(
                                std01_vis_dft_out, unnormalize=False, is_standarize_to_01=False)
                            save_image(std_filter_out_dft_image,
                                       current_dft_filt_channel_save_path)

                        if(is_print_ind_std_filter_out):
                            if not os.path.exists(current_save_folder):
                                os.makedirs(current_save_folder)
                            std_txt_save_folder = current_save_folder+"/std_filter_out.txt"
                            if(std_filter_out_image is None):
                                std_filter_out_image = recreate_image(
                                    std01_filter_out_image, unnormalize=False, is_standarize_to_01=False)
                            with open(std_txt_save_folder, "w") as f:
                                f.write("\n".join(
                                    ",".join(map(str, x)) for x in std_filter_out_image))

                        if(is_print_ind_raw_filter_out):
                            if not os.path.exists(current_save_folder):
                                os.makedirs(current_save_folder)
                            raw_txt_save_folder = current_save_folder+"/raw_filter_out.txt"
                            with open(raw_txt_save_folder, "w") as f:
                                f.write("\n".join(
                                    ",".join(map(str, x)) for x in each_fil_channel))

                        if(is_vis_ind_filter_out_diff_orig):
                            if not os.path.exists(current_save_folder):
                                os.makedirs(current_save_folder)
                            diff_fil_channel = each_fil_channel - orig_image
                            current_diff_save_path = current_save_folder + \
                                "filter_diff_" + \
                                str(each_fil_channel_indx)+".jpg"

                            std_diff_fil_channel = recreate_image(
                                diff_fil_channel, unnormalize=False)
                            save_image(std_diff_fil_channel,
                                       current_diff_save_path)

                    if(current_f_outs_norm is None):
                        current_f_outs_norm = torch.unsqueeze(
                            current_f_channel_norm_outs, 0)
                    else:
                        current_f_outs_norm = torch.vstack(
                            (current_f_outs_norm, torch.unsqueeze(current_f_channel_norm_outs, 0)))

                    if(current_f_outs_DFT_norm is None):
                        current_f_outs_DFT_norm = torch.unsqueeze(
                            current_f_channel_norm_dft_outs, 0)
                    else:
                        current_f_outs_DFT_norm = torch.vstack(
                            (current_f_outs_DFT_norm, torch.unsqueeze(current_f_channel_norm_dft_outs, 0)))

                if(current_layer_norm_outputs is None):
                    current_layer_norm_outputs = torch.unsqueeze(
                        current_f_outs_norm, 0)
                else:
                    current_layer_norm_outputs = torch.vstack(
                        (current_layer_norm_outputs, torch.unsqueeze(current_f_outs_norm, 0)))

                if(current_layer_DFT_norm_outputs is None):
                    current_layer_DFT_norm_outputs = torch.unsqueeze(
                        current_f_outs_DFT_norm, 0)
                else:
                    current_layer_DFT_norm_outputs = torch.vstack(
                        (current_layer_DFT_norm_outputs, torch.unsqueeze(current_f_outs_DFT_norm, 0)))

            current_layer_outputs = torch.transpose(
                current_layer_outputs, 0, 1)
            current_layer_norm_outputs = torch.transpose(
                current_layer_norm_outputs, 0, 1)
            current_layer_DFT_norm_outputs = torch.transpose(
                current_layer_DFT_norm_outputs, 0, 1)

            current_layer_HRelu_outputs = HardRelu()(current_layer_outputs.to(device))
            # print("current_layer_outputs size:", current_layer_outputs.size())
            # print("current_layer_norm_outputs size:",
            #       current_layer_norm_outputs.size())
            for each_b_indx in range(len(current_layer_outputs)):
                gr_batch_save_folder = grid_save_folder + \
                    "/BTCH_IND_" + str(each_b_indx)
                gr_current_save_folder = str(gr_batch_save_folder) + "/LAY_NUM_" + \
                    str(layer_num)
                if not os.path.exists(gr_current_save_folder):
                    os.makedirs(gr_current_save_folder)

                gr_current_b_fout_save_path = gr_current_save_folder + \
                    "/layer_level_std_gridded_filter_output.jpg"

                current_batch_layer_out = current_layer_outputs[each_b_indx]
                current_batch_layer_out = torch.squeeze(
                    current_batch_layer_out)
                # print("current_batch_layer_out size:",
                #       current_batch_layer_out.size())
                # print("gr_current_b_fout_save_path",
                #       gr_current_b_fout_save_path)
                generate_plain_image(
                    current_batch_layer_out, gr_current_b_fout_save_path, is_standarize=False)

                current_batch_layer_norm_out = current_layer_norm_outputs[each_b_indx]
                current_batch_layer_norm_out = torch.squeeze(
                    current_batch_layer_norm_out)

                gr_current_b_fout_save_path = gr_current_save_folder + \
                    "/lay_lev_individual_std_gridded_filter_output.jpg"

                generate_plain_image(
                    current_batch_layer_norm_out, gr_current_b_fout_save_path, is_standarize=False, is_standarize_01=False)

                current_batch_layer_DFT_norm_out = current_layer_DFT_norm_outputs[each_b_indx]
                current_batch_layer_DFT_norm_out = torch.squeeze(
                    current_batch_layer_DFT_norm_out)

                gr_current_b_fout_save_path = gr_current_save_folder + \
                    "/lay_lev_ind_std_DFT_filter_output.jpg"

                generate_plain_image(
                    current_batch_layer_DFT_norm_out, gr_current_b_fout_save_path, is_standarize=False, is_standarize_01=False)

                current_batch_layer_HRelu_outputs = current_layer_HRelu_outputs[each_b_indx]
                current_batch_layer_HRelu_outputs = torch.squeeze(
                    current_batch_layer_HRelu_outputs)

                gr_current_b_fout_save_path = gr_current_save_folder + \
                    "/lay_lev_HRelu_gridded_output.jpg"

                generate_plain_image(
                    current_batch_layer_HRelu_outputs, gr_current_b_fout_save_path, is_standarize=False)

        overall_indx_count += c_inputs.size()[0]
        if(not(num_batches_to_visualize is None) and batch_idx == num_batches_to_visualize - 1):
            break


def generate_dataset_from_images_folder(img_fold, reduce_dim_to_single, transform=None):
    class_list = os.listdir(img_fold)
    list_of_x = []
    list_of_y = []
    list_of_classes = []
    for c_ind in range(len(class_list)):
        c_label = class_list[c_ind]
        list_of_classes.append(c_label)
        class_path = os.path.join(img_fold, c_label)
        # print("class_path", class_path)
        img_list = os.listdir(class_path)
        num_imgs = len(img_list)
        for n in range(num_imgs):
            name = img_list[n]
            img_path = os.path.join(img_fold, c_label, name)
            # print("img_path", img_path)
            if os.path.isfile(img_path):
                if(reduce_dim_to_single):
                    im = cv2.imread(img_path, 0)
                else:
                    im = cv2.imread(img_path)

                if(transform is None):
                    im = torch.from_numpy(im)
                else:
                    # print("Before Image size:", im.shape)
                    im = transform(im)
                # print("Appended Image size:", im.size())
                list_of_x.append(im)
                list_of_y.append(c_ind)

    dataset = CustomSimpleDataset(list_of_x, list_of_y)

    return dataset, list_of_classes


def merge_conv_kernels(k1, k2):
    """
    :input k1: A tensor of shape ``(out1, in1, s1, s1)``
    :input k1: A tensor of shape ``(out2, in2, s2, s2)``
    :returns: A tensor of shape ``(out2, in1, s1+s2-1, s1+s2-1)``
      so that convolving with it equals convolving with k1 and
      then with k2.
    """
    if isinstance(k1, np.ndarray):
        k1 = torch.from_numpy(k1)
    if isinstance(k2, np.ndarray):
        k2 = torch.from_numpy(k2)
    padding = k2.shape[-1] - 1
    # Flip because this is actually correlation, and permute to adapt to BHCW
    k3 = torch.conv2d(k1.permute(1, 0, 2, 3), k2.flip(-1, -2),
                      padding=padding).permute(1, 0, 2, 3)
    return k3


def perform_sanity_check_over_merged_conv_filter(merged_conv, list_of_convs):
    b, c, h, w = 1, 1, 28, 28
    inp = torch.rand(b, c, h, w, dtype=torch.float32) * 10

    temp_seq_inp = inp.clone()
    for each_conv in list_of_convs:
        temp_seq_inp = torch.conv2d(
            temp_seq_inp, torch.from_numpy(each_conv), padding=each_conv.shape[-1]-1)

    seq_conv_out = temp_seq_inp

    temp_merged_inp = inp.clone()
    # merged_conv_out = torch.conv2d(
    #     temp_merged_inp, torch.from_numpy(merged_conv), padding=int(merged_conv.shape[-1]//2))
    merged_conv_out = torch.conv2d(
        temp_merged_inp, torch.from_numpy(merged_conv), padding=merged_conv.shape[-1]-1)

    print("merged_conv_out shape", merged_conv_out.size())
    print("seq_conv_out shape", seq_conv_out.size())
    print("seq_conv_out::::", seq_conv_out)
    print("merged_conv_out::::", merged_conv_out)


def generate_merged_convolution_weights_at_each_layer(list_of_weights):
    """
    f_hat(Current_layer,current_output_channel,current_input_channel) = Sum[(ou' from 0 to number of out-channels in current_layer)f_hat(current_layer-1,ou,current_input_channel) conv filt(current_layer , current_output_channel,ou)]
    """
    c_hat_sanity = [None] * len(list_of_weights)
    c_hat = [None]*len(list_of_weights)
    c_hat[0] = list_of_weights[0]
    c_hat_sanity[0] = list_of_weights[0]
    for current_lay in range(1, len(list_of_weights)):
        sanity_check_merged_conv = merge_conv_kernels(
            c_hat_sanity[current_lay-1], list_of_weights[current_lay]).numpy()
        c_hat_sanity[current_lay] = sanity_check_merged_conv
        # print("Layer:{} , shape:{} \n, sanity_check_merged_conv :{}".format(
        #     current_lay, sanity_check_merged_conv.shape, sanity_check_merged_conv))
        # perform_sanity_check_over_merged_conv_filter(
        # sanity_check_merged_conv, list_of_weights[0:current_lay+1])

    return c_hat_sanity


def get_modified_dataset(analyse_on, dataloader, adv_postfix_for_save, filter_vis_dataset, eval_dataset, batch_size, models_base_path, is_template_image_on_train, model, eps, adv_attack_type, number_of_adversarial_optimization_steps,
                         eps_step_size, adv_target, num_batches_to_visualize, is_save_adv):
    if(analyse_on == "ADVERSARIAL" or analyse_on == "ADVERSARIAL_PERTURB"):
        adv_save_path = models_base_path + "/RAW_ADV_SAVES/" + \
            filter_vis_dataset+"/" + adv_postfix_for_save+"/adv_dataset.npy"
        adv_dataset = load_or_generate_adv_examples(dataloader, models_base_path, is_template_image_on_train, model, eps, adv_attack_type, number_of_adversarial_optimization_steps,
                                                    eps_step_size, adv_target, number_of_batch_to_collect=num_batches_to_visualize, is_save_adv=is_save_adv, save_path=adv_save_path)
        adv_loader = torch.utils.data.DataLoader(
            adv_dataset, batch_size=batch_size, shuffle=False)

        if(analyse_on == "ADVERSARIAL_PERTURB"):
            coll_seed_gen = torch.Generator()
            coll_seed_gen.manual_seed(torch_seed)
            eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size,
                                                      shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)

            adv_dataset = generate_adversarial_perturbation_from_adv_orig(
                eval_loader, adv_loader)

    return adv_dataset


if __name__ == '__main__':
    # fashion_mnist , mnist , cifar10
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net , conv4_deep_gated_net_n16_small ,
    # conv4_deep_gated_net_with_actual_inp_in_wt_net , conv4_deep_gated_net_with_actual_inp_randomly_changed_in_wt_net
    # conv4_deep_gated_net_with_random_ones_in_wt_net , masked_conv4_dlgn , masked_conv4_dlgn_n16_small
    model_arch_type = 'plain_pure_conv4_dnn_n16_small'

    torch_seed = 2022

    # RAW_FILTERS_GEN , IMAGE_OUTPUTS_PER_FILTER , IMAGE_SEQ_OUTPUTS_PER_FILTER , IMAGE_OUT_PER_RES_FILTER
    # list_of_scheme_type = ["IMAGE_OUT_PER_RES_FILTER"]
    list_of_scheme_type = [
        "IMAGE_SEQ_OUTPUTS_PER_FILTER"]

    # std_image_preprocessing , mnist , fashion_mnist
    list_of_filter_vis_dataset = ["mnist"]

    batch_size = 14

    eps = 0.06
    adv_attack_type = 'PGD'
    number_of_adversarial_optimization_steps = 161
    eps_step_size = 0.01
    adv_target = None
    is_save_adv = True

    num_batches_to_visualize = 1

    # ORIGINAL, ADVERSARIAL , ADVERSARIAL_PERTURB
    analyse_on = "ADVERSARIAL_PERTURB"

    model_arch_type_str = model_arch_type
    mask_percentage = 0
    if("masked" in model_arch_type):
        mask_percentage = 50
        model_arch_type_str = model_arch_type_str + \
            "_PRC_"+str(mask_percentage)

    for filter_vis_dataset in list_of_filter_vis_dataset:
        print("Visualizing over " + str(filter_vis_dataset))
        if(filter_vis_dataset == "mnist"):
            batch_size = 4
            inp_channel = 1
            classes = [str(i) for i in range(0, 10)]
            num_classes = len(classes)

            mnist_config = DatasetConfig(
                'mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size)

            trainloader, _, testloader = preprocess_dataset_get_data_loader(
                mnist_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)
        elif(filter_vis_dataset == "cifar10"):
            inp_channel = 3
            classes = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            num_classes = len(classes)

            cifar10_config = DatasetConfig(
                'cifar10', is_normalize_data=False, valid_split_size=0.1, batch_size=batch_size)

            trainloader, _, testloader = preprocess_dataset_get_data_loader(
                cifar10_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)
        elif(filter_vis_dataset == "fashion_mnist"):
            batch_size = 4
            inp_channel = 1
            classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot')
            num_classes = len(classes)

            fashion_mnist_config = DatasetConfig(
                'fashion_mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size)

            trainloader, _, testloader = preprocess_dataset_get_data_loader(
                fashion_mnist_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)
        else:
            if(filter_vis_dataset == "std_image_preprocessing"):
                batch_size = 14
                inp_channel = 1
                reduce_dim_to_single = True
                transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize([512, 512]),
                    T.ToTensor()])
                train_ds_path = 'root/Datasets/std_image_preprocessing/train'
                test_ds_path = 'root/Datasets/std_image_preprocessing/test'

            train_dataset, classes = generate_dataset_from_images_folder(
                train_ds_path, reduce_dim_to_single, transform)
            test_dataset, classes = generate_dataset_from_images_folder(
                test_ds_path, reduce_dim_to_single, transform)

            print("train_dataset len", len(train_dataset))
            print("test_dataset len", len(test_dataset))
            print("classes", classes)

            num_classes = len(classes)
            trainloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False)
            testloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False)

        train_dataset = generate_dataset_from_loader(trainloader)
        test_dataset = generate_dataset_from_loader(testloader)

        for scheme_type in list_of_scheme_type:
            print("Running scheme", scheme_type)
            coll_seed_gen = torch.Generator()
            coll_seed_gen.manual_seed(torch_seed)
            orig_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                           shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)
            orig_testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                          shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)

            if(scheme_type != "RAW_FILTERS_GEN"):
                model_path = "root/model/save/mnist/V2_iterative_augmenting/DS_mnist/MT_plain_pure_conv4_dnn_n16_small_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.73/aug_conv4_dlgn_iter_1_dir.pt"
                model = get_model_from_path(
                    dataset, model_arch_type, model_path, mask_percentage=mask_percentage)

                save_prefix = get_prefix_for_save(
                    model_path, model_arch_type_str)

                models_base_path = model_path[0:model_path.rfind(".pt")]

            if(scheme_type == "RAW_FILTERS_GEN"):
                # IND , DIFF , START
                sub_scheme_type = 'IND'

                if(sub_scheme_type == 'IND'):
                    list_of_model_paths = []
                    models_base_path = None
                    # models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_deep_gated_net_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.93/"
                    list_of_model_paths = [
                        "root/model/save/mnist/adversarial_training/MT_conv4_dlgn_n16_small_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.06/batch_size_128/eps_stp_size_0.06/adv_steps_80/adv_model_dir.pt", "root/model/save/mnist/CLEAN_TRAINING/ST_2022/conv4_dlgn_n16_small_dir.pt"]
                    # models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/"
                    # models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_plain_pure_conv4_dnn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.91/"

                    num_iterations = 1
                    start_index = 1
                    for current_it_start in range(start_index, num_iterations + 1):
                        run_generate_raw_weight_analysis(
                            models_base_path, current_it_start, list_of_model_paths=list_of_model_paths)
                elif(sub_scheme_type == 'DIFF'):
                    model1_path = "root/model/save/mnist/CLEAN_TRAINING/ST_2022/conv4_deep_gated_net_n16_small_dir.pt"
                    model2_path = "root/model/save/mnist/adversarial_training/MT_conv4_deep_gated_net_n16_small_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.06/batch_size_128/eps_stp_size_0.06/adv_steps_80/adv_model_dir.pt"

                    run_generate_diff_raw_weight_analysis(
                        model1_path, model2_path)
                elif(sub_scheme_type == 'START'):
                    models_base_path = None
                    list_of_model_paths = []
                    num_iterations = 1
                    start_index = 1
                    for current_it_start in range(start_index, num_iterations + 1):
                        run_generate_raw_weight_analysis(
                            models_base_path, current_it_start, list_of_model_paths=list_of_model_paths)
            elif(scheme_type == "IMAGE_OUTPUTS_PER_FILTER"):
                # IND , DIFF , START
                sub_scheme_type = 'IND'

                if(sub_scheme_type == "IND"):

                    list_of_weights, list_of_bias = run_raw_weight_analysis_on_config(model, root_save_prefix=save_prefix, final_postfix_for_save="",
                                                                                      is_save_graph_visualizations=True)

                for is_template_image_on_train in [True, False]:
                    if(is_template_image_on_train):
                        evalloader = trainloader
                        final_postfix_for_save = 'TRAIN/'+analyse_on+"/"
                    else:
                        evalloader = testloader
                        final_postfix_for_save = "TEST/"+analyse_on+"/"
                    postfix_for_save = final_postfix_for_save
                    class_indx_to_visualize = [i for i in range(len(classes))]

                    if(len(class_indx_to_visualize) != 0):
                        input_data_list_per_class = true_segregation(
                            evalloader, num_classes)

                    for c_indx in class_indx_to_visualize:
                        class_label = classes[c_indx]
                        print(
                            "************************************************************ Class:", class_label)
                        per_class_dataset = PerClassDataset(
                            input_data_list_per_class[c_indx], c_indx)
                        if(analyse_on == "ADVERSARIAL" or analyse_on == "ADVERSARIAL_PERTURB"):
                            adv_postfix_for_save = "adv_type_{}/EPS_{}/eps_stp_size_{}/adv_steps_{}/on_train_{}/{}".format(
                                adv_attack_type, eps, eps_step_size, number_of_adversarial_optimization_steps, is_template_image_on_train, "")
                            postfix_for_save = final_postfix_for_save + adv_postfix_for_save

                            adv_postfix_for_save += "C_"+class_label+"/"
                            per_class_data_loader = torch.utils.data.DataLoader(
                                per_class_dataset, batch_size=batch_size, shuffle=False)

                            per_class_dataset = get_modified_dataset(analyse_on, per_class_data_loader, adv_postfix_for_save, filter_vis_dataset, per_class_dataset, batch_size, models_base_path, is_template_image_on_train, model, eps, adv_attack_type, number_of_adversarial_optimization_steps,
                                                                     eps_step_size, adv_target, num_batches_to_visualize, is_save_adv)

                        generate_filter_outputs_per_image(filter_vis_dataset, inp_channel, class_label, c_indx,
                                                          per_class_dataset, list_of_weights, save_prefix, num_batches_to_visualize,
                                                          final_postfix_for_save=postfix_for_save, scheme_type_tag="FILTER_OUTS")
            elif(scheme_type == "IMAGE_SEQ_OUTPUTS_PER_FILTER"):

                for is_template_image_on_train in [True, False]:
                    if(is_template_image_on_train):
                        evalloader = trainloader
                        final_postfix_for_save = 'TRAIN/'+analyse_on+"/"
                    else:
                        evalloader = testloader
                        final_postfix_for_save = "TEST/"+analyse_on+"/"
                    postfix_for_save = final_postfix_for_save
                    class_indx_to_visualize = [i for i in range(len(classes))]

                    if(len(class_indx_to_visualize) != 0):
                        input_data_list_per_class = true_segregation(
                            evalloader, num_classes)

                    for c_indx in class_indx_to_visualize:
                        class_label = classes[c_indx]
                        print(
                            "************************************************************ Class:", class_label)
                        per_class_dataset = PerClassDataset(
                            input_data_list_per_class[c_indx], c_indx)

                        if(analyse_on == "ADVERSARIAL" or analyse_on == "ADVERSARIAL_PERTURB"):
                            adv_postfix_for_save = "adv_type_{}/EPS_{}/eps_stp_size_{}/adv_steps_{}/on_train_{}/{}".format(
                                adv_attack_type, eps, eps_step_size, number_of_adversarial_optimization_steps, is_template_image_on_train, "")
                            postfix_for_save = final_postfix_for_save + adv_postfix_for_save

                            adv_postfix_for_save += "C_"+class_label+"/"
                            per_class_data_loader = torch.utils.data.DataLoader(
                                per_class_dataset, batch_size=batch_size, shuffle=False)

                            per_class_dataset = get_modified_dataset(analyse_on, per_class_data_loader, adv_postfix_for_save, filter_vis_dataset, per_class_dataset, batch_size, models_base_path, is_template_image_on_train, model, eps, adv_attack_type, number_of_adversarial_optimization_steps,
                                                                     eps_step_size, adv_target, num_batches_to_visualize, is_save_adv)

                        generate_seq_filter_outputs_per_image(model, filter_vis_dataset, class_label, c_indx,
                                                              per_class_dataset, save_prefix, num_batches_to_visualize,
                                                              final_postfix_for_save=postfix_for_save)
            elif(scheme_type == "IMAGE_OUT_PER_RES_FILTER"):
                sub_scheme_type = 'IND'

                if(sub_scheme_type == "IND"):
                    list_of_weights, list_of_bias = run_raw_weight_analysis_on_config(model, root_save_prefix=save_prefix, final_postfix_for_save="",
                                                                                      is_save_graph_visualizations=True)

                list_of_weights = generate_merged_convolution_weights_at_each_layer(
                    list_of_weights)

                output_params(list_of_weights, root_save_prefix=save_prefix,
                              final_postfix_for_save="MERGED_WEIGHTS")

                # for is_template_image_on_train in [True, False]:
                #     if(is_template_image_on_train):
                #         evalloader = trainloader
                #         final_postfix_for_save = 'TRAIN/'+analyse_on+"/"
                #     else:
                #         evalloader = testloader
                #         final_postfix_for_save = "TEST/"+analyse_on+"/"
                #     postfix_for_save = final_postfix_for_save
                #     class_indx_to_visualize = [i for i in range(len(classes))]

                #     if(len(class_indx_to_visualize) != 0):
                #         input_data_list_per_class = true_segregation(
                #             evalloader, num_classes)

                #     for c_indx in class_indx_to_visualize:
                #         class_label = classes[c_indx]
                #         print(
                #             "************************************************************ Class:", class_label)
                #         per_class_dataset = PerClassDataset(
                #             input_data_list_per_class[c_indx], c_indx)
                #         if(analyse_on == "ADVERSARIAL" or analyse_on == "ADVERSARIAL_PERTURB"):
                #             adv_postfix_for_save = "adv_type_{}/EPS_{}/eps_stp_size_{}/adv_steps_{}/on_train_{}/{}".format(
                #                 adv_attack_type, eps, eps_step_size, number_of_adversarial_optimization_steps, is_template_image_on_train, "")
                #             postfix_for_save = final_postfix_for_save + adv_postfix_for_save

                #             adv_postfix_for_save += "C_"+class_label+"/"
                #             per_class_data_loader = torch.utils.data.DataLoader(
                #                 per_class_dataset, batch_size=batch_size, shuffle=False)

                #             per_class_dataset = get_modified_dataset(analyse_on, per_class_data_loader, adv_postfix_for_save, filter_vis_dataset, per_class_dataset, batch_size, models_base_path, is_template_image_on_train, model, eps, adv_attack_type, number_of_adversarial_optimization_steps,
                #                                                      eps_step_size, adv_target, num_batches_to_visualize, is_save_adv)

                #         generate_filter_outputs_per_image(filter_vis_dataset, inp_channel, class_label, c_indx,
                #                                           per_class_dataset, list_of_weights, save_prefix, num_batches_to_visualize,
                #                                           final_postfix_for_save=postfix_for_save, scheme_type_tag="RES_FILT_OUTS")

    print("Finished execution!!!")

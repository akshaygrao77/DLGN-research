
import torch
import os
import numpy as np
from utils.weight_utils import get_gating_layer_weights
from utils.visualise_utils import generate_list_of_plain_images_from_data, generate_plain_image, generate_plain_image_data
from conv4_models import get_model_instance_from_dataset


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

    for i in range(len(list_of_weights)):
        current_weight_np = list_of_weights[i]

        print("Param {} size:{}".format(
            final_postfix_for_save, current_weight_np.shape))

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

    with open(txt_save_folder+"raw_params.txt", "w") as f:
        f.write("\n ************************************ Next Layer *********************************** \n".join(
            "\n -------------------- Next Filter ----------- \n".join(map(str, x)) for x in list_of_weights))

    with open(txt_save_folder+"formatted_raw_params.txt", "w") as f:
        f.write("\n ************************************ Next Layer *********************************** \n".join(
            "\n".join(map(str, generate_plain_image_data(np.squeeze(x)))) for x in list_of_weights))

    for i in range(len(list_of_weights)):

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
        generate_plain_image(
            current_weight_np, save_folder+"layer_num_"+str(i)+".jpg", is_standarize=False)


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
                "root/RAW_WEIGHT_ANALYSIS/"]
            list_of_save_postfixes = [None]
        else:
            list_of_save_prefixes = []
            list_of_save_postfixes = []
            for each_model_path in list_of_model_paths:
                base_path = each_model_path[0:each_model_path.rfind("/")+1]
                list_of_save_prefixes.append(
                    str(base_path)+"/"+str(model_arch_type)+"/RAW_WEIGHT_ANALYSIS/")
                list_of_save_postfixes.append("")

    for ind in range(len(list_of_model_paths)):
        each_model_path = list_of_model_paths[ind]
        each_save_prefix = list_of_save_prefixes[ind]
        each_save_postfix = list_of_save_postfixes[ind]
        analysed_model_path = each_model_path

        print("each_model_path", each_model_path)

        custom_model = get_model_instance_from_dataset(
            dataset, model_arch_type)

        custom_temp_model = torch.load(
            each_model_path, map_location=device)
        custom_model.load_state_dict(custom_temp_model.state_dict())

        print(" #*#*#*#*#*#*#*# Generating weights analysis for model path:{} with save prefix :{} and postfix:{}".format(
            each_model_path, each_save_prefix, each_save_postfix))
        with torch.no_grad():
            run_raw_weight_analysis_on_config(custom_model, root_save_prefix=each_save_prefix, final_postfix_for_save=each_save_postfix,
                                              is_save_graph_visualizations=True)


def run_generate_diff_raw_weight_analysis(model1_path, model2_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    custom1_model = get_model_instance_from_dataset(
        dataset, model_arch_type)

    custom_temp_model = torch.load(
        model1_path, map_location=device)
    custom1_model.load_state_dict(custom_temp_model.state_dict())

    custom2_model = get_model_instance_from_dataset(
        dataset, model_arch_type)

    custom_temp_model = torch.load(
        model2_path, map_location=device)
    custom2_model.load_state_dict(custom_temp_model.state_dict())

    base1_path = model1_path[0:model1_path.rfind("/")+1]
    base2_path = model2_path[0:model2_path.rfind("/")+1]

    save1_prefix = str(base1_path)+"/"+str(model_arch_type) + \
        "/RAW_WEIGHT_ANALYSIS/"
    save2_prefix = str(base2_path)+"/"+str(model_arch_type) + \
        "/RAW_WEIGHT_ANALYSIS/"

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


if __name__ == '__main__':
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net , conv4_deep_gated_net_n16_small ,
    # conv4_deep_gated_net_with_actual_inp_in_wt_net , conv4_deep_gated_net_with_actual_inp_randomly_changed_in_wt_net
    # conv4_deep_gated_net_with_random_ones_in_wt_net
    model_arch_type = 'conv4_deep_gated_net_n16_small'

    # IND , DIFF
    scheme_type = 'DIFF'

    if(scheme_type == 'IND'):
        list_of_model_paths = []
        models_base_path = None
        # models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_deep_gated_net_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.93/"
        list_of_model_paths = [
            "root/model/save/mnist/CLEAN_TRAINING/ST_2022/conv4_deep_gated_net_n16_small_dir.pt"]
        # models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/"
        # models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_plain_pure_conv4_dnn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.91/"

        num_iterations = 1
        start_index = 1
        for current_it_start in range(start_index, num_iterations + 1):
            run_generate_raw_weight_analysis(
                models_base_path, current_it_start, list_of_model_paths=list_of_model_paths)
    elif(scheme_type == 'DIFF'):
        model1_path = "root/model/save/mnist/CLEAN_TRAINING/ST_2022/conv4_deep_gated_net_n16_small_dir.pt"
        model2_path = "root/model/save/mnist/adversarial_training/MT_conv4_deep_gated_net_n16_small_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.06/batch_size_128/eps_stp_size_0.06/adv_steps_80/adv_model_dir.pt"

        run_generate_diff_raw_weight_analysis(model1_path, model2_path)

    print("Finished execution!!!")

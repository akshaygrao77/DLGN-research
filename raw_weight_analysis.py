
import torch
import os
import numpy as np
from utils.weight_utils import get_gating_layer_weights
from utils.visualise_utils import generate_list_of_plain_images_from_data, generate_plain_image, generate_plain_image_data, save_image, recreate_image
from conv4_models import get_model_instance_from_dataset
from utils.data_preprocessing import preprocess_dataset_get_data_loader, generate_dataset_from_loader, seed_worker, segregate_classes
from structure.generic_structure import PerClassDataset
from structure.dlgn_conv_config_structure import DatasetConfig
from tqdm import tqdm


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
                "root/RAW_WEIGHT_ANALYSIS/MT_"+str(model_arch_type)+"/"]
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
            dataset, model_arch_type, torch_seed)

        if(each_model_path is not None):
            custom_temp_model = torch.load(
                each_model_path, map_location=device)
            custom_model.load_state_dict(custom_temp_model.state_dict())

        print(" #*#*#*#*#*#*#*# Generating weights analysis for model path:{} with save prefix :{} and postfix:{}".format(
            each_model_path, each_save_prefix, each_save_postfix))
        with torch.no_grad():
            run_raw_weight_analysis_on_config(custom_model, root_save_prefix=each_save_prefix, final_postfix_for_save=each_save_postfix,
                                              is_save_graph_visualizations=True)


def get_model_from_path(dataset, model_arch_type, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_model = get_model_instance_from_dataset(
        dataset, model_arch_type)

    custom_temp_model = torch.load(
        model_path, map_location=device)
    custom_model.load_state_dict(custom_temp_model.state_dict())
    custom_model = custom_model.to(device)
    return custom_model


def get_prefix_for_save(model_path, model_arch_type):
    base_path = model_path[0:model_path.rfind("/")+1]

    save_prefix = str(base_path)+"/"+str(model_arch_type) + \
        "/RAW_WEIGHT_ANALYSIS/"
    return save_prefix


def run_generate_diff_raw_weight_analysis(model1_path, model2_path):

    custom1_model = get_model_from_path(dataset, model_arch_type, model1_path)

    custom2_model = get_model_from_path(dataset, model_arch_type, model2_path)

    save1_prefix = get_prefix_for_save(model1_path, model_arch_type)
    save2_prefix = get_prefix_for_save(model2_path, model_arch_type)

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


def generate_per_batch_filter_outs(filter_weights, batch_inputs):
    # print("filter_weights before size", filter_weights.shape)
    filter_weights = np.expand_dims(filter_weights, axis=1)
    # print("filter_weights after size", filter_weights.shape)
    filter_weights = torch.from_numpy(filter_weights)
    # print("filter_weights size", filter_weights.size())
    conv_obj = torch.nn.Conv2d(1, filter_weights.size()[
        0], filter_weights.size()[-1], padding=1)
    # print("conv_obj.weight size", conv_obj.weight.size())
    conv_obj.weight = torch.nn.Parameter(filter_weights)
    # print("filter_weights", filter_weights)
    # print("conv_obj.weight size", conv_obj.weight.size())
    # print("conv_obj.weight", conv_obj.weight)
    filter_out = conv_obj(batch_inputs)
    # print("filter_out size", filter_out.size())
    return filter_out


def generate_filter_outputs_per_image(filter_vis_dataset, class_label, c_indx,
                                      per_class_dataset, list_of_weights, save_prefix, num_batches_to_visualize, final_postfix_for_save):

    save_folder = save_prefix + "/DS_" + str(filter_vis_dataset)+"/" + \
        str(final_postfix_for_save)+"/FILTER_OUTS/C_"+str(class_label)+"/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    per_class_data_loader = torch.utils.data.DataLoader(
        per_class_dataset, batch_size=batch_size, shuffle=False)

    per_class_data_loader = tqdm(
        per_class_data_loader, desc='Outing filter outputs for class:'+str(class_label))
    overall_indx_count = 0
    for batch_idx, per_class_per_batch_data in enumerate(per_class_data_loader):
        c_inputs, _ = per_class_per_batch_data

        for layer_num in range(len(list_of_weights)):
            current_layer_weights = list_of_weights[layer_num]

            for filter_ind in range(len(current_layer_weights)):
                current_filter_weights = current_layer_weights[filter_ind]

                print("Layer_num:{} shape:{} =>Filter ind:{}=> shape:{}".format(
                    layer_num, current_layer_weights.shape, filter_ind, current_filter_weights.shape))

                f_outs = generate_per_batch_filter_outs(
                    current_filter_weights, c_inputs)

                for fil_ind in range(len(f_outs)):
                    each_filter_outs = f_outs[fil_ind]
                    batch_save_folder = save_folder + \
                        "/BTCH_IND_" + str(fil_ind)
                    current_save_folder = str(batch_save_folder) + "/LAY_NUM_" + \
                        str(layer_num)+"/FILT_IND_"+str(filter_ind) + "/"

                    if not os.path.exists(current_save_folder):
                        os.makedirs(current_save_folder)

                    current_original_save_path = batch_save_folder+"/original_img.jpg"
                    orig_image = c_inputs[fil_ind]
                    orig_image = orig_image[None, :]
                    std_orig_image = recreate_image(
                        orig_image, unnormalize=False)

                    save_image(std_orig_image, current_original_save_path)

                    for each_fil_channel_indx in range(len(each_filter_outs)):
                        each_fil_channel = each_filter_outs[each_fil_channel_indx]
                        each_fil_channel = each_fil_channel[None, :]
                        current_filt_channel_save_path = current_save_folder + \
                            "filter_out_channel_" + \
                            str(each_fil_channel_indx)+".jpg"

                        std_filter_out_image = recreate_image(
                            each_fil_channel, unnormalize=False)
                        save_image(std_filter_out_image,
                                   current_filt_channel_save_path)

                        diff_fil_channel = each_fil_channel - orig_image
                        current_diff_save_path = current_save_folder + \
                            "filter_diff_" + \
                            str(each_fil_channel_indx)+".jpg"

                        std_diff_fil_channel = recreate_image(
                            diff_fil_channel, unnormalize=False)
                        save_image(std_diff_fil_channel,
                                   current_diff_save_path)

        overall_indx_count += c_inputs.size()[0]
        if(not(num_batches_to_visualize is None) and batch_idx == num_batches_to_visualize - 1):
            break


if __name__ == '__main__':
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net , conv4_deep_gated_net_n16_small ,
    # conv4_deep_gated_net_with_actual_inp_in_wt_net , conv4_deep_gated_net_with_actual_inp_randomly_changed_in_wt_net
    # conv4_deep_gated_net_with_random_ones_in_wt_net
    model_arch_type = 'plain_pure_conv4_dnn_n16_small'

    torch_seed = 2022

    # RAW_FILTERS_GEN , IMAGE_OUTPUTS_PER_FILTER
    scheme_type = "IMAGE_OUTPUTS_PER_FILTER"

    if(scheme_type == "RAW_FILTERS_GEN"):
        # IND , DIFF , START
        sub_scheme_type = 'START'

        if(sub_scheme_type == 'IND'):
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
        elif(sub_scheme_type == 'DIFF'):
            model1_path = "root/model/save/mnist/CLEAN_TRAINING/ST_2022/conv4_deep_gated_net_n16_small_dir.pt"
            model2_path = "root/model/save/mnist/adversarial_training/MT_conv4_deep_gated_net_n16_small_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.06/batch_size_128/eps_stp_size_0.06/adv_steps_80/adv_model_dir.pt"

            run_generate_diff_raw_weight_analysis(model1_path, model2_path)
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

        filter_vis_dataset = "mnist"
        num_batches_to_visualize = 1
        batch_size = 32

        coll_seed_gen = torch.Generator()
        coll_seed_gen.manual_seed(torch_seed)

        model_path = "root/model/save/mnist/CLEAN_TRAINING/ST_2022/plain_pure_conv4_dnn_n16_small_dir.pt"
        model = get_model_from_path(dataset, model_arch_type, model_path)

        save_prefix = get_prefix_for_save(model_path, model_arch_type)

        if(sub_scheme_type == "IND"):

            list_of_weights, list_of_bias = run_raw_weight_analysis_on_config(model, root_save_prefix=save_prefix, final_postfix_for_save="",
                                                                              is_save_graph_visualizations=False)

        if(filter_vis_dataset == "mnist"):
            inp_channel = 1
            print("Training over MNIST")
            classes = [str(i) for i in range(0, 10)]
            num_classes = len(classes)

            mnist_config = DatasetConfig(
                'mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size)

            trainloader, _, testloader = preprocess_dataset_get_data_loader(
                mnist_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

        train_dataset = generate_dataset_from_loader(trainloader)
        test_dataset = generate_dataset_from_loader(testloader)

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                 shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)

        for is_template_image_on_train in [True, False]:
            if(is_template_image_on_train):
                final_postfix_for_save = 'TRAIN'
            else:
                final_postfix_for_save = "TEST"
            class_indx_to_visualize = [i for i in range(len(classes))]

            if(len(class_indx_to_visualize) != 0):
                input_data_list_per_class = segregate_classes(
                    model, trainloader, testloader, num_classes, is_template_image_on_train, is_class_segregation_on_ground_truth=True)

            for c_indx in class_indx_to_visualize:
                class_label = classes[c_indx]
                print(
                    "************************************************************ Class:", class_label)
                per_class_dataset = PerClassDataset(
                    input_data_list_per_class[c_indx], c_indx)
                generate_filter_outputs_per_image(filter_vis_dataset, class_label, c_indx,
                                                  per_class_dataset, list_of_weights, save_prefix, num_batches_to_visualize,
                                                  final_postfix_for_save=final_postfix_for_save)

    print("Finished execution!!!")

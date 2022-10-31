import torch
import os
from tqdm import tqdm
import wandb
import random
import numpy as np
import pickle


from utils.visualise_utils import save_image, recreate_image, add_lower_dimension_vectors_within_itself, construct_images_from_feature_maps, construct_heatmaps_from_data, determine_row_col_from_features, generate_plain_image
from utils.data_preprocessing import preprocess_dataset_get_data_loader, generate_dataset_from_loader
from configs.dlgn_conv_config import HardRelu
from utils.data_preprocessing import preprocess_dataset_get_data_loader, segregate_classes
from structure.generic_structure import PerClassDataset
from model.model_loader import get_model_from_loader
from configs.generic_configs import get_preprocessing_and_other_configs
from adversarial_attacks_tester import generate_adv_examples
from utils.generic_utils import save_dataset_into_path_from_loader
from structure.generic_structure import CustomSimpleDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_id + worker_seed)
    random.seed(worker_id - worker_seed)


def get_wandb_config(exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_act_collection_on_train,
                     is_class_segregation_on_ground_truth,
                     activation_calculation_batch_size, torch_seed, analysed_model_path,
                     number_of_batch_to_collect=None, collect_threshold=None):

    wandb_config = dict()
    wandb_config["class_label"] = class_label
    wandb_config["class_indx"] = class_indx
    wandb_config["classes"] = classes
    wandb_config["model_arch_type"] = model_arch_type
    wandb_config["dataset"] = dataset
    wandb_config["is_act_collection_on_train"] = is_act_collection_on_train
    wandb_config["is_class_segregation_on_ground_truth"] = is_class_segregation_on_ground_truth
    wandb_config["torch_seed"] = torch_seed
    wandb_config["analysed_model_path"] = analysed_model_path
    wandb_config["exp_type"] = exp_type
    wandb_config["activation_calculation_batch_size"] = activation_calculation_batch_size
    if(not(number_of_batch_to_collect is None)):
        wandb_config["number_of_batch_to_collect"] = number_of_batch_to_collect
    if(not(collect_threshold is None)):
        wandb_config["collect_threshold"] = collect_threshold

    return wandb_config


def print_array(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            print(arr[i][j], end=" ")
        print("")


class ActivationAnalyser():

    def __init__(self, model):
        self.model = model
        if(model is not None):
            self.model.eval()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.activation__save_prefix_folder = "root/activation_analysis/"

        self.reset_analyser_state()

    def reset_analyser_state(self):
        self.total_tcollect_img_count = 0
        # Count based states
        # Size: [Num layers,Num filter in current layer, W , H]
        self.active_counts_activation_map_list = None
        self.inactive_counts_activation_map_list = None
        self.active_inactive_diff_activation_map_list = None

        # Indicator based states
        # Size: [Num layers,Num filter in current layer, W , H]
        self.active_thresholded_indicator_activation_map_list = None
        self.inactive_thresholded_indicator_activation_map_list = None
        self.overall_thresholded_active_inactive_indicator_map_list = None

        # Statistics based states
        # Size: [Num layers,Num filter in current layer, W , H]
        self.mean_per_pixel_of_activations_map_list = None
        self.std_per_pixel_of_activations_map_list = None
        self.min_per_pixel_of_activations_map_list = None
        self.max_per_pixel_of_activations_map_list = None

        # Size: [Num layers,Num filter in current layer]
        self.avg_mean_per_activations_map_list = None
        self.avg_std_per_activations_map_list = None
        self.avg_min_per_activations_map_list = None
        self.avg_max_per_activations_map_list = None

        # Percent of active based states
        self.total_pixels = 0
        self.thresholded_active_pixel_count = 0
        self.thresholded_active_pixels_percentage = 0.

        self.unthresholded_active_pixel_count = 0
        self.unthresholded_active_pixels_percentage = 0.

        self.overall_average_active_percentage = 0.
        self.overall_std_active_percentage = 0.
        self.overall_min_active_percentage = None
        self.overall_max_active_percentage = None

    def diff_merge_with_another_activation(self, other_activation_state, root_save_prefix, final_postfix_for_save, wandb_group_name, wand_project_name=None, is_save_graph_visualizations=True, save_only_thres=False):
        merged_act_analyser = ActivationAnalyser(None)
        num_layers = len(self.mean_per_pixel_of_activations_map_list)
        merged_act_analyser.initialise_record_states(num_layers)

        for current_mean_per_pixel_activation_map in self.mean_per_pixel_of_activations_map_list:
            zero_initialised_current_activation_map = torch.zeros(
                size=current_mean_per_pixel_activation_map.size(), device=self.device)

            merged_act_analyser.active_counts_activation_map_list.append(
                zero_initialised_current_activation_map)
            merged_act_analyser.inactive_counts_activation_map_list.append(
                zero_initialised_current_activation_map.clone())

        merged_act_analyser.wandb_config = dict()

        merged_act_analyser.wandb_group_name = wandb_group_name
        if(wandb_group_name is None):
            merged_act_analyser.wandb_group_name = self.wandb_group_name + \
                "_*_" + other_activation_state.wandb_group_name

        merged_act_analyser.image_save_prefix_folder = self.image_save_prefix_folder.replace(self.final_postfix_for_save, "") + "/"+str(root_save_prefix) + "/" +\
            other_activation_state.image_save_prefix_folder + \
            "/"+str(final_postfix_for_save)

        wandb_run_name = merged_act_analyser.image_save_prefix_folder.replace(
            "/", "").replace(self.root_save_prefix, self.class_label).replace(other_activation_state.root_save_prefix, other_activation_state.class_label)

        merged_act_analyser.wandb_run_name = wandb_run_name

        merged_act_analyser.wandb_config["merge_type"] = merge_type
        for each_key in self.wandb_config:
            merged_act_analyser.wandb_config[each_key +
                                             "_1"] = self.wandb_config[each_key]

        for each_key in other_activation_state.wandb_config:
            merged_act_analyser.wandb_config[each_key +
                                             "_2"] = other_activation_state.wandb_config[each_key]

        with torch.no_grad():
            for indx in range(len(self.active_counts_activation_map_list)):
                merged_act_analyser.active_counts_activation_map_list[indx] = self.active_counts_activation_map_list[
                    indx] - other_activation_state.active_counts_activation_map_list[indx]
                merged_act_analyser.inactive_counts_activation_map_list[indx] = self.inactive_counts_activation_map_list[
                    indx] - other_activation_state.inactive_counts_activation_map_list[indx]
                merged_act_analyser.active_inactive_diff_activation_map_list[indx] = self.active_inactive_diff_activation_map_list[
                    indx] - other_activation_state.active_inactive_diff_activation_map_list[indx]

                merged_act_analyser.active_thresholded_indicator_activation_map_list[indx] = self.active_thresholded_indicator_activation_map_list[
                    indx] * other_activation_state.active_thresholded_indicator_activation_map_list[indx]
                merged_act_analyser.inactive_thresholded_indicator_activation_map_list[indx] = self.inactive_thresholded_indicator_activation_map_list[
                    indx] * other_activation_state.inactive_thresholded_indicator_activation_map_list[indx]
                merged_act_analyser.overall_thresholded_active_inactive_indicator_map_list[indx] = merged_act_analyser.active_thresholded_indicator_activation_map_list[
                    indx] - merged_act_analyser.inactive_thresholded_indicator_activation_map_list[indx]

                merged_act_analyser.mean_per_pixel_of_activations_map_list[indx] = self.mean_per_pixel_of_activations_map_list[
                    indx] - other_activation_state.mean_per_pixel_of_activations_map_list[indx]
                merged_act_analyser.std_per_pixel_of_activations_map_list[indx] = self.std_per_pixel_of_activations_map_list[
                    indx] - other_activation_state.std_per_pixel_of_activations_map_list[indx]
                merged_act_analyser.min_per_pixel_of_activations_map_list[indx] = self.min_per_pixel_of_activations_map_list[
                    indx] - other_activation_state.min_per_pixel_of_activations_map_list[indx]
                merged_act_analyser.max_per_pixel_of_activations_map_list[indx] = self.max_per_pixel_of_activations_map_list[
                    indx] - other_activation_state.max_per_pixel_of_activations_map_list[indx]

                merged_act_analyser.avg_mean_per_activations_map_list[indx] = self.avg_mean_per_activations_map_list[
                    indx] - other_activation_state.avg_mean_per_activations_map_list[indx]
                merged_act_analyser.avg_std_per_activations_map_list[indx] = self.avg_std_per_activations_map_list[
                    indx] - other_activation_state.avg_std_per_activations_map_list[indx]
                merged_act_analyser.avg_min_per_activations_map_list[indx] = self.avg_min_per_activations_map_list[
                    indx] - other_activation_state.avg_min_per_activations_map_list[indx]
                merged_act_analyser.avg_max_per_activations_map_list[indx] = self.avg_max_per_activations_map_list[
                    indx] - other_activation_state.avg_max_per_activations_map_list[indx]

            merged_act_analyser.total_pixels = self.total_pixels - \
                other_activation_state.total_pixels
            merged_act_analyser.thresholded_active_pixel_count = self.thresholded_active_pixel_count - \
                other_activation_state.thresholded_active_pixel_count
            merged_act_analyser.thresholded_active_pixels_percentage = self.thresholded_active_pixels_percentage - \
                other_activation_state.thresholded_active_pixels_percentage
            merged_act_analyser.unthresholded_active_pixel_count = self.unthresholded_active_pixel_count - \
                other_activation_state.unthresholded_active_pixel_count
            merged_act_analyser.unthresholded_active_pixels_percentage = self.unthresholded_active_pixels_percentage - \
                other_activation_state.unthresholded_active_pixels_percentage
            merged_act_analyser.overall_average_active_percentage = self.overall_average_active_percentage - \
                other_activation_state.overall_average_active_percentage
            merged_act_analyser.overall_std_active_percentage = self.overall_std_active_percentage - \
                other_activation_state.overall_std_active_percentage
            merged_act_analyser.overall_min_active_percentage = self.overall_min_active_percentage - \
                other_activation_state.overall_min_active_percentage
            merged_act_analyser.overall_max_active_percentage = self.overall_max_active_percentage - \
                other_activation_state.overall_max_active_percentage

        merged_act_analyser.root_save_prefix = root_save_prefix
        merged_act_analyser.final_postfix_for_save = final_postfix_for_save
        merged_act_analyser.class_label = self.class_label

        save_folder = merged_act_analyser.image_save_prefix_folder + \
            "class_"+str(merged_act_analyser.class_label)+"/"

        temp_model = merged_act_analyser.model
        merged_act_analyser.model = None
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        print("Save directory:", save_folder)
        with open(save_folder+'/analyser_state.pkl', 'wb') as out_file:
            pickle.dump(merged_act_analyser, out_file)
        merged_act_analyser.model = temp_model

        merged_act_analyser.save_and_log_states(
            wand_project_name, is_save_graph_visualizations=is_save_graph_visualizations, save_only_thres=save_only_thres)

        return merged_act_analyser

    def normalize_data(self, input):

        for indx in range(len(input)):
            input[indx] -= input[indx].mean(0)
            input[indx] /= input[indx].std(0)

        return input

    def shift_to_01_range(self, input):
        arr_max = None
        arr_min = None
        for each_input in input:
            current_max = torch.max(each_input).item()
            current_min = torch.min(each_input).item()
            if(arr_max is None):
                arr_max = current_max
            else:
                if(current_max > arr_max):
                    arr_max = current_max

            if(arr_min is None):
                arr_min = current_min
            else:
                if(current_min < arr_min):
                    arr_min = current_min

        for indx in range(len(input)):
            input[indx] = (
                input[indx]-arr_min)/(arr_max-arr_min)

        return input

    def standarize_all_recorded_activation_states(self):
        cpudevice = torch.device("cpu")

        self.active_thresholded_indicator_activation_map_list = self.shift_to_01_range(
            self.active_thresholded_indicator_activation_map_list)
        self.active_counts_activation_map_list = self.shift_to_01_range(
            self.active_counts_activation_map_list)
        self.active_inactive_diff_activation_map_list = self.shift_to_01_range(
            self.active_inactive_diff_activation_map_list)
        # print("Before mean_per_pixel_of_activations_map_list")
        # print_array(self.mean_per_pixel_of_activations_map_list[0])
        self.mean_per_pixel_of_activations_map_list = self.normalize_data(
            self.mean_per_pixel_of_activations_map_list)
        # print("After normalization mean_per_pixel_of_activations_map_list")
        # print_array(self.mean_per_pixel_of_activations_map_list[0])
        self.mean_per_pixel_of_activations_map_list = self.shift_to_01_range(
            self.mean_per_pixel_of_activations_map_list)
        # print("After mean_per_pixel_of_activations_map_list")
        # print_array(self.mean_per_pixel_of_activations_map_list[0])
        # temp = self.mean_per_pixel_of_activations_map_list[0] * 255
        # print("Before final mean_per_pixel_of_activations_map_list")
        # print_array(temp)
        # temp = np.clip(
        #     temp.to(cpudevice, non_blocking=True).numpy(), 0, 255).astype('uint8')
        # print("Final mean_per_pixel_of_activations_map_list", temp.shape)
        # print_array(temp)
        self.std_per_pixel_of_activations_map_list = self.normalize_data(
            self.std_per_pixel_of_activations_map_list)
        self.std_per_pixel_of_activations_map_list = self.normalize_data(
            self.std_per_pixel_of_activations_map_list)
        self.min_per_pixel_of_activations_map_list = self.normalize_data(
            self.min_per_pixel_of_activations_map_list)
        self.max_per_pixel_of_activations_map_list = self.normalize_data(
            self.max_per_pixel_of_activations_map_list)
        self.avg_mean_per_activations_map_list = self.normalize_data(
            self.avg_mean_per_activations_map_list)
        self.avg_std_per_activations_map_list = self.normalize_data(
            self.avg_std_per_activations_map_list)
        self.avg_min_per_activations_map_list = self.normalize_data(
            self.avg_min_per_activations_map_list)
        self.avg_max_per_activations_map_list = self.normalize_data(
            self.avg_max_per_activations_map_list)

        self.std_per_pixel_of_activations_map_list = self.shift_to_01_range(
            self.std_per_pixel_of_activations_map_list)
        self.std_per_pixel_of_activations_map_list = self.shift_to_01_range(
            self.std_per_pixel_of_activations_map_list)
        self.min_per_pixel_of_activations_map_list = self.shift_to_01_range(
            self.min_per_pixel_of_activations_map_list)
        self.max_per_pixel_of_activations_map_list = self.shift_to_01_range(
            self.max_per_pixel_of_activations_map_list)
        self.avg_mean_per_activations_map_list = self.shift_to_01_range(
            self.avg_mean_per_activations_map_list)
        self.avg_std_per_activations_map_list = self.shift_to_01_range(
            self.avg_std_per_activations_map_list)
        self.avg_min_per_activations_map_list = self.shift_to_01_range(
            self.avg_min_per_activations_map_list)
        self.avg_max_per_activations_map_list = self.shift_to_01_range(
            self.avg_max_per_activations_map_list)

    def save_recorded_activation_states_as_hm(self, base_save_folder, save_only_thres=False):
        cpudevice = torch.device("cpu")
        # For each conv layer
        for indx in range(len(self.active_counts_activation_map_list)):
            print("Saving visualization for layer:", indx)
            dict_full_path_to_saves = dict()

            final_save_dir = base_save_folder+"/layer_{}/".format(indx)
            if not os.path.exists(final_save_dir):
                os.makedirs(final_save_dir)

            current_active_thresholded_indicator_activation_map = self.active_thresholded_indicator_activation_map_list[
                indx].to(cpudevice, non_blocking=True).numpy()

            current_full_save_path = final_save_dir + \
                "thres_active_indicators_over_activation_map.jpg"
            dict_full_path_to_saves["thres_active_indictor_ov_act_map"] = current_full_save_path
            construct_heatmaps_from_data(current_active_thresholded_indicator_activation_map,
                                         title='Thresholded active pixel indicatoes over activation map', save_path=current_full_save_path)

            if(save_only_thres == False):
                current_active_counts_activation_map = self.active_counts_activation_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_active_inactive_diff_activation_map = self.active_inactive_diff_activation_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()

                current_mean_per_pixel_of_activations_map = self.mean_per_pixel_of_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_std_per_pixel_of_activations_map = self.std_per_pixel_of_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_min_per_pixel_of_activations_map = self.min_per_pixel_of_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_max_per_pixel_of_activations_map = self.max_per_pixel_of_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()

                current_avg_mean_per_activations_map = self.avg_mean_per_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_avg_std_per_activations_map = self.avg_std_per_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_avg_min_per_activations_map = self.avg_min_per_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_avg_max_per_activations_map = self.avg_max_per_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()

                current_full_save_path = final_save_dir+"active_counts_over_activation_map.jpg"
                dict_full_path_to_saves["active_count_ov_act_map"] = current_full_save_path
                construct_heatmaps_from_data(current_active_counts_activation_map, title='Active counts over activation map',
                                             save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "active_inact_diff_counts_over_activation_map.jpg"
                dict_full_path_to_saves["act_inact_diff_count_ov_act_map"] = current_full_save_path
                construct_heatmaps_from_data(current_active_inactive_diff_activation_map, title='Active - inactive difference counts over activation map',
                                             save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "mean_per_pixel_over_activation_map.jpg"
                dict_full_path_to_saves["mean_per_pxl_ov_act_map"] = current_full_save_path
                construct_heatmaps_from_data(
                    current_mean_per_pixel_of_activations_map, title="Mean per pixel over activation map", save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "std_dev_per_pixel_over_activation_map.jpg"
                dict_full_path_to_saves["std_dev_per_pxl_ov_act_map"] = current_full_save_path
                construct_heatmaps_from_data(
                    current_std_per_pixel_of_activations_map, title="Standard deviation per pixel over activation map", save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "min_per_pixel_over_activation_map.jpg"
                dict_full_path_to_saves["min_per_pxl_ov_act_map"] = current_full_save_path
                construct_heatmaps_from_data(
                    current_min_per_pixel_of_activations_map, title="Min per pixel over activation map", save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "max_per_pixel_over_activation_map.jpg"
                dict_full_path_to_saves["max_per_pxl_ov_act_map"] = current_full_save_path
                construct_heatmaps_from_data(
                    current_max_per_pixel_of_activations_map, title="Max per pixel over activation map", save_path=current_full_save_path)

                # Obtain the resize shape for num_filters dimension
                r, c = determine_row_col_from_features(
                    current_avg_mean_per_activations_map.shape[0])
                current_avg_mean_per_activations_map = np.reshape(
                    current_avg_mean_per_activations_map, (1, r, c))
                current_avg_std_per_activations_map = np.reshape(
                    current_avg_std_per_activations_map, (1, r, c))
                current_avg_min_per_activations_map = np.reshape(
                    current_avg_min_per_activations_map, (1, r, c))
                current_avg_max_per_activations_map = np.reshape(
                    current_avg_max_per_activations_map, (1, r, c))

                current_full_save_path = final_save_dir + \
                    "avg_mean_per_act_map.jpg"
                dict_full_path_to_saves["avg_mean_per_act_map"] = current_full_save_path
                construct_heatmaps_from_data(current_avg_mean_per_activations_map,
                                             title='Average of mean per activation map', save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "avg_std_per_act_map.jpg"
                dict_full_path_to_saves["avg_std_per_act_map"] = current_full_save_path
                construct_heatmaps_from_data(current_avg_std_per_activations_map,
                                             title='Average of std deviation per activation map', save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "avg_min_per_act_map.jpg"
                dict_full_path_to_saves["avg_min_per_act_map"] = current_full_save_path
                construct_heatmaps_from_data(current_avg_min_per_activations_map,
                                             title='Average of min per activation map', save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "avg_max_per_act_map.jpg"
                dict_full_path_to_saves["avg_max_per_act_map"] = current_full_save_path
                construct_heatmaps_from_data(current_avg_max_per_activations_map,
                                             title='Average of max per activation map', save_path=current_full_save_path)

            print("dict_full_path_to_saves in iteration {}=>{}".format(
                indx, dict_full_path_to_saves))

            # The image sizes are quite big and wandb might go out of space if this is enabled. So just local copy is enough
            # if(is_log_wandb == True):
            #     image_log_dict = dict()
            #     for each_key in dict_full_path_to_saves:
            #         print("Saving image with key:{} to wandb".format(each_key))
            #         full_path = dict_full_path_to_saves[each_key]
            #         image_log_dict[each_key] = wandb.Image(full_path)

            #     wandb.log(image_log_dict)

        return

    def save_recorded_activation_states(self, base_save_folder, save_only_thres=False):
        cpudevice = torch.device("cpu")
        # For each conv layer
        for indx in range(len(self.active_counts_activation_map_list)):
            print("Saving visualization for layer:", indx)
            dict_full_path_to_saves = dict()

            final_save_dir = base_save_folder+"/layer_{}/plain/".format(indx)
            if not os.path.exists(final_save_dir):
                os.makedirs(final_save_dir)

            current_active_thresholded_indicator_activation_map = self.active_thresholded_indicator_activation_map_list[
                indx].to(cpudevice, non_blocking=True).numpy()

            current_full_save_path = final_save_dir + \
                "thres_active_indicators_over_activation_map_im.jpg"
            dict_full_path_to_saves["thres_active_indictor_ov_act_map"] = current_full_save_path
            generate_plain_image(
                current_active_thresholded_indicator_activation_map, save_path=current_full_save_path)

            if(save_only_thres == False):
                current_active_counts_activation_map = self.active_counts_activation_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_active_inactive_diff_activation_map = self.active_inactive_diff_activation_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()

                current_mean_per_pixel_of_activations_map = self.mean_per_pixel_of_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_std_per_pixel_of_activations_map = self.std_per_pixel_of_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_min_per_pixel_of_activations_map = self.min_per_pixel_of_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_max_per_pixel_of_activations_map = self.max_per_pixel_of_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()

                current_avg_mean_per_activations_map = self.avg_mean_per_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_avg_std_per_activations_map = self.avg_std_per_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_avg_min_per_activations_map = self.avg_min_per_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()
                current_avg_max_per_activations_map = self.avg_max_per_activations_map_list[
                    indx].to(cpudevice, non_blocking=True).numpy()

                current_full_save_path = final_save_dir + \
                    "active_counts_over_activation_map_im.jpg"
                dict_full_path_to_saves["active_count_ov_act_map"] = current_full_save_path
                generate_plain_image(current_active_counts_activation_map,
                                     save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "active_inact_diff_counts_over_activation_map_im.jpg"
                dict_full_path_to_saves["act_inact_diff_count_ov_act_map"] = current_full_save_path
                generate_plain_image(current_active_inactive_diff_activation_map,
                                     save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "mean_per_pixel_over_activation_map_im_norm.jpg"
                dict_full_path_to_saves["mean_per_pxl_ov_act_map"] = current_full_save_path
                generate_plain_image(
                    current_mean_per_pixel_of_activations_map,  save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "std_dev_per_pixel_over_activation_map_im_norm.jpg"
                dict_full_path_to_saves["std_dev_per_pxl_ov_act_map"] = current_full_save_path
                generate_plain_image(
                    current_std_per_pixel_of_activations_map, save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "min_per_pixel_over_activation_map_im_norm.jpg"
                dict_full_path_to_saves["min_per_pxl_ov_act_map"] = current_full_save_path
                generate_plain_image(
                    current_min_per_pixel_of_activations_map, save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "max_per_pixel_over_activation_map_im_norm.jpg"
                dict_full_path_to_saves["max_per_pxl_ov_act_map"] = current_full_save_path
                generate_plain_image(
                    current_max_per_pixel_of_activations_map, save_path=current_full_save_path)

                # Obtain the resize shape for num_filters dimension
                r, c = determine_row_col_from_features(
                    current_avg_mean_per_activations_map.shape[0])
                current_avg_mean_per_activations_map = np.reshape(
                    current_avg_mean_per_activations_map, (1, r, c))
                current_avg_std_per_activations_map = np.reshape(
                    current_avg_std_per_activations_map, (1, r, c))
                current_avg_min_per_activations_map = np.reshape(
                    current_avg_min_per_activations_map, (1, r, c))
                current_avg_max_per_activations_map = np.reshape(
                    current_avg_max_per_activations_map, (1, r, c))

                current_full_save_path = final_save_dir + \
                    "avg_mean_per_act_map_im_norm.jpg"
                dict_full_path_to_saves["avg_mean_per_act_map"] = current_full_save_path
                generate_plain_image(current_avg_mean_per_activations_map,
                                     save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "avg_std_per_act_map_im_norm.jpg"
                dict_full_path_to_saves["avg_std_per_act_map"] = current_full_save_path
                generate_plain_image(current_avg_std_per_activations_map,
                                     save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "avg_min_per_act_map_im_norm.jpg"
                dict_full_path_to_saves["avg_min_per_act_map"] = current_full_save_path
                generate_plain_image(current_avg_min_per_activations_map,
                                     save_path=current_full_save_path)

                current_full_save_path = final_save_dir + \
                    "avg_max_per_act_map_im_norm.jpg"
                dict_full_path_to_saves["avg_max_per_act_map"] = current_full_save_path
                generate_plain_image(current_avg_max_per_activations_map,
                                     save_path=current_full_save_path)

            print("dict_full_path_to_saves in iteration {}=>{}".format(
                indx, dict_full_path_to_saves))

        return

    def record_min_max_average_std_active_pixels_per_batch(self, conv_outs):
        batch_size = conv_outs[0].size()[0]
        total_pixels = 0
        current_average_active_percentage = 0.0
        current_averagesq_active_percentage = 0.0
        with torch.no_grad():
            for i in range(batch_size):
                active_pixels = 0
                for indx in range(len(conv_outs)):
                    each_conv_output = conv_outs[indx][i]
                    active_conv_output = HardRelu()(each_conv_output)
                    if(i == 0):
                        total_pixels += torch.numel(
                            active_conv_output)
                    active_pixels += torch.count_nonzero(
                        HardRelu()(active_conv_output))

                current_active_percent = (
                    100. * (active_pixels/total_pixels))
                current_active_percent = current_active_percent.item()
                current_average_active_percentage += current_active_percent
                current_averagesq_active_percentage += (
                    current_active_percent ** 2)
                if(self.overall_min_active_percentage is None):
                    self.overall_min_active_percentage = current_active_percent
                elif(self.overall_min_active_percentage > current_active_percent):
                    self.overall_min_active_percentage = current_active_percent

                if(self.overall_max_active_percentage is None):
                    self.overall_max_active_percentage = current_active_percent
                elif(self.overall_max_active_percentage < current_active_percent):
                    self.overall_max_active_percentage = current_active_percent

            current_averagesq_active_percentage = current_averagesq_active_percentage / batch_size
            current_average_active_percentage = current_average_active_percentage / batch_size

            self.overall_average_active_percentage += current_average_active_percentage
            self.overall_std_active_percentage += current_averagesq_active_percentage

    def initialise_record_states(self, number_of_conv_layers=None):
        conv_outs = None
        if(self.model is not None):
            if(isinstance(self.model, torch.nn.DataParallel)):
                conv_outs = self.model.module.linear_conv_outputs
            else:
                conv_outs = self.model.linear_conv_outputs

        if(number_of_conv_layers is None):
            number_of_conv_layers = len(conv_outs)
        # Count based states
        self.active_counts_activation_map_list = []
        self.inactive_counts_activation_map_list = []
        self.active_inactive_diff_activation_map_list = [
            None] * number_of_conv_layers

        # Indicator based states
        self.active_thresholded_indicator_activation_map_list = [
            None] * number_of_conv_layers
        self.inactive_thresholded_indicator_activation_map_list = [
            None] * number_of_conv_layers
        self.overall_thresholded_active_inactive_indicator_map_list = [
            None] * number_of_conv_layers

        # Statistics based states
        self.mean_per_pixel_of_activations_map_list = [
            None] * number_of_conv_layers
        self.std_per_pixel_of_activations_map_list = [
            None] * number_of_conv_layers
        self.min_per_pixel_of_activations_map_list = [
            None] * number_of_conv_layers
        self.max_per_pixel_of_activations_map_list = [
            None] * number_of_conv_layers

        self.avg_mean_per_activations_map_list = [
            None] * number_of_conv_layers
        self.avg_std_per_activations_map_list = [
            None] * number_of_conv_layers
        self.avg_min_per_activations_map_list = [
            None] * number_of_conv_layers
        self.avg_max_per_activations_map_list = [
            None] * number_of_conv_layers

        self.total_tcollect_img_count = 0

        if(conv_outs is not None):
            for each_conv_output in conv_outs:
                zero_initialised_current_activation_map = torch.zeros(size=each_conv_output.size()[
                    1:], device=self.device)

                self.active_counts_activation_map_list.append(
                    zero_initialised_current_activation_map)
                self.inactive_counts_activation_map_list.append(
                    zero_initialised_current_activation_map.clone())

    def update_overall_recorded_states(self, collect_threshold, num_batches):

        with torch.no_grad():
            for indx in range(len(self.active_counts_activation_map_list)):
                each_active_count_activation_map = self.active_counts_activation_map_list[indx]
                each_inactive_count_activation_map = self.inactive_counts_activation_map_list[
                    indx]

                self.active_inactive_diff_activation_map_list[
                    indx] = each_active_count_activation_map - each_inactive_count_activation_map

                current_active_thresholded_indicator_activation_map = HardRelu()(each_active_count_activation_map - collect_threshold *
                                                                                 self.total_tcollect_img_count)
                current_inactive_thresholded_indicator_activation_map = HardRelu()(each_inactive_count_activation_map - collect_threshold *
                                                                                   self.total_tcollect_img_count)

                current_overall_thresholded_active_inactive_indicator_map = current_active_thresholded_indicator_activation_map - \
                    current_inactive_thresholded_indicator_activation_map

                self.overall_thresholded_active_inactive_indicator_map_list[
                    indx] = current_overall_thresholded_active_inactive_indicator_map
                self.active_thresholded_indicator_activation_map_list[
                    indx] = current_active_thresholded_indicator_activation_map
                self.inactive_thresholded_indicator_activation_map_list[
                    indx] = current_inactive_thresholded_indicator_activation_map

                self.mean_per_pixel_of_activations_map_list[
                    indx] = self.mean_per_pixel_of_activations_map_list[indx] / num_batches

                pre_variance = (
                    self.std_per_pixel_of_activations_map_list[indx] / num_batches - self.mean_per_pixel_of_activations_map_list[indx] ** 2)
                pre_variance[pre_variance < 0] = 0.

                self.std_per_pixel_of_activations_map_list[indx] = pre_variance ** 0.5

                if(torch.isnan(self.std_per_pixel_of_activations_map_list[indx]).any().item()):
                    print("STD Deviation has NAN entries at the end*****************")
                    print("num_batches", num_batches)
                    print("self.mean_per_pixel_of_activations_map_list[indx]",
                          self.mean_per_pixel_of_activations_map_list[indx])
                    print(
                        "self.std_per_pixel_of_activations_map_list[indx]", self.std_per_pixel_of_activations_map_list[indx])

                self.avg_mean_per_activations_map_list[
                    indx] = self.avg_mean_per_activations_map_list[indx] / num_batches
                self.avg_min_per_activations_map_list[
                    indx] = self.avg_min_per_activations_map_list[indx] / num_batches
                self.avg_max_per_activations_map_list[
                    indx] = self.avg_max_per_activations_map_list[indx] / num_batches

                pre_variance_per_map = (
                    self.avg_std_per_activations_map_list[indx] / num_batches - self.avg_mean_per_activations_map_list[indx] ** 2)
                pre_variance_per_map[pre_variance_per_map < 0] = 0.

                self.avg_std_per_activations_map_list[indx] = pre_variance_per_map ** 0.5

                if(torch.isnan(self.avg_std_per_activations_map_list[indx]).any().item()):
                    print(
                        "avg_std_per_activations_map_list has NAN entries at the end*****************")
                    print("num_batches", num_batches)
                    print("self.avg_mean_per_activations_map_list[indx]",
                          self.avg_mean_per_activations_map_list[indx])
                    print(
                        "self.avg_std_per_activations_map_list[indx]", self.avg_std_per_activations_map_list[indx])

                self.total_pixels += torch.numel(
                    current_overall_thresholded_active_inactive_indicator_map)
                current_thresholded_active_pixel = torch.count_nonzero(
                    HardRelu()(current_overall_thresholded_active_inactive_indicator_map))
                self.thresholded_active_pixel_count += current_thresholded_active_pixel.item()

                current_overall_unthresholded_active_inactive_indicator_map = HardRelu()(self.active_inactive_diff_activation_map_list[
                    indx])

                print(
                    "============================== INDX:{} =============================".format(indx))
                # print("total_tcollect_img_count:",
                #       self.total_tcollect_img_count)
                # print("current_overall_thresholded_active_inactive_indicator_map size:",
                #       current_overall_thresholded_active_inactive_indicator_map.size())
                # print("current_overall_unthresholded_active_inactive_indicator_map size:",
                #       current_overall_unthresholded_active_inactive_indicator_map.size())
                # print("mean_per_pixel_of_activations_map_list size:",
                #       self.mean_per_pixel_of_activations_map_list[indx].size())
                # print("min_per_pixel_of_activations_map_list size:",
                #       self.min_per_pixel_of_activations_map_list[indx].size())
                # print("std_per_pixel_of_activations_map_list size:",
                #       self.std_per_pixel_of_activations_map_list[indx].size())
                # print("avg_mean_per_activations_map_list size:",
                #       self.avg_mean_per_activations_map_list[indx].size())
                # print("avg_std_per_activations_map_list size:",
                #       self.avg_std_per_activations_map_list[indx].size())
                # print("avg_min_per_activations_map_list size:",
                #       self.avg_min_per_activations_map_list[indx].size())

                # print("current_overall_thresholded_active_inactive_indicator_map :",
                #       current_overall_thresholded_active_inactive_indicator_map)
                # print("current_overall_unthresholded_active_inactive_indicator_map :",
                #       current_overall_unthresholded_active_inactive_indicator_map)
                # print("mean_per_pixel_of_activations_map_list:",
                #       self.mean_per_pixel_of_activations_map_list[indx])
                # print("min_per_pixel_of_activations_map_list:",
                #       self.min_per_pixel_of_activations_map_list[indx])
                # print("std_per_pixel_of_activations_map_list:",
                #       self.std_per_pixel_of_activations_map_list[indx])
                # print("avg_mean_per_activations_map_list:",
                #       self.avg_mean_per_activations_map_list[indx])
                # print("avg_std_per_activations_map_list:",
                #       self.avg_std_per_activations_map_list[indx])
                # print("avg_min_per_activations_map_list:",
                #       self.avg_min_per_activations_map_list[indx])
                # print("avg_max_per_activations_map_list:",
                #       self.avg_max_per_activations_map_list[indx])

                current_unthresholded_active_pixel = torch.count_nonzero(
                    current_overall_unthresholded_active_inactive_indicator_map)
                self.unthresholded_active_pixel_count += current_unthresholded_active_pixel.item()

            self.overall_average_active_percentage = self.overall_average_active_percentage / num_batches
            self.overall_std_active_percentage = (
                self.overall_std_active_percentage/num_batches - self.overall_average_active_percentage ** 2) ** 0.5

            self.thresholded_active_pixels_percentage = (
                100. * (self.thresholded_active_pixel_count/self.total_pixels))
            self.unthresholded_active_pixels_percentage = (
                100. * (self.unthresholded_active_pixel_count/self.total_pixels))

    def update_record_states_per_batch(self):
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs = self.model.module.linear_conv_outputs
        else:
            conv_outs = self.model.linear_conv_outputs

        self.record_min_max_average_std_active_pixels_per_batch(conv_outs)
        with torch.no_grad():
            for indx in range(len(conv_outs)):
                each_conv_output = conv_outs[indx]

                current_max_activations_map = self.max_per_pixel_of_activations_map_list[
                    indx]
                max_activation_maps, _ = torch.max(each_conv_output, dim=0)
                if(current_max_activations_map is None):
                    self.max_per_pixel_of_activations_map_list[indx] = max_activation_maps
                else:
                    self.max_per_pixel_of_activations_map_list[indx] = torch.maximum(
                        current_max_activations_map, max_activation_maps)

                current_min_activations_map = self.min_per_pixel_of_activations_map_list[
                    indx]
                min_activation_maps, _ = torch.min(each_conv_output, dim=0)
                if(current_min_activations_map is None):
                    self.min_per_pixel_of_activations_map_list[indx] = min_activation_maps
                else:
                    self.min_per_pixel_of_activations_map_list[indx] = torch.minimum(
                        current_min_activations_map, min_activation_maps)

                # Until all batches are processed, mean tensor actually holds mean per batch(holding sum might result in overflow)
                current_mean_activations_map = self.mean_per_pixel_of_activations_map_list[
                    indx]
                mean_activation_maps = torch.mean(each_conv_output, 0)
                if(current_mean_activations_map is None):
                    self.mean_per_pixel_of_activations_map_list[indx] = mean_activation_maps
                else:
                    self.mean_per_pixel_of_activations_map_list[indx] = torch.add(
                        current_mean_activations_map, mean_activation_maps)

                # Until all batches are processed, std tensor actually holds the meansq per batch(holding sum might surely result in overflow)
                current_meansq_activations_map = self.std_per_pixel_of_activations_map_list[
                    indx]
                meansq_activation_maps = torch.mean(each_conv_output ** 2, 0)
                if(current_meansq_activations_map is None):
                    self.std_per_pixel_of_activations_map_list[indx] = meansq_activation_maps
                else:
                    self.std_per_pixel_of_activations_map_list[indx] = torch.add(
                        current_meansq_activations_map, meansq_activation_maps)
                    if(torch.isnan(self.std_per_pixel_of_activations_map_list[indx]).any().item()):
                        print("STD Deviation has NAN entries")
                        print("current_meansq_activations_map",
                              current_meansq_activations_map)
                        print("meansq_activation_maps", meansq_activation_maps)

                current_mean_per_activation_map = self.avg_mean_per_activations_map_list[
                    indx]
                # [B,C,W,H] take mean in C axis
                mean_per_activation_map = torch.mean(
                    each_conv_output, dim=[0, 2, 3])
                if(current_mean_per_activation_map is None):
                    self.avg_mean_per_activations_map_list[indx] = mean_per_activation_map
                else:
                    self.avg_mean_per_activations_map_list[indx] = torch.add(
                        current_mean_per_activation_map, mean_per_activation_map)

                currentsq_mean_per_activation_map = self.avg_std_per_activations_map_list[
                    indx]
                # [B,C,W,H] take mean in C axis
                meansq_per_activation_map = torch.mean(
                    each_conv_output**2, dim=[0, 2, 3])
                if(currentsq_mean_per_activation_map is None):
                    self.avg_std_per_activations_map_list[indx] = meansq_per_activation_map
                else:
                    self.avg_std_per_activations_map_list[indx] = torch.add(
                        currentsq_mean_per_activation_map, meansq_per_activation_map)

                current_max_per_activations_map = self.avg_max_per_activations_map_list[
                    indx]
                mean_max_per_activation_maps = torch.mean(
                    torch.max(torch.max(each_conv_output, dim=3)[0], dim=2)[0], 0)
                if(current_max_per_activations_map is None):
                    self.avg_max_per_activations_map_list[indx] = mean_max_per_activation_maps
                else:
                    self.avg_max_per_activations_map_list[indx] = torch.add(
                        current_max_per_activations_map, mean_max_per_activation_maps)

                current_min_per_activations_map = self.avg_min_per_activations_map_list[
                    indx]
                mean_min_per_activation_maps = torch.mean(
                    torch.min(torch.min(each_conv_output, dim=3)[0], dim=2)[0], 0)
                if(current_min_per_activations_map is None):
                    self.avg_min_per_activations_map_list[indx] = mean_min_per_activation_maps
                else:
                    self.avg_min_per_activations_map_list[indx] = torch.add(
                        current_min_per_activations_map, mean_min_per_activation_maps)

                # [B,C,W,H]
                current_active_indicator_map = HardRelu()(each_conv_output)
                # [C,W,H]
                current_active_count_activation_map = add_lower_dimension_vectors_within_itself(
                    current_active_indicator_map)
                self.active_counts_activation_map_list[indx] += current_active_count_activation_map

                current_inactive_indicator_map = HardRelu()(-each_conv_output)
                current_inactive_count_activation_map = add_lower_dimension_vectors_within_itself(
                    current_inactive_indicator_map)
                self.inactive_counts_activation_map_list[indx] += current_inactive_count_activation_map

    def record_activation_states_per_batch(self, per_class_per_batch_data):
        c_inputs, _ = per_class_per_batch_data
        c_inputs = c_inputs.to(self.device)
        current_batch_size = c_inputs.size()[0]

        # Forward pass to store layer outputs from hooks
        self.model(c_inputs)

        # Intiialise the structure to hold i's for which pixels are positive or negative
        if(self.active_counts_activation_map_list is None or self.inactive_counts_activation_map_list is None):
            self.initialise_record_states()

        self.update_record_states_per_batch()
        self.total_tcollect_img_count += current_batch_size

    def record_activation_states(self, per_class_data_loader, class_label, number_of_batch_to_collect, collect_threshold, is_save_original_image=True):
        self.reset_analyser_state()
        self.model.train(False)

        per_class_data_loader = tqdm(
            per_class_data_loader, desc='Recording activation stats for class label:'+str(class_label))
        num_batches = 0
        for i, per_class_per_batch_data in enumerate(per_class_data_loader):
            num_batches += 1
            torch.cuda.empty_cache()
            c_inputs, _ = per_class_per_batch_data
            if(i == 0 and c_inputs.size()[0] == 1 and is_save_original_image):
                with torch.no_grad():
                    temp_image = recreate_image(
                        c_inputs, False)
                    save_folder = self.image_save_prefix_folder + \
                        "class_"+str(class_label)+"/"
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    im_path = save_folder+'/original.jpg'

                    numpy_image = temp_image
                    save_image(numpy_image, im_path)

            self.record_activation_states_per_batch(
                per_class_per_batch_data)

            if(not(number_of_batch_to_collect is None) and i == number_of_batch_to_collect - 1):
                break

        self.update_overall_recorded_states(collect_threshold, num_batches)

    def get_raw_record_state_dict(self):
        raw_wandb_log_dict = dict()
        raw_wandb_log_dict["r_active_count_ov_act_map"] = self.active_counts_activation_map_list
        raw_wandb_log_dict["r_act_inact_diff_count_ov_act_map"] = self.active_inactive_diff_activation_map_list
        raw_wandb_log_dict["r_thres_active_indictor_ov_act_map"] = self.active_thresholded_indicator_activation_map_list
        raw_wandb_log_dict["r_mean_per_pxl_ov_act_map"] = self.mean_per_pixel_of_activations_map_list
        raw_wandb_log_dict["r_std_dev_per_pxl_ov_act_map"] = self.std_per_pixel_of_activations_map_list
        raw_wandb_log_dict["r_min_per_pxl_ov_act_map"] = self.min_per_pixel_of_activations_map_list
        raw_wandb_log_dict["r_max_per_pxl_ov_act_map"] = self.max_per_pixel_of_activations_map_list
        raw_wandb_log_dict["r_avg_mean_per_act_map"] = self.avg_mean_per_activations_map_list
        raw_wandb_log_dict["r_avg_std_per_act_map"] = self.avg_std_per_activations_map_list
        raw_wandb_log_dict["r_avg_min_per_act_map"] = self.avg_min_per_activations_map_list
        raw_wandb_log_dict["r_avg_max_per_act_map"] = self.avg_max_per_activations_map_list

        return raw_wandb_log_dict

    def get_wandb_log_dict(self):
        log_dict = {
            "total_pixels": self.total_pixels, "thres_active_pxl_count": self.thresholded_active_pixel_count, "thres_active_pxl_percent": self.thresholded_active_pixels_percentage,
            "unthres_active_pxl_count": self.unthresholded_active_pixel_count, "unthres_active_pxl_percent": self.unthresholded_active_pixels_percentage,
            "ovrall_avg_active_percent": self.overall_average_active_percentage, "ovrall_std_active_percent":  self.overall_std_active_percentage,
            "ovrall_min_active_percent": self.overall_min_active_percentage, "ovrall_max_active_percent": self.overall_max_active_percentage
        }
        return log_dict

    def generate_activation_stats_per_class(self, exp_type, per_class_dataset, class_label, class_indx, number_of_batch_to_collect, classes, model_arch_type, dataset,
                                            is_act_collection_on_train, is_class_segregation_on_ground_truth, activation_calculation_batch_size,
                                            wand_project_name, wandb_group_name, torch_seed, collect_threshold,
                                            root_save_prefix="root/ACT_PATTERN_PER_CLASS", final_postfix_for_save="", analysed_model_path="",
                                            is_save_graph_visualizations=True, is_save_activation_records=True, save_only_thres=False, wandb_config_additional_dict=None):
        self.root_save_prefix = root_save_prefix
        self.final_postfix_for_save = final_postfix_for_save
        self.class_label = class_label
        is_log_wandb = not(wand_project_name is None)

        # torch.manual_seed(torch_seed)
        coll_seed_gen = torch.Generator()
        coll_seed_gen.manual_seed(torch_seed)

        entr_seed_gen = torch.Generator()
        entr_seed_gen.manual_seed(torch_seed)

        self.model.train(False)
        per_class_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=activation_calculation_batch_size,
                                                            shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)

        tmp_image_over_what_str = 'test'
        if(is_act_collection_on_train):
            tmp_image_over_what_str = 'train'

        seg_over_what_str = 'MP'
        if(is_class_segregation_on_ground_truth):
            seg_over_what_str = 'GT'

        self.image_save_prefix_folder = str(root_save_prefix)+"/"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/_ACT_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
            seg_over_what_str)+"/TMP_COLL_BS_"+str(activation_calculation_batch_size)+"_NO_TO_COLL_"+str(number_of_batch_to_collect)+"/_torch_seed_"+str(torch_seed)+"_c_thres_"+str(collect_threshold)+"/" + str(final_postfix_for_save) + "/"

        if not os.path.exists(self.image_save_prefix_folder):
            os.makedirs(self.image_save_prefix_folder)

        wandb_run_name = self.image_save_prefix_folder.replace(
            "/", "").replace(root_save_prefix, class_label)
        self.wandb_run_name = wandb_run_name
        wandb_config = get_wandb_config(exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_act_collection_on_train,
                                        is_class_segregation_on_ground_truth,
                                        activation_calculation_batch_size, torch_seed, analysed_model_path,
                                        number_of_batch_to_collect=number_of_batch_to_collect, collect_threshold=collect_threshold)
        if(wandb_config_additional_dict is not None):
            wandb_config.update(wandb_config_additional_dict)

        self.wandb_config = wandb_config
        self.wandb_group_name = wandb_group_name

        self.record_activation_states(per_class_data_loader, class_label,
                                      number_of_batch_to_collect, collect_threshold, is_save_original_image=False)

        save_folder = self.image_save_prefix_folder + \
            "class_"+str(class_label)+"/"

        if(is_save_activation_records == True):
            temp_model = self.model
            self.model = None
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            with open(save_folder+'/analyser_state.pkl', 'wb') as out_file:
                pickle.dump(self, out_file)
            self.model = temp_model

        self.save_and_log_states(
            wand_project_name, is_save_graph_visualizations=is_save_graph_visualizations, save_only_thres=save_only_thres)

        if(is_log_wandb):
            wandb.finish()

    def save_and_log_states(self, wand_project_name, wandb_group_name=None, root_save_prefix=None, final_postfix_for_save=None, is_save_graph_visualizations=True, save_only_thres=False, wandb_config_additional_dict=None):
        is_log_wandb = not(wand_project_name is None)
        log_dict = self.get_wandb_log_dict()
        print("log_dict", log_dict)

        if(wandb_config_additional_dict is not None):
            self.wandb_config.update(wandb_config_additional_dict)

        if(is_log_wandb):
            if(wandb_group_name is not None):
                self.wandb_group_name = wandb_group_name
            wandb_run_name = self.wandb_run_name
            if(root_save_prefix is not None and final_postfix_for_save is not None):
                wandb_run_name = self.wandb_run_name.replace(self.root_save_prefix, root_save_prefix).replace(
                    self.final_postfix_for_save, final_postfix_for_save)
            raw_wandb_log_dict = self.get_raw_record_state_dict()
            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                group=f"{self.wandb_group_name}",
                config=self.wandb_config,
            )
            # Merge the two dictionary to log at one shot
            raw_wandb_log_dict.update(log_dict)
            wandb.log(raw_wandb_log_dict)

        if(is_save_graph_visualizations == True):
            save_folder = self.image_save_prefix_folder + \
                "class_"+str(self.class_label)+"/"
            if(root_save_prefix is not None and final_postfix_for_save is not None):
                save_folder = save_folder.replace(self.root_save_prefix, root_save_prefix).replace(
                    self.final_postfix_for_save, final_postfix_for_save)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            self.standarize_all_recorded_activation_states()
            self.save_recorded_activation_states(save_folder, save_only_thres)


def diff_merge_two_activation_analysis(merge_type, list_of_act_analyser1, list_of_act_analyser2, wandb_group_name=None, wand_project_name=None,
                                       root_save_prefix='/MERGE', final_postfix_for_save="",
                                       is_save_graph_visualizations=True):
    root_save_prefix += merge_type
    list_of_merged_act_analysis = []
    for ind in range(len(list_of_act_analyser1)):
        each_act_analyser1 = list_of_act_analyser1[ind]
        each_act_analyser2 = list_of_act_analyser2[ind]
        merged_act1_act2 = each_act_analyser1.diff_merge_with_another_activation(
            each_act_analyser2, root_save_prefix, final_postfix_for_save, wandb_group_name, wand_project_name, is_save_graph_visualizations)
        list_of_merged_act_analysis.append(merged_act1_act2)

    return list_of_merged_act_analysis


def run_activation_analysis_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth,
                                      activation_calculation_batch_size, number_of_batch_to_collect, wand_project_name, is_split_validation,
                                      valid_split_size, torch_seed, wandb_group_name, exp_type, collect_threshold,
                                      root_save_prefix='root/ACT_PATTERN_ANALYSIS', final_postfix_for_save="",
                                      custom_model=None, custom_data_loader=None, class_indx_to_visualize=None, analysed_model_path="",
                                      is_save_graph_visualizations=True, is_save_activation_records=True, wandb_config_additional_dict=None):
    if(root_save_prefix is None):
        root_save_prefix = 'root/ACT_PATTERN_ANALYSIS'
    if(final_postfix_for_save is None):
        final_postfix_for_save = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running for "+str(dataset))
    classes, num_classes, ret_config = get_preprocessing_and_other_configs(
        dataset, valid_split_size)

    if(custom_data_loader is None):
        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            ret_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=is_split_validation)
    else:
        trainloader, testloader = custom_data_loader

    print("Preprocessing and dataloader process completed of type:{} for dataset:{}".format(
        model_arch_type, dataset))

    if(custom_model is None):
        model, analysed_model_path = get_model_from_loader(
            model_arch_type, dataset)
        print("Model loaded is:", model)
    else:
        model = custom_model
        print("Custom model provided in arguments will be used")

    if(class_indx_to_visualize is None):
        class_indx_to_visualize = [i for i in range(len(classes))]

    list_of_act_analyser = []
    if(exp_type == "GENERATE_RECORD_STATS_PER_CLASS"):
        if(is_class_segregation_on_ground_truth == True):
            input_data_list_per_class = segregate_classes(
                model, trainloader, testloader, num_classes, is_template_image_on_train, is_class_segregation_on_ground_truth)

        if(is_class_segregation_on_ground_truth == False):
            input_data_list_per_class = segregate_classes(
                model, trainloader, testloader, num_classes, is_template_image_on_train, is_class_segregation_on_ground_truth)

        for c_indx in class_indx_to_visualize:
            class_label = classes[c_indx]
            print(
                "************************************************************ Class:", class_label)
            per_class_dataset = PerClassDataset(
                input_data_list_per_class[c_indx], c_indx)
            # per_class_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=32,
            #                                                shuffle=False)

            act_analyser = ActivationAnalyser(
                model)
            act_analyser.generate_activation_stats_per_class(exp_type, per_class_dataset, class_label, c_indx, number_of_batch_to_collect, classes, model_arch_type, dataset,
                                                             is_template_image_on_train, is_class_segregation_on_ground_truth, activation_calculation_batch_size,
                                                             wand_project_name, wandb_group_name, torch_seed, collect_threshold,
                                                             root_save_prefix, final_postfix_for_save, analysed_model_path, is_save_graph_visualizations, is_save_activation_records,
                                                             wandb_config_additional_dict=wandb_config_additional_dict)
            list_of_act_analyser.append(act_analyser)
    elif(exp_type == "GENERATE_RECORD_STATS_OVERALL"):
        class_label = 'ALL_CLASSES'
        c_indx = -1
        analyse_loader = trainloader
        if(is_template_image_on_train == False):
            analyse_loader = testloader
        analyse_dataset = generate_dataset_from_loader(analyse_loader)

        act_analyser = ActivationAnalyser(
            model)
        act_analyser.generate_activation_stats_per_class(exp_type, analyse_dataset, class_label, c_indx, number_of_batch_to_collect, classes, model_arch_type, dataset,
                                                         is_template_image_on_train, is_class_segregation_on_ground_truth, activation_calculation_batch_size,
                                                         wand_project_name, wandb_group_name, torch_seed, collect_threshold,
                                                         root_save_prefix, final_postfix_for_save, analysed_model_path, is_save_graph_visualizations, is_save_activation_records,
                                                         wandb_config_additional_dict=wandb_config_additional_dict)
        list_of_act_analyser.append(act_analyser)

    return list_of_act_analyser


def load_and_save_activation_analysis_on_config(dataset, exp_type, wand_project_name, load_analyser_base_folder, wandb_group_name=None,
                                                root_save_prefix=None, final_postfix_for_save=None,
                                                class_indx_to_visualize=None, is_save_graph_visualizations=True, save_only_thres=False, wandb_config_additional_dict=None):
    is_log_wandb = not(wand_project_name is None)

    print("Running for "+str(dataset))
    classes, _, _ = get_preprocessing_and_other_configs(
        dataset, valid_split_size)

    print("load_and_save_activation_analysis_on_config of type:{} for dataset:{}".format(
        model_arch_type, dataset))

    if(class_indx_to_visualize is None):
        class_indx_to_visualize = [i for i in range(len(classes))]

    list_of_act_analyser = []
    if(exp_type == "GENERATE_RECORD_STATS_PER_CLASS"):

        for c_indx in class_indx_to_visualize:
            class_label = classes[c_indx]
            print(
                "************************************************************ Class:", class_label)

            load_folder = load_analyser_base_folder + \
                "/class_"+str(class_label)
            with open(load_folder+'/analyser_state.pkl', 'rb') as in_file:
                act_analyser = pickle.load(in_file)
                act_analyser.save_and_log_states(
                    wand_project_name, wandb_group_name, root_save_prefix, final_postfix_for_save, is_save_graph_visualizations, save_only_thres, wandb_config_additional_dict)
                if(is_log_wandb):
                    wandb.finish()
                list_of_act_analyser.append(act_analyser)

    elif(exp_type == "GENERATE_RECORD_STATS_OVERALL"):
        class_label = 'ALL_CLASSES'
        c_indx = -1
        load_folder = load_analyser_base_folder + "/class_"+str(class_label)
        with open(load_folder+'/analyser_state.pkl', 'rb') as in_file:
            act_analyser = pickle.load(in_file)
            act_analyser.save_and_log_states(
                wand_project_name, wandb_group_name, root_save_prefix, final_postfix_for_save, is_save_graph_visualizations, save_only_thres, wandb_config_additional_dict)
            if(is_log_wandb):
                wandb.finish()

        list_of_act_analyser.append(act_analyser)

    return list_of_act_analyser


if __name__ == '__main__':
    # THIS OPERATION IS MEMORY HUNGRY! #
    # Because of the selected image is very large
    # If it gives out of memory error or locks the computer
    # Try it with a smaller image
    print("Start")
    # mnist , cifar10
    dataset = 'mnist'
    # cifar10_conv4_dlgn , cifar10_vgg_dlgn_16 , dlgn_fc_w_128_d_4 , random_conv4_dlgn , random_vggnet_dlgn
    # random_conv4_dlgn_sim_vgg_wo_bn , cifar10_conv4_dlgn_sim_vgg_wo_bn , cifar10_conv4_dlgn_sim_vgg_with_bn
    # random_conv4_dlgn_sim_vgg_with_bn , cifar10_conv4_dlgn_with_inbuilt_norm , random_cifar10_conv4_dlgn_with_inbuilt_norm
    # cifar10_vgg_dlgn_16_with_inbuilt_norm , random_cifar10_vgg_dlgn_16_with_inbuilt_norm
    # random_cifar10_conv4_dlgn_with_bn_with_inbuilt_norm , cifar10_conv4_dlgn_with_bn_with_inbuilt_norm
    # cifar10_conv4_dlgn_with_inbuilt_norm_with_flip_crop
    # cifar10_conv4_dlgn_with_bn_with_inbuilt_norm_with_flip_crop
    # cifar10_vgg_dlgn_16_with_inbuilt_norm_wo_bn
    # plain_pure_conv4_dnn , conv4_dlgn
    model_arch_type = 'conv4_dlgn'
    # If False, then on test
    is_act_collection_on_train = True
    # If False, then segregation is over model prediction
    is_class_segregation_on_ground_truth = True
    activation_calculation_batch_size = 64
    number_of_batch_to_collect = None
    # wand_project_name = 'test_activation_analyser'
    # wand_project_name = 'activation_analysis_class'
    # wand_project_name = "new_activation_analysis_class"
    wand_project_name = None
    wand_project_name_for_gen = None
    # wand_project_name_for_merge = "merge_diff_activation_analysis_class"
    wand_project_name_for_merge = None
    wandb_group_name = "activation_analysis_augmented_mnist_conv4_dlgn_adv_m95_c95"
    is_split_validation = False
    valid_split_size = 0.1
    torch_seed = 2022
    # GENERATE_RECORD_STATS_PER_CLASS ,  GENERATE_RECORD_STATS_OVERALL
    exp_type = "GENERATE_RECORD_STATS_PER_CLASS"
    is_save_graph_visualizations = True
    # GENERATE , LOAD_AND_SAVE , LOAD_AND_GENERATE_MERGE
    scheme_type = "LOAD_AND_GENERATE_MERGE"
    # OVER_RECONSTRUCTED , OVER_ADVERSARIAL , OVER_ORIGINAL
    sub_scheme_type = 'OVER_ADVERSARIAL'
    collect_threshold = 0.95

    if(not(wand_project_name is None)):
        wandb.login()

    if(scheme_type == "GENERATE"):
        list_of_model_paths = []
        models_base_path = None

        models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/"

        if(models_base_path != None):
            list_of_save_prefixes = []
            list_of_save_postfixes = []

            num_iterations = 3
            # for i in range(4, 6):
            for i in range(1, num_iterations+1):
                each_model_prefix = "aug_conv4_dlgn_iter_{}_dir.pt".format(i)
                # each_model_prefix = "aug_conv4_dlgn_iter_{}_dir.pt".format(i)
                list_of_model_paths.append(models_base_path+each_model_prefix)
                list_of_save_prefixes.append(
                    str(models_base_path)+"/ACT_ANALYSIS/"+str(sub_scheme_type))
                list_of_save_postfixes.append("/aug_indx_{}".format(i))

        else:
            list_of_model_paths = [None]
            list_of_save_prefixes = [
                "root/ACT_PATTERN_ANALYSIS/"+str(sub_scheme_type)]
            list_of_save_postfixes = [None]

        for ind in range(len(list_of_model_paths)):
            each_model_path = list_of_model_paths[ind]
            each_save_prefix = list_of_save_prefixes[ind]
            each_save_postfix = list_of_save_postfixes[ind]
            analysed_model_path = each_model_path

            if(each_model_path is None):
                custom_model = None
                analysed_model_path = ''
            else:
                custom_model = torch.load(each_model_path)
                print(" #*#*#*#*#*#*#*# Generating activation analysis for model path:{} with save prefix :{} and postfix:{}".format(
                    each_model_path, each_save_prefix, each_save_postfix))

            if(sub_scheme_type == 'OVER_ORIGINAL'):
                list_of_act_analyser = run_activation_analysis_on_config(dataset, model_arch_type, is_act_collection_on_train, is_class_segregation_on_ground_truth,
                                                                         activation_calculation_batch_size, number_of_batch_to_collect, wand_project_name_for_gen, is_split_validation,
                                                                         valid_split_size, torch_seed, wandb_group_name, exp_type, collect_threshold,
                                                                         custom_model=custom_model, root_save_prefix=each_save_prefix, final_postfix_for_save=each_save_postfix,
                                                                         is_save_graph_visualizations=is_save_graph_visualizations, analysed_model_path=analysed_model_path)
            elif(sub_scheme_type == 'OVER_RECONSTRUCTED'):
                pass
            elif(sub_scheme_type == 'OVER_ADVERSARIAL'):
                is_save_adv = True
                eps = 0.02
                adv_attack_type = 'PGD'
                number_of_adversarial_optimization_steps = 161
                eps_step_size = 0.01
                adv_target = None

                final_postfix_for_save = "/RAW_ADV_SAVES/adv_type_{}/EPS_{}/eps_stp_size_{}/adv_steps_{}/on_train_{}/{}".format(
                    adv_attack_type, eps, eps_step_size, number_of_adversarial_optimization_steps, is_act_collection_on_train, each_save_postfix)
                save_path = models_base_path + final_postfix_for_save+"/adv_dataset.npy"
                print("Adversarial examples will be saved at: ", save_path)

                wandb_config_additional_dict = {"eps": eps, "adv_atack_type": adv_attack_type, "num_of_adversarial_optim_stps":
                                                number_of_adversarial_optimization_steps, "eps_stp_size": eps_step_size, "adv_target": adv_target}

                each_save_postfix = "EPS_{}/ADV_TYPE_{}/NUM_ADV_STEPS_{}/eps_step_size_{}/".format(
                    eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size) + each_save_postfix

                classes, num_classes, ret_config = get_preprocessing_and_other_configs(
                    dataset, valid_split_size)
                trainloader, _, testloader = preprocess_dataset_get_data_loader(
                    ret_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=is_split_validation)
                if(is_act_collection_on_train):
                    to_be_analysed_dataloader = trainloader
                else:
                    to_be_analysed_dataloader = testloader

                is_current_adv_aug_available = os.path.exists(save_path)
                if(is_current_adv_aug_available):
                    with open(save_path, 'rb') as file:
                        npzfile = np.load(save_path)
                        list_of_adv_images = npzfile['x']
                        list_of_labels = npzfile['y']
                        adv_dataset = CustomSimpleDataset(
                            list_of_adv_images, list_of_labels)
                        print("Loading adversarial examples from path:", save_path)
                else:
                    adv_dataset = generate_adv_examples(
                        to_be_analysed_dataloader, custom_model, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, number_of_batch_to_collect, is_save_adv=is_save_adv, save_path=save_path)

                to_be_analysed_adversarial_dataloader = torch.utils.data.DataLoader(
                    adv_dataset, shuffle=False, batch_size=128)

                if(is_act_collection_on_train):
                    custom_loader = to_be_analysed_adversarial_dataloader, None
                else:
                    custom_loader = None, to_be_analysed_adversarial_dataloader

                list_of_act_analyser = run_activation_analysis_on_config(dataset, model_arch_type, is_act_collection_on_train, is_class_segregation_on_ground_truth,
                                                                         activation_calculation_batch_size, number_of_batch_to_collect, wand_project_name_for_gen,
                                                                         is_split_validation, valid_split_size, torch_seed, wandb_group_name, exp_type, collect_threshold,
                                                                         custom_data_loader=custom_loader, custom_model=custom_model, root_save_prefix=each_save_prefix,
                                                                         final_postfix_for_save=each_save_postfix, is_save_graph_visualizations=is_save_graph_visualizations,
                                                                         analysed_model_path=analysed_model_path, wandb_config_additional_dict=wandb_config_additional_dict)

    elif(scheme_type == "LOAD_AND_SAVE"):
        # class_ind_visualize = [9]
        # class_ind_visualize = [2, 4, 6, 8]
        class_ind_visualize = None
        # class_ind_visualize = [3, 5,7,9]
        # class_ind_visualize = [7, 9]

        list_of_load_paths = []
        loader_base_path = None

        save_only_thres = False

        loader_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/ACT_ANALYSIS/OVER_ADVERSARIAL/mnist/MT_conv4_dlgn_ET_GENERATE_RECORD_STATS_PER_CLASS/_ACT_OV_train/SEG_GT/TMP_COLL_BS_64_NO_TO_COLL_None/_torch_seed_2022_c_thres_0.95/EPS_0.02/ADV_TYPE_PGD/NUM_ADV_STEPS_161/eps_step_size_0.01/"

        if(loader_base_path != None):
            num_iterations = 3
            for i in range(1, num_iterations+1):
                # for i in range(1, 2):
                each_model_prefix = "aug_indx_{}".format(i)
                list_of_load_paths.append(loader_base_path+each_model_prefix)

        else:
            list_of_load_paths = ["root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.5/ACT_ANALYSIS/OVER_ORIGINAL/mnist/MT_conv4_dlgn_ET_GENERATE_RECORD_STATS_PER_CLASS/_ACT_OV_train/SEG_GT/TMP_COLL_BS_64_NO_TO_COLL_None/_torch_seed_2022_c_thres_0.95/"]

        wandb_config_additional_dict = dict()
        wandb_config_additional_dict["sub_scheme_type"] = "OVER_ADVERSARIAL"
        for ind in range(len(list_of_load_paths)):
            current_analyser_loader_path = list_of_load_paths[ind]
            list_of_act_analyser = load_and_save_activation_analysis_on_config(dataset, exp_type, wand_project_name, current_analyser_loader_path, wandb_group_name=wandb_group_name,
                                                                               class_indx_to_visualize=class_ind_visualize, is_save_graph_visualizations=True, save_only_thres=save_only_thres, wandb_config_additional_dict=wandb_config_additional_dict)

    elif(scheme_type == "LOAD_AND_GENERATE_MERGE"):
        merge_type = "DIFF"
        # class_ind_visualize = [9]
        # class_ind_visualize = [2, 4, 6, 8]
        class_ind_visualize = None
        # class_ind_visualize = [3, 5,7,9]
        # class_ind_visualize = [7, 9]

        list_of_load_paths1 = []
        list_of_load_paths2 = []
        loader_base_path1 = None
        loader_base_path2 = None

        save_only_thres = False

        loader_base_path1 = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/ACT_ANALYSIS/OVER_ADVERSARIAL/mnist/MT_conv4_dlgn_ET_GENERATE_RECORD_STATS_PER_CLASS/_ACT_OV_train/SEG_GT/TMP_COLL_BS_64_NO_TO_COLL_None/_torch_seed_2022_c_thres_0.95/EPS_0.02/ADV_TYPE_PGD/NUM_ADV_STEPS_161/eps_step_size_0.01/"
        loader_base_path2 = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/ACT_ANALYSIS/OVER_ORIGINAL/mnist/MT_conv4_dlgn_ET_GENERATE_RECORD_STATS_PER_CLASS/_ACT_OV_train/SEG_GT/TMP_COLL_BS_64_NO_TO_COLL_None/_torch_seed_2022_c_thres_0.95/"

        if(loader_base_path1 is not None and loader_base_path2 is not None):
            num_iterations = 3
            for i in range(1, num_iterations+1):
                # for i in range(5, 6):
                each_model_prefix = "aug_indx_{}".format(i)
                list_of_load_paths1.append(loader_base_path1+each_model_prefix)
                list_of_load_paths2.append(loader_base_path2+each_model_prefix)

        else:
            list_of_load_paths1 = ["root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.5/ACT_ANALYSIS/OVER_ORIGINAL/mnist/MT_conv4_dlgn_ET_GENERATE_RECORD_STATS_PER_CLASS/_ACT_OV_train/SEG_GT/TMP_COLL_BS_64_NO_TO_COLL_None/_torch_seed_2022_c_thres_0.95/"]
            list_of_load_paths2 = ["root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.5/ACT_ANALYSIS/OVER_ORIGINAL/mnist/MT_conv4_dlgn_ET_GENERATE_RECORD_STATS_PER_CLASS/_ACT_OV_train/SEG_GT/TMP_COLL_BS_64_NO_TO_COLL_None/_torch_seed_2022_c_thres_0.95/"]

        if(merge_type == "DIFF"):
            for ind in range(len(list_of_load_paths1)):
                current_analyser_loader_path1 = list_of_load_paths1[ind]
                current_analyser_loader_path2 = list_of_load_paths2[ind]
                print("current_analyser_loader_path1",
                      current_analyser_loader_path1)
                print("current_analyser_loader_path2",
                      current_analyser_loader_path2)
                wandb_config_additional_dict = dict()
                wandb_config_additional_dict["sub_scheme_type"] = "OVER_ADVERSARIAL"
                list_of_act_analyser1 = load_and_save_activation_analysis_on_config(dataset, exp_type, wand_project_name=wand_project_name, load_analyser_base_folder=current_analyser_loader_path1, wandb_group_name=wandb_group_name,
                                                                                    class_indx_to_visualize=class_ind_visualize, is_save_graph_visualizations=True, wandb_config_additional_dict=wandb_config_additional_dict)

                wandb_config_additional_dict["sub_scheme_type"] = "OVER_ORIGINAL"
                list_of_act_analyser2 = load_and_save_activation_analysis_on_config(dataset, exp_type, wand_project_name=wand_project_name, load_analyser_base_folder=current_analyser_loader_path2, wandb_group_name=wandb_group_name,
                                                                                    class_indx_to_visualize=class_ind_visualize, is_save_graph_visualizations=True, wandb_config_additional_dict=wandb_config_additional_dict)
                list_of_merged_act1_act2 = diff_merge_two_activation_analysis(merge_type,
                                                                              list_of_act_analyser1, list_of_act_analyser2, wand_project_name=wand_project_name_for_merge, is_save_graph_visualizations=is_save_graph_visualizations)

    print("Finished execution!!!")

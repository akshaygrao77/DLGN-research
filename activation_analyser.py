import torch
import os
from tqdm import tqdm, trange
import wandb
import random
import numpy as np
import pickle


from utils.visualise_utils import save_image, recreate_image, add_lower_dimension_vectors_within_itself, construct_images_from_feature_maps, construct_heatmaps_from_data, determine_row_col_from_features
from utils.data_preprocessing import preprocess_dataset_get_data_loader, generate_dataset_from_loader
from configs.dlgn_conv_config import HardRelu
from utils.data_preprocessing import preprocess_dataset_get_data_loader, segregate_classes
from structure.generic_structure import PerClassDataset
from model.model_loader import get_model_from_loader
from configs.generic_configs import get_preprocessing_and_other_configs


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


class ActivationAnalyser():

    def __init__(self, model):
        self.model = model
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

    def save_recorded_activation_states(self, base_save_folder):
        cpudevice = torch.device("cpu")
        # For each conv layer
        for indx in range(len(self.active_counts_activation_map_list)):
            print("Saving visualization for layer:", indx)
            dict_full_path_to_saves = dict()

            current_active_counts_activation_map = self.active_counts_activation_map_list[
                indx].to(cpudevice, non_blocking=True).numpy()
            current_active_inactive_diff_activation_map = self.active_inactive_diff_activation_map_list[
                indx].to(cpudevice, non_blocking=True).numpy()
            current_active_thresholded_indicator_activation_map = self.active_thresholded_indicator_activation_map_list[
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

            final_save_dir = base_save_folder+"/layer_{}/".format(indx)
            if not os.path.exists(final_save_dir):
                os.makedirs(final_save_dir)

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
                "thres_active_indicators_over_activation_map.jpg"
            dict_full_path_to_saves["thres_active_indictor_ov_act_map"] = current_full_save_path
            construct_heatmaps_from_data(current_active_thresholded_indicator_activation_map,
                                         title='Thresholded active pixel indicatoes over activation map', save_path=current_full_save_path)

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

    def initialise_record_states(self):
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs = self.model.module.linear_conv_outputs
        else:
            conv_outs = self.model.linear_conv_outputs

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
                                            is_save_graph_visualizations=True, is_save_activation_records=True):
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
            wand_project_name, is_save_graph_visualizations=is_save_graph_visualizations)

        if(is_log_wandb):
            wandb.finish()

    def save_and_log_states(self, wand_project_name, root_save_prefix=None, final_postfix_for_save=None, is_save_graph_visualizations=True):
        is_log_wandb = not(wand_project_name is None)
        log_dict = self.get_wandb_log_dict()
        print("log_dict", log_dict)

        if(is_log_wandb):
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
            self.save_recorded_activation_states(save_folder)


def run_activation_analysis_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth,
                                      activation_calculation_batch_size, number_of_batch_to_collect, wand_project_name, is_split_validation,
                                      valid_split_size, torch_seed, wandb_group_name, exp_type, collect_threshold,
                                      root_save_prefix='root/ACT_PATTERN_ANALYSIS', final_postfix_for_save="",
                                      custom_model=None, custom_data_loader=None, class_indx_to_visualize=None, analysed_model_path="",
                                      is_save_graph_visualizations=True, is_save_activation_records=True):
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
                                                             root_save_prefix, final_postfix_for_save, analysed_model_path, is_save_graph_visualizations, is_save_activation_records)
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
                                                         root_save_prefix, final_postfix_for_save, analysed_model_path, is_save_graph_visualizations, is_save_activation_records)
        list_of_act_analyser.append(act_analyser)

    return list_of_act_analyser


def load_and_save_activation_analysis_on_config(dataset, exp_type, wand_project_name, load_analyser_base_folder,
                                                root_save_prefix=None, final_postfix_for_save=None,
                                                class_indx_to_visualize=None, is_save_graph_visualizations=True):
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
                    wand_project_name, root_save_prefix, final_postfix_for_save, is_save_graph_visualizations)
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
                wand_project_name, root_save_prefix, final_postfix_for_save, is_save_graph_visualizations)
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
    wand_project_name = 'activation_analysis_class'
    # wand_project_name = None
    wand_project_name_for_gen = None
    wandb_group_name = "activation_analysis_augmented_mnist_conv4_dlgn"
    is_split_validation = False
    valid_split_size = 0.1
    torch_seed = 2022
    # GENERATE_RECORD_STATS_PER_CLASS ,  GENERATE_RECORD_STATS_OVERALL
    exp_type = "GENERATE_RECORD_STATS_PER_CLASS"
    is_save_graph_visualizations = False
    # GENERATE , LOAD
    scheme_type = "LOAD"
    # OVER_RECONSTRUCTED , OVER_ADVERSARIAL , OVER_ORIGINAL
    sub_scheme_type = 'OVER_ORIGINAL'
    collect_threshold = 0.95

    if(not(wand_project_name is None)):
        wandb.login()

    if(scheme_type == "GENERATE"):
        list_of_model_paths = []
        models_base_path = None

        models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.5/"

        if(models_base_path != None):
            list_of_save_prefixes = []
            list_of_save_postfixes = []

            num_iterations = 3
            for i in range(1, num_iterations+1):
                each_model_prefix = "aug_conv4_dlgn_iter_{}_dir.pt".format(i)
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

    elif(scheme_type == "LOAD"):
        class_ind_visualize = [0, 2, 4, 6, 8]
        # class_ind_visualize = None
        # class_ind_visualize = [1, 3, 5, 7, 9]
        list_of_load_paths = []
        loader_base_path = None

        loader_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.5/ACT_ANALYSIS/OVER_ORIGINAL/mnist/MT_conv4_dlgn_ET_GENERATE_RECORD_STATS_OVERALL/_ACT_OV_train/SEG_GT/TMP_COLL_BS_64_NO_TO_COLL_None/_torch_seed_2022_c_thres_0.95/"

        if(loader_base_path != None):
            num_iterations = 3
            # for i in range(1, num_iterations+1):
            for i in range(3, 4):
                each_model_prefix = "aug_indx_{}".format(i)
                list_of_load_paths.append(loader_base_path+each_model_prefix)

        else:
            list_of_load_paths = ["root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.5/ACT_ANALYSIS/OVER_ORIGINAL/mnist/MT_conv4_dlgn_ET_GENERATE_RECORD_STATS_PER_CLASS/_ACT_OV_train/SEG_GT/TMP_COLL_BS_64_NO_TO_COLL_None/_torch_seed_2022_c_thres_0.95/"]

        for ind in range(len(list_of_load_paths)):
            current_analyser_loader_path = list_of_load_paths[ind]
            list_of_act_analyser = load_and_save_activation_analysis_on_config(dataset, exp_type, wand_project_name, current_analyser_loader_path,
                                                                               class_indx_to_visualize=class_ind_visualize, is_save_graph_visualizations=True)

    print("Finished execution!!!")

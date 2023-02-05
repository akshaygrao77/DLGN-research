import torch
import os
from tqdm import tqdm
import wandb
import random
import numpy as np
import pickle
import itertools
import xlsxwriter

from utils.visualise_utils import save_image, recreate_image, generate_list_of_plain_images_from_data, generate_plain_image, generate_list_of_images_from_data, construct_images_from_feature_maps, construct_heatmaps_from_data, generate_video_of_image_from_data
from utils.data_preprocessing import preprocess_dataset_get_data_loader, generate_dataset_from_loader, seed_worker
from structure.generic_structure import CustomSimpleDataset
from utils.data_preprocessing import preprocess_dataset_get_data_loader, segregate_classes
from structure.generic_structure import PerClassDataset
from model.model_loader import get_model_from_loader
from configs.generic_configs import get_preprocessing_and_other_configs
from adversarial_attacks_tester import generate_adv_examples
from configs.dlgn_conv_config import HardRelu


def unroll_print_torch_3D_array(torch_arr):
    (x, y, z) = torch_arr.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                print(torch_arr[i][j][k].item(), end=" ")
            print("")
        print("---------------------")


def get_wandb_config(exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_act_collection_on_train,
                     is_class_segregation_on_ground_truth,
                     activation_calculation_batch_size, torch_seed, analysed_model_path,
                     number_of_batch_to_collect=None):

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

    return wandb_config


def get_mean_sd(input_tensor):
    overall_std = torch.std(input_tensor)
    overall_mean = torch.mean(input_tensor)
    return overall_mean, overall_std


def generate_stats(input_tensor):
    min_per_pixel = torch.min(input_tensor, dim=0).values
    max_per_pixel = torch.max(input_tensor, dim=0).values
    std_per_pixel = torch.std(input_tensor, dim=0)
    mean_per_pixel = torch.mean(input_tensor, dim=0)
    overall_min = torch.min(min_per_pixel)
    overall_max = torch.max(max_per_pixel)
    overall_mean, overall_std = get_mean_sd(input_tensor)

    mean_min_per_pixel, sd_min_per_pixel = get_mean_sd(min_per_pixel)
    mean_max_per_pixel, sd_max_per_pixel = get_mean_sd(max_per_pixel)
    mean_std_per_pixel, sd_std_per_pixel = get_mean_sd(std_per_pixel)
    mean_mean_per_pixel, sd_mean_per_pixel = get_mean_sd(
        mean_per_pixel)

    return overall_min, overall_max, overall_std, overall_mean, min_per_pixel, max_per_pixel, std_per_pixel, mean_per_pixel, mean_min_per_pixel, sd_min_per_pixel, mean_max_per_pixel, sd_max_per_pixel, mean_std_per_pixel, sd_std_per_pixel, mean_mean_per_pixel, sd_mean_per_pixel


class RawActivationAnalyser():

    def __init__(self, model):
        self.model = model
        if(model is not None):
            self.model.eval()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.activation__save_prefix_folder = "root/raw_activation_analysis/"

        self.reset_raw_analyser_state()

    # [B,128,28,28]
    def reset_raw_analyser_state(self):
        self.total_tcollect_img_count = 0
        # Count based states
        # Size: [Num of points(batch dim, Num filters aggregated over layer, W , H]
        self.post_activation_values_all_batches = None
        self.post_activation_values_positive_hrelu_counts = None
        self.post_activation_values_negative_hrelu_counts = None
        self.post_activation_values_gate_entropy = None
        self.post_activation_values_entropy_per_sample = None

    def diff_merge_with_another_activation(self, other_activation_state, root_save_prefix, final_postfix_for_save, wandb_group_name, wand_project_name=None, is_save_graph_visualizations=True):
        merged_act_analyser = RawActivationAnalyser(None)
        merged_act_analyser.initialise_raw_record_states()

        merged_act_analyser.wandb_config = dict()

        merged_act_analyser.wandb_group_name = wandb_group_name
        if(wandb_group_name is None):
            merged_act_analyser.wandb_group_name = self.wandb_group_name + \
                "_*_" + other_activation_state.wandb_group_name

        merged_act_analyser.image_save_prefix_folder = self.image_save_prefix_folder + "/"+str(root_save_prefix) + "/" +\
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
            merged_act_analyser.total_tcollect_img_count = self.total_tcollect_img_count
            temp1 = self.post_activation_values_all_batches
            temp2 = other_activation_state.post_activation_values_all_batches
            if(not isinstance(temp1, torch.Tensor)):
                temp1 = torch.from_numpy(temp1)
            if(not isinstance(temp2, torch.Tensor)):
                temp2 = torch.from_numpy(temp2)
            merged_act_analyser.post_activation_values_all_batches = HardRelu()(temp1) - \
                HardRelu()(temp2)
            merged_act_analyser.post_activation_values_positive_hrelu_counts = self.post_activation_values_positive_hrelu_counts - \
                other_activation_state.post_activation_values_positive_hrelu_counts
            merged_act_analyser.post_activation_values_negative_hrelu_counts = self.post_activation_values_negative_hrelu_counts - \
                other_activation_state.post_activation_values_negative_hrelu_counts
            merged_act_analyser.post_activation_values_gate_entropy = self.post_activation_values_gate_entropy - \
                other_activation_state.post_activation_values_gate_entropy
            merged_act_analyser.post_activation_values_entropy_per_sample = self.post_activation_values_entropy_per_sample - \
                other_activation_state.post_activation_values_entropy_per_sample

            temp1 = merged_act_analyser.post_activation_values_all_batches
            pos_diff = HardRelu()(temp1)
            neg_diff = HardRelu()(-temp1)
            merged_act_analyser.diff_counts_post_activation_values_all_layers = torch.sum(
                pos_diff, dim=0) + torch.sum(neg_diff, dim=0)

        merged_act_analyser.root_save_prefix = root_save_prefix
        merged_act_analyser.final_postfix_for_save = final_postfix_for_save
        merged_act_analyser.class_label = self.class_label

        save_folder = merged_act_analyser.image_save_prefix_folder + \
            "class_"+str(merged_act_analyser.class_label)+"/"

        # temp_model = merged_act_analyser.model
        # merged_act_analyser.model = None
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)
        # print("Save directory:", save_folder)
        # with open(save_folder+'/raw_analyser_state.pkl', 'wb') as out_file:
        #     pickle.dump(merged_act_analyser, out_file)
        # merged_act_analyser.model = temp_model

        merged_act_analyser.save_and_log_states(
            wand_project_name, is_save_graph_visualizations=is_save_graph_visualizations)

        return merged_act_analyser

    def save_raw_recorded_activation_states(self, base_save_folder, is_save_video_data=False):
        hrelu_base_folder = base_save_folder + "/PlainImages/HardRelu/"
        hrelu_video_base_folder = base_save_folder + "/VideoImages/HardRelu/"
        if not os.path.exists(hrelu_base_folder):
            os.makedirs(hrelu_base_folder)

        dict_full_path_to_saves = dict()

        temp1 = self.post_activation_values_all_batches
        if(not isinstance(temp1, torch.Tensor)):
            temp1 = torch.from_numpy(temp1)
        current_post_activation_values = HardRelu()(temp1)

        current_full_save_path = base_save_folder+"/Video/HardRelu/" + \
            "hardrelu_raw_postactivation_images_*.mp4"
        dict_full_path_to_saves["raw_post_activation"] = current_full_save_path
        print("current_full_save_path:", current_full_save_path)

        current_full_img_save_path = hrelu_video_base_folder + \
            "hard_relu_post_act_img_b_*.jpg"

        print("current_full_img_save_path:", current_full_img_save_path)

        generate_plain_image(self.post_activation_values_positive_hrelu_counts,
                             hrelu_base_folder+"c_Positive_HRelu_counts.jpg", is_standarize=False, is_standarize_01=True)
        generate_plain_image(self.post_activation_values_negative_hrelu_counts,
                             hrelu_base_folder+"c_Negative_HRelu_counts.jpg", is_standarize=False, is_standarize_01=True)
        generate_plain_image(self.post_activation_values_gate_entropy,
                             hrelu_base_folder+"c_Entropy.jpg", is_standarize=False, is_standarize_01=True)
        current_full_txt_save_path = hrelu_base_folder + \
            "entropy_stats.txt"
        with open(current_full_txt_save_path, "w") as myfile:
            ovrl_diff_entr_min, ovrl_diff_entr_max, ovrl_diff_entr_std, ovrl_diff_entr_mean, diff_entr_min_per_pxl, diff_entr_max_per_pxl, diff_entr_std_per_pxl, diff_entr_mean_per_pxl, mean_diff_entr_min_per_pxl, sd_diff_entr_min_per_pxl, mean_diff_entr_max_per_pxl, sd_diff_entr_max_per_pxl, mean_diff_entr_std_per_pxl, sd_diff_entr_std_per_pxl, mean_diff_entr_mean_per_pxl, sd_diff_entr_mean_per_pxl = generate_stats(
                self.post_activation_values_gate_entropy)

            ovrl_sample_entr_mean, ovrl_sample_entr_std = get_mean_sd(
                self.post_activation_values_entropy_per_sample)
            ovrl_sample_entr_min = torch.min(
                self.post_activation_values_entropy_per_sample)
            ovrl_sample_entr_max = torch.max(
                self.post_activation_values_entropy_per_sample)

            myfile.write("Overall Sample Entr Min = %s\n" %
                         ovrl_sample_entr_min)
            myfile.write("Overall Sample Entr Max = %s\n" %
                         ovrl_sample_entr_max)
            myfile.write("Overall Sample Entr STD = %s\n" %
                         ovrl_sample_entr_std)
            myfile.write("Overall Sample Entr mean = %s \n" %
                         ovrl_sample_entr_mean)

            myfile.write("Overall Entr Min = %s\n" %
                         ovrl_diff_entr_min)
            myfile.write("Overall Entr Max = %s\n" %
                         ovrl_diff_entr_max)
            myfile.write("Overall Entr STD = %s\n" %
                         ovrl_diff_entr_std)
            myfile.write("Overall Entr mean = %s \n" %
                         ovrl_diff_entr_mean)

            myfile.write(
                "======================== Min per pixel stats ==================== \n")
            myfile.write("Mean = %s \t" %
                         mean_diff_entr_min_per_pxl)
            myfile.write("SD = %s\n" %
                         sd_diff_entr_min_per_pxl)
            myfile.write(
                "======================== Max per pixel stats ==================== \n")
            myfile.write("Mean = %s \t" %
                         mean_diff_entr_max_per_pxl)
            myfile.write("SD = %s\n" %
                         sd_diff_entr_max_per_pxl)
            myfile.write(
                "======================== Std per pixel stats ==================== \n")
            myfile.write("Mean = %s \t" %
                         mean_diff_entr_std_per_pxl)
            myfile.write("SD = %s\n" %
                         sd_diff_entr_std_per_pxl)
            myfile.write(
                "======================== Mean per pixel stats ==================== \n")
            myfile.write("Mean = %s \t" %
                         mean_diff_entr_mean_per_pxl)
            myfile.write("SD = %s\n" %
                         sd_diff_entr_mean_per_pxl)

            myfile.write("Overall Entr Per Sample = %s \n" %
                         self.post_activation_values_entropy_per_sample)

            myfile.write("Entr Min per pixel = %s\n" %
                         diff_entr_min_per_pxl)
            myfile.write("Entr Max per pixel = %s\n" %
                         diff_entr_max_per_pxl)
            myfile.write("Entr STD per pixel = %s\n" %
                         diff_entr_std_per_pxl)
            myfile.write("Entr Mean per pixel = %s \n" %
                         diff_entr_mean_per_pxl)

        if hasattr(self, 'diff_counts_post_activation_values_all_layers'):
            current_full_txt_save_path = hrelu_base_folder + \
                "diff_counts.txt"
            with open(current_full_txt_save_path, "w") as myfile:
                ovrl_diff_count_min, ovrl_diff_count_max, ovrl_diff_count_std, ovrl_diff_count_mean, diff_count_min_per_pxl, diff_count_max_per_pxl, diff_count_std_per_pxl, diff_count_mean_per_pxl, mean_diff_count_min_per_pxl, sd_diff_count_min_per_pxl, mean_diff_count_max_per_pxl, sd_diff_count_max_per_pxl, mean_diff_count_std_per_pxl, sd_diff_count_std_per_pxl, mean_diff_count_mean_per_pxl, sd_diff_count_mean_per_pxl = generate_stats(
                    self.diff_counts_post_activation_values_all_layers)

                myfile.write("Overall Diff Count Min = %s\n" %
                             ovrl_diff_count_min)
                myfile.write("Overall Diff Count Max = %s\n" %
                             ovrl_diff_count_max)
                myfile.write("Overall Diff Count STD = %s\n" %
                             ovrl_diff_count_std)
                myfile.write("Overall Diff Count mean = %s \n" %
                             ovrl_diff_count_mean)
                myfile.write(
                    "======================== Min per pixel stats ==================== \n")
                myfile.write("Mean = %s \t" %
                             mean_diff_count_min_per_pxl)
                myfile.write("SD = %s\n" %
                             sd_diff_count_min_per_pxl)
                myfile.write(
                    "======================== Max per pixel stats ==================== \n")
                myfile.write("Mean = %s \t" %
                             mean_diff_count_max_per_pxl)
                myfile.write("SD = %s\n" %
                             sd_diff_count_max_per_pxl)
                myfile.write(
                    "======================== Std per pixel stats ==================== \n")
                myfile.write("Mean = %s \t" %
                             mean_diff_count_std_per_pxl)
                myfile.write("SD = %s\n" %
                             sd_diff_count_std_per_pxl)
                myfile.write(
                    "======================== Mean per pixel stats ==================== \n")
                myfile.write("Mean = %s \t" %
                             mean_diff_count_mean_per_pxl)
                myfile.write("SD = %s\n" %
                             sd_diff_count_mean_per_pxl)

                myfile.write("Diff Count Min per pixel = %s\n" %
                             diff_count_min_per_pxl)
                myfile.write("Diff Count Max per pixel = %s\n" %
                             diff_count_max_per_pxl)
                myfile.write("Diff Count STD per pixel = %s\n" %
                             diff_count_std_per_pxl)
                myfile.write("Diff Count Mean per pixel = %s \n" %
                             diff_count_mean_per_pxl)

                for f_ind in range(self.diff_counts_post_activation_values_all_layers.size()[0]):
                    curr_filter = self.diff_counts_post_activation_values_all_layers[f_ind]
                    # print("curr_filter size", curr_filter.size())
                    sum_curr_filter = torch.sum(curr_filter)
                    myfile.write(
                        "\n ************************************ Next Filter:{} = {} *********************************** \n".format(f_ind, sum_curr_filter))
                    myfile.write("\n%s\n" % curr_filter)
            current_full_diff_img_save_path = hrelu_base_folder + \
                "diff_counts_image.jpg"
            generate_plain_image(self.diff_counts_post_activation_values_all_layers,
                                 current_full_diff_img_save_path, is_standarize=False, is_standarize_01=True)

        # generate_list_of_images_from_data(current_post_activation_values, 200, 300,
        #                                   "Hard relu Raw post activation video", save_each_img_path=current_full_img_save_path, cmap='binary')
        if(is_save_video_data):
            if not os.path.exists(hrelu_video_base_folder):
                os.makedirs(hrelu_video_base_folder)
            generate_list_of_plain_images_from_data(
                current_post_activation_values, save_each_img_path=current_full_img_save_path, is_standarize=False)
        # generate_video_of_image_from_data(
        #     current_post_activation_values, 200, 300, "Hard relu Raw post activation video", save_path=current_full_save_path, save_each_img_path=current_full_img_save_path, cmap='binary')

        return

    def initialise_raw_record_states(self):
        self.total_tcollect_img_count = 0
        self.post_activation_values_all_batches = None
        self.post_activation_values_positive_hrelu_counts = None
        self.post_activation_values_negative_hrelu_counts = None
        self.post_activation_values_gate_entropy = None
        self.post_activation_values_entropy_per_sample = None

    def update_raw_record_states_per_batch(self):
        cpudevice = torch.device("cpu")
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs = self.model.module.linear_conv_outputs
        else:
            conv_outs = self.model.linear_conv_outputs

        beta = 4
        current_post_activation_values = None
        with torch.no_grad():
            for indx in range(len(conv_outs)):
                each_conv_output = conv_outs[indx]

                # each_conv_output = torch.nn.Sigmoid()(beta * each_conv_output)
                # each_conv_output = each_conv_output
                each_conv_output = each_conv_output.to(
                    cpudevice, non_blocking=True).numpy()

                if(current_post_activation_values is None):
                    current_post_activation_values = each_conv_output
                else:
                    current_post_activation_values = np.concatenate(
                        (current_post_activation_values, each_conv_output), axis=1)

            if(self.post_activation_values_all_batches is None):
                self.post_activation_values_all_batches = current_post_activation_values
            else:
                self.post_activation_values_all_batches = np.concatenate(
                    (self.post_activation_values_all_batches, current_post_activation_values), axis=0)

    def record_raw_activation_states_per_batch(self, per_class_per_batch_data):
        c_inputs, _ = per_class_per_batch_data
        c_inputs = c_inputs.to(self.device)
        current_batch_size = c_inputs.size()[0]

        # Forward pass to store layer outputs from hooks
        self.model(c_inputs)

        # Intiialise the structure to hold i's for which pixels are positive or negative
        if(self.post_activation_values_all_batches is None):
            self.initialise_raw_record_states()

        self.update_raw_record_states_per_batch()
        self.total_tcollect_img_count += current_batch_size

    def record_raw_activation_states(self, per_class_data_loader, class_label, number_of_batch_to_collect, is_save_original_image=True):
        self.reset_raw_analyser_state()
        self.model.train(False)

        per_class_data_loader = tqdm(
            per_class_data_loader, desc='Recording raw activation stats for class label:'+str(class_label))
        num_batches = 0
        with torch.no_grad():
            for i, per_class_per_batch_data in enumerate(per_class_data_loader):
                num_batches += 1
                torch.cuda.empty_cache()
                c_inputs, _ = per_class_per_batch_data
                if(i == 0 and c_inputs.size()[0] == 1 and is_save_original_image):
                    temp_image = recreate_image(
                        c_inputs, False)
                    save_folder = self.image_save_prefix_folder + \
                        "class_"+str(class_label)+"/"
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    im_path = save_folder+'/original.jpg'

                    numpy_image = temp_image
                    save_image(numpy_image, im_path)

                self.record_raw_activation_states_per_batch(
                    per_class_per_batch_data)

                if(not(number_of_batch_to_collect is None) and i == number_of_batch_to_collect - 1):
                    break

            temp = torch.from_numpy(
                self.post_activation_values_all_batches)
            pos_hrelu = HardRelu()(temp)
            neg_hrelu = HardRelu()(-temp)

            temp = pos_hrelu + neg_hrelu
            tones = torch.ones(temp.size())
            neg_hrelu = torch.where(temp == 0, tones, neg_hrelu)

            self.post_activation_values_positive_hrelu_counts = torch.sum(
                pos_hrelu, dim=0)
            self.post_activation_values_negative_hrelu_counts = torch.sum(
                neg_hrelu, dim=0)

            prob_pos = self.post_activation_values_positive_hrelu_counts / \
                self.total_tcollect_img_count
            prob_neg = self.post_activation_values_negative_hrelu_counts / \
                self.total_tcollect_img_count
            prob_zero = 1 - (prob_pos + prob_neg)
            entropy_bin_list = [prob_pos, prob_neg, prob_zero]

            self.post_activation_values_gate_entropy = calculate_entropy(
                entropy_bin_list)
            print("self.post_activation_values_all_batches",
                  self.post_activation_values_all_batches.shape)
            print("self.post_activation_values_positive_hrelu_counts",
                  self.post_activation_values_positive_hrelu_counts.size())

            post_act_values_pos_hrelu_counts = torch.sum(
                pos_hrelu, dim=[i for i in range(1, len(pos_hrelu.size()))])
            post_act_values_neg_hrelu_counts = torch.sum(
                neg_hrelu, dim=[i for i in range(1, len(neg_hrelu.size()))])

            each_sample_layer_out_size = torch.numel(
                pos_hrelu[0])
            prob_pos = post_act_values_pos_hrelu_counts / each_sample_layer_out_size
            prob_neg = post_act_values_neg_hrelu_counts / each_sample_layer_out_size
            prob_zero = 1 - (prob_pos + prob_neg)
            overall_entropy_bin = torch.vstack((prob_pos, prob_neg, prob_zero))
            overall_entropy_bin = torch.transpose(overall_entropy_bin, 0, 1)
            self.post_activation_values_entropy_per_sample = torch.zeros(
                self.total_tcollect_img_count)
            # print("overall_entropy_bin size", overall_entropy_bin.size())
            for indx in range(len(overall_entropy_bin)):
                each_entropy_bin_list = overall_entropy_bin[indx]
                temp_entr = calculate_entropy(
                    each_entropy_bin_list)
                self.post_activation_values_entropy_per_sample[indx] = temp_entr

            print("self.post_activation_values_entropy_per_sample",
                  self.post_activation_values_entropy_per_sample.size())
            print("each_sample_layer_out_size", each_sample_layer_out_size)

        # print("self.post_activation_values_all_batches",
        #       self.post_activation_values_all_batches)

    def get_wandb_log_dict(self):
        log_dict = {
            "total_tcollect_img_count": self.total_tcollect_img_count
        }
        return log_dict

    def generate_raw_activation_stats_per_class(self, exp_type, per_class_dataset, class_label, class_indx, number_of_batch_to_collect, classes, model_arch_type, dataset,
                                                is_act_collection_on_train, is_class_segregation_on_ground_truth, activation_calculation_batch_size,
                                                wand_project_name, wandb_group_name, torch_seed,
                                                root_save_prefix="root/ACT_PATTERN_PER_CLASS", final_postfix_for_save="", analysed_model_path="",
                                                is_save_graph_visualizations=True, is_save_activation_records=True, wandb_config_additional_dict=None):
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

        self.image_save_prefix_folder = str(root_save_prefix)+"/"+str(dataset_str)+"/"+list_of_classes_to_train_on_str+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/_ACT_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
            seg_over_what_str)+"/TMP_COLL_BS_"+str(activation_calculation_batch_size)+"_NO_TO_COLL_"+str(number_of_batch_to_collect)+"/_torch_seed_"+str(torch_seed)+"/" + str(final_postfix_for_save) + "/"

        if not os.path.exists(self.image_save_prefix_folder):
            os.makedirs(self.image_save_prefix_folder)

        wandb_run_name = self.image_save_prefix_folder.replace(
            "/", "").replace(root_save_prefix, class_label)
        self.wandb_run_name = wandb_run_name
        wandb_config = get_wandb_config(exp_type, class_label, class_indx, classes, model_arch_type, dataset_str, is_act_collection_on_train,
                                        is_class_segregation_on_ground_truth,
                                        activation_calculation_batch_size, torch_seed, analysed_model_path,
                                        number_of_batch_to_collect=number_of_batch_to_collect)
        if(wandb_config_additional_dict is not None):
            wandb_config.update(wandb_config_additional_dict)

        self.wandb_config = wandb_config
        self.wandb_group_name = wandb_group_name

        self.record_raw_activation_states(per_class_data_loader, class_label,
                                          number_of_batch_to_collect, is_save_original_image=False)

        save_folder = self.image_save_prefix_folder + \
            "class_"+str(class_label)+"/"

        # if(is_save_activation_records == True):
        #     temp_model = self.model
        #     self.model = None
        #     if not os.path.exists(save_folder):
        #         os.makedirs(save_folder)
        #     with open(save_folder+'/raw_analyser_state.pkl', 'wb') as out_file:
        #         pickle.dump(self, out_file)
        #     self.model = temp_model

        self.save_and_log_states(
            wand_project_name, is_save_graph_visualizations=is_save_graph_visualizations)

        if(is_log_wandb):
            wandb.finish()

    def save_and_log_states(self, wand_project_name, wandb_group_name=None, root_save_prefix=None, final_postfix_for_save=None, is_save_graph_visualizations=True):
        is_log_wandb = not(wand_project_name is None)
        log_dict = self.get_wandb_log_dict()
        print("log_dict", log_dict)

        if(is_log_wandb):
            if(wandb_group_name is not None):
                self.wandb_group_name = wandb_group_name
            wandb_run_name = self.wandb_run_name
            if(root_save_prefix is not None and final_postfix_for_save is not None):
                wandb_run_name = self.wandb_run_name.replace(self.root_save_prefix, root_save_prefix).replace(
                    self.final_postfix_for_save, final_postfix_for_save)

            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                group=f"{self.wandb_group_name}",
                config=self.wandb_config,
            )

            wandb.log(log_dict)

        if(is_save_graph_visualizations == True):
            save_folder = self.image_save_prefix_folder + \
                "class_"+str(self.class_label)+"/"
            if(root_save_prefix is not None and final_postfix_for_save is not None):
                save_folder = save_folder.replace(self.root_save_prefix, root_save_prefix).replace(
                    self.final_postfix_for_save, final_postfix_for_save)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            self.save_raw_recorded_activation_states(
                save_folder, is_save_video_data=is_save_video_data)


def calculate_entropy(entropy_bin_list):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    entropy = torch.zeros(
        size=entropy_bin_list[0].size(), device=device)
    zero_default = torch.zeros(
        size=entropy_bin_list[0].size(), device=device)

    for each_bin_value in entropy_bin_list:
        each_bin_value = each_bin_value.to(device)

        pre_entropy = torch.where(
            each_bin_value == 0., zero_default, (each_bin_value * torch.log2(each_bin_value)))
        entropy += pre_entropy

    return -entropy


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


def run_raw_activation_analysis_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth,
                                          activation_calculation_batch_size, number_of_batch_to_collect, wand_project_name, is_split_validation,
                                          valid_split_size, torch_seed, wandb_group_name, exp_type,
                                          root_save_prefix='root/RAW_ACT_PATTERN_ANALYSIS', final_postfix_for_save="",
                                          custom_model=None, custom_data_loader=None, class_indx_to_visualize=None, analysed_model_path="",
                                          is_save_graph_visualizations=True, is_save_activation_records=True, wandb_config_additional_dict=None):
    if(root_save_prefix is None):
        root_save_prefix = 'root/RAW_ACT_PATTERN_ANALYSIS'
    if(final_postfix_for_save is None):
        final_postfix_for_save = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running for "+str(dataset_str))

    coll_seed_gen = torch.Generator()
    coll_seed_gen.manual_seed(torch_seed)

    if(custom_data_loader is None):
        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            ret_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=is_split_validation)
    else:
        trainloader, testloader = custom_data_loader

    if(trainloader is not None):
        train_dataset = generate_dataset_from_loader(trainloader)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=ret_config.batch_size,
                                                  shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)
    if(testloader is not None):
        test_dataset = generate_dataset_from_loader(testloader)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=ret_config.batch_size,
                                                 shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)

    print("Preprocessing and dataloader process completed of type:{} for dataset:{}".format(
        model_arch_type, dataset_str))

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

            act_analyser = RawActivationAnalyser(
                model)
            act_analyser.generate_raw_activation_stats_per_class(exp_type, per_class_dataset, class_label, c_indx, number_of_batch_to_collect, classes, model_arch_type, dataset,
                                                                 is_template_image_on_train, is_class_segregation_on_ground_truth, activation_calculation_batch_size,
                                                                 wand_project_name, wandb_group_name, torch_seed,
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

        act_analyser = RawActivationAnalyser(
            model)
        act_analyser.generate_raw_activation_stats_per_class(exp_type, analyse_dataset, class_label, c_indx, number_of_batch_to_collect, classes, model_arch_type, dataset,
                                                             is_template_image_on_train, is_class_segregation_on_ground_truth, activation_calculation_batch_size,
                                                             wand_project_name, wandb_group_name, torch_seed,
                                                             root_save_prefix, final_postfix_for_save, analysed_model_path, is_save_graph_visualizations, is_save_activation_records,
                                                             wandb_config_additional_dict=wandb_config_additional_dict)
        list_of_act_analyser.append(act_analyser)

    return list_of_act_analyser


def load_and_save_activation_analysis_on_config(dataset, valid_split_size, model_arch_type, exp_type, wand_project_name, load_analyser_base_folder, wandb_group_name=None,
                                                root_save_prefix=None, final_postfix_for_save=None,
                                                class_indx_to_visualize=None, is_save_graph_visualizations=True):
    is_log_wandb = not(wand_project_name is None)

    print("Running for "+str(dataset_str))
    classes, _, _ = get_preprocessing_and_other_configs(
        dataset, valid_split_size)

    print("load_and_save_activation_analysis_on_config of type:{} for dataset:{}".format(
        model_arch_type, dataset_str))

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
            with open(load_folder+'/raw_analyser_state.pkl', 'rb') as in_file:
                act_analyser = pickle.load(in_file)
                act_analyser.save_and_log_states(
                    wand_project_name, wandb_group_name, root_save_prefix, final_postfix_for_save, is_save_graph_visualizations)
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
                wand_project_name, wandb_group_name, root_save_prefix, final_postfix_for_save, is_save_graph_visualizations)
            if(is_log_wandb):
                wandb.finish()

        list_of_act_analyser.append(act_analyser)

    return list_of_act_analyser


def run_generate_scheme(models_base_path, to_be_analysed_dataloader, custom_data_loader, it_start=1, num_iter=None, direct_model_path=None):
    if(num_iter is None):
        num_iter = it_start + 1
    list_of_list_of_act_analyser = []
    list_of_save_prefixes = []
    list_of_save_postfixes = []

    if(direct_model_path is None):
        list_of_model_paths = []
        if(models_base_path != None):
            for i in range(it_start, num_iter):
                each_model_prefix = "aug_conv4_dlgn_iter_{}_dir.pt".format(i)
                # each_model_prefix = "aug_conv4_dlgn_iter_{}_dir.pt".format(i)
                list_of_model_paths.append(models_base_path+each_model_prefix)
                list_of_save_prefixes.append(
                    str(models_base_path)+"/RAW_ACT_ANALYSIS/"+str(sub_scheme_type))
                list_of_save_postfixes.append("/aug_indx_{}".format(i))

        else:
            list_of_model_paths = [None]
            list_of_save_prefixes = [
                "root/RAW_ACT_PATTERN_ANALYSIS/"+str(sub_scheme_type)]
            list_of_save_postfixes = [None]
    else:
        list_of_model_paths = [direct_model_path]
        if('CLEAN' in direct_model_path):
            models_base_path = direct_model_path[0:direct_model_path.rfind(
                ".pt")]
        else:
            models_base_path = direct_model_path[0:direct_model_path.rfind(
                "/")+1]

        temp_base_path = models_base_path
        if("CLEAN_TRAINING" in direct_model_path or 'epoch' in direct_model_path or 'aug' in direct_model_path):
            temp_base_path = direct_model_path[0:direct_model_path.rfind(
                ".pt")]+"/"
        list_of_save_prefixes.append(
            str(temp_base_path)+"/RAW_ACT_ANALYSIS/"+str(sub_scheme_type))
        list_of_save_postfixes.append("")

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
            list_of_act_analyser = run_raw_activation_analysis_on_config(dataset, model_arch_type, is_act_collection_on_train, is_class_segregation_on_ground_truth,
                                                                         activation_calculation_batch_size, number_of_batch_to_collect, wand_project_name_for_gen, is_split_validation,
                                                                         valid_split_size, torch_seed, wandb_group_name, exp_type, is_save_activation_records=is_save_activation_records,
                                                                         class_indx_to_visualize=class_ind_visualize, custom_data_loader=custom_data_loader,
                                                                         custom_model=custom_model, root_save_prefix=each_save_prefix, final_postfix_for_save=each_save_postfix,
                                                                         is_save_graph_visualizations=is_save_graph_visualizations, analysed_model_path=analysed_model_path)
        elif(sub_scheme_type == 'OVER_RECONSTRUCTED'):
            pass
        elif(sub_scheme_type == 'OVER_ADVERSARIAL'):

            final_postfix_for_save = "/RAW_ADV_SAVES/adv_type_{}/EPS_{}/eps_stp_size_{}/adv_steps_{}/on_train_{}/{}".format(
                adv_attack_type, eps, eps_step_size, number_of_adversarial_optimization_steps, is_act_collection_on_train, each_save_postfix)
            adv_save_path = models_base_path + final_postfix_for_save+"/adv_dataset.npy"

            wandb_config_additional_dict = {"eps": eps, "adv_atack_type": adv_attack_type, "num_of_adversarial_optim_stps":
                                            number_of_adversarial_optimization_steps, "eps_stp_size": eps_step_size, "adv_target": adv_target}

            each_save_postfix = "EPS_{}/ADV_TYPE_{}/NUM_ADV_STEPS_{}/eps_step_size_{}/".format(
                eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size) + each_save_postfix

            is_current_adv_aug_available = os.path.exists(adv_save_path)
            if(is_current_adv_aug_available):
                with open(adv_save_path, 'rb') as file:
                    npzfile = np.load(adv_save_path)
                    list_of_adv_images = npzfile['x']
                    list_of_labels = npzfile['y']
                    adv_dataset = CustomSimpleDataset(
                        list_of_adv_images, list_of_labels)
                    print("Loading adversarial examples from path:",
                          adv_save_path)
            else:
                adv_dataset = generate_adv_examples(
                    to_be_analysed_dataloader, custom_model, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, number_of_batch_to_collect, is_save_adv=is_save_adv, save_path=adv_save_path)

            to_be_analysed_adversarial_dataloader = torch.utils.data.DataLoader(
                adv_dataset, shuffle=False, batch_size=128)

            if(is_act_collection_on_train):
                custom_loader = to_be_analysed_adversarial_dataloader, None
            else:
                custom_loader = None, to_be_analysed_adversarial_dataloader

            list_of_act_analyser = run_raw_activation_analysis_on_config(dataset, model_arch_type, is_act_collection_on_train, is_class_segregation_on_ground_truth,
                                                                         activation_calculation_batch_size, number_of_batch_to_collect, wand_project_name_for_gen,
                                                                         is_split_validation, valid_split_size, torch_seed, wandb_group_name, exp_type, is_save_activation_records=is_save_activation_records,
                                                                         custom_data_loader=custom_loader, custom_model=custom_model, root_save_prefix=each_save_prefix,
                                                                         final_postfix_for_save=each_save_postfix, is_save_graph_visualizations=is_save_graph_visualizations,
                                                                         class_indx_to_visualize=class_ind_visualize,
                                                                         analysed_model_path=analysed_model_path, wandb_config_additional_dict=wandb_config_additional_dict)

        list_of_list_of_act_analyser.append(list_of_act_analyser)

    return list_of_list_of_act_analyser


def generate_per_class_combination_stats(list_of_act_analyser_of_class_comb):
    # min_common_batch = list_of_act_analyser_of_class_comb[0].total_tcollect_img_count
    # for each_class_act_analyser in list_of_act_analyser_of_class_comb:
    #     if(min_common_batch < each_class_act_analyser.total_tcollect_img_count):
    #         min_common_batch = each_class_act_analyser.total_tcollect_img_count

    common_class_pos_hrelu_difference = list_of_act_analyser_of_class_comb[
        0].post_activation_values_positive_hrelu_counts.clone()
    common_class_neg_hrelu_difference = list_of_act_analyser_of_class_comb[
        0].post_activation_values_negative_hrelu_counts.clone()

    common_class_total_pos_hrelu_counts = common_class_pos_hrelu_difference.clone()
    common_class_total_neg_hrelu_counts = common_class_neg_hrelu_difference.clone()

    for ii in range(1, len(list_of_act_analyser_of_class_comb)):
        each_class_act_analyser = list_of_act_analyser_of_class_comb[ii]
        common_class_pos_hrelu_difference = common_class_pos_hrelu_difference - \
            each_class_act_analyser.post_activation_values_positive_hrelu_counts
        common_class_neg_hrelu_difference = common_class_neg_hrelu_difference - \
            each_class_act_analyser.post_activation_values_negative_hrelu_counts
        common_class_total_pos_hrelu_counts = common_class_total_pos_hrelu_counts + \
            each_class_act_analyser.post_activation_values_positive_hrelu_counts
        common_class_total_neg_hrelu_counts = common_class_total_neg_hrelu_counts + \
            each_class_act_analyser.post_activation_values_negative_hrelu_counts

    pos_hrelu_entropy_bin_list = []
    neg_hrelu_entropy_bin_list = []
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    zeros_t = torch.zeros(
        list_of_act_analyser_of_class_comb[0].post_activation_values_positive_hrelu_counts.size())
    pos_sum = torch.zeros(
        list_of_act_analyser_of_class_comb[0].post_activation_values_positive_hrelu_counts.size(), device=device)
    neg_sum = torch.zeros(
        list_of_act_analyser_of_class_comb[0].post_activation_values_positive_hrelu_counts.size(), device=device)
    for ind in range(len(list_of_act_analyser_of_class_comb)-1):
        each_class_act_analyser = list_of_act_analyser_of_class_comb[ind]
        temp_pos = each_class_act_analyser.post_activation_values_positive_hrelu_counts / \
            common_class_total_pos_hrelu_counts
        temp_pos = torch.where(
            common_class_total_pos_hrelu_counts > 0., temp_pos, zeros_t)
        pos_hrelu_entropy_bin_list.append(temp_pos)
        pos_sum += temp_pos.to(device)
        temp_neg = each_class_act_analyser.post_activation_values_negative_hrelu_counts / \
            common_class_total_neg_hrelu_counts
        temp_neg = torch.where(
            common_class_total_neg_hrelu_counts > 0., temp_neg, zeros_t)
        neg_hrelu_entropy_bin_list.append(temp_neg)
        neg_sum += temp_neg.to(device)

    pos_hrelu_entropy_bin_list.append(1-pos_sum)
    neg_hrelu_entropy_bin_list.append(1-neg_sum)

    pos_hrelu_entr_per_pixel = calculate_entropy(pos_hrelu_entropy_bin_list)
    neg_hrelu_entr_per_pixel = calculate_entropy(neg_hrelu_entropy_bin_list)

    return common_class_pos_hrelu_difference, common_class_neg_hrelu_difference, pos_hrelu_entr_per_pixel, neg_hrelu_entr_per_pixel


def write_stats_to_file(write_to_file_path, input_tensor, append_title):
    with open(write_to_file_path, "w") as myfile:
        ovrl_min, ovrl_max, ovrl_std, ovrl_mean, min_per_pxl, max_per_pxl, std_per_pxl, mean_per_pxl, mean_min_per_pxl, sd_min_per_pxl, mean_max_per_pxl, sd_max_per_pxl, mean_std_per_pxl, sd_std_per_pxl, mean_mean_per_pxl, sd_mean_per_pxl = generate_stats(
            input_tensor)

        myfile.write("Overall {} Min = {} \n".format(append_title, ovrl_min))
        myfile.write("Overall {} Max = {} \n".format(append_title, ovrl_max))
        myfile.write("Overall {} STD = {} \n".format(append_title, ovrl_std))
        myfile.write("Overall {} Mean = {} \n".format(append_title, ovrl_mean))
        myfile.write(
            "======================== Min per pixel stats ==================== \n")
        myfile.write("Mean = {} \t".format(mean_min_per_pxl))
        myfile.write("SD = {}\n".format(sd_min_per_pxl))
        myfile.write(
            "======================== Max per pixel stats ==================== \n")
        myfile.write("Mean = {} \t".format(mean_max_per_pxl))
        myfile.write("SD = {} \n".format(sd_max_per_pxl))
        myfile.write(
            "======================== Std per pixel stats ==================== \n")
        myfile.write("Mean = {} \t".format(mean_std_per_pxl))
        myfile.write("SD = {} \n".format(sd_std_per_pxl))
        myfile.write(
            "======================== Mean per pixel stats ==================== \n")
        myfile.write("Mean = {} \t".format(mean_mean_per_pxl))
        myfile.write("SD = {} \n".format(sd_mean_per_pxl))

        myfile.write("{} Min per pixel = {} \n".format(
            append_title, min_per_pxl))
        myfile.write("{} Max per pixel = {} \n".format(
            append_title, max_per_pxl))
        myfile.write("{} STD per pixel = {} \n".format(
            append_title, std_per_pxl))
        myfile.write("{} Mean per pixel = {} \n".format(
            append_title, mean_per_pxl))


def generate_class_combination_statistics(list_of_list_of_act_analyser, class_combination_tuple_list):
    for each_list_of_act_analyser in list_of_list_of_act_analyser:
        for (class_indx_to_generate_combinations, class_combination_length) in class_combination_tuple_list:
            for each_class_combination in itertools.combinations(class_indx_to_generate_combinations, class_combination_length):
                comb_tag = str('_'.join(map(str, each_class_combination)))
                print("each_class_combination: {}".format(comb_tag))
                current_combination_act_analysers = [
                    None] * class_combination_length
                for ii in range(len(each_class_combination)):
                    each_class_in_comb = each_class_combination[ii]
                    current_combination_act_analysers[ii] = each_list_of_act_analyser[each_class_in_comb]

                common_class_pos_hrelu_difference, common_class_neg_hrelu_difference, pos_hrelu_entr_per_pixel, neg_hrelu_entr_per_pixel = generate_per_class_combination_stats(
                    current_combination_act_analysers)

                save_folder = current_combination_act_analysers[0].image_save_prefix_folder + "/CLASS_COMPARISION/COMB_LENGTH_" + str(class_combination_length) +\
                    "/class_" + comb_tag \
                    + "/"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                if(is_save_graph_visualizations):
                    print("Writing class combination:{} under path:{}".format(
                        each_class_combination, save_folder))
                    generate_plain_image(common_class_pos_hrelu_difference,
                                         save_folder+"c_Diff_Positive_HRelu_counts.jpg", is_standarize=False, is_standarize_01=True)
                    generate_plain_image(common_class_neg_hrelu_difference,
                                         save_folder+"c_Diff_Negative_HRelu_counts.jpg", is_standarize=False, is_standarize_01=True)
                    generate_plain_image(pos_hrelu_entr_per_pixel,
                                         save_folder+"c_PosHrelu_Entropy.jpg", is_standarize=False, is_standarize_01=True)
                    generate_plain_image(neg_hrelu_entr_per_pixel,
                                         save_folder+"c_NegHrelu_Entropy.jpg", is_standarize=False, is_standarize_01=True)

                write_stats_to_file(save_folder+"c_Diff_Positive_HRelu_counts.txt",
                                    common_class_pos_hrelu_difference, "Diff PHrelu Count")
                write_stats_to_file(save_folder+"c_PosHrelu_Entropy.txt",
                                    pos_hrelu_entr_per_pixel, "PHrelu Entropy")
                write_stats_to_file(save_folder+"c_NegHrelu_Entropy.txt",
                                    neg_hrelu_entr_per_pixel, "NHrelu Entropy")


def generate_class_pair_wise_stats_array(list_of_act_analyser, classes):
    current_combination_act_analysers = [None] * 2
    pairwise_class_stats = []
    for each_source_c_indx in range(len(classes)):
        current_class_pairing_stats = []
        current_combination_act_analysers[0] = list_of_act_analyser[each_source_c_indx]
        for each_dest_c_indx in range(len(classes)):
            current_combination_act_analysers[1] = list_of_act_analyser[each_dest_c_indx]
            common_class_pos_hrelu_diff, common_class_neg_hrelu_diff, pos_hrelu_entr_per_pixel, neg_hrelu_entr_per_pixel = generate_per_class_combination_stats(
                current_combination_act_analysers)
            ovrl_phrelu_diff_min, ovrl_phrelu_diff_max, ovrl_phrelu_diff_std, ovrl_phrelu_diff_mean, _, _, _, _, _, _, _, _, _, _, _, _ = generate_stats(
                common_class_pos_hrelu_diff)
            ovrl_nhrelu_diff_min, ovrl_nhrelu_diff_max, ovrl_nhrelu_diff_std, ovrl_nhrelu_diff_mean, _, _, _, _, _, _, _, _, _, _, _, _ = generate_stats(
                common_class_neg_hrelu_diff)
            ovrl_phrelu_entr_min, ovrl_phrelu_entr_max, ovrl_phrelu_entr_std, ovrl_phrelu_entr_mean, _, _, _, _, _, _, _, _, _, _, _, _ = generate_stats(
                pos_hrelu_entr_per_pixel)
            ovrl_nhrelu_entr_min, ovrl_nhrelu_entr_max, ovrl_nhrelu_entr_std, ovrl_nhrelu_entr_mean, _, _, _, _, _, _, _, _, _, _, _, _ = generate_stats(
                neg_hrelu_entr_per_pixel)
            current_pair_stats = [ovrl_phrelu_diff_min.cpu().numpy(), ovrl_phrelu_diff_max.cpu().numpy(), ovrl_phrelu_diff_std.cpu().numpy(), ovrl_phrelu_diff_mean.cpu().numpy(), ovrl_phrelu_entr_min.cpu().numpy(), ovrl_phrelu_entr_max.cpu().numpy(), ovrl_phrelu_entr_std.cpu().numpy(), ovrl_phrelu_entr_mean.cpu().numpy(),
                                  ovrl_nhrelu_diff_min.cpu().numpy(), ovrl_nhrelu_diff_max.cpu().numpy(), ovrl_nhrelu_diff_std.cpu().numpy(), ovrl_nhrelu_diff_mean.cpu().numpy(), ovrl_nhrelu_entr_min.cpu().numpy(), ovrl_nhrelu_entr_max.cpu().numpy(), ovrl_nhrelu_entr_std.cpu().numpy(), ovrl_nhrelu_entr_mean.cpu().numpy()]
            current_class_pairing_stats.append(current_pair_stats)
        pairwise_class_stats.append(current_class_pairing_stats)
    return pairwise_class_stats


def generate_all_class_stats(list_of_act_analyser):
    all_class_stats = []
    for each_act_analyser in list_of_act_analyser:
        ovrl_phrelu_count_min, ovrl_phrelu_count_max, ovrl_phrelu_count_std, ovrl_phrelu_count_mean, _, _, _, _, _, _, _, _, _, _, _, _ = generate_stats(
            each_act_analyser.post_activation_values_positive_hrelu_counts)
        ovrl_nhrelu_count_min, ovrl_nhrelu_count_max, ovrl_nhrelu_count_std, ovrl_nhrelu_count_mean, _, _, _, _, _, _, _, _, _, _, _, _ = generate_stats(
            each_act_analyser.post_activation_values_negative_hrelu_counts)
        ovrl_phrelu_entr_min, ovrl_phrelu_entr_max, ovrl_phrelu_entr_std, ovrl_phrelu_entr_mean, _, _, _, _, _, _, _, _, _, _, _, _ = generate_stats(
            each_act_analyser.post_activation_values_gate_entropy)
        if hasattr(each_act_analyser, 'diff_counts_post_activation_values_all_layers'):
            ovrl_nhrelu_entr_min, ovrl_nhrelu_entr_max, ovrl_nhrelu_entr_std, ovrl_nhrelu_entr_mean, _, _, _, _, _, _, _, _, _, _, _, _ = generate_stats(
                each_act_analyser.diff_counts_post_activation_values_all_layers)
        else:
            temp = torch.zeros(1)
            ovrl_nhrelu_entr_min, ovrl_nhrelu_entr_max, ovrl_nhrelu_entr_std, ovrl_nhrelu_entr_mean = temp, temp, temp, temp
        current_per_class_stats = [ovrl_phrelu_count_min.item(), ovrl_phrelu_count_max.item(), ovrl_phrelu_count_std.item(), ovrl_phrelu_count_mean.item(), ovrl_phrelu_entr_min.item(), ovrl_phrelu_entr_max.item(), ovrl_phrelu_entr_std.item(), ovrl_phrelu_entr_mean.item(),
                                   ovrl_nhrelu_count_min.item(), ovrl_nhrelu_count_max.item(), ovrl_nhrelu_count_std.item(), ovrl_nhrelu_count_mean.item(), ovrl_nhrelu_entr_min.item(), ovrl_nhrelu_entr_max.item(), ovrl_nhrelu_entr_std.item(), ovrl_nhrelu_entr_mean.item()]
        all_class_stats.append(current_per_class_stats)

    return all_class_stats


def merge_all_class_stats(all_stats_arr):
    merged_all_class_stats = []
    for each_class_ind in range(len(all_stats_arr[0])):
        pc_stats = all_stats_arr[0][each_class_ind]
        per_class_stats = []
        for each_col_ind in range(len(pc_stats)):
            for f_in in range(len(all_stats_arr)):
                per_class_stats.append(
                    all_stats_arr[f_in][each_class_ind][each_col_ind])
        merged_all_class_stats.append(per_class_stats)
    return merged_all_class_stats


def generate_all_classes_excel_report(csv_file_name, std_model_merged_all_classes_stats, adv_model_merged_all_classes_stats):
    # Open an Excel workbook
    workbook = xlsxwriter.Workbook(csv_file_name+'.xlsx')

    # Set up a format
    adv_example_of_std_model_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'red', 'bg_color': 'white'})
    orig_example_of_std_model_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'black', 'bg_color': 'white'})
    diff_example_of_std_model_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'blue', 'bg_color': 'white'})
    adv_example_of_adv_model_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'red', 'bg_color': 'yellow'})
    orig_example_of_adv_model_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'black', 'bg_color': 'yellow'})
    diff_example_of_adv_model_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'blue', 'bg_color': 'yellow'})

    header_format1 = workbook.add_format(
        properties={'bold': True, 'font_color': 'pink'})
    header_format2 = workbook.add_format(
        properties={'bold': True, 'font_color': 'green'})

    header_format = [header_format1, header_format2]

    property_names = ["MIN-Ovral Act Gate Diff", "MAX-Ovral Act Gate Diff", "STD-Ovral Act Gate Diff", "MEAN-Ovral Active Gate Diff", "MIN-Ovral Act Gate Entropy", "MAX-Ovral Act Gate Entropy", "STD-Ovral Act Gate Entropy", "MEAN-Ovral Act Gate Entropy",
                      "MIN-Ovral INAct Gate Diff", "MAX-Ovral INAct Gate Diff", "STD-Ovral INAct Gate Diff", "MEAN-Ovral INAct Gate Diff", "MIN-Ovral Diff Gate Count", "MAX-Ovral Diff Gate Count", "STD-Ovral Diff Gate Count", "MEAN-Ovral Diff Gate Count"]
    property_types = ["ADV_EX", "ORG_EX", "DIFF_EX"]

    # Create a sheet
    worksheet = workbook.add_worksheet('Stats_Sheet')
    worksheet.freeze_panes(1, 0)

    headers = ["Type of Model", "Class Label"]
    for each_property_name in property_names:
        for each_prop_type in property_types:
            headers.append(each_prop_type+"_"+each_property_name)

    # Write the headers
    for col_num, each_header in enumerate(headers):
        worksheet.write(0, col_num, each_header,
                        header_format[((col_num - 2) // 12) % 2])

    row_num = 0
    model_type_arr = ["ADV Model", "STD Model"]
    format_arr = [[adv_example_of_adv_model_format, orig_example_of_adv_model_format, diff_example_of_adv_model_format],
                  [adv_example_of_std_model_format, orig_example_of_std_model_format, diff_example_of_std_model_format]]
    for out_in, current_model_merged_all_classes_stats in enumerate([adv_model_merged_all_classes_stats, std_model_merged_all_classes_stats]):
        for r_num, r_data in enumerate(current_model_merged_all_classes_stats):
            row_num += 1
            worksheet.write(
                row_num, 0, model_type_arr[out_in], format_arr[out_in][1])
            worksheet.write(row_num, 1, "Cl_Ind_" +
                            str(r_num), format_arr[out_in][1])
            for c_num, col_data in enumerate(r_data):
                worksheet.write(row_num, c_num+2, col_data,
                                format_arr[out_in][c_num % 3])
        row_num += 1

    workbook.close()


def generate_all_classes_only_diff_excel_report(csv_file_name, model_merged_all_classes_stats):
    # Open an Excel workbook
    workbook = xlsxwriter.Workbook(csv_file_name+'.xlsx')

    # Set up a format
    diff_example_of_std_model_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'black', 'bg_color': 'white'})
    diff_example_of_adv_model_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'red', 'bg_color': 'yellow'})

    header_format1 = workbook.add_format(
        properties={'bold': True, 'font_color': 'pink'})
    header_format2 = workbook.add_format(
        properties={'bold': True, 'font_color': 'green'})

    header_format = [header_format1, header_format2]

    property_names = ["MIN-Ovral Act Gate Diff", "MAX-Ovral Act Gate Diff", "STD-Ovral Act Gate Diff", "MEAN-Ovral Active Gate Diff", "MIN-Ovral Act Gate Entropy", "MAX-Ovral Act Gate Entropy", "STD-Ovral Act Gate Entropy", "MEAN-Ovral Act Gate Entropy",
                      "MIN-Ovral INAct Gate Diff", "MAX-Ovral INAct Gate Diff", "STD-Ovral INAct Gate Diff", "MEAN-Ovral INAct Gate Diff", "MIN-Ovral Diff Gate Count", "MAX-Ovral Diff Gate Count", "STD-Ovral Diff Gate Count", "MEAN-Ovral Diff Gate Count"]
    property_types = ["ADV_MOD", "STD_MOD"]

    # Create a sheet
    worksheet = workbook.add_worksheet('Adv_vs_orig_Sheet')
    worksheet.freeze_panes(1, 0)

    headers = ["Class Label"]
    for each_property_name in property_names:
        for each_prop_type in property_types:
            headers.append(each_prop_type+"_"+each_property_name)

    # Write the headers
    for col_num, each_header in enumerate(headers):
        worksheet.write(0, col_num, each_header,
                        header_format[((col_num - 1) // 8) % 2])

    row_num = 0
    format_arr = [diff_example_of_adv_model_format,
                  diff_example_of_std_model_format]
    for r_num, r_data in enumerate(model_merged_all_classes_stats):
        row_num += 1
        worksheet.write(row_num, 0, "Cl_Ind_" +
                        str(r_num))
        for c_num, col_data in enumerate(r_data):
            worksheet.write(row_num, c_num + 1, col_data,
                            format_arr[c_num % 2])

    workbook.close()


def merge_two_pairwise_class_stats_in_alternative_manner(stats1, stats2):
    merged_pairwise_class_stats = []
    for source_class_indx in range(len(stats1)):
        per_class_stats1 = stats1[source_class_indx]
        per_class_stats2 = stats2[source_class_indx]
        current_class_merged_pairing_stats = []
        for dest_class_indx in range(len(per_class_stats1)):
            per_class_pairing_stats1 = per_class_stats1[dest_class_indx]
            per_class_pairing_stats2 = per_class_stats2[dest_class_indx]
            current_class_pairing_stats = []
            for each_stat_indx in range(len(per_class_pairing_stats1)):
                current_class_pairing_stats.append(
                    per_class_pairing_stats1[each_stat_indx])
                current_class_pairing_stats.append(
                    per_class_pairing_stats2[each_stat_indx])
            current_class_merged_pairing_stats.append(
                current_class_pairing_stats)
        merged_pairwise_class_stats.append(current_class_merged_pairing_stats)
    return merged_pairwise_class_stats


def generate_class_pairwise_excel_report(csv_file_name, std_model_merged_classpair_stats, adv_model_merged_classpair_stats):
    # Open an Excel workbook
    workbook = xlsxwriter.Workbook(csv_file_name+'.xlsx')

    # Set up a format
    adv_example_of_std_model_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'red', 'bg_color': 'white'})
    orig_example_of_std_model_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'black', 'bg_color': 'white'})
    adv_example_of_adv_model_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'red', 'bg_color': 'yellow'})
    orig_example_of_adv_model_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'black', 'bg_color': 'yellow'})
    header_format = workbook.add_format(
        properties={'bold': True, 'font_color': 'red'})

    property_format1 = workbook.add_format(
        properties={'bold': True, 'font_color': 'pink'})
    property_format2 = workbook.add_format(
        properties={'bold': True, 'font_color': 'green'})

    property_format = [property_format1, property_format2]

    property_names = ["MIN-Ovral Act Gate Diff", "MAX-Ovral Act Gate Diff", "STD-Ovral Act Gate Diff", "MEAN-Ovral Active Gate Diff", "MIN-Ovral Act Gate Entropy", "MAX-Ovral Act Gate Entropy", "STD-Ovral Act Gate Entropy", "MEAN-Ovral Act Gate Entropy",
                      "MIN-Ovral INAct Gate Diff", "MAX-Ovral INAct Gate Diff", "STD-Ovral INAct Gate Diff", "MEAN-Ovral INAct Gate Diff", "MIN-Ovral INAct Gate Entropy", "MAX-Ovral INAct Gate Entropy", "STD-Ovral INAct Gate Entropy", "MEAN-Ovral INAct Gate Entropy"]

    # Create a sheet
    worksheet = workbook.add_worksheet('Stats_Sheet')
    worksheet.freeze_panes(1, 0)

    headers = ["Type of Model", "Source Class Label", "Property Measured"]
    for ind in range(len(std_model_merged_classpair_stats)):
        headers.append(ind)

    # Write the headers
    for col_num, each_header in enumerate(headers):
        worksheet.write(0, col_num, each_header, header_format)

    is_currently_adv = False
    row_num = 0
    for each_source_class_indx in range(len(adv_model_merged_classpair_stats)):
        per_class_adv_model_cp_stats = adv_model_merged_classpair_stats[each_source_class_indx]
        per_class_std_model_cp_stats = std_model_merged_classpair_stats[each_source_class_indx]
        per_class_adv_model_cp_stats = np.transpose(
            per_class_adv_model_cp_stats, (1, 0))
        per_class_std_model_cp_stats = np.transpose(
            per_class_std_model_cp_stats, (1, 0))
        for internal_row in range(len(per_class_adv_model_cp_stats)):
            each_pc_stat_prop = per_class_adv_model_cp_stats[internal_row]
            if(internal_row % 2 == 0):
                current_format = adv_example_of_adv_model_format
            else:
                current_format = orig_example_of_adv_model_format
            worksheet.write(row_num+1, 0, "ADV Model", current_format)
            worksheet.write(row_num+1, 1, "Class_Ind_" +
                            str(each_source_class_indx), current_format)
            worksheet.write(
                row_num+1, 2, property_names[internal_row//2], property_format[((internal_row) // 8) % 2])
            for col_num, cell_data in enumerate(each_pc_stat_prop):
                worksheet.write(row_num+1, col_num+3,
                                cell_data, current_format)
            row_num += 1

        for internal_row in range(len(per_class_std_model_cp_stats)):
            each_pc_stat_prop = per_class_std_model_cp_stats[internal_row]
            if(internal_row % 2 == 0):
                current_format = adv_example_of_std_model_format
            else:
                current_format = orig_example_of_std_model_format
            worksheet.write(row_num+1, 0, "STD Model", current_format)
            worksheet.write(row_num+1, 1, "Class_Ind_" +
                            str(each_source_class_indx), current_format)
            worksheet.write(
                row_num+1, 2, property_names[internal_row//2], property_format[((internal_row) // 8) % 2])
            for col_num, cell_data in enumerate(each_pc_stat_prop):
                worksheet.write(row_num+1, col_num+3,
                                cell_data, current_format)
            row_num += 1
        row_num += 1

    # Close the workbook
    workbook.close()


if __name__ == '__main__':
    # THIS OPERATION IS MEMORY HUNGRY! #
    # Because of the selected image is very large
    # If it gives out of memory error or locks the computer
    # Try it with a smaller image
    print("Start")
    # mnist , cifar10 , fashion_mnist
    dataset = 'mnist'
    # cifar10_conv4_dlgn , cifar10_vgg_dlgn_16 , dlgn_fc_w_128_d_4 , random_conv4_dlgn , random_vggnet_dlgn
    # random_conv4_dlgn_sim_vgg_wo_bn , cifar10_conv4_dlgn_sim_vgg_wo_bn , cifar10_conv4_dlgn_sim_vgg_with_bn
    # random_conv4_dlgn_sim_vgg_with_bn , cifar10_conv4_dlgn_with_inbuilt_norm , random_cifar10_conv4_dlgn_with_inbuilt_norm
    # cifar10_vgg_dlgn_16_with_inbuilt_norm , random_cifar10_vgg_dlgn_16_with_inbuilt_norm
    # random_cifar10_conv4_dlgn_with_bn_with_inbuilt_norm , cifar10_conv4_dlgn_with_bn_with_inbuilt_norm
    # cifar10_conv4_dlgn_with_inbuilt_norm_with_flip_crop
    # cifar10_conv4_dlgn_with_bn_with_inbuilt_norm_with_flip_crop
    # cifar10_vgg_dlgn_16_with_inbuilt_norm_wo_bn
    # plain_pure_conv4_dnn , conv4_dlgn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net_n16_small
    # fc_dnn , fc_dlgn , fc_dgn
    model_arch_type = 'fc_dnn'
    is_save_video_data = False
    # If False, then on test
    is_act_collection_on_train = True
    # If False, then segregation is over model prediction
    is_class_segregation_on_ground_truth = True
    activation_calculation_batch_size = 64
    number_of_batch_to_collect = None
    # wand_project_name = 'raw_activation_analyser'
    # wand_project_name = 'new_raw_activation_analysis_class'
    wand_project_name = None
    wand_project_name_for_gen = None
    wand_project_name_for_merge = None
    # wand_project_name_for_merge = "raw_activation_analysis_augmented_mnist_conv4_dlgn_n16_small"
    wandb_group_name = "raw_activation_analysis_augmented_mnist_conv4_dlgn"
    is_split_validation = False
    valid_split_size = 0.1
    torch_seed = 2022
    # GENERATE_RECORD_STATS_PER_CLASS ,  GENERATE_RECORD_STATS_OVERALL
    exp_type = "GENERATE_RECORD_STATS_PER_CLASS"
    is_save_graph_visualizations = True
    is_save_activation_records = False
    # GENERATE , LOAD_AND_SAVE , LOAD_AND_GENERATE_MERGE , GENERATE_MERGE_AND_SAVE , ADV_VS_ORIG_REPORT , CLASS_WISE_REPORT
    scheme_type = "CLASS_WISE_REPORT"
    # OVER_RECONSTRUCTED , OVER_ADVERSARIAL , OVER_ORIGINAL
    sub_scheme_type = 'OVER_ORIGINAL'
    # OVER_ORIGINAL_VS_ADVERSARIAL , TWO_CUSTOM_MODELS
    merge_scheme_type = "OVER_ORIGINAL_VS_ADVERSARIAL"

    # None means that train on all classes
    list_of_classes_to_train_on = None
    list_of_classes_to_train_on = [3, 8]

    classes, num_classes, ret_config = get_preprocessing_and_other_configs(
        dataset, valid_split_size)
    ret_config.list_of_classes = list_of_classes_to_train_on
    trainloader, _, testloader = preprocess_dataset_get_data_loader(
        ret_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=is_split_validation)

    dataset_str = dataset

    list_of_classes_to_train_on_str = ""
    if(list_of_classes_to_train_on is not None):
        for each_class_to_train_on in list_of_classes_to_train_on:
            list_of_classes_to_train_on_str += \
                str(each_class_to_train_on)+"_"
        dataset_str += "_"+str(list_of_classes_to_train_on_str)
        list_of_classes_to_train_on_str = "TR_ON_" + \
            list_of_classes_to_train_on_str[0:-1]
        num_classes = len(list_of_classes_to_train_on)
        temp_classes = []
        for ea_c in list_of_classes_to_train_on:
            temp_classes.append(classes[ea_c])
        classes = temp_classes

    model_arch_type_str = model_arch_type
    if("masked" in model_arch_type):
        mask_percentage = 90
        model_arch_type_str = model_arch_type_str + \
            "_PRC_"+str(mask_percentage)
    elif("fc" in model_arch_type):
        fc_width = 128
        fc_depth = 4
        nodes_in_each_layer_list = [fc_width] * fc_depth
        model_arch_type_str = model_arch_type_str + \
            "_W_"+str(fc_width)+"_D_"+str(fc_depth)

    class_combination_tuple_list = None
    class_combination_tuple_list = [
        ([i for i in range(len(classes))], 2)
    ]

    if(is_act_collection_on_train):
        custom_data_loader = trainloader, None
        to_be_analysed_dataloader = trainloader
    else:
        custom_data_loader = None, testloader
        to_be_analysed_dataloader = testloader

    if(not(wand_project_name is None)):
        wandb.login()

    if(scheme_type == "GENERATE"):

        models_base_path = None
        is_save_graph_visualizations = True

        is_save_adv = True
        eps = 0.02
        adv_attack_type = 'PGD'
        number_of_adversarial_optimization_steps = 161
        eps_step_size = 0.01
        adv_target = None

        c_indices = [i for i in range(10)]

        # class_ind_visualize = None
        if(class_combination_tuple_list is not None):
            loop_range = [0]
        else:
            loop_range = c_indices

        for c_i in loop_range:
            if(class_combination_tuple_list is not None):
                class_ind_visualize = None
            else:
                class_ind_visualize = [c_i]

            models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/"

            direct_model_path = None

            direct_model_path = "root/model/save/mnist/V2_iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.73/aug_conv4_dlgn_iter_1_dir.pt"

            list_of_list_of_act_analyser = run_generate_scheme(
                models_base_path, to_be_analysed_dataloader, custom_data_loader, direct_model_path=direct_model_path)
            if(class_combination_tuple_list is not None):
                generate_class_combination_statistics(
                    list_of_list_of_act_analyser, class_combination_tuple_list)

    elif(scheme_type == "LOAD_AND_SAVE"):
        class_ind_visualize = [9]
        # class_ind_visualize = [2, 4, 6, 8]
        # class_ind_visualize = None
        # class_ind_visualize = [3, 5,7,9]
        # class_ind_visualize = [7, 9]

        list_of_load_paths = []
        loader_base_path = None

        save_only_thres = False

        loader_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/ACT_ANALYSIS/OVER_ADVERSARIAL/mnist/MT_conv4_dlgn_ET_GENERATE_RECORD_STATS_PER_CLASS/_ACT_OV_train/SEG_GT/TMP_COLL_BS_64_NO_TO_COLL_None/_torch_seed_2022_c_thres_0.95/EPS_0.02/ADV_TYPE_PGD/NUM_ADV_STEPS_161/eps_step_size_0.01/"

        if(loader_base_path != None):
            num_iterations = 5
            # for i in range(1, num_iterations+1):
            for i in range(1, 2):
                each_model_prefix = "aug_indx_{}".format(i)
                list_of_load_paths.append(loader_base_path+each_model_prefix)

        else:
            list_of_load_paths = ["root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.5/ACT_ANALYSIS/OVER_ORIGINAL/mnist/MT_conv4_dlgn_ET_GENERATE_RECORD_STATS_PER_CLASS/_ACT_OV_train/SEG_GT/TMP_COLL_BS_64_NO_TO_COLL_None/_torch_seed_2022_c_thres_0.95/"]

        for ind in range(len(list_of_load_paths)):
            current_analyser_loader_path = list_of_load_paths[ind]
            list_of_act_analyser = load_and_save_activation_analysis_on_config(dataset, valid_split_size, model_arch_type, exp_type, wand_project_name, current_analyser_loader_path, wandb_group_name=wandb_group_name,
                                                                               class_indx_to_visualize=class_ind_visualize, is_save_graph_visualizations=True, save_only_thres=save_only_thres)

    elif(scheme_type == "GENERATE_MERGE_AND_SAVE"):
        merge_type = "DIFF"
        is_save_video_data = False

        c_indices = [i for i in range(10)]

        # class_ind_visualize = None
        if(class_combination_tuple_list is not None):
            loop_range = [0]
        else:
            loop_range = c_indices

        for c_i in loop_range:
            if(class_combination_tuple_list is not None):
                class_ind_visualize = None
            else:
                class_ind_visualize = [c_i]

            list_of_model_paths = []
            models_base_path = None

            models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_plain_pure_conv4_dnn_n16_small_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/"

            direct_model_path = None

            direct_model_path = "root/model/save/fashion_mnist/V2_iterative_augmenting/DS_fashion_mnist/MT_conv4_deep_gated_net_n16_small_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.73/aug_conv4_dlgn_iter_1_dir_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.06/batch_size_128/eps_stp_size_0.06/adv_steps_80/adv_model_dir.pt"

            if(merge_scheme_type == "OVER_ORIGINAL_VS_ADVERSARIAL"):
                num_iterations = 1
                it_start = 1
                for current_it_start in range(it_start, num_iterations + 1):
                    sub_scheme_type = 'OVER_ORIGINAL'
                    is_save_graph_visualizations = True

                    list_of_list_of_act_analyser_orig = run_generate_scheme(
                        models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=direct_model_path)
                    if(class_combination_tuple_list is not None):
                        generate_class_combination_statistics(
                            list_of_list_of_act_analyser_orig, class_combination_tuple_list)

                    sub_scheme_type = 'OVER_ADVERSARIAL'
                    is_save_adv = True
                    eps = 0.06
                    adv_attack_type = 'PGD'
                    number_of_adversarial_optimization_steps = 161
                    eps_step_size = 0.01
                    adv_target = None

                    list_of_list_of_act_analyser_adv = run_generate_scheme(
                        models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=direct_model_path)
                    if(class_combination_tuple_list is not None):
                        generate_class_combination_statistics(
                            list_of_list_of_act_analyser_adv, class_combination_tuple_list)

                    for ind in range(len(list_of_list_of_act_analyser_adv)):
                        list_of_act_analyser1 = list_of_list_of_act_analyser_adv[ind]
                        list_of_act_analyser2 = list_of_list_of_act_analyser_orig[ind]

                        if(merge_type == "DIFF"):
                            is_save_video_data = True
                            is_save_graph_visualizations = True
                            list_of_merged_act1_act2 = diff_merge_two_activation_analysis(merge_type,
                                                                                          list_of_act_analyser1, list_of_act_analyser2, wand_project_name=wand_project_name_for_merge, is_save_graph_visualizations=is_save_graph_visualizations)
                            if(class_combination_tuple_list is not None):
                                generate_class_combination_statistics(
                                    [list_of_merged_act1_act2], class_combination_tuple_list)

            elif(merge_scheme_type == "TWO_CUSTOM_MODELS"):
                direct_model_path1 = "root/model/save/mnist/adversarial_training/MT_conv4_dlgn_n16_small_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.06/batch_size_128/eps_stp_size_0.06/adv_steps_80/adv_model_dir.pt"
                direct_model_path2 = "root/model/save/mnist/V2_iterative_augmenting/DS_mnist/MT_conv4_dlgn_n16_small_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.73/aug_conv4_dlgn_iter_1_dir.pt"
                num_iterations = 1
                it_start = 1
                for current_it_start in range(it_start, num_iterations + 1):
                    sub_scheme_type = 'OVER_ADVERSARIAL'
                    # sub_scheme_type = 'OVER_ORIGINAL'
                    is_save_graph_visualizations = True

                    is_save_adv = True
                    eps = 0.06
                    adv_attack_type = 'PGD'
                    number_of_adversarial_optimization_steps = 161
                    eps_step_size = 0.01
                    adv_target = None

                    list_of_list_of_act_analyser_m1 = run_generate_scheme(
                        models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=direct_model_path1)
                    if(class_combination_tuple_list is not None):
                        generate_class_combination_statistics(
                            list_of_list_of_act_analyser_m1, class_combination_tuple_list)

                    list_of_list_of_act_analyser_m2 = run_generate_scheme(
                        models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=direct_model_path2)
                    if(class_combination_tuple_list is not None):
                        generate_class_combination_statistics(
                            list_of_list_of_act_analyser_m2, class_combination_tuple_list)

                    for ind in range(len(list_of_list_of_act_analyser_m2)):
                        list_of_act_analyser1 = list_of_list_of_act_analyser_m2[ind]
                        list_of_act_analyser2 = list_of_list_of_act_analyser_m1[ind]

                        if(merge_type == "DIFF"):
                            is_save_graph_visualizations = True
                            list_of_merged_act1_act2 = diff_merge_two_activation_analysis(merge_type,
                                                                                          list_of_act_analyser1, list_of_act_analyser2, wand_project_name=wand_project_name_for_merge, is_save_graph_visualizations=is_save_graph_visualizations)
                            if(class_combination_tuple_list is not None):
                                generate_class_combination_statistics(
                                    [list_of_merged_act1_act2], class_combination_tuple_list)

    elif(scheme_type == "LOAD_AND_GENERATE_MERGE"):
        merge_type = "DIFF"
        # class_ind_visualize = [0]
        # class_ind_visualize = [2, 4, 6, 8]
        class_ind_visualize = None
        # class_ind_visualize = [3, 5,7,9]
        # class_ind_visualize = [7, 9]

        list_of_load_paths1 = []
        list_of_load_paths2 = []
        loader_base_path1 = None
        loader_base_path2 = None

        save_only_thres = False

        loader_base_path1 = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_n16_small_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/RAW_ACT_ANALYSIS/OVER_ADVERSARIAL/mnist/MT_conv4_dlgn_n16_small_ET_GENERATE_RECORD_STATS_PER_CLASS/_ACT_OV_train/SEG_GT/TMP_COLL_BS_64_NO_TO_COLL_None/_torch_seed_2022/EPS_0.02/ADV_TYPE_PGD/NUM_ADV_STEPS_161/eps_step_size_0.01/"
        loader_base_path2 = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_n16_small_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/RAW_ACT_ANALYSIS/OVER_ORIGINAL/mnist/MT_conv4_dlgn_n16_small_ET_GENERATE_RECORD_STATS_PER_CLASS/_ACT_OV_train/SEG_GT/TMP_COLL_BS_64_NO_TO_COLL_None/_torch_seed_2022/"

        if(loader_base_path1 is not None and loader_base_path2 is not None):
            num_iterations = 1
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
                list_of_act_analyser1 = load_and_save_activation_analysis_on_config(dataset, valid_split_size, model_arch_type, exp_type, wand_project_name=None, load_analyser_base_folder=current_analyser_loader_path1,
                                                                                    class_indx_to_visualize=class_ind_visualize, is_save_graph_visualizations=False)
                list_of_act_analyser2 = load_and_save_activation_analysis_on_config(dataset, valid_split_size, model_arch_type, exp_type, wand_project_name=None, load_analyser_base_folder=current_analyser_loader_path2,
                                                                                    class_indx_to_visualize=class_ind_visualize, is_save_graph_visualizations=False)
                list_of_merged_act1_act2 = diff_merge_two_activation_analysis(merge_type,
                                                                              list_of_act_analyser1, list_of_act_analyser2, wand_project_name=wand_project_name_for_merge, is_save_graph_visualizations=is_save_graph_visualizations)
                if(class_combination_tuple_list is not None and class_ind_visualize is None):
                    generate_class_combination_statistics(
                        list_of_act_analyser1, class_combination_tuple_list)
                    generate_class_combination_statistics(
                        list_of_act_analyser2, class_combination_tuple_list)
                    generate_class_combination_statistics(
                        list_of_merged_act1_act2, class_combination_tuple_list)
    elif(scheme_type == "CLASS_WISE_REPORT"):
        is_save_adv = True
        eps = 0.06
        adv_attack_type = 'PGD'
        number_of_adversarial_optimization_steps = 161
        eps_step_size = 0.01
        adv_target = None
        class_ind_visualize = None
        is_save_graph_visualizations = False
        current_it_start = 1
        models_base_path = None
        std_model_path = "root/model/save/mnist/CLEAN_TRAINING/TR_ON_3_8/ST_2022/fc_dnn_W_128_D_4_dir.pt"
        adv_model_path = "root/model/save/mnist/CLEAN_TRAINING/TR_ON_3_8/ST_2022/fc_dnn_W_128_D_4_dir_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.06/batch_size_128/eps_stp_size_0.06/adv_steps_80/adv_model_dir.pt"

        tmp_image_over_what_str = 'test'
        if(is_act_collection_on_train):
            tmp_image_over_what_str = 'train'

        seg_over_what_str = 'MP'
        if(is_class_segregation_on_ground_truth):
            seg_over_what_str = 'GT'

        if('CLEAN' in std_model_path):
            first_prefix = std_model_path[0:std_model_path.rfind(
                ".pt")]
        else:
            first_prefix = std_model_path[0:std_model_path.rfind(
                "/")+1]

        final_postfix_for_save = "/{}/EPS_{}/ADV_TYPE_{}/NUM_ADV_STEPS_{}/eps_step_size_{}/".format(
            first_prefix, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size)

        if('CLEAN' in adv_model_path):
            prefix = adv_model_path[0:adv_model_path.rfind(
                ".pt")]
        else:
            prefix = adv_model_path[0:adv_model_path.rfind(
                "/")+1]

        image_save_prefix_folder = str(prefix)+"/RAW_ACT_ANALYSIS/CLASS_PAIRWISE_REPORT/"+str(dataset_str)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/_ACT_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
            seg_over_what_str)+"/TMP_COLL_BS_"+str(activation_calculation_batch_size)+"_NO_TO_COLL_"+str(number_of_batch_to_collect)+"/_torch_seed_"+str(torch_seed)+"/" + str(final_postfix_for_save) + "/"

        if not os.path.exists(image_save_prefix_folder):
            os.makedirs(image_save_prefix_folder)

        std_model_merged_classwise_pair_stats_store_location = image_save_prefix_folder + \
            "/std_model_cp_stats.npy"
        print("std_model_merged_classwise_pair_stats_store_location",
              std_model_merged_classwise_pair_stats_store_location)
        adv_model_merged_classwise_pair_stats_store_location = image_save_prefix_folder + \
            "/adv_model_cp_stats.npy"
        xls_save_location = image_save_prefix_folder + \
            "/adv_std_model_over_adv_orig_examples_report"

        if(os.path.exists(std_model_merged_classwise_pair_stats_store_location) and os.path.exists(adv_model_merged_classwise_pair_stats_store_location)):
            with open(std_model_merged_classwise_pair_stats_store_location, 'rb') as file:
                npzfile = np.load(
                    std_model_merged_classwise_pair_stats_store_location)
                std_model_merged_classpair_stats = npzfile['arr']

            with open(adv_model_merged_classwise_pair_stats_store_location, 'rb') as file:
                npzfile = np.load(
                    adv_model_merged_classwise_pair_stats_store_location)
                adv_model_merged_classpair_stats = npzfile['arr']
        else:
            sub_scheme_type = 'OVER_ORIGINAL'
            list_of_list_of_act_analyser = run_generate_scheme(
                models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=std_model_path)
            if(class_combination_tuple_list is not None):
                generate_class_combination_statistics(
                    list_of_list_of_act_analyser, class_combination_tuple_list)
            std_model_pairwise_class_stats_for_orig = generate_class_pair_wise_stats_array(
                list_of_list_of_act_analyser[0], classes)
            print("std_model_pairwise_class_stats_for_orig shape:",
                  np.array(std_model_pairwise_class_stats_for_orig).shape)
            list_of_list_of_act_analyser = run_generate_scheme(
                models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=adv_model_path)
            if(class_combination_tuple_list is not None):
                generate_class_combination_statistics(
                    list_of_list_of_act_analyser, class_combination_tuple_list)
            adv_model_pairwise_class_stats_for_orig = generate_class_pair_wise_stats_array(
                list_of_list_of_act_analyser[0], classes)
            print("adv_model_pairwise_class_stats_for_orig shape:",
                  np.array(adv_model_pairwise_class_stats_for_orig).shape)

            sub_scheme_type = 'OVER_ADVERSARIAL'

            list_of_list_of_act_analyser = run_generate_scheme(
                models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=std_model_path)
            if(class_combination_tuple_list is not None):
                generate_class_combination_statistics(
                    list_of_list_of_act_analyser, class_combination_tuple_list)
            std_model_pairwise_class_stats_for_adv = generate_class_pair_wise_stats_array(
                list_of_list_of_act_analyser[0], classes)
            print("std_model_pairwise_class_stats_for_adv shape:",
                  np.array(std_model_pairwise_class_stats_for_adv).shape)
            list_of_list_of_act_analyser = run_generate_scheme(
                models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=adv_model_path)
            if(class_combination_tuple_list is not None):
                generate_class_combination_statistics(
                    list_of_list_of_act_analyser, class_combination_tuple_list)
            adv_model_pairwise_class_stats_for_adv = generate_class_pair_wise_stats_array(
                list_of_list_of_act_analyser[0], classes)
            print("adv_model_pairwise_class_stats_for_adv shape:",
                  np.array(adv_model_pairwise_class_stats_for_adv).shape)

            std_model_merged_classpair_stats = np.array(merge_two_pairwise_class_stats_in_alternative_manner(
                std_model_pairwise_class_stats_for_adv, std_model_pairwise_class_stats_for_orig))
            adv_model_merged_classpair_stats = np.array(merge_two_pairwise_class_stats_in_alternative_manner(
                adv_model_pairwise_class_stats_for_adv, adv_model_pairwise_class_stats_for_orig))

            with open(adv_model_merged_classwise_pair_stats_store_location, 'wb') as file:
                np.savez(
                    file, arr=adv_model_merged_classpair_stats)

            with open(std_model_merged_classwise_pair_stats_store_location, 'wb') as file:
                np.savez(
                    file, arr=std_model_merged_classpair_stats)

        print("std_model_merged_classpair_stats shape:",
              std_model_merged_classpair_stats.shape)
        print("adv_model_merged_classpair_stats shape:",
              adv_model_merged_classpair_stats.shape)
        generate_class_pairwise_excel_report(
            xls_save_location, std_model_merged_classpair_stats, adv_model_merged_classpair_stats)
    elif(scheme_type == "ADV_VS_ORIG_REPORT"):
        merge_type = "DIFF"
        is_save_adv = True
        eps = 0.06
        adv_attack_type = 'PGD'
        number_of_adversarial_optimization_steps = 161
        eps_step_size = 0.01
        adv_target = None
        class_ind_visualize = None
        is_save_graph_visualizations = False
        current_it_start = 1
        models_base_path = None
        std_model_path = "root/model/save/mnist/CLEAN_TRAINING/TR_ON_3_8/ST_2022/fc_dnn_W_128_D_4_dir.pt"
        adv_model_path = "root/model/save/mnist/CLEAN_TRAINING/TR_ON_3_8/ST_2022/fc_dnn_W_128_D_4_dir_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.06/batch_size_128/eps_stp_size_0.06/adv_steps_80/adv_model_dir.pt"

        tmp_image_over_what_str = 'test'
        if(is_act_collection_on_train):
            tmp_image_over_what_str = 'train'

        seg_over_what_str = 'MP'
        if(is_class_segregation_on_ground_truth):
            seg_over_what_str = 'GT'

        if('CLEAN' in std_model_path):
            first_prefix = std_model_path[0:std_model_path.rfind(
                ".pt")]
        else:
            first_prefix = std_model_path[0:std_model_path.rfind(
                "/")+1]
        final_postfix_for_save = "/{}/EPS_{}/ADV_TYPE_{}/NUM_ADV_STEPS_{}/eps_step_size_{}/".format(
            first_prefix, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size)

        if('CLEAN' in adv_model_path):
            prefix = adv_model_path[0:adv_model_path.rfind(
                ".pt")]
        else:
            prefix = adv_model_path[0:adv_model_path.rfind(
                "/")+1]
        image_save_prefix_folder = str(prefix)+"/RAW_ACT_ANALYSIS/ADV_VS_ORIG_REPORT/"+str(dataset_str)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/_ACT_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
            seg_over_what_str)+"/TMP_COLL_BS_"+str(activation_calculation_batch_size)+"_NO_TO_COLL_"+str(number_of_batch_to_collect)+"/_torch_seed_"+str(torch_seed)+"/" + str(final_postfix_for_save) + "/"

        if not os.path.exists(image_save_prefix_folder):
            os.makedirs(image_save_prefix_folder)

        std_model_merged_adv_vs_orig_stats_store_location = image_save_prefix_folder + \
            "/std_model_adv_vs_orig_stats.npy"
        print("std_model_merged_classwise_pair_stats_store_location",
              std_model_merged_adv_vs_orig_stats_store_location)
        adv_model_merged_adv_vs_orig_stats_store_location = image_save_prefix_folder + \
            "/adv_model_adv_vs_orig_stats.npy"
        xls_save_location = image_save_prefix_folder + \
            "/adv_std_model_adv_vs_origs_report"
        only_diff_xls_save_location = image_save_prefix_folder + \
            "/adv_std_model_only_adv_vs_origs_report"

        if(os.path.exists(std_model_merged_adv_vs_orig_stats_store_location) and os.path.exists(adv_model_merged_adv_vs_orig_stats_store_location)):
            with open(std_model_merged_adv_vs_orig_stats_store_location, 'rb') as file:
                npzfile = np.load(
                    std_model_merged_adv_vs_orig_stats_store_location, allow_pickle=True)
                adv_ex_stats = npzfile['adv']
                orig_ex_stats = npzfile['orig']
                std_model_adv_vs_orig_ex_stats = npzfile['adv_orig']
                std_model_adv_orig_diff_stats = merge_all_class_stats(
                    [adv_ex_stats, orig_ex_stats, std_model_adv_vs_orig_ex_stats])

            with open(adv_model_merged_adv_vs_orig_stats_store_location, 'rb') as file:
                npzfile = np.load(
                    adv_model_merged_adv_vs_orig_stats_store_location, allow_pickle=True)
                adv_ex_stats = npzfile['adv']
                orig_ex_stats = npzfile['orig']
                adv_model_adv_vs_orig_ex_stats = npzfile['adv_orig']
                adv_model_adv_orig_diff_stats = merge_all_class_stats(
                    [adv_ex_stats, orig_ex_stats, adv_model_adv_vs_orig_ex_stats])
        else:
            sub_scheme_type = 'OVER_ADVERSARIAL'
            list_of_list_of_act_analyser_adv = run_generate_scheme(
                models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=std_model_path)
            adv_ex_stats = np.array(generate_all_class_stats(
                list_of_list_of_act_analyser_adv[0]))
            sub_scheme_type = 'OVER_ORIGINAL'
            list_of_list_of_act_analyser_orig = run_generate_scheme(
                models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=std_model_path)
            orig_ex_stats = np.array(generate_all_class_stats(
                list_of_list_of_act_analyser_orig[0]))
            is_save_graph_visualizations = False
            list_of_merged_act1_act2 = diff_merge_two_activation_analysis(
                merge_type, list_of_list_of_act_analyser_adv[0], list_of_list_of_act_analyser_orig[0], wand_project_name=wand_project_name_for_merge, is_save_graph_visualizations=is_save_graph_visualizations)
            std_model_adv_vs_orig_ex_stats = np.array(generate_all_class_stats(
                list_of_merged_act1_act2))
            std_model_adv_orig_diff_stats = merge_all_class_stats(
                [adv_ex_stats, orig_ex_stats, std_model_adv_vs_orig_ex_stats])
            with open(std_model_merged_adv_vs_orig_stats_store_location, 'wb') as file:
                np.savez(
                    file, adv=adv_ex_stats, orig=orig_ex_stats, adv_orig=std_model_adv_vs_orig_ex_stats)

            sub_scheme_type = 'OVER_ADVERSARIAL'
            list_of_list_of_act_analyser_adv = run_generate_scheme(
                models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=adv_model_path)
            adv_ex_stats = np.array(generate_all_class_stats(
                list_of_list_of_act_analyser_adv[0]))
            sub_scheme_type = 'OVER_ORIGINAL'
            list_of_list_of_act_analyser_orig = run_generate_scheme(
                models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=adv_model_path)
            orig_ex_stats = np.array(generate_all_class_stats(
                list_of_list_of_act_analyser_orig[0]))
            is_save_graph_visualizations = False
            list_of_merged_act1_act2 = diff_merge_two_activation_analysis(
                merge_type, list_of_list_of_act_analyser_adv[0], list_of_list_of_act_analyser_orig[0], wand_project_name=wand_project_name_for_merge, is_save_graph_visualizations=is_save_graph_visualizations)
            adv_model_adv_vs_orig_ex_stats = np.array(generate_all_class_stats(
                list_of_merged_act1_act2))
            adv_model_adv_orig_diff_stats = merge_all_class_stats(
                [adv_ex_stats, orig_ex_stats, adv_model_adv_vs_orig_ex_stats])

            with open(adv_model_merged_adv_vs_orig_stats_store_location, 'wb') as file:
                np.savez(
                    file, adv=adv_ex_stats, orig=orig_ex_stats, adv_orig=adv_model_adv_vs_orig_ex_stats)

        print("adv_ex_stats shape", adv_ex_stats.shape)
        print("orig_ex_stats shape", orig_ex_stats.shape)
        print("adv_model_adv_vs_orig_ex_stats shape",
              adv_model_adv_vs_orig_ex_stats.shape)
        # print("adv_ex_stats ", adv_ex_stats)
        # print("orig_ex_stats ", orig_ex_stats)
        # print("adv_model_adv_vs_orig_ex_stats ",
        #       adv_model_adv_vs_orig_ex_stats)

        print("adv_model_adv_orig_diff_stats shape:",
              np.array(adv_model_adv_orig_diff_stats).shape)
        print("std_model_adv_orig_diff_stats shape",
              np.array(std_model_adv_orig_diff_stats).shape)
        generate_all_classes_excel_report(
            xls_save_location, std_model_adv_orig_diff_stats, adv_model_adv_orig_diff_stats)
        all_model_adv_vs_orig_ex_stats = merge_all_class_stats(
            [adv_model_adv_vs_orig_ex_stats, std_model_adv_vs_orig_ex_stats])
        print("all_model_adv_vs_orig_ex_stats shape:",
              np.array(all_model_adv_vs_orig_ex_stats).shape)
        generate_all_classes_only_diff_excel_report(
            only_diff_xls_save_location, all_model_adv_vs_orig_ex_stats)
        print("xls_save_location", xls_save_location)

    print("Finished execution!!!")

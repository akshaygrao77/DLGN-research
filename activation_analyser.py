import torch
import os
from tqdm import tqdm, trange
import wandb
import random
import numpy as np


from utils.visualise_utils import save_image, recreate_image, add_lower_dimension_vectors_within_itself
from utils.data_preprocessing import preprocess_dataset_get_data_loader
from structure.dlgn_conv_config_structure import DatasetConfig
from configs.dlgn_conv_config import HardRelu
from utils.data_preprocessing import preprocess_dataset_get_data_loader, segregate_classes
from structure.generic_structure import PerClassDataset
from model.model_loader import get_model_from_loader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_id + worker_seed)
    random.seed(worker_id - worker_seed)


def get_wandb_config(exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                     is_class_segregation_on_ground_truth,
                     activation_calculation_batch_size, torch_seed,
                     number_of_batch_to_collect=None, collect_threshold=None):

    wandb_config = dict()
    wandb_config["class_label"] = class_label
    wandb_config["class_indx"] = class_indx
    wandb_config["classes"] = classes
    wandb_config["model_arch_type"] = model_arch_type
    wandb_config["dataset"] = dataset
    wandb_config["is_template_image_on_train"] = is_template_image_on_train
    wandb_config["is_class_segregation_on_ground_truth"] = is_class_segregation_on_ground_truth
    wandb_config["torch_seed"] = torch_seed
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
        # Count based states
        self.active_counts_activation_map_list = None
        self.inactive_counts_activation_map_list = None
        self.active_inactive_diff_activation_map_list = None

        # Indicator based states
        self.active_thresholded_indicator_activation_map_list = None
        self.inactive_thresholded_indicator_activation_map_list = None
        self.overall_thresholded_active_inactive_indicator_map_list = None

        # Statistics based states
        self.mean_activations_map_list = None
        self.std_activations_map_list = None
        self.min_activations_map_list = None
        self.max_activations_map_list = None

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
        self.mean_activations_map_list = [None] * number_of_conv_layers
        self.std_activations_map_list = [None] * number_of_conv_layers
        self.min_activations_map_list = [None] * number_of_conv_layers
        self.max_activations_map_list = [None] * number_of_conv_layers

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

                self.mean_activations_map_list[indx] = self.mean_activations_map_list[indx] / num_batches

                pre_variance = (
                    self.std_activations_map_list[indx] / num_batches - self.mean_activations_map_list[indx] ** 2)
                pre_variance[pre_variance < 0] = 0.

                self.std_activations_map_list[indx] = pre_variance ** 0.5

                if(torch.isnan(self.std_activations_map_list[indx]).any().item()):
                    print("STD Deviation has NAN entries at the end*****************")
                    print("num_batches", num_batches)
                    print("self.mean_activations_map_list[indx]",
                          self.mean_activations_map_list[indx])
                    print(
                        "self.std_activations_map_list[indx]", self.std_activations_map_list[indx])

                self.total_pixels += torch.numel(
                    current_overall_thresholded_active_inactive_indicator_map)
                current_thresholded_active_pixel = torch.count_nonzero(
                    HardRelu()(current_overall_thresholded_active_inactive_indicator_map))
                self.thresholded_active_pixel_count += current_thresholded_active_pixel.item()

                current_overall_unthresholded_active_inactive_indicator_map = HardRelu()(self.active_inactive_diff_activation_map_list[
                    indx])

                print(
                    "============================== INDX:{} =============================".format(indx))
                print("total_tcollect_img_count:",
                      self.total_tcollect_img_count)
                print("current_overall_thresholded_active_inactive_indicator_map size:",
                      current_overall_thresholded_active_inactive_indicator_map.size())
                print("current_overall_unthresholded_active_inactive_indicator_map size:",
                      current_overall_unthresholded_active_inactive_indicator_map.size())
                print("mean_activations_map_list size:",
                      self.mean_activations_map_list[indx].size())
                print("min_activations_map_list size:",
                      self.min_activations_map_list[indx].size())
                print("std_activations_map_list size:",
                      self.std_activations_map_list[indx].size())

                print("current_overall_thresholded_active_inactive_indicator_map :",
                      current_overall_thresholded_active_inactive_indicator_map)
                print("current_overall_unthresholded_active_inactive_indicator_map :",
                      current_overall_unthresholded_active_inactive_indicator_map)
                print("mean_activations_map_list:",
                      self.mean_activations_map_list[indx])
                print("min_activations_map_list:",
                      self.min_activations_map_list[indx])
                print("std_activations_map_list:",
                      self.std_activations_map_list[indx])

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

                for each_image_conv_output in each_conv_output:
                    current_max_activations_map = self.max_activations_map_list[indx]
                    if(current_max_activations_map is None):
                        self.max_activations_map_list[indx] = each_image_conv_output
                    else:
                        self.max_activations_map_list[indx] = torch.maximum(
                            current_max_activations_map, each_image_conv_output)

                    current_min_activations_map = self.min_activations_map_list[indx]
                    if(current_min_activations_map is None):
                        self.min_activations_map_list[indx] = each_image_conv_output
                    else:
                        self.min_activations_map_list[indx] = torch.minimum(
                            current_min_activations_map, each_image_conv_output)

                # Until all batches are processed, mean tensor actually holds mean per batch(holding sum might result in overflow)
                current_mean_activations_map = self.mean_activations_map_list[indx]
                mean_activation_maps = torch.mean(each_conv_output, 0)
                if(current_mean_activations_map is None):
                    self.mean_activations_map_list[indx] = mean_activation_maps
                else:
                    self.mean_activations_map_list[indx] = torch.add(
                        current_mean_activations_map, mean_activation_maps)

                # Until all batches are processed, std tensor actually holds the meansq per batch(holding sum might surely result in overflow)
                current_meansq_activations_map = self.std_activations_map_list[indx]
                meansq_activation_maps = torch.mean(each_conv_output ** 2, 0)
                if(current_meansq_activations_map is None):
                    self.std_activations_map_list[indx] = meansq_activation_maps
                else:
                    self.std_activations_map_list[indx] = torch.add(
                        current_meansq_activations_map, meansq_activation_maps)
                    if(torch.isnan(self.std_activations_map_list[indx]).any().item()):
                        print("STD Deviation has NAN entries")
                        print("current_meansq_activations_map",
                              current_meansq_activations_map)
                        print("meansq_activation_maps", meansq_activation_maps)
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

    def generate_activation_stats_per_class(self, exp_type, per_class_dataset, class_label, class_indx, number_of_batch_to_collect, classes, model_arch_type, dataset,
                                            is_template_image_on_train, is_class_segregation_on_ground_truth, activation_calculation_batch_size,
                                            wand_project_name, wandb_group_name, torch_seed, collect_threshold,
                                            root_save_prefix="root", final_postfix_for_save=""):
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
        if(is_template_image_on_train):
            tmp_image_over_what_str = 'train'

        seg_over_what_str = 'MP'
        if(is_class_segregation_on_ground_truth):
            seg_over_what_str = 'GT'

        self.image_save_prefix_folder = str(root_save_prefix)+"/"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/_COLL_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
            seg_over_what_str)+"/TMP_COLL_BS_"+str(activation_calculation_batch_size)+"_NO_TO_COLL_"+str(number_of_batch_to_collect)+"/_torch_seed_"+str(torch_seed)+"_c_thres_"+str(collect_threshold)+"/" + str(final_postfix_for_save) + "/"

        if(is_log_wandb):
            wandb_run_name = self.image_save_prefix_folder.replace(
                "/", "").replace(root_save_prefix, class_label)
            wandb_config = get_wandb_config(exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                                            is_class_segregation_on_ground_truth,
                                            activation_calculation_batch_size, torch_seed,
                                            number_of_batch_to_collect=number_of_batch_to_collect, collect_threshold=collect_threshold)

            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                group=f"{wandb_group_name}",
                config=wandb_config,
            )

        self.record_activation_states(per_class_data_loader, class_label,
                                      number_of_batch_to_collect, collect_threshold, is_save_original_image=False)

        log_dict = {
            "total_pixels": self.total_pixels, "thres_active_pxl_count": self.thresholded_active_pixel_count, "thres_active_pxl_percent": self.thresholded_active_pixels_percentage,
            "unthres_active_pxl_count": self.unthresholded_active_pixel_count, "unthres_active_pxl_percent": self.unthresholded_active_pixels_percentage,
            "ovrall_avg_active_percent": self.overall_average_active_percentage, "ovrall_std_active_percent":  self.overall_std_active_percentage,
            "ovrall_min_active_percent": self.overall_min_active_percentage, "ovrall_max_active_percent": self.overall_max_active_percentage
        }

        print("log_dict", log_dict)

        if(is_log_wandb):
            wandb.log(log_dict)
            wandb.finish()


def run_activation_analysis_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth,
                                      activation_calculation_batch_size, number_of_batch_to_collect, wand_project_name, is_split_validation,
                                      valid_split_size, torch_seed, wandb_group_name, exp_type, collect_threshold,
                                      root_save_prefix='root', final_postfix_for_save="aug_indx_1",
                                      custom_model=None, custom_data_loader=None, class_indx_to_visualize=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if(dataset == "cifar10"):
        print("Running for CIFAR 10")
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = len(classes)

        if(custom_data_loader is None):
            cifar10_config = DatasetConfig(
                'cifar10', is_normalize_data=False, valid_split_size=valid_split_size, batch_size=128)

            trainloader, _, testloader = preprocess_dataset_get_data_loader(
                cifar10_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=is_split_validation)
        else:
            trainloader, testloader = custom_data_loader

    elif(dataset == "mnist"):
        print("Running for MNIST")
        classes = [str(i) for i in range(0, 10)]
        num_classes = len(classes)
        if(custom_data_loader is None):
            mnist_config = DatasetConfig(
                'mnist', is_normalize_data=True, valid_split_size=valid_split_size, batch_size=128)

            trainloader, _, testloader = preprocess_dataset_get_data_loader(
                mnist_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=is_split_validation)
        else:
            trainloader, testloader = custom_data_loader

    print("Preprocessing and dataloader process completed of type:{} for dataset:{}".format(
        model_arch_type, dataset))

    if(custom_model is None):
        model = get_model_from_loader(model_arch_type, dataset)
        print("Model loaded is:", model)
    else:
        model = custom_model
        print("Custom model provided in arguments will be used")

    if(class_indx_to_visualize is None):
        # class_indx_to_visualize = [i for i in range(len(classes))]
        class_indx_to_visualize = [i for i in range(1)]

    if(is_class_segregation_on_ground_truth == True):
        input_data_list_per_class = segregate_classes(
            model, trainloader, testloader, num_classes, is_template_image_on_train, is_class_segregation_on_ground_truth)

    if(is_class_segregation_on_ground_truth == False):
        input_data_list_per_class = segregate_classes(
            model, trainloader, testloader, num_classes, is_template_image_on_train, is_class_segregation_on_ground_truth)

    list_of_act_analyser = []
    for c_indx in class_indx_to_visualize:
        class_label = classes[c_indx]
        print("************************************************************ Class:", class_label)
        per_class_dataset = PerClassDataset(
            input_data_list_per_class[c_indx], c_indx)
        # per_class_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=32,
        #                                                shuffle=False)

        act_analyser = ActivationAnalyser(
            model)
        if(exp_type == "GENERATE_RECORD_STATS"):
            act_analyser.generate_activation_stats_per_class(exp_type, per_class_dataset, class_label, c_indx, number_of_batch_to_collect, classes, model_arch_type, dataset,
                                                             is_template_image_on_train, is_class_segregation_on_ground_truth, activation_calculation_batch_size,
                                                             wand_project_name, wandb_group_name, torch_seed, collect_threshold,
                                                             root_save_prefix, final_postfix_for_save)
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
    is_template_image_on_train = True
    # If False, then segregation is over model prediction
    is_class_segregation_on_ground_truth = True
    activation_calculation_batch_size = 64
    number_of_batch_to_collect = None
    # wand_project_name = 'test_activation_analyser'
    wand_project_name = None
    wandb_group_name = "activation_analysis_mnist_conv4_dlgn"
    is_split_validation = False
    valid_split_size = 0.1
    torch_seed = 2022
    # GENERATE_RECORD_STATS
    exp_type = "GENERATE_RECORD_STATS"
    collect_threshold = 0.95

    if(not(wand_project_name is None)):
        wandb.login()

    list_of_act_analyser = run_activation_analysis_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth,
                                                             activation_calculation_batch_size, number_of_batch_to_collect, wand_project_name, is_split_validation,
                                                             valid_split_size, torch_seed, wandb_group_name, exp_type, collect_threshold)

    print("Finished execution!!!")

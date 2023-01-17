import torch
import os
from tqdm import tqdm
import wandb
import random
import numpy as np
import pickle


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
            merged_act_analyser.post_activation_values_all_batches = self.post_activation_values_all_batches - \
                other_activation_state.post_activation_values_all_batches
            temp1 = torch.from_numpy(
                merged_act_analyser.post_activation_values_all_batches)
            pos_diff = HardRelu()(temp1)
            neg_diff = HardRelu()(-temp1)
            merged_act_analyser.diff_counts_post_activation_values_all_layers = torch.sum(
                pos_diff, dim=0)+torch.sum(neg_diff, dim=0)

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

    def save_raw_recorded_activation_states(self, base_save_folder):

        dict_full_path_to_saves = dict()

        current_post_activation_values = self.post_activation_values_all_batches

        current_full_save_path = base_save_folder+"/Video/HardRelu/" + \
            "hardrelu_raw_postactivation_images_*.mp4"
        dict_full_path_to_saves["raw_post_activation"] = current_full_save_path
        print("current_full_save_path:", current_full_save_path)

        # current_full_img_save_path = base_save_folder + \
        #     "/Images/HardRelu/" + \
        #     "hard_relu_post_act_img_b_*.jpg"

        current_full_img_save_path = base_save_folder + \
            "/PlainImages/HardRelu/" + \
            "hard_relu_post_act_img_b_*.jpg"

        print("current_full_img_save_path:", current_full_img_save_path)

        if hasattr(self, 'diff_counts_post_activation_values_all_layers'):
            current_full_txt_save_path = base_save_folder + "/PlainImages/HardRelu/" + \
                "diff_counts.txt"
            sfolder = current_full_txt_save_path[0:current_full_txt_save_path.rfind(
                "/")+1]
            if not os.path.exists(sfolder):
                os.makedirs(sfolder)
            # print("self.diff_counts_post_activation_values_all_layers",
            #       self.diff_counts_post_activation_values_all_layers.size())
            with open(current_full_txt_save_path, "w") as myfile:
                for f_ind in range(self.diff_counts_post_activation_values_all_layers.size()[0]):
                    curr_filter = self.diff_counts_post_activation_values_all_layers[f_ind]
                    # print("curr_filter size", curr_filter.size())
                    sum_curr_filter = torch.sum(curr_filter)
                    myfile.write(
                        "\n ************************************ Next Filter:{} = {} *********************************** \n".format(f_ind, sum_curr_filter))
                    myfile.write("%s" % curr_filter)
            current_full_diff_img_save_path = base_save_folder + "/PlainImages/HardRelu/" + \
                "diff_counts_image.jpg"
            generate_plain_image(self.diff_counts_post_activation_values_all_layers,
                                 current_full_diff_img_save_path, is_standarize=False, is_standarize_01=True)

        # generate_list_of_images_from_data(current_post_activation_values, 200, 300,
        #                                   "Hard relu Raw post activation video", save_each_img_path=current_full_img_save_path, cmap='binary')
        generate_list_of_plain_images_from_data(
            current_post_activation_values, save_each_img_path=current_full_img_save_path, is_standarize=False)
        # generate_video_of_image_from_data(
        #     current_post_activation_values, 200, 300, "Hard relu Raw post activation video", save_path=current_full_save_path, save_each_img_path=current_full_img_save_path, cmap='binary')

        return

    def initialise_raw_record_states(self):
        self.total_tcollect_img_count = 0
        self.post_activation_values_all_batches = None

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
                each_conv_output = HardRelu()(each_conv_output)
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
        print("self.post_activation_values_all_batches",
              self.post_activation_values_all_batches.shape)
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

        self.image_save_prefix_folder = str(root_save_prefix)+"/"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/_ACT_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
            seg_over_what_str)+"/TMP_COLL_BS_"+str(activation_calculation_batch_size)+"_NO_TO_COLL_"+str(number_of_batch_to_collect)+"/_torch_seed_"+str(torch_seed)+"/" + str(final_postfix_for_save) + "/"

        if not os.path.exists(self.image_save_prefix_folder):
            os.makedirs(self.image_save_prefix_folder)

        wandb_run_name = self.image_save_prefix_folder.replace(
            "/", "").replace(root_save_prefix, class_label)
        self.wandb_run_name = wandb_run_name
        wandb_config = get_wandb_config(exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_act_collection_on_train,
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
                save_folder)


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
    print("Running for "+str(dataset))
    classes, num_classes, ret_config = get_preprocessing_and_other_configs(
        dataset, valid_split_size)

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
        models_base_path = direct_model_path[0:direct_model_path.rfind("/")+1]
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
    # plain_pure_conv4_dnn , conv4_dlgn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small
    model_arch_type = 'conv4_dlgn_n16_small'
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
    # GENERATE , LOAD_AND_SAVE , LOAD_AND_GENERATE_MERGE , GENERATE_MERGE_AND_SAVE
    scheme_type = "GENERATE_MERGE_AND_SAVE"
    # OVER_RECONSTRUCTED , OVER_ADVERSARIAL , OVER_ORIGINAL
    sub_scheme_type = 'OVER_ORIGINAL'
    # OVER_ORIGINAL_VS_ADVERSARIAL , TWO_CUSTOM_MODELS
    merge_scheme_type = "OVER_ORIGINAL_VS_ADVERSARIAL"

    classes, num_classes, ret_config = get_preprocessing_and_other_configs(
        dataset, valid_split_size)
    trainloader, _, testloader = preprocess_dataset_get_data_loader(
        ret_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=is_split_validation)
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
        is_save_graph_visualizations = False

        is_save_adv = True
        eps = 0.02
        adv_attack_type = 'PGD'
        number_of_adversarial_optimization_steps = 161
        eps_step_size = 0.01
        adv_target = None

        models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/"

        list_of_list_of_act_analyser = run_generate_scheme(
            models_base_path, to_be_analysed_dataloader, custom_data_loader)

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

        c_indices = [i for i in range(10)]

        # class_ind_visualize = None

        for c_i in c_indices:
            class_ind_visualize = [c_i]

            list_of_model_paths = []
            models_base_path = None

            models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_plain_pure_conv4_dnn_n16_small_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/"

            direct_model_path = None

            direct_model_path = "root/model/save/mnist/V2_iterative_augmenting/DS_mnist/MT_conv4_dlgn_n16_small_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.73/aug_conv4_dlgn_iter_1_dir.pt"

            if(merge_scheme_type == "OVER_ORIGINAL_VS_ADVERSARIAL"):
                num_iterations = 1
                it_start = 1
                for current_it_start in range(it_start, num_iterations + 1):
                    sub_scheme_type = 'OVER_ORIGINAL'
                    is_save_graph_visualizations = False

                    list_of_list_of_act_analyser_orig = run_generate_scheme(
                        models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=direct_model_path)

                    sub_scheme_type = 'OVER_ADVERSARIAL'
                    is_save_adv = True
                    eps = 0.06
                    adv_attack_type = 'PGD'
                    number_of_adversarial_optimization_steps = 161
                    eps_step_size = 0.01
                    adv_target = None

                    list_of_list_of_act_analyser_adv = run_generate_scheme(
                        models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=direct_model_path)

                    for ind in range(len(list_of_list_of_act_analyser_adv)):
                        list_of_act_analyser1 = list_of_list_of_act_analyser_adv[ind]
                        list_of_act_analyser2 = list_of_list_of_act_analyser_orig[ind]

                        if(merge_type == "DIFF"):
                            is_save_graph_visualizations = True
                            list_of_merged_act1_act2 = diff_merge_two_activation_analysis(merge_type,
                                                                                          list_of_act_analyser1, list_of_act_analyser2, wand_project_name=wand_project_name_for_merge, is_save_graph_visualizations=is_save_graph_visualizations)
            elif(merge_scheme_type == "TWO_CUSTOM_MODELS"):
                direct_model_path1 = "root/model/save/mnist/adversarial_training/MT_plain_pure_conv4_dnn_n16_small_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.06/batch_size_128/eps_stp_size_0.06/adv_steps_80/adv_model_dir_epoch_5.pt"
                direct_model_path2 = "root/model/save/mnist/CLEAN_TRAINING/ST_2022/plain_pure_conv4_dnn_n16_small_dir.pt"
                num_iterations = 1
                it_start = 1
                for current_it_start in range(it_start, num_iterations + 1):
                    # sub_scheme_type = 'OVER_ORIGINAL'
                    sub_scheme_type = 'OVER_ADVERSARIAL'
                    is_save_graph_visualizations = False

                    is_save_adv = True
                    eps = 0.06
                    adv_attack_type = 'PGD'
                    number_of_adversarial_optimization_steps = 161
                    eps_step_size = 0.01
                    adv_target = None

                    list_of_list_of_act_analyser_m1 = run_generate_scheme(
                        models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=direct_model_path1)

                    list_of_list_of_act_analyser_m2 = run_generate_scheme(
                        models_base_path, to_be_analysed_dataloader, custom_data_loader, current_it_start, direct_model_path=direct_model_path2)

                    for ind in range(len(list_of_list_of_act_analyser_m2)):
                        list_of_act_analyser1 = list_of_list_of_act_analyser_m2[ind]
                        list_of_act_analyser2 = list_of_list_of_act_analyser_m1[ind]

                        if(merge_type == "DIFF"):
                            is_save_graph_visualizations = True
                            list_of_merged_act1_act2 = diff_merge_two_activation_analysis(merge_type,
                                                                                          list_of_act_analyser1, list_of_act_analyser2, wand_project_name=wand_project_name_for_merge, is_save_graph_visualizations=is_save_graph_visualizations)

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

    print("Finished execution!!!")

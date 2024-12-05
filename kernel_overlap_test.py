import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import math
import os
import wandb
from structure.dlgn_conv_config_structure import DatasetConfig
import torch.backends.cudnn as cudnn

from external_utils import format_time
from utils.data_preprocessing import preprocess_dataset_get_data_loader, generate_dataset_from_loader
from structure.generic_structure import CustomSimpleDataset
from conv4_models import Plain_CONV4_Net, Conv4_DLGN_Net, get_model_instance, get_model_instance_from_dataset
from adversarial_attacks_tester import load_or_generate_adv_examples
from configs.dlgn_conv_config import HardRelu
from statistics import mean
import csv


def get_wandb_config(exp_type, adv_attack_type, model_arch_type, dataset, is_analysis_on_train,
                     eps, number_of_adversarial_optimization_steps, eps_step_size, is_targetted, seed, adv_target=None):

    wandb_config = dict()
    wandb_config["adv_attack_type"] = adv_attack_type
    wandb_config["model_arch_type"] = model_arch_type
    wandb_config["dataset"] = dataset
    wandb_config["is_analysis_on_train"] = is_analysis_on_train
    wandb_config["eps"] = eps
    wandb_config["number_of_adversarial_optimization_steps"] = number_of_adversarial_optimization_steps
    wandb_config["eps_step_size"] = eps_step_size
    wandb_config["is_targetted"] = is_targetted
    wandb_config["exp_type"] = exp_type
    wandb_config["seed"] = seed
    if(not(adv_target is None)):
        wandb_config["adv_target"] = adv_target

    return wandb_config


def obtain_kernel_overlap(net, analyse_dataset,is_consider_true_npk=False):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X, Y = analyse_dataset
        if(not isinstance(X, torch.Tensor)):
            X = torch.from_numpy(X)
        if(not isinstance(Y, torch.Tensor)):
            Y = torch.from_numpy(Y)
        X, Y = X.to(
            device), Y.to(device)
        # Replace the lesser class index with -1 and another with +1
        Y = torch.where(Y == 0, -1, 1)
        Y = torch.unsqueeze(Y, 1)
        Y = Y.type(torch.float32)
        X = torch.flatten(X, 1)

        print("Size before overlap calculation: X:{},Y:{} Types: X:{},Y:{}".format(
            X.size(), Y.size(), X.dtype, Y.dtype))

        net(X)

        if(isinstance(net, torch.nn.DataParallel)):
            conv_outs = net.module.linear_conv_outputs
        else:
            conv_outs = net.linear_conv_outputs

        if is_consider_true_npk:
            tmp_X = X
        else:
            tmp_X = HardRelu()(X)
        npk_kernel = torch.matmul(tmp_X, torch.transpose(tmp_X, 0, 1))
        print("Norm XTX", torch.norm(npk_kernel))
        for each_conv_out in conv_outs:
            gate_out = HardRelu()(each_conv_out)
            npk_kernel = npk_kernel * \
                (torch.matmul(gate_out,  torch.transpose(gate_out, 0, 1)))
            print("Norm npk_kernel", torch.norm(npk_kernel))

        print("Size of npk kernel is:{} , dtype:{}".format(
            npk_kernel.size(), npk_kernel.dtype))
        # npk_kernel_inverse = torch.linalg.pinv(npk_kernel)

        # print("Size of npk kernel inverse is {}, dtype:{}".format(
        #     npk_kernel_inverse.size(), npk_kernel_inverse.dtype))
        width = conv_outs[0].size()[1]
        depth = len(conv_outs)
        npk_kernel = npk_kernel / pow(width, depth)
        # npk_kernel = npk_kernel / torch.trace(npk_kernel)

        print("Final Norm npk_kernel:{} for width:{} depth:{}".format(
            torch.norm(npk_kernel), width, depth))
        tmp = torch.matmul(torch.matmul(
            torch.transpose(Y, 0, 1), npk_kernel), Y)
        Y_gram = torch.matmul(Y, torch.transpose(Y, 0, 1))
        print("Y_gram ",Y_gram.shape)
        overlap = npk_kernel * Y_gram
        # print(torch.numel(overlap[overlap > 0]),torch.numel(overlap[overlap<0]))
        # overlap_diff = torch.sum(overlap)
        # assert tmp==overlap_diff, "Error:{}".format(tmp-overlap_diff)
        overlap_same_class = torch.sum(torch.where(overlap > 0.,overlap.float(),torch.zeros(1,dtype=torch.float32).to(device=overlap.device)))
        overlap_diff_class = -torch.sum(torch.where(overlap < 0.,overlap.float(),torch.zeros(1,dtype=torch.float32).to(device=overlap.device)))

        return tmp,overlap_same_class,overlap_diff_class

def obtain_adv_orig_kernel_overlap(net, orig_dataset,adv_dataset,is_consider_true_npk=False):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_orig, Y_orig = orig_dataset
        X_adv, _ = adv_dataset
        if(not isinstance(X_orig, torch.Tensor)):
            X_orig = torch.from_numpy(X_orig)
        if(not isinstance(Y_orig, torch.Tensor)):
            Y_orig = torch.from_numpy(Y_orig)
        X_orig, Y_orig = X_orig.to(
            device), Y_orig.to(device)
        
        if(not isinstance(X_adv, torch.Tensor)):
            X_adv = torch.from_numpy(X_adv)
        X_adv = X_adv.to(device)

        # Replace the lesser class index with -1 and another with +1
        Y_orig = torch.where(Y_orig == 0, -1, 1)
        Y_orig = torch.unsqueeze(Y_orig, 1)
        Y_orig = Y_orig.type(torch.float32)
        X_orig = torch.flatten(X_orig, 1)

        print("Size before overlap calculation: X_orig:{},Y_orig:{} Types: X_orig:{},Y_orig:{}".format(
            X_orig.size(), Y_orig.size(), X_orig.dtype, Y_orig.dtype))

        net(X_orig)

        if(isinstance(net, torch.nn.DataParallel)):
            conv_outs_orig = net.module.linear_conv_outputs
        else:
            conv_outs_orig = net.linear_conv_outputs

        if is_consider_true_npk:
            tmp_X_orig = X_orig
        else:
            tmp_X_orig = HardRelu()(X_orig)
        
        print("Size before overlap calculation: X_adv:{}, Types: X_adv:{}".format(
            X_adv.size(), X_adv.dtype))
        
        X_adv = torch.flatten(X_adv, 1)
        
        net(X_adv)

        if(isinstance(net, torch.nn.DataParallel)):
            conv_outs_adv = [a.clone().detach() for a in net.module.linear_conv_outputs]
        else:
            conv_outs_adv = [a.clone().detach() for a in net.linear_conv_outputs]

        if is_consider_true_npk:
            tmp_X_adv = X_adv
        else:
            tmp_X_adv = HardRelu()(X_adv)
        

        npk_kernel = torch.matmul(tmp_X_orig, torch.transpose(tmp_X_adv, 0, 1))
        print("Norm XTX", torch.norm(npk_kernel))
        for i in range(len(conv_outs_orig)):
            each_conv_out_orig = conv_outs_orig[i]
            gate_out_orig = HardRelu()(each_conv_out_orig)
            each_conv_out_adv = conv_outs_adv[i]
            gate_out_adv = HardRelu()(each_conv_out_adv)
            npk_kernel = npk_kernel * \
                (torch.matmul(gate_out_orig,  torch.transpose(gate_out_adv, 0, 1)))
            print("Norm npk_kernel", torch.norm(npk_kernel))

        print("Size of npk kernel is:{} , dtype:{}".format(
            npk_kernel.size(), npk_kernel.dtype))
        # npk_kernel_inverse = torch.linalg.pinv(npk_kernel)

        # print("Size of npk kernel inverse is {}, dtype:{}".format(
        #     npk_kernel_inverse.size(), npk_kernel_inverse.dtype))
        width = conv_outs_orig[0].size()[1]
        depth = len(conv_outs_orig)
        npk_kernel = npk_kernel / pow(width, depth)
        # npk_kernel = npk_kernel / torch.trace(npk_kernel)

        print("Final Norm npk_kernel:{} for width:{} depth:{}".format(
            torch.norm(npk_kernel), width, depth))
        tmp = torch.matmul(torch.matmul(
            torch.transpose(Y_orig, 0, 1), npk_kernel), Y_orig)
        Y_gram = torch.matmul(Y_orig, torch.transpose(Y_orig, 0, 1))
        print("Y_gram ",Y_gram.shape)
        overlap = npk_kernel * Y_gram
        # print(torch.numel(overlap[overlap > 0]),torch.numel(overlap[overlap<0]))
        # overlap_diff = torch.sum(overlap)
        # assert tmp==overlap_diff, "Error:{}".format(tmp-overlap_diff)
        overlap_same_class = torch.sum(torch.where(overlap > 0.,overlap.float(),torch.zeros(1,dtype=torch.float32).to(device=overlap.device)))
        overlap_diff_class = -torch.sum(torch.where(overlap < 0.,overlap.float(),torch.zeros(1,dtype=torch.float32).to(device=overlap.device)))

        return tmp,overlap_same_class,overlap_diff_class

if __name__ == '__main__':
    # fashion_mnist , mnist
    dataset = 'fashion_mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net , conv4_deep_gated_net_n16_small
    # fc_dnn , fc_dlgn , fc_dgn , bc_fc_dlgn , bc_fc_sf_dlgn
    model_arch_type = 'bc_fc_dlgn'

    batch_size = 64

    torch_seed = 2022

    is_analysis_on_train = True

    # None means that train on all classes
    list_of_classes_to_train_on = None
    # list_of_classes_to_train_on = [1,9]

    map_of_mtype_paths = {
        "STD": "root/model/save/fashion_mnist/CLEAN_TRAINING/TR_ON_6_7/ST_2022/bc_fc_dlgn_W_256_D_4_dir.pt",
        "ADFS": "root/model/save/fashion_mnist/adversarial_training/TR_ON_6_7/MT_bc_fc_dlgn_W_256_D_4_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.3/OPT_Adam (Parameter Group 0    amsgrad: False    betas: (0.9, 0.999)    eps: 1e-08    lr: 0.0001    weight_decay: 0)/batch_size_64/eps_stp_size_0.01/adv_steps_40/update_on_all/R_init_True/norm_inf/use_ytrue_True/out_lossfn_BCEWithLogitsLoss()/inner_lossfn_Y_Logits_Binary_class_Loss()/adv_model_dir.pt",
    }

    wand_project_name = "thesis_kernel_overlap_experiments"
    # wand_project_name = None

    torch_seed = 2022
    number_of_adversarial_optimization_steps = 40
    adv_attack_type = "PGD"
    adv_target = None
    # K_OVERLAP , All_PAIRS_K_OVERLAP
    exp_type = "K_OVERLAP"
    is_consider_NPK = False
    percentage_of_dataset_for_analysis = 100
    eps_step_size = 0.01
    eps = 0.3

    if(dataset == "cifar10"):
        inp_channel = 3
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = len(classes)

        cifar10_config = DatasetConfig(
            'cifar10', is_normalize_data=False, valid_split_size=0.1, batch_size=batch_size, list_of_classes=list_of_classes_to_train_on)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            cifar10_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    elif(dataset == "mnist"):
        inp_channel = 1
        classes = [str(i) for i in range(0, 10)]
        num_classes = len(classes)

        mnist_config = DatasetConfig(
            'mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size, list_of_classes=list_of_classes_to_train_on)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            mnist_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    elif(dataset == "fashion_mnist"):
        inp_channel = 1
        classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot')
        num_classes = len(classes)

        fashion_mnist_config = DatasetConfig(
            'fashion_mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size, list_of_classes=list_of_classes_to_train_on)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            fashion_mnist_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    print("Training over "+dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes_trained_on = num_classes
    dataset_str = dataset

    list_of_classes_to_train_on_str = ""
    if(list_of_classes_to_train_on is not None):
        for each_class_to_train_on in list_of_classes_to_train_on:
            list_of_classes_to_train_on_str += \
                str(each_class_to_train_on)+"_"
        dataset_str += "_"+str(list_of_classes_to_train_on_str)
        list_of_classes_to_train_on_str = "TR_ON_" + \
            list_of_classes_to_train_on_str[0:-1]
        num_classes_trained_on = len(list_of_classes_to_train_on)
        temp_classes = []
        for ea_c in list_of_classes_to_train_on:
            temp_classes.append(classes[ea_c])
        classes = temp_classes

    model_arch_type_str = model_arch_type
    if("masked" in model_arch_type):
        mask_percentage = 90
        model_arch_type_str = model_arch_type_str + \
            "_PRC_"+str(mask_percentage)
        net = get_model_instance(
            model_arch_type, inp_channel, mask_percentage=mask_percentage, seed=torch_seed, num_classes=num_classes_trained_on)
    elif("fc" in model_arch_type):
        fc_width = 256
        fc_depth = 4
        nodes_in_each_layer_list = [fc_width] * fc_depth
        model_arch_type_str = model_arch_type_str + \
            "_W_"+str(fc_width)+"_D_"+str(fc_depth)
        net = get_model_instance_from_dataset(dataset,
                                              model_arch_type, seed=torch_seed, num_classes=num_classes_trained_on, nodes_in_each_layer_list=nodes_in_each_layer_list)
    else:
        net = get_model_instance(model_arch_type, inp_channel,
                                 seed=torch_seed, num_classes=num_classes_trained_on)

    is_targetted = adv_target is not None
    is_log_wandb = not(wand_project_name is None)

    if(is_log_wandb):
        wandb.login()

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_str == 'cuda':
        cudnn.benchmark = True

    if(is_analysis_on_train == True):
        eval_loader = trainloader
    else:
        eval_loader = testloader

    if(exp_type == "K_OVERLAP"):
        number_of_trails = 1

        orig_dataset = generate_dataset_from_loader(eval_loader)
        if(isinstance(orig_dataset.list_of_x[0], torch.Tensor)):
            orig_dataset = torch.stack(
                orig_dataset.list_of_x), torch.stack(orig_dataset.list_of_y)
        else:
            orig_dataset = np.array(orig_dataset.list_of_x), np.array(
                orig_dataset.list_of_y)

        number_of_samples = orig_dataset[0].shape[0]
        print("Total number_of_samples", number_of_samples)

        number_of_samples_used_for_analysis = int(
            (percentage_of_dataset_for_analysis/100)*number_of_samples)

        print("Total number of filtered samples for analysis",
              number_of_samples_used_for_analysis)

        if(is_log_wandb):
            wandb_group_name = "DS_"+str(dataset_str) + \
                "ST_"+str(torch_seed)+"_PROB_"+str(percentage_of_dataset_for_analysis)+"_NSAMP_" + \
                str(number_of_samples_used_for_analysis) + \
                "_K_OV_"+str(model_arch_type)
            wandb_run_name = "DS_"+str(dataset_str) + \
                "ST_"+str(torch_seed)+"_NSAMP_"+str(number_of_samples_used_for_analysis) + \
                "_K_OV_"+str(model_arch_type)+"TS_"+str(number_of_samples)
            wandb_config = get_wandb_config(exp_type, adv_attack_type, model_arch_type_str, dataset_str, is_analysis_on_train,
                                            eps, number_of_adversarial_optimization_steps, eps_step_size, is_targetted, adv_target)
            wandb_config["seed"] = torch_seed
            wandb_config["percent_used_for_analysis"] = percentage_of_dataset_for_analysis
            wandb_config["total_sampls"] = number_of_samples
            wandb_config["sampl_for_analysis"] = number_of_samples_used_for_analysis
            wandb_config["mtype_paths_map"] = map_of_mtype_paths
            wandb_config["number_of_trails"] = number_of_trails
            wandb_config["is_consider_NPK"] = is_consider_NPK
            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                group=f"{wandb_group_name}",
                config=wandb_config,
            )

        np.random.seed(torch_seed)
        indices = np.arange(0, number_of_samples)

        orig_overlap_map = dict()
        orig_overlap_sameclass_map = dict()
        orig_overlap_diffclass_map = dict()
        adv_overlap_map = dict()
        adv_overlap_sameclass_map = dict()
        adv_overlap_diffclass_map = dict()
        adv_orig_overlap_map = dict()
        adv_orig_overlap_sameclass_map = dict()
        adv_orig_overlap_diffclass_map = dict()
        for key in map_of_mtype_paths:
            orig_overlap_map[key] = []
            orig_overlap_sameclass_map[key] = []
            orig_overlap_diffclass_map[key] = []
            adv_overlap_map[key] = []
            adv_overlap_sameclass_map[key] = []
            adv_overlap_diffclass_map[key] = []
            adv_orig_overlap_map[key] = []
            adv_orig_overlap_sameclass_map[key] = []
            adv_orig_overlap_diffclass_map[key] = []

        for trail_num in range(number_of_trails):
            np.random.shuffle(indices)

            filtered_indices = indices[:number_of_samples_used_for_analysis]

            print("Orig dataset size: X=>{},Y=>{}".format(
                orig_dataset[0].shape, orig_dataset[1].shape))
            analyse_orig_dataset = orig_dataset[0][filtered_indices], orig_dataset[1][filtered_indices]
            print("Analyse orig dataset size: X=>{},Y=>{}".format(
                analyse_orig_dataset[0].shape, analyse_orig_dataset[1].shape))

            for current_model_type in map_of_mtype_paths:
                current_direct_model_path = map_of_mtype_paths[current_model_type]

                isExist = os.path.exists(current_direct_model_path)
                assert isExist == True, 'Model path does not have saved model'

                custom_temp_model = torch.load(current_direct_model_path)
                net.load_state_dict(custom_temp_model.state_dict())

                net = net.to(device)

                if('CLEAN' in current_direct_model_path or 'APR_TRAINING' in current_direct_model_path):
                    model_and_data_save_prefix = current_direct_model_path[0:current_direct_model_path.rfind(
                        ".pt")]
                else:
                    model_and_data_save_prefix = current_direct_model_path[0:current_direct_model_path.rfind(
                        "/")+1]

                adv_dataset = load_or_generate_adv_examples(eval_loader, model_and_data_save_prefix, is_analysis_on_train, net, eps, adv_attack_type,
                                                            number_of_adversarial_optimization_steps, eps_step_size, adv_target, is_save_adv=False)
                print("Adv dataset size: X=>{},Y=>{}".format(
                    adv_dataset.list_of_x[0].shape, adv_dataset.list_of_y[0].shape))
                print("Adv dataset size: X=>{},Y=>{}".format(
                    adv_dataset.list_of_x.shape, adv_dataset.list_of_y.shape))
                if(not isinstance(adv_dataset.list_of_x, torch.Tensor) and isinstance(adv_dataset.list_of_x[0], torch.Tensor)):
                    adv_dataset = torch.stack(
                        adv_dataset.list_of_x), torch.stack(adv_dataset.list_of_y)
                elif(isinstance(adv_dataset.list_of_x, torch.Tensor) and isinstance(adv_dataset.list_of_x[0], torch.Tensor)):
                    adv_dataset = adv_dataset.list_of_x, adv_dataset.list_of_y
                else:
                    adv_dataset = np.array(adv_dataset.list_of_x), np.array(
                        adv_dataset.list_of_y)

                print("Adv dataset size: X=>{},Y=>{}".format(
                    adv_dataset[0].shape, adv_dataset[1].shape))
                analyse_adv_dataset = adv_dataset[0][filtered_indices], adv_dataset[1][filtered_indices]
                print("Analyse adv dataset size: X=>{},Y=>{}".format(
                    analyse_adv_dataset[0].shape, analyse_adv_dataset[1].shape))

                orig_overlap,orig_sameclass_overlap,orig_diffclass_overlap = obtain_kernel_overlap(net, analyse_orig_dataset,is_consider_NPK)
                print("current_model_type:{}=>orig_overlap:{}".format(
                    current_model_type, orig_overlap))
                orig_overlap_map[current_model_type].append(
                    orig_overlap.item())
                orig_overlap_sameclass_map[current_model_type].append(
                    orig_sameclass_overlap.item())
                orig_overlap_diffclass_map[current_model_type].append(
                    orig_diffclass_overlap.item())
                adv_overlap,adv_sameclass_overlap,adv_diffclass_overlap = obtain_kernel_overlap(net, analyse_adv_dataset,is_consider_NPK)
                print("current_model_type:{}=>adv_overlap:{}".format(
                    current_model_type, adv_overlap))
                adv_overlap_map[current_model_type].append(adv_overlap.item())
                adv_overlap_sameclass_map[current_model_type].append(
                    adv_sameclass_overlap.item())
                adv_overlap_diffclass_map[current_model_type].append(
                    adv_diffclass_overlap.item())
                
                adv_orig_overlap,adv_orig_sameclass_overlap,adv_orig_diffclass_overlap = obtain_adv_orig_kernel_overlap(net,analyse_orig_dataset, analyse_adv_dataset,is_consider_NPK)
                print("current_model_type:{}=>adv_orig_overlap:{}".format(
                    current_model_type, adv_orig_overlap))
                adv_orig_overlap_map[current_model_type].append(adv_orig_overlap.item())
                adv_orig_overlap_sameclass_map[current_model_type].append(
                    adv_orig_sameclass_overlap.item())
                adv_orig_overlap_diffclass_map[current_model_type].append(
                    adv_orig_diffclass_overlap.item())

        for current_model_type in map_of_mtype_paths:
            if(is_log_wandb):
                mean_current_orig_overlap = mean(
                    orig_overlap_map[current_model_type])
                mean_current_orig_overlap_sameclass = mean(
                    orig_overlap_sameclass_map[current_model_type])
                mean_current_orig_overlap_diffclass = mean(
                    orig_overlap_diffclass_map[current_model_type])
                mean_current_adv_overlap = mean(
                    adv_overlap_map[current_model_type])
                mean_current_adv_overlap_sameclass = mean(
                    adv_overlap_sameclass_map[current_model_type])
                mean_current_adv_overlap_diffclass = mean(
                    adv_overlap_diffclass_map[current_model_type])
                mean_current_adv_orig_overlap = mean(
                    adv_orig_overlap_map[current_model_type])
                mean_current_adv_orig_overlap_sameclass = mean(
                    adv_orig_overlap_sameclass_map[current_model_type])
                mean_current_adv_orig_overlap_diffclass = mean(
                    adv_orig_overlap_diffclass_map[current_model_type])
                wandb.log({str(current_model_type)+"_or_k_lap": mean_current_orig_overlap,
                           str(current_model_type)+"_ad_k_lap": mean_current_adv_overlap,
                           str(current_model_type)+"same_or_k_lap": mean_current_orig_overlap_sameclass,
                           str(current_model_type)+"same_ad_k_lap": mean_current_adv_overlap_sameclass,
                           str(current_model_type)+"diff_or_k_lap": mean_current_orig_overlap_diffclass,
                           str(current_model_type)+"diff_ad_k_lap": mean_current_adv_overlap_diffclass,
                           str(current_model_type)+"_or_k_lap_L2": math.log2(mean_current_orig_overlap),
                           str(current_model_type)+"_ad_k_lap_L2": math.log2(mean_current_adv_overlap),
                           "same"+str(current_model_type)+"_or_k_lap_L2": math.log2(mean_current_orig_overlap_sameclass),
                           "same"+str(current_model_type)+"_ad_k_lap_L2": math.log2(mean_current_adv_overlap_sameclass),
                           "diff"+str(current_model_type)+"_or_k_lap_L2": math.log2(mean_current_orig_overlap_diffclass),
                           "diff"+str(current_model_type)+"_ad_k_lap_L2": math.log2(mean_current_adv_overlap_diffclass),
                           str(current_model_type)+"_ad_org_k_lap_L2": np.sign(mean_current_adv_orig_overlap) * math.log2(abs(mean_current_adv_orig_overlap)),
                           "ado_same"+str(current_model_type)+"_ad_org_k_lap_L2": math.log2(mean_current_adv_orig_overlap_sameclass),
                           "ado_diff"+str(current_model_type)+"_ad_org_k_lap_L2": math.log2(mean_current_adv_orig_overlap_diffclass)})

        if(is_log_wandb):
            wandb.finish()
    elif(exp_type == "All_PAIRS_K_OVERLAP"):
        tar_classes = [i for i in range(10)]
        rows = []
        headers = ["SRC Class","Model Type"]
        metrics = ["same_or_k_lap_L2","same_ad_k_lap_L2","diff_or_k_lap_L2","diff_ad_k_lap_L2","ado_same_ad_org_k_lap_L2","ado_diff_ad_org_k_lap_L2"]
        ext_header=[]
        for tar_class in tar_classes:
            ext_header.extend([str(tar_class)]*len(metrics))
        second_row = [""]*len(headers) + metrics*len(tar_classes)
        overall_header = headers + ext_header
        allrows = []
        allrows.append(overall_header)
        allrows.append(second_row)
        for src_class in range(10):
            for current_model_type in map_of_mtype_paths:
                current_direct_model_path = map_of_mtype_paths[current_model_type]

                isExist = os.path.exists(current_direct_model_path)
                assert isExist == True, 'Model path does not have saved model'

                custom_temp_model = torch.load(current_direct_model_path)
                net.load_state_dict(custom_temp_model.state_dict())

                net = net.to(device)

                if('CLEAN' in current_direct_model_path or 'APR_TRAINING' in current_direct_model_path):
                    model_and_data_save_prefix = current_direct_model_path[0:current_direct_model_path.rfind(
                        ".pt")]
                else:
                    model_and_data_save_prefix = current_direct_model_path[0:current_direct_model_path.rfind(
                        "/")+1]
                cur_rows = [str(src_class),str(current_model_type)]
                for tar_class in tar_classes:
                    print("SRC Class:{} *********************************** Tar Class:{}".format(src_class,tar_class))
                    if(tar_class == src_class):
                        cur_rows.extend(["0"]*len(metrics))
                        continue
                    d_config = DatasetConfig(
                        dataset, is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size, list_of_classes=[src_class,tar_class])

                    trainloader, _, testloader = preprocess_dataset_get_data_loader(
                        d_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

                    if(is_analysis_on_train == True):
                        eval_loader = trainloader
                    else:
                        eval_loader = testloader

                    number_of_trails = 1

                    orig_dataset = generate_dataset_from_loader(eval_loader)
                    if(isinstance(orig_dataset.list_of_x[0], torch.Tensor)):
                        orig_dataset = torch.stack(
                            orig_dataset.list_of_x), torch.stack(orig_dataset.list_of_y)
                    else:
                        orig_dataset = np.array(orig_dataset.list_of_x), np.array(
                            orig_dataset.list_of_y)

                    number_of_samples = orig_dataset[0].shape[0]
                    print("Total number_of_samples", number_of_samples)

                    number_of_samples_used_for_analysis = int(
                        (percentage_of_dataset_for_analysis/100)*number_of_samples)

                    print("Total number of filtered samples for analysis",
                        number_of_samples_used_for_analysis)

                    np.random.seed(torch_seed)
                    indices = np.arange(0, number_of_samples)

                    orig_overlap_map = []
                    orig_overlap_sameclass_map = []
                    orig_overlap_diffclass_map = []
                    adv_overlap_map = []
                    adv_overlap_sameclass_map = []
                    adv_overlap_diffclass_map = []
                    adv_orig_overlap_map = []
                    adv_orig_overlap_sameclass_map = []
                    adv_orig_overlap_diffclass_map = []

                    for trail_num in range(number_of_trails):
                        np.random.shuffle(indices)

                        filtered_indices = indices[:number_of_samples_used_for_analysis]

                        print("Orig dataset size: X=>{},Y=>{}".format(
                            orig_dataset[0].shape, orig_dataset[1].shape))
                        analyse_orig_dataset = orig_dataset[0][filtered_indices], orig_dataset[1][filtered_indices]
                        print("Analyse orig dataset size: X=>{},Y=>{}".format(
                            analyse_orig_dataset[0].shape, analyse_orig_dataset[1].shape))


                        adv_dataset = load_or_generate_adv_examples(eval_loader, "", is_analysis_on_train, net, eps, adv_attack_type,
                                                                    number_of_adversarial_optimization_steps, eps_step_size, adv_target, is_save_adv=False)
                        print("Adv dataset size: X=>{},Y=>{}".format(
                            adv_dataset.list_of_x[0].shape, adv_dataset.list_of_y[0].shape))
                        print("Adv dataset size: X=>{},Y=>{}".format(
                            adv_dataset.list_of_x.shape, adv_dataset.list_of_y.shape))
                        if(not isinstance(adv_dataset.list_of_x, torch.Tensor) and isinstance(adv_dataset.list_of_x[0], torch.Tensor)):
                            adv_dataset = torch.stack(
                                adv_dataset.list_of_x), torch.stack(adv_dataset.list_of_y)
                        elif(isinstance(adv_dataset.list_of_x, torch.Tensor) and isinstance(adv_dataset.list_of_x[0], torch.Tensor)):
                            adv_dataset = adv_dataset.list_of_x, adv_dataset.list_of_y
                        else:
                            adv_dataset = np.array(adv_dataset.list_of_x), np.array(
                                adv_dataset.list_of_y)

                        print("Adv dataset size: X=>{},Y=>{}".format(
                            adv_dataset[0].shape, adv_dataset[1].shape))
                        analyse_adv_dataset = adv_dataset[0][filtered_indices], adv_dataset[1][filtered_indices]
                        print("Analyse adv dataset size: X=>{},Y=>{}".format(
                            analyse_adv_dataset[0].shape, analyse_adv_dataset[1].shape))

                        orig_overlap,orig_sameclass_overlap,orig_diffclass_overlap = obtain_kernel_overlap(net, analyse_orig_dataset,is_consider_NPK)
                        print("current_model_type:{}=>orig_overlap:{}".format(
                            current_model_type, orig_overlap))
                        orig_overlap_map.append(
                            orig_overlap.item())
                        orig_overlap_sameclass_map.append(
                            orig_sameclass_overlap.item())
                        orig_overlap_diffclass_map.append(
                            orig_diffclass_overlap.item())
                        adv_overlap,adv_sameclass_overlap,adv_diffclass_overlap = obtain_kernel_overlap(net, analyse_adv_dataset,is_consider_NPK)
                        print("current_model_type:{}=>adv_overlap:{}".format(
                            current_model_type, adv_overlap))
                        adv_overlap_map.append(adv_overlap.item())
                        adv_overlap_sameclass_map.append(
                            adv_sameclass_overlap.item())
                        adv_overlap_diffclass_map.append(
                            adv_diffclass_overlap.item())
                        
                        adv_orig_overlap,adv_orig_sameclass_overlap,adv_orig_diffclass_overlap = obtain_adv_orig_kernel_overlap(net,analyse_orig_dataset, analyse_adv_dataset,is_consider_NPK)
                        print("current_model_type:{}=>adv_orig_overlap:{}".format(
                            current_model_type, adv_orig_overlap))
                        adv_orig_overlap_map.append(adv_orig_overlap.item())
                        adv_orig_overlap_sameclass_map.append(
                            adv_orig_sameclass_overlap.item())
                        adv_orig_overlap_diffclass_map.append(
                            adv_orig_diffclass_overlap.item())

                        mean_current_orig_overlap = mean(orig_overlap_map)
                        mean_current_orig_overlap_sameclass = mean(orig_overlap_sameclass_map)
                        mean_current_orig_overlap_diffclass = mean(orig_overlap_diffclass_map)
                        mean_current_adv_overlap = mean(adv_overlap_map)
                        mean_current_adv_overlap_sameclass = mean(adv_overlap_sameclass_map)
                        mean_current_adv_overlap_diffclass = mean(adv_overlap_diffclass_map)
                        mean_current_adv_orig_overlap = mean(adv_orig_overlap_map)
                        mean_current_adv_orig_overlap_sameclass = mean(adv_orig_overlap_sameclass_map)
                        mean_current_adv_orig_overlap_diffclass = mean(adv_orig_overlap_diffclass_map)

                        cur_rows.append(math.log2(mean_current_orig_overlap_sameclass))
                        cur_rows.append(math.log2(mean_current_adv_overlap_sameclass))
                        cur_rows.append(math.log2(mean_current_orig_overlap_diffclass))
                        cur_rows.append(math.log2(mean_current_adv_overlap_diffclass))
                        cur_rows.append(math.log2(mean_current_adv_orig_overlap_sameclass))
                        cur_rows.append(math.log2(mean_current_adv_orig_overlap_diffclass))

                allrows.append(cur_rows)
        
        with open("tmp/kernel_overlap_all_classes_mnist.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(allrows)

    print("Finished execution!!!")

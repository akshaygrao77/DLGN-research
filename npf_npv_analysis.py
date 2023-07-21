import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import time
import os
import wandb
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt

from external_utils import format_time
from utils.data_preprocessing import preprocess_dataset_get_data_loader, generate_merged_dataset_from_two_loader, generate_dataset_from_loader,preprocess_mnist_fmnist,get_data_loader
from structure.dlgn_conv_config_structure import DatasetConfig
from collections import OrderedDict

from visualization import recreate_image, save_image,  PerClassDataset
from utils.data_preprocessing import true_segregation
from structure.generic_structure import CustomSimpleDataset
from adversarial_attacks_tester import generate_adv_examples
from configs.dlgn_conv_config import HardRelu

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from keras.datasets import mnist, fashion_mnist

from conv4_models import get_model_instance, get_model_instance_from_dataset


def get_model_from_path(dataset, model_arch_type, model_path, mask_percentage=40,custom_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_model = torch.load(model_path, map_location=device)
    if(custom_model is None):
        custom_model = get_model_instance_from_dataset(
            dataset, model_arch_type)
    if("masked" in model_arch_type):
        custom_model = get_model_instance_from_dataset(
            dataset, model_arch_type, mask_percentage=mask_percentage)
    if(isinstance(temp_model, dict)):
        if("module." in [*temp_model['state_dict'].keys()][0]):
            new_state_dict = OrderedDict()
            for k, v in temp_model['state_dict'].items():
                name = k[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v
            custom_model.load_state_dict(new_state_dict)
        else:
            custom_model.load_state_dict(temp_model['state_dict'])
    else:
        custom_model.load_state_dict(temp_model.state_dict())

    return custom_model

def output_PIL_image(images,path):
    std01_vis_images = recreate_image(
        images, unnormalize=False)
    save_image(std01_vis_images,path)

def output_plt_image(images,path,title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    cax = ax.imshow(images.detach().cpu().numpy(), interpolation='nearest')
    ax.set_title(title)
    cbar = fig.colorbar(cax)
    fig.tight_layout()
    fig.savefig(path,bbox_inches='tight')

def get_fixedNPVs(model):
    prev = None
    for each_module in model.value_network.list_of_modules:
        cur_w = each_module.weight
        if(prev is None):
            prev = torch.transpose(cur_w,0,1)
        else:
            tmp=[]
            for i in range(prev.size()[0]):
                t1 = torch.unsqueeze(prev[i],0)
                rtmp = torch.transpose(cur_w*t1,0,1)
                tmp.append(torch.reshape(rtmp,(np.prod(list(rtmp.size())),1)))
                
            prev = torch.squeeze(torch.stack(tmp))
    return torch.squeeze(torch.reshape(prev,(np.prod(list(prev.size())),1)))

def get_NPFs(outlists):
    prev = None
    for cur_w in outlists:
        if(prev is None):
            prev = torch.transpose(cur_w,0,1)
        else:
            cur_w = torch.transpose(cur_w,0,1)
            tmp=[]
            for i in range(prev.size()[0]):
                t1 = torch.unsqueeze(prev[i],0)
                rtmp = torch.transpose(t1*cur_w,0,1)
                tmp.append(rtmp)
                
            prev = torch.transpose(torch.concat(tmp,dim=1),0,1)
    
    return prev

def generate_NPF_stats(model,dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = tqdm.tqdm(dataloader, desc='Generating NPF')
    agg_npfs = 0
    tot_size = 0
    hrelu = HardRelu()
    for _, data in enumerate(loader, 0):
        inputs, _ = data
        inputs = inputs.to(
            device)
        model(inputs)
        tot_size += inputs.size()[0]
        for indx in range(len(model.linear_conv_outputs)):
            model.linear_conv_outputs[indx] = hrelu(model.linear_conv_outputs[indx])
        model.linear_conv_outputs.insert(0,torch.flatten(inputs, 1))
        npfs = get_NPFs(model.linear_conv_outputs)
        agg_npfs += torch.sum(npfs,1)
    agg_npfs = agg_npfs/tot_size

    return agg_npfs

def output_bar_graph(xlist,ylist,path,title=""):
    # col = ['blue'if i%2==0 else 'red' for i in range(len(xlist))]
    # fig = plt.figure(figsize =(len(xlist)//10, 7))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # xticks = [0]
    # while(xticks[-1]<len(xlist)):
    #     xticks.append(xticks[-1]+10)
    # ax.set_xticks(xticks)
    ax.bar(xlist,ylist,align='edge')
    ax.set_title(title)
    # fig.tight_layout()
    fig.savefig(path,bbox_inches='tight')

if __name__ == '__main__':
    # fashion_mnist , mnist, cifar10
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net , conv4_deep_gated_net_n16_small
    # fc_dnn , fc_dlgn , fc_dgn , dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias__
    model_arch_type = 'fc_dlgn'
    batch_size = 64

    is_analysis_on_train = False

    torch_seed = 2022

    # None means that train on all classes
    list_of_classes_to_train_on = None
    list_of_classes_to_train_on = [3,8]

    # Percentage of information retention during PCA (values between 0-1)
    pca_exp_percent = None
    # pca_exp_percent = 0.85

    is_analyse_adv = True

    wandb_config_additional_dict = None
    # wandb_config_additional_dict = {"type_of_APR": "APRS"}

    # direct_model_path = "root/model/save/mnist/CLEAN_TRAINING/TR_ON_3_8/ST_2022/fc_dlgn_W_10_D_2_dir.pt"
    direct_model_path = "root/model/save/mnist/adversarial_training/TR_ON_3_8/MT_fc_dlgn_W_10_D_2_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.06/batch_size_128/eps_stp_size_0.06/adv_steps_80/adv_model_dir.pt"

    custom_dataset_path = None
    # custom_dataset_path = "data/custom_datasets/freq_band_dataset/mnist__MB_HB.npy"

    if(dataset == "cifar10"):
        inp_channel = 3
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = len(classes)

        cifar10_config = DatasetConfig(
            'cifar10', is_normalize_data=False, valid_split_size=0.1, batch_size=batch_size, 
            list_of_classes=list_of_classes_to_train_on,custom_dataset_path=custom_dataset_path)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            cifar10_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    elif(dataset == "mnist"):
        inp_channel = 1
        classes = [str(i) for i in range(0, 10)]
        num_classes = len(classes)

        ds_config = DatasetConfig(
            'mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size, 
            list_of_classes=list_of_classes_to_train_on,custom_dataset_path=custom_dataset_path)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            ds_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    elif(dataset == "fashion_mnist"):
        inp_channel = 1
        classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot')
        num_classes = len(classes)

        ds_config = DatasetConfig(
            'fashion_mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size, 
            list_of_classes=list_of_classes_to_train_on,custom_dataset_path=custom_dataset_path)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            ds_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)
    
    if(custom_dataset_path is not None):
        dataset = custom_dataset_path[custom_dataset_path.rfind("/")+1:custom_dataset_path.rfind(".npy")]
    
    print("Testing over "+dataset)
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
        fc_width = 10
        fc_depth = 2
        nodes_in_each_layer_list = [fc_width] * fc_depth
        model_arch_type_str = model_arch_type_str + \
            "_W_"+str(fc_width)+"_D_"+str(fc_depth)
        net = get_model_instance_from_dataset(dataset,
                                              model_arch_type, seed=torch_seed, num_classes=num_classes_trained_on, nodes_in_each_layer_list=nodes_in_each_layer_list)
    else:
        net = get_model_instance(model_arch_type, inp_channel,
                                 seed=torch_seed, num_classes=num_classes_trained_on)

    if(pca_exp_percent is not None):
        dataset_for_pca = generate_dataset_from_loader(trainloader)
        if(isinstance(dataset_for_pca.list_of_x[0], torch.Tensor)):
            dataset_for_pca = torch.stack(
                dataset_for_pca.list_of_x), torch.stack(dataset_for_pca.list_of_y)
        else:
            dataset_for_pca = np.array(dataset_for_pca.list_of_x), np.array(
                dataset_for_pca.list_of_y)
        number_of_components_for_pca = net.initialize_PCA_transformation(
            dataset_for_pca[0], pca_exp_percent)
        model_arch_type_str = model_arch_type_str + \
            "_PCA_K"+str(number_of_components_for_pca) + \
            "_P_"+str(pca_exp_percent)

    if('CLEAN' in direct_model_path or 'APR_TRAINING' in direct_model_path or 'adv_model_dir_epoch' in direct_model_path):
        data_save_prefix = direct_model_path[0:direct_model_path.rfind(
            ".pt")]
    else:
        data_save_prefix = direct_model_path[0:direct_model_path.rfind(
            "/")+1]
    
    isExist = os.path.exists(direct_model_path)
    assert isExist == True, 'Model path does not have saved model'

    net = get_model_from_path(
        dataset, model_arch_type, direct_model_path,custom_model=net)

    net = net.to(device)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_str == 'cuda':
        cudnn.benchmark = True
    
    if(is_analysis_on_train == True):
        eval_loader = trainloader
    else:
        eval_loader = testloader
    
    class_indx_to_visualize = [i for i in range(len(classes))]
    # class_indx_to_visualize= []

    if(is_analyse_adv):
        number_of_adversarial_optimization_steps = 161
        adv_attack_type = "PGD"
        adv_target = None
        eps_step_size = 0.06
        eps = 0.06
        is_adv_attack_on_train = is_analysis_on_train

        final_adv_postfix_for_save = "/RAW_ADV_SAVES/adv_type_{}/EPS_{}/eps_stp_size_{}/adv_steps_{}/on_train_{}/".format(
                adv_attack_type, eps, eps_step_size, number_of_adversarial_optimization_steps, is_adv_attack_on_train)
        adv_save_path = data_save_prefix + \
            final_adv_postfix_for_save+"/adv_dataset.npy"
        is_current_adv_aug_available = os.path.exists(
            adv_save_path)
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
            print("adv_save_path:", adv_save_path)
            adv_dataset = generate_adv_examples(
                eval_loader, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, is_save_adv=True, save_path=adv_save_path)
        
        to_be_analysed_adversarial_dataloader = torch.utils.data.DataLoader(
                        adv_dataset, shuffle=False, batch_size=batch_size)
        true_tobe_analysed_dataset_per_class = true_segregation(
                        to_be_analysed_adversarial_dataloader, num_classes_trained_on)
    
    true_eval_dataset_per_class = true_segregation(
        eval_loader, num_classes_trained_on)
    
    final_data_save_postfix = "/NPF_NPV_ANALYSIS/ON_TRAIN_{}/dataset_{}/allclasses/".format(
                    is_analysis_on_train,dataset)
    save_folder = data_save_prefix + final_data_save_postfix
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # print("Will be saved in folder:"+save_folder)
    npvs = get_fixedNPVs(net)
    pos_npv_indx = (npvs >0).nonzero()
    neg_npv_indx = (npvs <0).nonzero()
    pos_npvs = torch.squeeze(npvs[pos_npv_indx])
    neg_npvs = torch.squeeze(npvs[neg_npv_indx])
    print("npvs:{} Max NPV:{} Min NPV:{}  npvs positive count:{} npvs negative count:{}".format(npvs.size(),torch.max(npvs),torch.min(npvs),pos_npvs.numel(),neg_npvs.numel()))
    
    # agg_npfs,npvs = generate_NPF_stats_on_fixedNPVs(net,eval_loader)
    # print("agg_npfs:{} npvs:{}".format(agg_npfs.size(),npvs.size()))

    # output_bar_graph(npvs.tolist(),agg_npfs.tolist(),save_folder+'/NPV_NPF.jpeg',"Frequency of NPFs over NPVs")

    for c_indx in class_indx_to_visualize:
        class_label = classes[c_indx]
        print(
            "************************************************************ Class:", class_label)
        per_class_eval_dataset = PerClassDataset(
            true_eval_dataset_per_class[c_indx], c_indx)
        if(is_analyse_adv):
            per_class_tobe_analysed_dataset = PerClassDataset(
                                    true_tobe_analysed_dataset_per_class[c_indx], c_indx)
            per_class_tobe_analysed_data_loader = torch.utils.data.DataLoader(per_class_tobe_analysed_dataset, batch_size=256,
                                                                        shuffle=False)
        
        final_data_save_postfix = "/NPF_NPV_ANALYSIS/ON_TRAIN_{}/c_indx_{}_class_label_{}/".format(
                            is_analysis_on_train, c_indx, class_label)

        per_class_eval_data_loader = torch.utils.data.DataLoader(per_class_eval_dataset, batch_size=256,
                                                                    shuffle=False)
        save_folder = data_save_prefix + final_data_save_postfix
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with torch.no_grad():
            agg_npfs = generate_NPF_stats(net,per_class_eval_data_loader)
            print("agg_npfs size:{} max:{} min:{} nonzero:{} agg_npfs:{}".format(agg_npfs.size(),torch.max(agg_npfs),torch.min(agg_npfs),torch.count_nonzero(agg_npfs),agg_npfs))
            agg_npfs_of_posnpv = torch.squeeze(agg_npfs[pos_npv_indx])
            print("agg_npfs_of_negnpv size:{} max:{} min:{} nonzero:{} agg_npfs_of_negnpv:{}".format(agg_npfs_of_posnpv.size(),torch.max(agg_npfs_of_posnpv),torch.min(agg_npfs_of_posnpv),torch.count_nonzero(agg_npfs_of_posnpv),agg_npfs_of_posnpv))
            output_bar_graph(pos_npvs.tolist(),agg_npfs_of_posnpv.tolist(),save_folder+'/NPF_of_pos_NPV.jpeg',"NPFs of positive NPVs")
            agg_npfs_of_negnpv = torch.squeeze(agg_npfs[neg_npv_indx])
            print("agg_npfs_of_negnpv size:{} max:{} min:{} nonzero:{} agg_npfs_of_negnpv:{}".format(agg_npfs_of_negnpv.size(),torch.max(agg_npfs_of_negnpv),torch.min(agg_npfs_of_negnpv),torch.count_nonzero(agg_npfs_of_negnpv),agg_npfs_of_negnpv))
            output_bar_graph(neg_npvs.tolist(),agg_npfs_of_negnpv.tolist(),save_folder+'/NPF_of_neg_NPV.jpeg',"NPFs of negative NPVs")
        
        if(is_analyse_adv):
            with torch.no_grad():
                agg_npfs = generate_NPF_stats(net,per_class_tobe_analysed_data_loader)
                print("agg_npfs size:{} max:{} min:{} nonzero:{} agg_npfs:{}".format(agg_npfs.size(),torch.max(agg_npfs),torch.min(agg_npfs),torch.count_nonzero(agg_npfs),agg_npfs))
                agg_npfs_of_posnpv = torch.squeeze(agg_npfs[pos_npv_indx])
                print("agg_npfs_of_negnpv size:{} max:{} min:{} nonzero:{} agg_npfs_of_negnpv:{}".format(agg_npfs_of_posnpv.size(),torch.max(agg_npfs_of_posnpv),torch.min(agg_npfs_of_posnpv),torch.count_nonzero(agg_npfs_of_posnpv),agg_npfs_of_posnpv))
                output_bar_graph(pos_npvs.tolist(),agg_npfs_of_posnpv.tolist(),save_folder+'/Adv_NPF_of_pos_NPV.jpeg',"Adv NPFs of positive NPVs")
                agg_npfs_of_negnpv = torch.squeeze(agg_npfs[neg_npv_indx])
                print("agg_npfs_of_negnpv size:{} max:{} min:{} nonzero:{} agg_npfs_of_negnpv:{}".format(agg_npfs_of_negnpv.size(),torch.max(agg_npfs_of_negnpv),torch.min(agg_npfs_of_negnpv),torch.count_nonzero(agg_npfs_of_negnpv),agg_npfs_of_negnpv))
                output_bar_graph(neg_npvs.tolist(),agg_npfs_of_negnpv.tolist(),save_folder+'/Adv_NPF_of_neg_NPV.jpeg',"Adv NPFs of negative NPVs")

    print("Completed!")
    
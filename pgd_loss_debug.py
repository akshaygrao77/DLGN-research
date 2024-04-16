import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import time
import math
import os
import wandb
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import csv
from collections import OrderedDict

from external_utils import format_time
from utils.data_preprocessing import preprocess_dataset_get_dataset, generate_dataset_from_loader,preprocess_dataset_get_data_loader,get_data_loader
from structure.dlgn_conv_config_structure import DatasetConfig
from conv4_models import get_model_instance, get_model_save_path, get_model_instance_from_dataset, get_img_size
import numpy as np
from utils.generic_utils import Y_True_Loss

from attacks import cleverhans_projected_gradient_descent,cleverhans_fast_gradient_method,get_locuslab_adv_per_batch,get_residue_adv_per_batch

def generate_table_row(model,X,y,sorted_list_steps,torch_seed,batch_idx,alpha_folder,residue_vname,fast_adv_attack_type,target_indx=16):
    relu=torch.nn.ReLU()
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    # tr_loss_fn = Y_True_Loss()
    eps = 0.3
    eps_step_size = 0.01
    # eps_step_size = 0.3
    number_of_adversarial_optimization_steps = sorted_list_steps[0]
    update_on = 'all'
    rand_init = True
    clip_min = 0.0
    clip_max = 1.0
    eta_growth_reduced_rate = 1

    cur_row = []
    all_losses = []
    l2_grads=[]
    pos_grads = []
    eps_norms = []

    kargs = {"criterion":loss_fn,"eps":eps,"eps_step_size":eps_step_size,"steps":number_of_adversarial_optimization_steps,"update_on":update_on,'rand_init':rand_init,'clip_min':clip_min,'clip_max':clip_max,'targeted':False,'norm':np.inf,'residue_vname':residue_vname,"num_of_restrts":number_of_restarts}
    lr_sched_rates = [(40,0.005),(100,0.0025)]
    kargs["labels"] = y
    init_loss = loss_fn(model(X),y).item()
    cur_row.append(init_loss)
    adv_x = X.clone()
    debugs_losses = []
    alphas = np.arange(0.0, 2.0, 0.001)
    for i,cur_s in enumerate(sorted_list_steps):
        torch.manual_seed(torch_seed)
        kargs['steps'] = cur_s
        
        # kargs['residue_vname'] = None
        # # kargs['lr_sched'] = None
        # end_adv_x = cleverhans_projected_gradient_descent(model,X,kargs)
        # reach_end_loss = loss_fn(model(end_adv_x),y)
        # cur_row.append(reach_end_loss.item())

        kargs['residue_vname'] = residue_vname
        # kargs['lr_sched'] = lr_sched_rates
        torch.manual_seed(torch_seed)
        if fast_adv_attack_type == 'PGD':
            adv_x = cleverhans_projected_gradient_descent(model,X,kargs)
        elif fast_adv_attack_type == 'residual_PGD':
            kargs["eta_growth_reduced_rate"] = eta_growth_reduced_rate
            adv_x = get_residue_adv_per_batch(model,X,kargs)
        adv_x = adv_x.clone().detach().to(torch.float).requires_grad_(True)
        pgd_at_end_loss = loss_fn(model(adv_x),y)
        pgd_at_end_loss.backward()
        all_losses.append(pgd_at_end_loss.item())
        # print(adv_x.grad)
        
        l2_grads.append(torch.norm(adv_x.grad,p=2).item())
        
        # Ratio of pos grad sign
        pos_grads.append((torch.sum(torch.where(torch.sign(adv_x.grad)==1,1,0))/adv_x.grad.numel()).item())
        
        assert (((adv_x - X) > -(eps+1e-5)).all() and ((adv_x - X) < (eps+1e-5)).all()) , 'Wrong'
        tttmp = torch.where(adv_x > X,(torch.clamp(X + eps,clip_min,clip_max) - adv_x),(adv_x - torch.clamp(X - eps,clip_min,clip_max)))
        eps_norm = torch.norm(tttmp,p=1)
        eps_norms.append(eps_norm.item())

        if(batch_idx == target_indx):
            save_folder = "{}/target_indx_{}/".format(alpha_folder,str(batch_idx))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            # print("cur_s:{} pgd_at_end_loss:{} norm(adv_x.grad):{} adv_x.grad:{}".format(cur_s,pgd_at_end_loss.item(),torch.norm(adv_x.grad,p=2).item(),torch.flatten(adv_x.grad)))
            cur_debug_losses = []
            with torch.no_grad():
                for alpha in alphas:
                    if(residue_vname is not None and residue_vname == "L2_norm_grad_unitnorm"):
                        cgrad = (adv_x.grad / (torch.norm(adv_x.grad,p=2,dim=[1,2]).unsqueeze(1).unsqueeze(2) + 10e-8))
                    elif(residue_vname is not None and residue_vname == "L2_norm_grad_scale"):
                        cgrad = (adv_x.grad / (torch.norm(adv_x.grad,p=2,dim=[1,2]).unsqueeze(1).unsqueeze(2) + 10e-8)) * math.sqrt(adv_x[0].numel())
                    elif(residue_vname is not None and residue_vname == "PGD_unit_norm"):
                        tmp = torch.sign(adv_x.grad)
                        cgrad = (tmp / (torch.norm(tmp,p=2,dim=[1,2]).unsqueeze(1).unsqueeze(2) + 10e-8))
                    else:
                        cgrad = torch.sign(adv_x.grad)

                    # cgrad = (adv_x.grad / (torch.norm(adv_x.grad,p=2,dim=[1,2]).unsqueeze(1).unsqueeze(2) + 10e-8)) * math.sqrt(adv_x[0].numel())
                    tmp_X = torch.clamp(adv_x + alpha * cgrad,X-eps,X+eps)
                    cur_debug_losses.append(loss_fn(model(tmp_X),y).item())
            plt.plot(alphas, cur_debug_losses, label = str(cur_s))
            plt.legend()
            plt.savefig("{}/{}_{}_alpha_debug_fc_dnn_pgdat_model.jpg".format(save_folder,str(cur_s),residue_vname))
            plt.close()
            debugs_losses.append(cur_debug_losses)
    
    if(batch_idx == target_indx):
        plt.figure(figsize=(25,25))
        for i in range(len(debugs_losses)):
            cur_deb_loss = debugs_losses[i]
            plt.plot(alphas, cur_deb_loss, label = str(sorted_list_steps[i]))
        plt.legend()
        plt.savefig("{}/{}_alpha_debug_fc_dnn_pgdat_model.jpg".format(save_folder,residue_vname))
        plt.close()
        debugs_losses.insert(0,alphas)
        with open("{}/{}_alpha_debug_fc_dnn_pgdat_model.csv".format(save_folder,residue_vname), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(debugs_losses)
    cur_row.extend(all_losses)
    cur_row.extend(l2_grads)
    cur_row.extend(pos_grads)
    cur_row.extend(eps_norms)
    return cur_row
           

def generate_table(model,loader,sorted_list_steps,num_batches,alpha_folder,residue_vname,fast_adv_attack_type):
    rows = []
    all_losses = []
    l2_grads=[]
    pos_grads = []
    eps_norms = []
    header = ['init_loss']
    for ii in sorted_list_steps:
        # header.append('FGSM@{}'.format(ii))
        # header.append('{}@{}'.format(residue_vname,ii))
        all_losses.append('{}@{}'.format(residue_vname,ii))
        # header.append('S_{}_L2_grad_of{}'.format(ii,residue_vname))
        l2_grads.append('S_{}_L2_grad_of{}'.format(ii,residue_vname))
        # header.append('S_{}_Pos_ratio_grad_of_{}'.format(ii,residue_vname))
        pos_grads.append('S_{}_Pos_ratio_grad_of_{}'.format(ii,residue_vname))
        # header.append('S_{}_EPS_norm_of_{}'.format(ii,residue_vname))
        eps_norms.append('S_{}_EPS_norm_of_{}'.format(ii,residue_vname))
    header.extend(all_losses)
    header.extend(l2_grads)
    header.extend(pos_grads)
    header.extend(eps_norms)
    rows.append(header)
    loader = tqdm.tqdm(loader, desc='Generating Table Data')
    for batch_idx, data in enumerate(loader, 0):
        (X, y) = data
        X, y = X.cuda(), y.cuda()
        
        # if(batch_idx == 16):
        cur_row = generate_table_row(model,X,y,sorted_list_steps,200+batch_idx,batch_idx,alpha_folder,residue_vname,fast_adv_attack_type)
        rows.append(cur_row)
        
        if(batch_idx > num_batches):
            break
    
    return rows



if __name__ == '__main__':
    model_arch_type = "fc_dnn"
    fast_adv_attack_type = "residual_PGD"
    # L2_norm_grad_unitnorm , L2_norm_grad_scale , PGD_unit_norm , plain_grad_without_sign
    # eta_growth , max_eps
    residue_vname = "eta_growth"
    number_of_restarts = 1
    num_batches = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = "mnist"
    # None means that train on all classes
    list_of_classes_to_train_on = None
    # list_of_classes_to_train_on = [4,9]

    train_transforms = None
    is_normalize_data = True

    custom_dataset_path = None
    batch_size = 128

    # torch_seed = ""
    torch_seed = 2022

    if(dataset == "cifar10"):
        inp_channel = 3
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = len(classes)
        
        data_config = DatasetConfig(
            'cifar10', is_normalize_data=is_normalize_data, valid_split_size=0.1, batch_size=batch_size, list_of_classes=list_of_classes_to_train_on,
            train_transforms=train_transforms,custom_dataset_path=custom_dataset_path)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            data_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    elif(dataset == "mnist"):
        inp_channel = 1
        classes = [str(i) for i in range(0, 10)]
        num_classes = len(classes)
        
        data_config = DatasetConfig(
            'mnist', is_normalize_data=is_normalize_data, valid_split_size=0.1, batch_size=batch_size, list_of_classes=list_of_classes_to_train_on, 
            train_transforms=train_transforms,custom_dataset_path=custom_dataset_path)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            data_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    elif(dataset == "fashion_mnist"):
        inp_channel = 1
        classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot')
        num_classes = len(classes)
        
        data_config = DatasetConfig(
            'fashion_mnist', is_normalize_data=is_normalize_data, valid_split_size=0.1, batch_size=batch_size, list_of_classes=list_of_classes_to_train_on, 
            train_transforms=train_transforms,custom_dataset_path=custom_dataset_path)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            data_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    if(custom_dataset_path is not None):
        dataset = custom_dataset_path[custom_dataset_path.rfind("/")+1:custom_dataset_path.rfind(".npy")]

    print("Training over "+dataset)

    num_classes_trained_on = num_classes

    data_config = DatasetConfig(
                'mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=1, list_of_classes=None,custom_dataset_path=None)

    trainloader, _, testloader = preprocess_dataset_get_data_loader(
                data_config, "fc_dnn", verbose=1, dataset_folder="./Datasets/", is_split_validation=False)
    
    model_path = "root/model/save/mnist/adversarial_training/MT_fc_dnn_W_128_D_4_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.3/batch_size_64/eps_stp_size_0.01/adv_steps_40/update_on_all/R_init_True/norm_inf/use_ytrue_True/adv_model_dir_epoch_20.pt"
    # model_path = "root/model/save/mnist/adversarial_training/MT_fc_dnn_W_128_D_4_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_residual_PGD/adv_type_PGD/EPS_0.3/batch_size_64/eps_stp_size_0.01/adv_steps_40/update_on_all/R_init_True/norm_inf/use_ytrue_True/residue_vname_eta_growth/adv_model_dir_epoch_20.pt"
    
    # model_arch_type_str = model_arch_type
    # fc_width = 128
    # fc_depth = 4
    # nodes_in_each_layer_list = [fc_width] * fc_depth
    # model_arch_type_str = model_arch_type_str + \
    #     "_W_"+str(fc_width)+"_D_"+str(fc_depth)
    # net = get_model_instance_from_dataset("mnist",
    #                                         "fc_dnn", seed=2022, num_classes=num_classes_trained_on, nodes_in_each_layer_list=nodes_in_each_layer_list)

    net = torch.load(model_path)
    net = net.to(device)

    # edge_random_start , Y_True_Loss , EPS_0.27
    alpha_folder = "{}/alpha_debug/red_rate0.1/residue_{}/num_restrt_{}/".format(model_path.replace(".pt","/"),residue_vname,number_of_restarts)
    if not os.path.exists(alpha_folder):
        os.makedirs(alpha_folder)

    sorted_list_steps = [i for i in range(0,120,5)]
    loss_table = generate_table(net,trainloader,sorted_list_steps,num_batches,alpha_folder,residue_vname,fast_adv_attack_type)
    
    with open("{}/res_{}_nb_{}.csv".format(alpha_folder,residue_vname,num_batches), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(loss_table)
    
    print("Finished exec")

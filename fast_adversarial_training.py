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

from external_utils import format_time
from utils.data_preprocessing import preprocess_dataset_get_data_loader, generate_dataset_from_loader
from adversarial_attacks_tester import adv_evaluate_model, evaluate_model_via_reconstructed, plain_evaluate_model_via_reconstructed
from model_trainer import evaluate_model
from visualization import run_visualization_on_config
from structure.dlgn_conv_config_structure import DatasetConfig
from collections import OrderedDict
from attacks import cleverhans_projected_gradient_descent,cleverhans_fast_gradient_method,get_locuslab_adv_per_batch,get_residue_adv_per_batch,get_gateflip_adv_per_batch


from conv4_models import get_model_instance, get_model_instance_from_dataset

def get_wandb_config(exp_type,fast_adv_attack_type, adv_attack_type, model_arch_type_str, dataset_str,batch_size,epochs,
                    eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted,update_on,rand_init,norm,use_ytrue):
    wandb_config = dict()
    wandb_config["exp_type"] = exp_type
    wandb_config["adv_attack_type"] = adv_attack_type
    wandb_config["model_arch_type"] = model_arch_type_str
    wandb_config["dataset"] = dataset_str
    wandb_config["eps"] = eps
    wandb_config["number_of_adversarial_optimization_steps"] = number_of_adversarial_optimization_steps
    wandb_config["epochs"] = epochs
    wandb_config["batch_size"] = batch_size
    wandb_config["fast_adv_attack_type"] = fast_adv_attack_type
    wandb_config["eps_step_size"] = eps_step_size
    wandb_config["model_save_path"] = model_save_path
    wandb_config["update_on"] = update_on
    wandb_config["is_targetted"] = is_targetted
    wandb_config["rand_init"] = rand_init
    wandb_config["norm"] = norm
    wandb_config["use_ytrue"] = use_ytrue

    return wandb_config

def perform_adversarial_training(model, train_loader, test_loader, eps_step_size, adv_target, eps, fast_adv_attack_type, adv_attack_type, number_of_adversarial_optimization_steps, model_save_path, epochs=32, wand_project_name=None, lr_type='cyclic', lr_max=5e-3,dataset=None,npk_reg=0,update_on='all',rand_init=True,norm=np.inf,use_ytrue=True,clip_min=0.0,clip_max=1.0,residue_vname='std',opt=None,eta_growth_reduced_rate=1):
    targeted = adv_target is not None
    print("Model will be saved at", model_save_path)
    save_adv_image_prefix = model_save_path[0:model_save_path.rfind("/")+1]
    if not os.path.exists(save_adv_image_prefix):
        os.makedirs(save_adv_image_prefix)
    
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_str == 'cuda':
        if(torch.cuda.device_count() > 1):
            print("Parallelizing model")
            model = torch.nn.DataParallel(model)

        cudnn.benchmark = True
    is_log_wandb = not(wand_project_name is None)
    best_test_acc = 0
    best_rob_orig_acc = 0

    criterion = nn.CrossEntropyLoss()
    if("bc_" in model_arch_type):
        criterion = nn.BCELoss()
    
    epoch = 0
    if(dataset is not None and dataset == "cifar10" or (residue_vname is not None and residue_vname == "cyclic_lr")):
        print("Using cyclic scheduler")
        opt = torch.optim.Adam(model.parameters(), lr=lr_max)
        if lr_type == 'cyclic':
            def lr_schedule(t): return np.interp(
                [t], [0, epochs * 2//5, epochs], [0, lr_max, 0])[0]
        elif lr_type == 'flat':
            def lr_schedule(t): return lr_max
        else:
            raise ValueError('Unknown lr_type')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

    if fast_adv_attack_type == 'FGSM':
        kargs = {"criterion":criterion,"eps":eps,"eps_step_size":eps,"steps":1,"update_on":update_on,'rand_init':rand_init,'clip_min':clip_min,'clip_max':clip_max,'targeted':targeted,'norm':norm}
    elif fast_adv_attack_type == 'PGD':
        kargs = {"criterion":criterion,"eps":eps,"eps_step_size":eps_step_size,"steps":number_of_adversarial_optimization_steps,"update_on":update_on,'rand_init':rand_init,'clip_min':clip_min,'clip_max':clip_max,'targeted':targeted,'norm':norm,'residue_vname':residue_vname}
    elif fast_adv_attack_type == 'residual_PGD':
        kargs = {"criterion":criterion,"eps":eps,"eps_step_size":eps_step_size,"steps":number_of_adversarial_optimization_steps,"update_on":update_on,'rand_init':rand_init,'clip_min':clip_min,'clip_max':clip_max,'targeted':targeted,'norm':norm,'residue_vname':residue_vname}
    elif fast_adv_attack_type == 'FEATURE_FLIP':
        kargs = {"criterion":criterion,"eps":eps,"eps_step_size":eps_step_size,"steps":number_of_adversarial_optimization_steps,"update_on":update_on,'rand_init':rand_init,'clip_min':clip_min,'clip_max':clip_max,'targeted':targeted,'norm':norm,'residue_vname':residue_vname}
    while(epoch < epochs and (start_net_path is None or (stop_at_adv_test_acc is None or best_test_acc < stop_at_adv_test_acc))):
        correct = 0
        total = 0
        overall_eps_norm_mean = 0

        running_loss = 0.0
        running_before_adv_loss = 0.0
        loader = tqdm.tqdm(train_loader, desc='Training')
        for batch_idx, data in enumerate(loader, 0):
            begin_time = time.time()
            loader.set_description(f"Epoch {epoch+1}")
            (X, y) = data
            X, y = X.cuda(), y.cuda()
            if(dataset is not None and dataset == "cifar10" or (residue_vname is not None and residue_vname == "cyclic_lr")):
                lr = lr_schedule(epoch + (batch_idx+1)/len(train_loader))
                opt.param_groups[0].update(lr=lr)
            
            before_adv_loss = criterion(model(X), y)
            running_before_adv_loss += before_adv_loss.item() * y.size(0)

            kargs["labels"] = y if use_ytrue else None  
            if fast_adv_attack_type == 'FGSM':
                inputs = cleverhans_fast_gradient_method(model,X,kargs)
            elif fast_adv_attack_type == 'PGD':
                inputs = cleverhans_projected_gradient_descent(model,X,kargs)
            elif fast_adv_attack_type == 'residual_PGD':
                kargs["eta_growth_reduced_rate"] = eta_growth_reduced_rate
                inputs = get_residue_adv_per_batch(model,X,kargs)
            elif fast_adv_attack_type == 'FEATURE_FLIP':
                inputs = get_gateflip_adv_per_batch(model,X,kargs)
            
            cur_eps_norm_left = torch.sum(torch.norm(torch.where(inputs > X,(torch.clamp(X + eps,clip_min,clip_max) - inputs),(inputs - torch.clamp(X - eps,clip_min,clip_max))),p=1,dim=[1,2]))
            overall_eps_norm_mean += cur_eps_norm_left
            output = model(inputs)
            if(len(output.size())==1):
                predicted = output.data.round()
            else:
                _, predicted = torch.max(output.data, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

            loss = criterion(output, y)
            if(npk_reg != 0):
                beta=4
                if(isinstance(net, torch.nn.DataParallel)):
                    conv_outs = net.module.linear_conv_outputs
                    if(hasattr(net.module,"beta")):
                        beta = net.module.beta
                else:
                    conv_outs = net.linear_conv_outputs
                    if(hasattr(net,"beta")):
                        beta = net.beta
                    
                y = torch.where(y == 0, -1, 1)
                y = torch.unsqueeze(y, 1)
                y = y.type(torch.float32)
                inputs = torch.flatten(inputs, 1)
                npk_kernel = torch.matmul(inputs, torch.transpose(inputs, 0, 1))
                for each_conv_out in conv_outs:
                    gate_out = nn.Sigmoid()(beta * each_conv_out)
                    npk_kernel = npk_kernel * \
                        (torch.matmul(gate_out,  torch.transpose(gate_out, 0, 1)))
                width = conv_outs[0].size()[1]
                depth = len(conv_outs)
                npk_kernel = npk_kernel / (pow(width, depth)*npk_kernel.numel())
                overlap = nn.ReLU()(torch.matmul(torch.matmul(
                    torch.transpose(y, 0, 1), npk_kernel), y))
                # print("Loss:{} npk_reg*overlap:{}".format(loss,npk_reg*overlap))
                loss = loss + npk_reg*overlap
            
            opt.zero_grad()
            loss.backward()
            # for ip,pp in enumerate(model.parameters()):
            #     print("Param grad {} max:{} mean:{}".format(ip,torch.max(pp.grad),torch.mean(pp.grad)))
            opt.step()

            running_loss += loss.item() * y.size(0)

            cur_time = time.time()
            step_time = cur_time - begin_time
            loader.set_postfix(train_loss=running_loss/(batch_idx+1),before_adv_loss=running_before_adv_loss/(batch_idx+1),
                               train_acc=100.*correct/total, ratio="{}/{}".format(correct, total), stime=format_time(step_time))

        live_train_acc = 100. * correct / total
        overall_eps_norm_mean = overall_eps_norm_mean / total
        org_test_acc,_ = evaluate_model(net, test_loader)
        test_acc = adv_evaluate_model(net, test_loader, classes, eps, adv_attack_type)
        if(is_log_wandb):
            wandb.log({"live_train_acc": live_train_acc,"overall_eps_norm_mean":overall_eps_norm_mean,"tr_loss":running_loss/(batch_idx+1),'before_adv_tr_loss':running_before_adv_loss/(batch_idx+1),
                      "current_epoch": epoch, "test_acc": test_acc,"org_test_acc":org_test_acc})
        if(epoch % 5 == 0):
            per_epoch_save_model_path = model_save_path.replace(
                ".pt", '_epoch_{}.pt'.format(epoch))
            save_adv_image_prefix = per_epoch_save_model_path[0:per_epoch_save_model_path.rfind("/")+1]
            if not os.path.exists(save_adv_image_prefix):
                os.makedirs(save_adv_image_prefix)
            torch.save(model, per_epoch_save_model_path)

        if(test_acc > best_test_acc):
            best_test_acc = test_acc
            best_rob_orig_acc = org_test_acc
            if(is_log_wandb):
                wandb.log({"adv_tr_best_test_acc": best_test_acc,"adv_tr_org_test_acc":best_rob_orig_acc})
            torch.save(model, model_save_path)
            print("Saved model at", model_save_path)
        epoch += 1
        if(dataset is not None and dataset == "cifar10"):
            scheduler.step()

    train_acc = adv_evaluate_model(net, train_loader, classes, eps, adv_attack_type,save_adv_image_prefix=save_adv_image_prefix)
    if(is_log_wandb):
        wandb.log({"train_acc": train_acc, "test_acc": test_acc})

    print('Finished adversarial Training: Best saved model test acc is:', best_test_acc)
    return best_test_acc,best_rob_orig_acc, torch.load(model_save_path)

def get_model_from_path(dataset, model_arch_type, model_path, mask_percentage=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_model = torch.load(model_path, map_location=device)
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

if __name__ == '__main__':
    # fashion_mnist , mnist,cifar10
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net , conv4_deep_gated_net_n16_small ,
    # conv4_deep_gated_net_with_actual_inp_in_wt_net , conv4_deep_gated_net_with_actual_inp_randomly_changed_in_wt_net
    # conv4_deep_gated_net_with_random_ones_in_wt_net , masked_conv4_dlgn , masked_conv4_dlgn_n16_small , fc_dnn , fc_dlgn , fc_dgn,dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias__
    # bc_fc_dnn , fc_sf_dlgn , gal_fc_dnn , gal_plain_pure_conv4_dnn , madry_mnist_conv4_dnn , small_dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias__ ,
    # plain_pure_conv4_dnn_n16_pad_k_1_st1_bn_wo_bias__ , plain_pure_conv4_dnn_n16_small_pad_k_1_st1_bn_wo_bias__ , plain_pure_conv4_dnn_with_bn , plain_pure_conv4_dnn_pad_k_1_st1_with_bn__
    model_arch_type = 'fc_dnn'
    # batch_size = 128
    wand_project_name = None
    # wand_project_name = "fast_adv_tr_visualisation"
    # wand_project_name = "common_model_init_exps"
    # wand_project_name = "benchmarking_adv_exps"
    # wand_project_name = "model_band_frequency_experiments"
    # wand_project_name = "frequency_augmentation_experiments"
    # wand_project_name = "NPK_reg"
    # wand_project_name = "Pruning-exps"
    # wand_project_name = "Part_training_for_robustness"
    wand_project_name = "Residual_training"
    # wand_project_name = "madry's_benchmarking"
    # wand_project_name = "reach_end_plot"
    # wand_project_name = "SVM_Adv_training"
    
    # ADV_TRAINING ,  RECONST_EVAL_ADV_TRAINED_MODEL , VIS_ADV_TRAINED_MODEL , PART_ADV_TRAINING , GATE_FREEZE_ADV_TRAINING , VALUE_FREEZE_ADV_TRAINING
    exp_type = "ADV_TRAINING"

    npk_reg = 0
    # npk_reg = 0.01

    adv_attack_type = "PGD"
    adv_target = None
    is_targetted = adv_target is not None
    
    # Best adv-tr params are  update_on='all' rand_init=True norm=np.inf use_ytrue=True
    update_on='all'
    rand_init=True
    norm=np.inf
    use_ytrue=True
    # Applied only for "eta_growth" version
    eta_growth_reduced_rate = 1

    # eta_growth , max_eps , std , eq , reach_edge_at_end , add_rand_at__X__X ,max_dwnscld_eta_growth, None , cyclic_lr(this is not actually on inner maximization but on outer minimization)
    # L2_norm_grad_scale , L1_norm_grad_scale , feature_norm , feature_norm_sign_preserve
    residue_vname = "feature_norm_sign_preserve"
    # residue_vname = 'all_tanh_gate_flip'

    # If False, then segregation is over model prediction
    is_class_segregation_on_ground_truth = True
    template_initial_image_type = 'zero_init_image'
    # TANH_TEMP_LOSS , TEMP_LOSS
    template_loss_type = "TEMP_LOSS"
    is_split_validation = False
    valid_split_size = 0.1

    # torch_seed = ""
    torch_seed = 2022

    if(torch_seed == ""):
        torch_seed_str = ""
    else:
        torch_seed_str = "/ST_"+str(torch_seed)+"/"

    number_of_image_optimization_steps = 171
    collect_threshold = 0.73
    entropy_calculation_batch_size = 64
    number_of_batches_to_calculate_entropy_on = None

    is_log_wandb = not(wand_project_name is None)
    if(is_log_wandb):
        wandb.login(key=os.environ["WANDB_API_KEY"])

    # batch_size_list = [256, 128, 64]
    batch_size_list = [64]

    # None means that train on all classes
    list_of_classes_to_train_on = None
    # list_of_classes_to_train_on = [4, 9]

    # Percentage of information retention during PCA (values between 0-1)
    pca_exp_percent = None
    # pca_exp_percent = 0.85

    custom_dataset_path = None
    # custom_dataset_path = "data/custom_datasets/freq_band_dataset/mnist__LB_MB_HB.npy"

    for batch_size in batch_size_list:
        if(dataset == "cifar10"):
            inp_channel = 3
            classes = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            num_classes = len(classes)

            cifar10_config = DatasetConfig(
                'cifar10', is_normalize_data=False, valid_split_size=0.1, batch_size=batch_size, list_of_classes=list_of_classes_to_train_on,custom_dataset_path=custom_dataset_path)

            trainloader, _, testloader = preprocess_dataset_get_data_loader(
                cifar10_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

        elif(dataset == "mnist"):
            inp_channel = 1
            classes = [str(i) for i in range(0, 10)]
            num_classes = len(classes)

            mnist_config = DatasetConfig(
                'mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size, list_of_classes=list_of_classes_to_train_on,custom_dataset_path=custom_dataset_path)

            trainloader, _, testloader = preprocess_dataset_get_data_loader(
                mnist_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

        elif(dataset == "fashion_mnist"):
            inp_channel = 1
            classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot')
            num_classes = len(classes)
            
            fashion_mnist_config = DatasetConfig(
                'fashion_mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size,list_of_classes=list_of_classes_to_train_on,custom_dataset_path=custom_dataset_path)

            trainloader, _, testloader = preprocess_dataset_get_data_loader(
                fashion_mnist_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

        if(custom_dataset_path is not None):
            dataset = custom_dataset_path[custom_dataset_path.rfind("/")+1:custom_dataset_path.rfind(".npy")]

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
            fc_width = 128
            fc_depth = 4
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

        if(npk_reg!=0):
            model_arch_type_str = model_arch_type_str + "_NPKREG_"+str(npk_reg)
        start_net_path = None

        # start_net_path = "root/model/save/mnist/CLEAN_TRAINING/ST_2022/fc_sf_dlgn_W_128_D_4_dir/MARGIN_ANALYSIS/svm_gated_C_0.01_sf_dlgn.pt"
        if(start_net_path is not None):
            custom_temp_model = torch.load(start_net_path)
            net.load_state_dict(custom_temp_model.state_dict())
            stop_at_adv_test_acc = None
            # stop_at_adv_test_acc = 70.68

        net = net.to(device)

        # eps_list = [0.03, 0.06, 0.1]
        fast_adv_attack_type_list = ["PGD"]
        # fast_adv_attack_type_list = ['FGSM', 'PGD' ,'residual_PGD' , 'FEATURE_FLIP']
        if("mnist" in dataset):
            number_of_adversarial_optimization_steps_list = [40]
            eps_list = [0.3]
            eps_step_size = 0.01
            epochs = 36
        elif("cifar10" in dataset):
            number_of_adversarial_optimization_steps_list = [10]
            eps_list = [8/255]
            eps_step_size = 2/255
            epochs = 200
        # fast_adv_attack_type_list = ['FGSM', 'PGD']
        # number_of_adversarial_optimization_steps_list = [80]

        for fast_adv_attack_type in fast_adv_attack_type_list:
            for number_of_adversarial_optimization_steps in number_of_adversarial_optimization_steps_list:
                for eps in eps_list:
                    # eps_step_size = 1 * eps
                    print("iters:{} eps:{} step_size:{},epochs:{}".format(number_of_adversarial_optimization_steps,eps,eps_step_size,epochs))
                    root_save_prefix = "root/ADVER_RECONS_SAVE/"
                    init_prefix = "root/model/save/" + \
                        str(dataset)+"/adversarial_training/"+str(list_of_classes_to_train_on_str)+"/MT_" + \
                        str(model_arch_type_str)
                    if(start_net_path is not None):
                        init_prefix = start_net_path[0:start_net_path.rfind(
                            ".pt")]
                        root_save_prefix = init_prefix+"/ADVER_RECONS_SAVE/"
                    model_save_prefix = str(
                        init_prefix)+"_ET_ADV_TRAINING/"
                    tttmp=""
                    if(residue_vname is not None):
                        tttmp = "/residue_vname_"+str(residue_vname)
                    if(residue_vname == "eta_growth" and eta_growth_reduced_rate != 1):
                        tttmp += "/eta_growth_reduced_rate_"+str(eta_growth_reduced_rate)
                    prefix2 = str(torch_seed_str)+"fast_adv_attack_type_{}/adv_type_{}/EPS_{}/batch_size_{}/eps_stp_size_{}/adv_steps_{}/update_on_{}/R_init_{}/norm_{}/use_ytrue_{}/{}/".format(
                        fast_adv_attack_type, adv_attack_type, eps, batch_size, eps_step_size, number_of_adversarial_optimization_steps,update_on,rand_init,norm,use_ytrue,tttmp)
                    wandb_group_name = "DS_"+str(dataset_str) + "_EXP_"+str(exp_type) +\
                        "_fast_adv_training_TYP_"+str(model_arch_type_str)
                    model_save_prefix += prefix2
                    model_save_path = model_save_prefix + "adv_model_dir.pt"

                    isExist = os.path.exists(model_save_prefix)
                    if not os.path.exists(model_save_prefix):
                        os.makedirs(model_save_prefix)

                    if(exp_type == "ADV_TRAINING"):
                        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
                        if(is_log_wandb):
                            wandb_run_name = str(
                                model_arch_type_str)+prefix2.replace(
                                "/", "_")
                            wandb_config = get_wandb_config(exp_type,fast_adv_attack_type, adv_attack_type, model_arch_type_str, dataset_str,batch_size,epochs,
                                eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted,update_on,rand_init,norm,use_ytrue)
                            wandb_config["optimizer"]=opt
                            wandb_config["start_net_path"] = start_net_path
                            wandb_config["torch_seed"] = torch_seed
                            if(residue_vname is not None):
                                wandb_config["residue_vname"] = residue_vname
                            if(residue_vname == "eta_growth"):
                                wandb_config["eta_growth_reduced_rate"] = eta_growth_reduced_rate
                            if(npk_reg != 0):
                                wandb_config["npk_reg"]=npk_reg
                            if(pca_exp_percent is not None):
                                wandb_config["pca_exp_percent"] = pca_exp_percent
                                wandb_config["num_comp_pca"] = number_of_components_for_pca

                            wandb.init(
                                project=f"{wand_project_name}",
                                name=f"{wandb_run_name}",
                                group=f"{wandb_group_name}",
                                config=wandb_config,
                            )
                        # wandb.watch(net, log='all')
                        best_test_acc,best_rob_orig_acc, best_model = perform_adversarial_training(net, trainloader, testloader, eps_step_size, adv_target,
                                                                                 eps, fast_adv_attack_type, adv_attack_type, number_of_adversarial_optimization_steps, model_save_path, epochs, wand_project_name,dataset=dataset,npk_reg=npk_reg,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue,residue_vname=residue_vname,opt=opt,eta_growth_reduced_rate=eta_growth_reduced_rate)
                        if(is_log_wandb):
                            wandb.log({"adv_tr_best_test_acc": best_test_acc,"adv_tr_org_test_acc":best_rob_orig_acc})
                            wandb.finish()
                    
                    elif(exp_type == "GATE_FREEZE_ADV_TRAINING" or exp_type == "VALUE_FREEZE_ADV_TRAINING"):
                        if(exp_type == "GATE_FREEZE_ADV_TRAINING"):
                            ordict = net.get_gate_layers_ordered_dict()
                        elif(exp_type == "VALUE_FREEZE_ADV_TRAINING"):
                            ordict = net.get_value_layers_ordered_dict()

                        for key in ordict:
                            for param in  ordict[key].parameters():
                                param.requires_grad = False
                        
                        net = net.to(device)
                        print("net: ",net)
                        model_save_path = model_save_path.replace(
                            "ADV_TRAINING", exp_type)
                        print("model_save_path: ", model_save_path)
                        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
                        if(is_log_wandb):
                            wandb_run_name = str(
                                model_arch_type_str)+prefix2.replace(
                                "/", "_")
                            wandb_config = get_wandb_config(exp_type,fast_adv_attack_type, adv_attack_type, model_arch_type_str, dataset_str,batch_size,epochs,
                                eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted,update_on,rand_init,norm,use_ytrue)
                            wandb_config["optimizer"] = opt
                            wandb_config["start_net_path"] = start_net_path
                            wandb_config["torch_seed"] = torch_seed
                            if(residue_vname is not None):
                                wandb_config["residue_vname"] = residue_vname

                            wandb.init(
                                project=f"{wand_project_name}",
                                name=f"{wandb_run_name}",
                                group=f"{wandb_group_name}",
                                config=wandb_config,
                            )

                        best_test_acc,best_rob_orig_acc, best_model = perform_adversarial_training(net, trainloader, testloader, eps_step_size, adv_target,
                                                                                 eps, fast_adv_attack_type, adv_attack_type, number_of_adversarial_optimization_steps, model_save_path, epochs, wand_project_name,dataset=dataset,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue,residue_vname=residue_vname,opt=opt)
                        if(is_log_wandb):
                            wandb.log({"adv_tr_best_test_acc": best_test_acc,"adv_tr_org_test_acc":best_rob_orig_acc})
                            wandb.finish()

                    elif(exp_type == "PART_ADV_TRAINING"):
                        # GATE_NET_FREEZE , VAL_NET_FREEZE
                        transfer_mode = "VAL_NET_FREEZE"

                        teacher_model_path = "root/model/save/fashion_mnist/CLEAN_TRAINING/ST_2022/dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias___dir.pt"
                        net = get_model_from_path(
                            dataset, model_arch_type, teacher_model_path)
                        
                        if(transfer_mode == "GATE_NET_FREEZE"):
                            net.init_value_net()
                            ordict = net.get_gate_layers_ordered_dict()

                        elif(transfer_mode == "VAL_NET_FREEZE"):
                            net.init_gate_net()
                            ordict = net.get_value_layers_ordered_dict()
                        
                        for key in ordict:
                            for param in  ordict[key].parameters():
                                param.requires_grad = False
                            
                        net = net.to(device)
                        print("net",net)
                        model_save_path = model_save_path.replace(
                            "ADV_TRAINING", "PART_ADV_TRAINING/TEACHER__"+teacher_model_path.replace("/","-")+"/TYP_"+str(transfer_mode))
                        print("model_save_path: ", model_save_path)
                        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
                        if(is_log_wandb):
                            wandb_run_name = str(
                                model_arch_type_str)+prefix2.replace(
                                "/", "_")
                            wandb_config = get_wandb_config(exp_type,fast_adv_attack_type, adv_attack_type, model_arch_type_str, dataset_str,batch_size,epochs,
                                eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted,update_on,rand_init,norm,use_ytrue)
                            wandb_config["optimizer"]=opt
                            wandb_config["torch_seed"] = torch_seed
                            wandb_config["teacher_model_path"] = teacher_model_path
                            wandb_config["transfer_mode"] = transfer_mode
                            if(pca_exp_percent is not None):
                                wandb_config["pca_exp_percent"] = pca_exp_percent
                                wandb_config["num_comp_pca"] = number_of_components_for_pca

                            wandb.init(
                                project=f"{wand_project_name}",
                                name=f"{wandb_run_name}",
                                group=f"{wandb_group_name}",
                                config=wandb_config,
                            )

                        best_test_acc,best_rob_orig_acc, best_model = perform_adversarial_training(net, trainloader, testloader, eps_step_size, adv_target,
                                                                                 eps, fast_adv_attack_type, adv_attack_type, number_of_adversarial_optimization_steps, model_save_path, epochs, wand_project_name,dataset=dataset,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue,residue_vname=residue_vname,opt=opt)
                        if(is_log_wandb):
                            wandb.log({"adv_tr_best_test_acc": best_test_acc,"adv_tr_org_test_acc":best_rob_orig_acc})
                            wandb.finish()
                    elif(exp_type == "RECONST_EVAL_ADV_TRAINED_MODEL"):
                        final_postfix_for_save = prefix2
                        final_postfix_for_overall_save = prefix2 + "overall_template/"

                        print("Loading model from:", model_save_path)
                        best_model = torch.load(model_save_path)
                        print("Loaded model from:", model_save_path)

                        if(is_log_wandb):
                            wandb_run_name = str(
                                model_arch_type_str)+prefix2.replace(
                                "/", "_")
                            wandb_config = get_wandb_config("EVAL_VIA_RECONST",fast_adv_attack_type, adv_attack_type, model_arch_type_str, dataset_str,batch_size,epochs,
                                eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted,update_on,rand_init,norm,use_ytrue)
                            if(pca_exp_percent is not None):
                                wandb_config["pca_exp_percent"] = pca_exp_percent
                                wandb_config["num_comp_pca"] = number_of_components_for_pca

                        if(is_log_wandb):
                            wandb_run_name = str(
                                model_arch_type_str)+prefix2.replace(
                                "/", "_")

                            wandb.init(
                                project=f"{wand_project_name}",
                                name=f"{wandb_run_name}",
                                group=f"{wandb_group_name}",
                                config=wandb_config,
                            )
                            acc_over_orig_via_reconst = plain_evaluate_model_via_reconstructed(
                                model_arch_type_str, net, testloader, classes, dataset, template_initial_image_type, number_of_image_optimization_steps, template_loss_type, adv_target)
                            acc_over_adv_via_reconst = evaluate_model_via_reconstructed(model_arch_type_str, net, testloader, classes, eps, adv_attack_type, dataset, exp_type, template_initial_image_type, number_of_image_optimization_steps,
                                                                                        template_loss_type, number_of_adversarial_optimization_steps=number_of_adversarial_optimization_steps, eps_step_size=eps_step_size, adv_target=None, save_adv_image_prefix=model_save_prefix,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue)

                            wandb.log(
                                {"adv_tr_ad_test_acc_via_reconst": acc_over_adv_via_reconst, "adv_tr_orig_test_acc_via_reconst": acc_over_orig_via_reconst})
                            wandb.finish()
                    elif(exp_type == "VIS_ADV_TRAINED_MODEL"):
                        final_postfix_for_save = prefix2
                        final_postfix_for_overall_save = prefix2 + "overall_template/"

                        print("Loading model from:", model_save_path)
                        best_model = torch.load(model_save_path)
                        print("Loaded model from:", model_save_path)

                        if(is_log_wandb):
                            wandb_run_name = str(
                                model_arch_type_str)+prefix2.replace(
                                "/", "_")
                            wandb_config = get_wandb_config(exp_type,fast_adv_attack_type, adv_attack_type, model_arch_type_str, dataset_str,batch_size,epochs,
                                eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted,update_on,rand_init,norm,use_ytrue)
                            if(pca_exp_percent is not None):
                                wandb_config["pca_exp_percent"] = pca_exp_percent
                                wandb_config["num_comp_pca"] = number_of_components_for_pca

                        for is_template_image_on_train in [True]:
                            wandb_config["is_template_image_on_train"] = is_template_image_on_train
                            output_template_list = run_visualization_on_config(dataset, model_arch_type_str, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
                                                                               template_image_calculation_batch_size=1, template_loss_type=template_loss_type,
                                                                               number_of_batch_to_collect=1, wand_project_name=wand_project_name, is_split_validation=False,
                                                                               valid_split_size=None, torch_seed=torch_seed, number_of_image_optimization_steps=number_of_image_optimization_steps,
                                                                               wandb_group_name=wandb_group_name, exp_type="GENERATE_ALL_FINAL_TEMPLATE_IMAGES", collect_threshold=collect_threshold,
                                                                               entropy_calculation_batch_size=entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on=number_of_batches_to_calculate_entropy_on,
                                                                               root_save_prefix=root_save_prefix, final_postfix_for_save=final_postfix_for_save,
                                                                               custom_model=best_model, custom_data_loader=(trainloader, testloader), wandb_config_additional_dict=wandb_config)
                            # TO get one template image per class
                            run_visualization_on_config(dataset, model_arch_type_str, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
                                                        template_image_calculation_batch_size=32, template_loss_type=template_loss_type, number_of_batch_to_collect=None,
                                                        wand_project_name=wand_project_name, is_split_validation=is_split_validation, valid_split_size=valid_split_size,
                                                        torch_seed=torch_seed, number_of_image_optimization_steps=number_of_image_optimization_steps, wandb_group_name=wandb_group_name,
                                                        exp_type="GENERATE_TEMPLATE_IMAGES", collect_threshold=collect_threshold, entropy_calculation_batch_size=entropy_calculation_batch_size,
                                                        number_of_batches_to_calculate_entropy_on=number_of_batches_to_calculate_entropy_on, root_save_prefix=root_save_prefix,
                                                        final_postfix_for_save=final_postfix_for_overall_save, custom_model=best_model,
                                                        custom_data_loader=(trainloader, testloader), wandb_config_additional_dict=wandb_config)
                print("Finished fast_adv_attack_type:{} ,eps{}".format(
                    fast_adv_attack_type, eps))
print("Finished execution!!!")

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
from adversarial_attacks_tester import evaluate_model, evaluate_model_via_reconstructed, plain_evaluate_model_via_reconstructed
from visualization import run_visualization_on_config
from structure.dlgn_conv_config_structure import DatasetConfig

from conv4_models import get_model_instance, get_model_instance_from_dataset


def perform_adversarial_training(model, train_loader, test_loader, eps_step_size, adv_target, eps, fast_adv_attack_type, adv_attack_type, number_of_adversarial_optimization_steps, model_save_path, epochs=32, wand_project_name=None, lr_type='cyclic', lr_max=5e-3, alpha=0.375,dataset=None):
    print("Model will be saved at", model_save_path)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_str == 'cuda':
        if(torch.cuda.device_count() > 1):
            print("Parallelizing model")
            model = torch.nn.DataParallel(model)

        cudnn.benchmark = True
    is_log_wandb = not(wand_project_name is None)
    best_test_acc = 0
    opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    if lr_type == 'cyclic':
        def lr_schedule(t): return np.interp(
            [t], [0, epochs * 2//5, epochs], [0, lr_max, 0])[0]
    elif lr_type == 'flat':
        def lr_schedule(t): return lr_max
    else:
        raise ValueError('Unknown lr_type')

    criterion = nn.CrossEntropyLoss()
    epoch = 0
    if(dataset is not None and dataset == "cifar10"):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=200)
    while(epoch < epochs and (start_net_path is None or (stop_at_adv_test_acc is None or best_test_acc < stop_at_adv_test_acc))):
        correct = 0
        total = 0

        running_loss = 0.0
        loader = tqdm.tqdm(train_loader, desc='Training')
        for batch_idx, data in enumerate(loader, 0):
            begin_time = time.time()
            loader.set_description(f"Epoch {epoch+1}")
            (X, y) = data
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch + (batch_idx+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)

            if fast_adv_attack_type == 'FGSM':
                delta = torch.zeros_like(
                    X).uniform_(-eps, eps).cuda()
                delta.requires_grad = True
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(
                    delta + alpha * torch.sign(grad), -eps, eps)
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                delta = delta.detach()
            elif fast_adv_attack_type == 'PGD':
                delta = torch.zeros_like(
                    X).uniform_(-eps, eps)
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                for _ in range(number_of_adversarial_optimization_steps):
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    I = output.max(1)[1] == y
                    delta.data[I] = torch.clamp(
                        delta + alpha * torch.sign(grad), -eps, eps)[I]
                    delta.data[I] = torch.max(
                        torch.min(1-X, delta.data), 0-X)[I]
                delta = delta.detach()
            else:
                delta = torch.zeros_like(X)

            output = model(torch.clamp(X + delta, 0, 1))
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * y.size(0)
            correct += (output.max(1)[1] == y).sum().item()
            total += y.size(0)

            cur_time = time.time()
            step_time = cur_time - begin_time
            loader.set_postfix(train_loss=running_loss/(batch_idx+1),
                               train_acc=100.*correct/total, ratio="{}/{}".format(correct, total), stime=format_time(step_time))

        live_train_acc = 100. * correct/total
        test_acc = evaluate_model(net, test_loader, classes, eps, adv_attack_type,
                                  number_of_adversarial_optimization_steps, eps_step_size, adv_target)
        if(is_log_wandb):
            wandb.log({"live_train_acc": live_train_acc,
                      "current_epoch": epoch, "test_acc": test_acc})
        if(epoch % 2 == 0):
            per_epoch_save_model_path = model_save_path.replace(
                ".pt", '_epoch_{}.pt'.format(epoch))
            torch.save(model, per_epoch_save_model_path)

        if(test_acc > best_test_acc):
            best_test_acc = test_acc
            torch.save(model, model_save_path)
            print("Saved model at", model_save_path)
        epoch += 1
        if(dataset is not None and dataset == "cifar10"):
            scheduler.step()

    save_adv_image_prefix = model_save_path[0:model_save_path.rfind("/")+1]
    if not os.path.exists(save_adv_image_prefix):
        os.makedirs(save_adv_image_prefix)
    train_acc = evaluate_model(net, train_loader, classes, eps, adv_attack_type,
                               number_of_adversarial_optimization_steps, eps_step_size, adv_target, save_adv_image_prefix)
    if(is_log_wandb):
        wandb.log({"train_acc": train_acc, "test_acc": test_acc})

    print('Finished adversarial Training: Best saved model test acc is:', best_test_acc)
    return best_test_acc, torch.load(model_save_path)


if __name__ == '__main__':
    # fashion_mnist , mnist,cifar10
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net , conv4_deep_gated_net_n16_small ,
    # conv4_deep_gated_net_with_actual_inp_in_wt_net , conv4_deep_gated_net_with_actual_inp_randomly_changed_in_wt_net
    # conv4_deep_gated_net_with_random_ones_in_wt_net , masked_conv4_dlgn , masked_conv4_dlgn_n16_small , fc_dnn , fc_dlgn , fc_dgn,dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias__
    model_arch_type = 'dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias__'
    # batch_size = 128
    # wand_project_name = "fast_adv_tr_visualisation"
    # wand_project_name = "common_model_init_exps"
    wand_project_name = "model_band_frequency_experiments"
    # wand_project_name = None
    # ADV_TRAINING ,  RECONST_EVAL_ADV_TRAINED_MODEL , VIS_ADV_TRAINED_MODEL
    exp_type = "ADV_TRAINING"

    adv_attack_type = "PGD"
    adv_target = None

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
        wandb.login()

    # batch_size_list = [256, 128, 64]
    batch_size_list = [128]

    # None means that train on all classes
    list_of_classes_to_train_on = None
    # list_of_classes_to_train_on = [3, 8]

    # Percentage of information retention during PCA (values between 0-1)
    pca_exp_percent = None
    # pca_exp_percent = 0.85

    custom_dataset_path = None
    custom_dataset_path = "data/custom_datasets/freq_band_dataset/fashion_mnist__MB.npy"

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

        start_net_path = None

        # start_net_path = "root/model/save/mnist/CLEAN_TRAINING/ST_2022/conv4_deep_gated_net_n16_small_PCA_K12_P_0.5_dir.pt"
        if(start_net_path is not None):
            custom_temp_model = torch.load(start_net_path)
            net.load_state_dict(custom_temp_model.state_dict())
            stop_at_adv_test_acc = None
            stop_at_adv_test_acc = 70.68

        net = net.to(device)

        # eps_list = [0.03, 0.06, 0.1]
        fast_adv_attack_type_list = ['PGD']
        # fast_adv_attack_type_list = ['FGSM', 'PGD']
        if("mnist" in dataset):
            number_of_adversarial_optimization_steps_list = [80]
            eps_list = [0.06]
            eps_step_size = 0.06
            epochs = 100
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
                    prefix2 = str(torch_seed_str)+"fast_adv_attack_type_{}/adv_type_{}/EPS_{}/batch_size_{}/eps_stp_size_{}/adv_steps_{}/".format(
                        fast_adv_attack_type, adv_attack_type, eps, batch_size, eps_step_size, number_of_adversarial_optimization_steps)
                    wandb_group_name = "DS_"+str(dataset_str) + "_EXP_"+str(exp_type) +\
                        "_fast_adv_training_TYP_"+str(model_arch_type_str)
                    model_save_prefix += prefix2
                    model_save_path = model_save_prefix + "adv_model_dir.pt"

                    isExist = os.path.exists(model_save_prefix)
                    if not os.path.exists(model_save_prefix):
                        os.makedirs(model_save_prefix)

                    if(exp_type == "ADV_TRAINING"):
                        if(is_log_wandb):
                            wandb_run_name = str(
                                model_arch_type_str)+prefix2.replace(
                                "/", "_")
                            wandb_config = dict()
                            wandb_config["exp_type"] = exp_type
                            wandb_config["adv_attack_type"] = adv_attack_type
                            wandb_config["model_arch_type"] = model_arch_type_str
                            wandb_config["dataset"] = dataset_str
                            wandb_config["eps"] = eps
                            wandb_config["number_of_adversarial_optimization_steps"] = number_of_adversarial_optimization_steps
                            wandb_config["epochs"] = epochs
                            wandb_config["batch_size"] = batch_size
                            wandb_config["torch_seed"] = torch_seed
                            wandb_config["fast_adv_attack_type"] = fast_adv_attack_type
                            wandb_config["eps_step_size"] = eps_step_size
                            wandb_config["model_save_path"] = model_save_path
                            wandb_config["start_net_path"] = start_net_path
                            if(pca_exp_percent is not None):
                                wandb_config["pca_exp_percent"] = pca_exp_percent
                                wandb_config["num_comp_pca"] = number_of_components_for_pca

                            wandb.init(
                                project=f"{wand_project_name}",
                                name=f"{wandb_run_name}",
                                group=f"{wandb_group_name}",
                                config=wandb_config,
                            )

                        best_test_acc, best_model = perform_adversarial_training(net, trainloader, testloader, eps_step_size, adv_target,
                                                                                 eps, fast_adv_attack_type, adv_attack_type, number_of_adversarial_optimization_steps, model_save_path, epochs, wand_project_name, alpha=eps_step_size,dataset=dataset)
                        if(is_log_wandb):
                            wandb.log({"adv_tr_best_test_acc": best_test_acc})
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
                            wandb_config = dict()
                            wandb_config["exp_type"] = "EVAL_VIA_RECONST"
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
                                                                                        template_loss_type, number_of_adversarial_optimization_steps=number_of_adversarial_optimization_steps, eps_step_size=eps_step_size, adv_target=None, save_adv_image_prefix=model_save_prefix)

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
                            wandb_config = dict()
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

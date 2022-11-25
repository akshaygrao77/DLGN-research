import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import time
import os
import wandb

from external_utils import format_time
from utils.data_preprocessing import preprocess_dataset_get_data_loader
from adversarial_attacks_tester import evaluate_model, evaluate_model_via_reconstructed
from visualization import run_visualization_on_config
from structure.dlgn_conv_config_structure import DatasetConfig
from configs.generic_configs import get_preprocessing_and_other_configs

from conv4_models import get_model_instance


def perform_adversarial_training(model, train_loader, test_loader, eps_step_size, adv_target, eps, fast_adv_attack_type, adv_attack_type, number_of_adversarial_optimization_steps, model_save_path, epochs=32, wand_project_name=None, lr_type='cyclic', lr_max=5e-3, alpha=0.375):
    print("Model will be saved at", model_save_path)
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

    for epoch in range(epochs):
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
        if(epoch % 5 == 0):
            per_epoch_save_model_path = model_save_path.replace(
                ".pt", '_epoch_{}.pt'.format(epoch))
            torch.save(model, per_epoch_save_model_path)

        if(test_acc > best_test_acc):
            best_test_acc = test_acc
            torch.save(model, model_save_path)
            print("Saved model at", model_save_path)

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
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net , conv4_deep_gated_net_n16_small ,
    # conv4_deep_gated_net_with_actual_inp_in_wt_net , conv4_deep_gated_net_with_actual_inp_randomly_changed_in_wt_net
    # conv4_deep_gated_net_with_random_ones_in_wt_net
    model_arch_type = 'conv4_deep_gated_net_n16_small'
    # scheme_type = ''
    # batch_size = 128
    # wand_project_name = "fast_adv_training_and_visualisation"
    wand_project_name = "common_model_init_exps"
    # wand_project_name = None
    # ADV_TRAINING ,  RECONST_VIS_ADV_TRAINED_MODEL
    exp_type = "ADV_TRAINING"

    epochs = 32
    adv_attack_type = "PGD"
    adv_target = None

    # If False, then segregation is over model prediction
    is_class_segregation_on_ground_truth = True
    template_initial_image_type = 'zero_init_image'
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
    collect_threshold = 0.95
    entropy_calculation_batch_size = 64
    number_of_batches_to_calculate_entropy_on = None

    is_log_wandb = not(wand_project_name is None)
    if(is_log_wandb):
        wandb.login()

    # batch_size_list = [256, 128, 64]
    batch_size_list = [128]

    for batch_size in batch_size_list:
        if(dataset == "cifar10"):
            inp_channel = 3
            print("Training over CIFAR 10")
            classes = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            num_classes = len(classes)

            cifar10_config = DatasetConfig(
                'cifar10', is_normalize_data=False, valid_split_size=0.1, batch_size=batch_size)

            trainloader, _, testloader = preprocess_dataset_get_data_loader(
                cifar10_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

        elif(dataset == "mnist"):
            inp_channel = 1
            print("Training over MNIST")
            classes = [str(i) for i in range(0, 10)]
            num_classes = len(classes)

            mnist_config = DatasetConfig(
                'mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size)

            trainloader, _, testloader = preprocess_dataset_get_data_loader(
                mnist_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = get_model_instance(model_arch_type, inp_channel, seed=torch_seed)
        start_net_path = None

        # start_net_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/aug_conv4_dlgn_iter_3_dir.pt"
        # net = torch.load(start_net_path)

        net.to(device)

        # eps_list = [0.03, 0.06, 0.1]
        fast_adv_attack_type_list = ['PGD']
        # fast_adv_attack_type_list = ['FGSM', 'PGD']
        number_of_adversarial_optimization_steps_list = [80]

        eps_list = [0.06]
        # fast_adv_attack_type_list = ['FGSM', 'PGD']
        # number_of_adversarial_optimization_steps_list = [80]

        for fast_adv_attack_type in fast_adv_attack_type_list:
            for number_of_adversarial_optimization_steps in number_of_adversarial_optimization_steps_list:
                for eps in eps_list:
                    eps_step_size = 1 * eps

                    root_save_prefix = "root/ADVER_RECONS_SAVE/"
                    init_prefix = "root/model/save/" + \
                        str(dataset)+"/adversarial_training/MT_" + \
                        str(model_arch_type)
                    if(start_net_path is not None):
                        init_prefix = start_net_path[0:start_net_path.rfind(
                            "/")+1]
                        root_save_prefix = init_prefix+"/ADVER_RECONS_SAVE/"
                    model_save_prefix = str(
                        init_prefix)+"_ET_ADV_TRAINING/"
                    prefix2 = str(torch_seed_str)+"fast_adv_attack_type_{}/adv_type_{}/EPS_{}/batch_size_{}/eps_stp_size_{}/adv_steps_{}/".format(
                        fast_adv_attack_type, adv_attack_type, eps, batch_size, eps_step_size, number_of_adversarial_optimization_steps)
                    wandb_group_name = "DS_"+str(dataset) + "_EXP_"+str(exp_type) +\
                        "_fast_adv_training_TYP_"+str(model_arch_type)
                    model_save_prefix += prefix2
                    model_save_path = model_save_prefix + "adv_model_dir.pt"

                    isExist = os.path.exists(model_save_prefix)
                    if not os.path.exists(model_save_prefix):
                        os.makedirs(model_save_prefix)

                    if(exp_type == "ADV_TRAINING"):
                        if(is_log_wandb):
                            wandb_run_name = str(
                                model_arch_type)+prefix2.replace(
                                "/", "_")
                            wandb_config = dict()
                            wandb_config["exp_type"] = exp_type
                            wandb_config["adv_attack_type"] = adv_attack_type
                            wandb_config["model_arch_type"] = model_arch_type
                            wandb_config["dataset"] = dataset
                            wandb_config["eps"] = eps
                            wandb_config["number_of_adversarial_optimization_steps"] = number_of_adversarial_optimization_steps
                            wandb_config["epochs"] = epochs
                            wandb_config["batch_size"] = batch_size
                            wandb_config["torch_seed"] = torch_seed
                            wandb_config["fast_adv_attack_type"] = fast_adv_attack_type
                            wandb_config["eps_step_size"] = eps_step_size
                            wandb_config["model_save_path"] = model_save_path
                            wandb_config["start_net_path"] = start_net_path

                            wandb.init(
                                project=f"{wand_project_name}",
                                name=f"{wandb_run_name}",
                                group=f"{wandb_group_name}",
                                config=wandb_config,
                            )

                        best_test_acc, best_model = perform_adversarial_training(net, trainloader, testloader, eps_step_size, adv_target,
                                                                                 eps, fast_adv_attack_type, adv_attack_type, number_of_adversarial_optimization_steps, model_save_path, epochs, wand_project_name)
                        if(is_log_wandb):
                            wandb.log({"adv_tr_best_test_acc": best_test_acc})
                            wandb.finish()

                    elif(exp_type == "RECONST_VIS_ADV_TRAINED_MODEL"):
                        final_postfix_for_save = prefix2
                        final_postfix_for_overall_save = prefix2 + "overall_template/"

                        print("Loading model from:", model_save_path)
                        best_model = torch.load(model_save_path)
                        print("Loaded model from:", model_save_path)

                        if(is_log_wandb):
                            wandb_run_name = str(
                                model_arch_type)+prefix2.replace(
                                "/", "_")
                            wandb_config = dict()
                            wandb_config["exp_type"] = "EVAL_VIA_RECONST"
                            wandb_config["adv_attack_type"] = adv_attack_type
                            wandb_config["model_arch_type"] = model_arch_type
                            wandb_config["dataset"] = dataset
                            wandb_config["eps"] = eps
                            wandb_config["number_of_adversarial_optimization_steps"] = number_of_adversarial_optimization_steps
                            wandb_config["epochs"] = epochs
                            wandb_config["batch_size"] = batch_size
                            wandb_config["fast_adv_attack_type"] = fast_adv_attack_type
                            wandb_config["eps_step_size"] = eps_step_size
                            wandb_config["model_save_path"] = model_save_path

                        if(is_log_wandb):
                            wandb_run_name = str(
                                model_arch_type)+prefix2.replace(
                                "/", "_")

                            wandb.init(
                                project=f"{wand_project_name}",
                                name=f"{wandb_run_name}",
                                group=f"{wandb_group_name}",
                                config=wandb_config,
                            )
                            acc_with_orig_via_reconst = evaluate_model_via_reconstructed(net, testloader, classes, eps, adv_attack_type, dataset, exp_type, template_initial_image_type, number_of_image_optimization_steps,
                                                                                         template_loss_type, number_of_adversarial_optimization_steps=number_of_adversarial_optimization_steps, eps_step_size=eps_step_size, adv_target=None, save_adv_image_prefix=model_save_prefix)

                            wandb.log(
                                {"adv_tr_test_acc_via_reconst": acc_with_orig_via_reconst})
                            wandb.finish()

                        for is_template_image_on_train in [True]:
                            wandb_config["is_template_image_on_train"] = is_template_image_on_train
                            output_template_list = run_visualization_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
                                                                               template_image_calculation_batch_size=1, template_loss_type=template_loss_type,
                                                                               number_of_batch_to_collect=1, wand_project_name=wand_project_name, is_split_validation=False,
                                                                               valid_split_size=None, torch_seed=torch_seed, number_of_image_optimization_steps=number_of_image_optimization_steps,
                                                                               wandb_group_name=wandb_group_name, exp_type="GENERATE_ALL_FINAL_TEMPLATE_IMAGES", collect_threshold=collect_threshold,
                                                                               entropy_calculation_batch_size=entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on=number_of_batches_to_calculate_entropy_on,
                                                                               root_save_prefix=root_save_prefix, final_postfix_for_save=final_postfix_for_save,
                                                                               custom_model=best_model, custom_data_loader=(trainloader, testloader), wandb_config_additional_dict=wandb_config)
                            # TO get one template image per class
                            run_visualization_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
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

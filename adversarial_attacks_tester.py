from cProfile import label
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
from data_preprocessing import preprocess_dataset_get_data_loader
from structure.dlgn_conv_config_structure import DatasetConfig

from visualization import recreate_image, save_image

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from conv4_models import Plain_CONV4_Net, Conv4_DLGN_Net


def apply_adversarial_attack_on_input(input_data, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, is_targetted):
    adv_inp = None
    if(adv_attack_type == "PGD"):
        adv_inp = projected_gradient_descent(
            net, input_data, eps, eps_step_size, number_of_adversarial_optimization_steps, np.inf, y=adv_target, targeted=is_targetted)

    return adv_inp


def get_wandb_config(exp_type, adv_attack_type, model_arch_type, dataset, is_adv_attack_on_train,
                     eps, number_of_adversarial_optimization_steps, eps_step_size, model_attacked_path, is_targetted, adv_target=None):

    wandb_config = dict()
    wandb_config["adv_attack_type"] = adv_attack_type
    wandb_config["model_arch_type"] = model_arch_type
    wandb_config["dataset"] = dataset
    wandb_config["is_adv_attack_on_train"] = is_adv_attack_on_train
    wandb_config["eps"] = eps
    wandb_config["number_of_adversarial_optimization_steps"] = number_of_adversarial_optimization_steps
    wandb_config["eps_step_size"] = eps_step_size
    wandb_config["is_targetted"] = is_targetted
    wandb_config["model_attacked_path"] = model_attacked_path
    wandb_config["exp_type"] = exp_type
    if(not(adv_target is None)):
        wandb_config["adv_target"] = adv_target

    return wandb_config


def evaluate_model(net, dataloader, classes, eps, adv_attack_type, number_of_adversarial_optimization_steps=40, eps_step_size=0.01, adv_target=None, save_adv_image_prefix="root/adv_images"):
    correct = 0
    total = 0
    acc = 0.
    is_targetted = adv_target is not None

    net.train(False)

    loader = tqdm.tqdm(dataloader, desc='Evaluating')
    # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    for batch_idx, data in enumerate(loader, 0):
        begin_time = time.time()
        images, labels = data
        images, labels = images.to(
            device), labels.to(device)

        adv_images = apply_adversarial_attack_on_input(
            images, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, is_targetted)
        # calculate outputs by running images through the network
        outputs = net(adv_images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if(batch_idx % 10 == 0):
            orig_image = recreate_image(
                images[0], unnormalize=False)
            # print("orig_image.shape::",
            #       orig_image.shape)

            adv_image = recreate_image(
                adv_images[0], unnormalize=False)
            # print("adv_image.shape::",
            #       adv_image.shape)

            orig_save_folder = save_adv_image_prefix + "/orig/"
            adv_save_folder = save_adv_image_prefix + "/adver/"
            if not os.path.exists(orig_save_folder):
                os.makedirs(orig_save_folder)
            orig_im_path = orig_save_folder+'/original_c' + \
                str(classes[labels[0]])+'_batch_ind_' + \
                str(batch_idx) + '.jpg'

            if not os.path.exists(adv_save_folder):
                os.makedirs(adv_save_folder)
            adv_im_path = adv_save_folder+'/adv_c' + \
                str(classes[labels[0]])+'_batch_ind_' + \
                str(batch_idx) + '.jpg'

            save_image(orig_image, orig_im_path)
            save_image(adv_image, adv_im_path)

        cur_time = time.time()
        step_time = cur_time - begin_time
        acc = 100.*correct/total
        loader.set_postfix(
            acc=acc, ratio="{}/{}".format(correct, total), stime=format_time(step_time))

    return acc


if __name__ == '__main__':
    dataset = 'mnist'
    model_arch_type = 'conv4_dlgn'
    scheme_type = 'iterative_augmented_model_attack'
    # scheme_type = ''
    batch_size = 64

    if(dataset == "cifar10"):
        inp_channel = 3
        print("Evaluating over CIFAR 10")
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = len(classes)

        cifar10_config = DatasetConfig(
            'cifar10', is_normalize_data=False, valid_split_size=0.1, batch_size=batch_size)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            cifar10_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    elif(dataset == "mnist"):
        inp_channel = 1
        print("Evaluating over MNIST")
        classes = [str(i) for i in range(0, 10)]
        num_classes = len(classes)

        mnist_config = DatasetConfig(
            'mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            mnist_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if(scheme_type == 'iterative_augmented_model_attack'):
        dataset = 'mnist'
        # wand_project_name = "cifar10_all_images_based_template_visualizations"
        # wand_project_name = "template_images_visualization-test"
        wand_project_name = 'adv_attack_on_reconst_augmentation'
        # wand_project_name = None
        wandb_group_name = "DS_"+str(dataset) + \
            "_adv_attack_over_aug_"+str(model_arch_type)

        number_of_adversarial_optimization_steps = 161
        adv_attack_type = "PGD"
        adv_target = None
        exp_type = "ADV_ATTACK"
        is_adv_attack_on_train = False
        eps_step_size = 0.01

        model_and_data_save_prefix = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_test/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/"

        number_of_augment_iterations = 5

        is_targetted = adv_target is not None
        is_log_wandb = not(wand_project_name is None)

        if(is_log_wandb):
            wandb.login()
        for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
            for current_aug_iter_num in range(1, number_of_augment_iterations+1):

                model_save_path = model_and_data_save_prefix+'aug_conv4_dlgn_iter_{}_dir.pt'.format(
                    current_aug_iter_num)
                isExist = os.path.exists(model_save_path)
                if not os.path.exists(model_and_data_save_prefix):
                    os.makedirs(model_and_data_save_prefix)

                assert isExist == True, 'Model path does not have saved model'

                net = torch.load(model_save_path)
                net.to(device)
                device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
                if device_str == 'cuda':
                    cudnn.benchmark = True

                print("Loaded previously trained model for augment iteration:",
                      current_aug_iter_num)
                if(is_adv_attack_on_train == True):
                    eval_loader = trainloader
                else:
                    eval_loader = testloader

                if(is_log_wandb):
                    wandb_run_name = str(
                        model_arch_type)+"_aug_it_"+str(current_aug_iter_num)+"adv_at_"+str(adv_attack_type)
                    wandb_config = get_wandb_config(exp_type, adv_attack_type, model_arch_type, dataset, is_adv_attack_on_train,
                                                    eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted, adv_target)
                    wandb.init(
                        project=f"{wand_project_name}",
                        name=f"{wandb_run_name}",
                        group=f"{wandb_group_name}",
                        config=wandb_config,
                    )

                print("Net:", net)
                final_postfix_for_save = "ADV_SAVES/adv_type_{}/EPS_{}/eps_stp_size_{}/adv_steps_{}/aug_indx_{}".format(
                    adv_attack_type, eps, eps_step_size, number_of_adversarial_optimization_steps, current_aug_iter_num)
                save_folder = model_and_data_save_prefix + final_postfix_for_save
                acc = evaluate_model(net, eval_loader, classes, eps, adv_attack_type,
                                     number_of_adversarial_optimization_steps, eps_step_size, adv_target, save_folder)

                if(is_log_wandb):
                    wandb.log({"eval_acc": acc})
                    wandb.finish()

    print("Finished execution!!!")

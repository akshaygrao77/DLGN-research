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
from utils.data_preprocessing import preprocess_dataset_get_data_loader, generate_merged_dataset_from_two_loader, generate_dataset_from_loader,preprocess_mnist_fmnist,get_data_loader
from structure.dlgn_conv_config_structure import DatasetConfig
from collections import OrderedDict

from visualization import recreate_image, save_image,  PerClassDataset, TemplateImageGenerator, seed_worker, quick_visualization_on_config
from utils.data_preprocessing import true_segregation
from structure.generic_structure import CustomSimpleDataset

from utils.visualise_utils import calculate_common_among_two_activation_patterns, save_images_from_dataloader,save_bifurcated_images_from_dataloader

from attacks import cleverhans_projected_gradient_descent,cleverhans_fast_gradient_method,get_locuslab_adv_per_batch,get_gateflip_adv_per_batch
from keras.datasets import mnist, fashion_mnist
from freq_dataset_generator import modify_bandpass_freq_get_dataset
from utils.generic_utils import Y_Logits_Binary_class_Loss

from conv4_models import Plain_CONV4_Net, Conv4_DLGN_Net, get_model_instance, get_model_instance_from_dataset


def apply_adversarial_attack_on_input(input_data, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, labels, is_targetted,update_on,rand_init,norm,use_ytrue,criterion=None,clip_min=0.0,clip_max=1.0,residue_vname=None):
    adv_inp = None
    if(adv_attack_type == "PGD"):
        kargs = {"criterion":criterion,"eps":eps,"eps_step_size":eps_step_size,"steps":number_of_adversarial_optimization_steps,"labels":labels if use_ytrue or is_targetted else None,"update_on":update_on,'rand_init':rand_init,'clip_min':clip_min,'clip_max':clip_max,'targeted':is_targetted,'norm':norm,"residue_vname":residue_vname}
        # adv_inp = get_locuslab_adv_per_batch(net, input_data,kargs)
        adv_inp = cleverhans_projected_gradient_descent(
            net, input_data,kargs)
    elif(adv_attack_type == "FGSM"):
        kargs = {"criterion":criterion,"eps":eps,"eps_step_size":eps,"steps":1,"labels":labels if use_ytrue or is_targetted else None,"update_on":update_on,'rand_init':rand_init,'clip_min':clip_min,'clip_max':clip_max,'targeted':is_targetted,'norm':norm,"residue_vname":residue_vname}
        adv_inp = cleverhans_fast_gradient_method(net,input_data, kargs)
    elif(adv_attack_type == "FEATURE_FLIP"):
        kargs = {"criterion":criterion,"eps":eps,"eps_step_size":eps_step_size,"steps":number_of_adversarial_optimization_steps,"labels":labels if use_ytrue or is_targetted else None,"update_on":update_on,'rand_init':rand_init,'clip_min':clip_min,'clip_max':clip_max,'targeted':is_targetted,'norm':norm,"residue_vname":residue_vname}
        adv_inp = get_gateflip_adv_per_batch(net,input_data, kargs)

    return adv_inp


def get_wandb_config(exp_type, adv_attack_type, model_arch_type, dataset, is_adv_attack_on_train,
                     eps, number_of_adversarial_optimization_steps, eps_step_size, model_attacked_path, is_targetted, adv_target=None,update_on='all',rand_init=True,norm=np.inf,use_ytrue=True):

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
    wandb_config["update_on"] = update_on
    wandb_config["rand_init"] = rand_init
    wandb_config["norm"] = norm
    wandb_config["use_ytrue"] = use_ytrue
    if(not(adv_target is None)):
        wandb_config["adv_target"] = adv_target

    return wandb_config


def plain_evaluate_model(net, dataloader, classes=None,is_get_classwise_acc=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    acc = 0.

    net.train(False)

    frequency_pred = None
    if(classes is not None):
        num_classes = len(classes)
        frequency_pred = torch.zeros(num_classes)
        frequency_pred = frequency_pred.to(device, non_blocking=True)
        all_classes = torch.arange(0, num_classes)
        all_classes = all_classes.to(device, non_blocking=True)
        if is_get_classwise_acc:
            classwise_acc = torch.zeros(num_classes)
            classwise_acc = classwise_acc.to(device, non_blocking=True)
            classwise_count = torch.zeros(num_classes)
            classwise_count = classwise_count.to(device, non_blocking=True)

    loader = tqdm.tqdm(dataloader, desc='Evaluating')
    # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    for batch_idx, data in enumerate(loader, 0):
        begin_time = time.time()
        images, labels = data
        images, labels = images.to(
            device), labels.to(device)

        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        if(len(outputs.size())==1):
            predicted = torch.sigmoid(outputs.data).round()
        else:
            _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        with torch.no_grad():
            if(classes is not None):
                temp = torch.cat((predicted.float(), all_classes))
                temp = temp.to(device)
                frequency_pred += torch.histc(temp, num_classes) - 1
                if is_get_classwise_acc:
                    corr_mask = torch.where(predicted == labels,1,0)
                    for cur_class in range(num_classes):
                        class_mask = torch.where(labels==cur_class,1,0)
                        classwise_count[cur_class] += class_mask.sum().item()
                        classwise_acc[cur_class] += (corr_mask * class_mask).sum().item()

        cur_time = time.time()
        step_time = cur_time - begin_time
        acc = 100.*correct/total
        loader.set_postfix(
            acc=acc, ratio="{}/{}".format(correct, total), stime=format_time(step_time))
    if frequency_pred is not None:
        frequency_pred = (frequency_pred/total)*100
        frequency_pred = torch.round(frequency_pred)
        if is_get_classwise_acc:
            for cur_class in range(num_classes):
                classwise_acc[cur_class] = (classwise_acc[cur_class] / classwise_count[cur_class])*100
            classwise_acc = torch.round(classwise_acc)
            return acc,frequency_pred,classwise_acc
    return acc, frequency_pred


def adv_evaluate_model(net, dataloader,classes, eps, adv_attack_type, number_of_adversarial_optimization_steps=40, eps_step_size=0.01,adv_target=None, save_adv_image_prefix=None, update_on='all',rand_init=True,norm=np.inf,use_ytrue=True,lossfn=None,residue_vname=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    acc = 0.
    is_targetted = adv_target is not None

    net.train(False)

    loader = tqdm.tqdm(dataloader, desc='Adversarial Evaluating')
    # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    for batch_idx, data in enumerate(loader, 0):
        begin_time = time.time()
        images, labels = data
        images, labels = images.to(
            device), labels.to(device)

        adv_images = apply_adversarial_attack_on_input(
            images, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, labels, is_targetted,update_on,rand_init,norm,use_ytrue,lossfn,residue_vname=residue_vname)
        # calculate outputs by running images through the network
        outputs = net(adv_images)
        # the class with the highest energy is what we choose as prediction
        if(len(outputs.size())==1):
            predicted = torch.sigmoid(outputs.data).round()
        else:
            _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if(batch_idx % 1 == 0 and save_adv_image_prefix is not None):
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
                str(classes[int(labels[0])])+'_batch_ind_' + \
                str(batch_idx) + '.jpg'

            if not os.path.exists(adv_save_folder):
                os.makedirs(adv_save_folder)
            adv_im_path = adv_save_folder+'/adv_c' + \
                str(classes[int(labels[0])])+'_batch_ind_' + \
                str(batch_idx) + '.jpg'

            save_image(orig_image, orig_im_path)
            save_image(adv_image, adv_im_path)

        cur_time = time.time()
        step_time = cur_time - begin_time
        acc = 100.*correct/total
        loader.set_postfix(
            acc=acc, ratio="{}/{}".format(correct, total), stime=format_time(step_time))

    return acc


def plain_evaluate_model_via_reconstructed(model_arch_type, net, dataloader, classes, dataset, template_initial_image_type, number_of_image_optimization_steps, template_loss_type, adv_target, save_image_prefix=None, postfix_folder_for_save="/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    acc = 0.
    is_targetted = adv_target is not None

    net.train(False)

    loader = tqdm.tqdm(dataloader, desc='Evaluating via reconstruction')
    # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    for batch_idx, data in enumerate(loader, 0):
        begin_time = time.time()
        images, labels = data
        images, labels = images.to(
            device), labels.to(device)

        reconst_adv_images = quick_visualization_on_config(
            model_arch_type, net, dataset, exp_type="GENERATE_TEMPLATE_GIVEN_BATCH_OF_IMAGES", template_initial_image_type=template_initial_image_type,
            images_to_collect_upon=images, number_of_image_optimization_steps=number_of_image_optimization_steps, template_loss_type=template_loss_type)

        # calculate outputs by running images through the network
        outputs = net(reconst_adv_images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if(batch_idx % 20 == 0 and save_image_prefix is not None):

            reconst_adv_image = recreate_image(
                reconst_adv_images[0], unnormalize=False)

            recon_adv_save_folder = save_image_prefix + "/" + postfix_folder_for_save

            if not os.path.exists(recon_adv_save_folder):
                os.makedirs(recon_adv_save_folder)
            recon_adv_im_path = recon_adv_save_folder+'/_c' + \
                str(classes[labels[0]])+'_batch_ind_' + \
                str(batch_idx) + '.jpg'

            save_image(reconst_adv_image, recon_adv_im_path)

        cur_time = time.time()
        step_time = cur_time - begin_time
        acc = 100.*correct/total
        loader.set_postfix(
            acc=acc, ratio="{}/{}".format(correct, total), stime=format_time(step_time))

    return acc


def evaluate_model_via_reconstructed(model_arch_type, net, dataloader, classes, eps, adv_attack_type, dataset, exp_type, template_initial_image_type, number_of_image_optimization_steps, template_loss_type, number_of_adversarial_optimization_steps=40, eps_step_size=0.01, adv_target=None, save_adv_image_prefix=None,update_on='all',rand_init=True,norm=np.inf,use_ytrue=True,lossfn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    acc = 0.
    is_targetted = adv_target is not None

    net.train(False)

    loader = tqdm.tqdm(dataloader, desc='Evaluating via reconstruction')
    # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    for batch_idx, data in enumerate(loader, 0):
        begin_time = time.time()
        images, labels = data
        images, labels = images.to(
            device), labels.to(device)

        adv_images = apply_adversarial_attack_on_input(
            images, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, labels, is_targetted,update_on,rand_init,norm,use_ytrue,lossfn)

        reconst_adv_images = quick_visualization_on_config(
            model_arch_type, net, dataset, exp_type="GENERATE_TEMPLATE_GIVEN_BATCH_OF_IMAGES", template_initial_image_type=template_initial_image_type,
            images_to_collect_upon=adv_images, number_of_image_optimization_steps=number_of_image_optimization_steps, template_loss_type=template_loss_type)

        # calculate outputs by running images through the network
        outputs = net(reconst_adv_images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if(batch_idx % 20 == 0 and save_adv_image_prefix is not None):
            orig_image = recreate_image(
                images[0], unnormalize=False)

            adv_image = recreate_image(
                adv_images[0], unnormalize=False)

            reconst_adv_image = recreate_image(
                reconst_adv_images[0], unnormalize=False)

            orig_save_folder = save_adv_image_prefix + "/orig/"
            adv_save_folder = save_adv_image_prefix + "/adver/"
            recon_adv_save_folder = save_adv_image_prefix + "/recons_adver/"
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

            if not os.path.exists(recon_adv_save_folder):
                os.makedirs(recon_adv_save_folder)
            recon_adv_im_path = recon_adv_save_folder+'/adv_c' + \
                str(classes[labels[0]])+'_batch_ind_' + \
                str(batch_idx) + '.jpg'

            save_image(orig_image, orig_im_path)
            save_image(adv_image, adv_im_path)
            save_image(reconst_adv_image, recon_adv_im_path)

        cur_time = time.time()
        step_time = cur_time - begin_time
        acc = 100.*correct/total
        loader.set_postfix(
            acc=acc, ratio="{}/{}".format(correct, total), stime=format_time(step_time))

    return acc

def get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_adv_attack_on_train,update_on='all',rand_init=True,norm=np.inf,use_ytrue=False,residue_vname=None,criterion=None):
    if(residue_vname is None):
        residue_vname = ""
    if(criterion is None):
        crit = ""
    else:
        crit = "INNER_CRIT_"+str(criterion)
    no_default_str = ""
    if(not(update_on=='all' and rand_init==True and norm==np.inf and use_ytrue==False)):
        no_default_str = "/update_on_{}/R_init_{}/norm_{}/use_ytrue_{}/".format(update_on,rand_init,str(norm),use_ytrue)
    final_adv_postfix_for_save = "/RAW_ADV_SAVES/adv_type_{}/{}/EPS_{}/eps_stp_size_{}/{}/adv_steps_{}/on_train_{}/{}".format(adv_attack_type , residue_vname , eps, eps_step_size, crit,number_of_adversarial_optimization_steps, is_adv_attack_on_train,no_default_str)

    return final_adv_postfix_for_save
        

def load_or_generate_adv_examples(to_be_analysed_dataloader, models_base_path, is_act_collection_on_train, model, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, number_of_batch_to_collect=None, is_save_adv=False, save_path=None, each_save_postfix="",update_on='all',rand_init=True,norm=np.inf,use_ytrue=True,lossfn=None):
    if(save_path is None):
        final_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_act_collection_on_train,update_on,rand_init,norm,use_ytrue)+str(each_save_postfix)
        adv_save_path = models_base_path + final_postfix_for_save+"/adv_dataset.npy"
    else:
        adv_save_path = save_path

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
            to_be_analysed_dataloader, model, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, number_of_batch_to_collect, is_save_adv=is_save_adv, save_path=adv_save_path,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue,lossfn=lossfn)
    return adv_dataset


def generate_adversarial_perturbation_from_adv_orig(orig_dataloader, adv_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged_dataset = generate_merged_dataset_from_two_loader(
        orig_dataloader, adv_dataloader)
    merged_data_loader = torch.utils.data.DataLoader(
        merged_dataset, batch_size=128, shuffle=False)
    # merged_data_loader = tqdm(
    #     merged_data_loader, desc='Generating adversarial perturbation')
    list_of_x = []
    list_of_y = []
    for _, inp_data in enumerate(merged_data_loader):
        x1, x2, y1, y2 = inp_data
        x1, x2, y1, y2 = x1.to(device), x2.to(
            device), y1.to(device), y2.to(device)
        for ind in range(len(x1)):
            list_of_x.append(x2[ind] - x1[ind])
        for each_y in y1:
            list_of_y.append(each_y)

    perturb_dataset = CustomSimpleDataset(
        list_of_x, list_of_y)
    return perturb_dataset


def generate_adv_examples(data_loader, model, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, number_of_batch_to_collect=None, is_save_adv=False, save_path=None, update_on='all',rand_init=True,norm=np.inf,use_ytrue=True,lossfn=None,residue_vname=None):
    print("Adversarial will be saved at:", save_path)
    cpudevice = torch.device("cpu")
    is_targetted = adv_target is not None
    list_of_adv_images = None
    list_of_labels = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = tqdm.tqdm(
        data_loader, desc='Generating adversarial examples for current class')
    for i, per_class_per_batch_data in enumerate(data_loader):
        images, labels = per_class_per_batch_data
        images, labels = images.to(
            device), labels.to(device)

        current_adv_image = apply_adversarial_attack_on_input(
            images, model, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, labels, is_targetted,update_on,rand_init,norm,use_ytrue,lossfn,residue_vname=residue_vname)

        if(list_of_adv_images is None):
            list_of_adv_images = current_adv_image
        else:
            list_of_adv_images = torch.vstack(
                (list_of_adv_images, current_adv_image))

        if(list_of_labels is None):
            list_of_labels = labels
        else:
            list_of_labels = torch.cat(
                (list_of_labels, labels))

        if(not(number_of_batch_to_collect is None) and i == number_of_batch_to_collect - 1):
            break

    per_class_adv_dataset = CustomSimpleDataset(
        list_of_adv_images, list_of_labels)

    if(is_save_adv == True):
        sfolder = save_path[0:save_path.rfind("/")+1]
        if not os.path.exists(sfolder):
            os.makedirs(sfolder)
        with open(save_path, 'wb') as file:
            np.savez(file, x=list_of_adv_images.to(cpudevice, non_blocking=True).detach().numpy(
            ), y=list_of_labels.to(cpudevice, non_blocking=True).detach().numpy())

    return per_class_adv_dataset


def get_reconstructed_template_images(search_path, class_label):

    np_save_filename = search_path + \
        '/class_'+str(class_label) + '.npy'
    with open(np_save_filename, 'rb') as file:
        npzfile = np.load(np_save_filename)
        each_class_output_template_list = npzfile['x']
        current_y_s = npzfile['y']

    current_reconstructed_dataset = CustomSimpleDataset(
        each_class_output_template_list, current_y_s)

    return current_reconstructed_dataset


def extract_common_activation_patterns_between_adv_and_normal(true_input_data_list_per_class, model,
                                                              template_image_calculation_batch_size, number_of_batch_to_collect, collect_threshold, torch_seed, classes,
                                                              eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target,
                                                              wand_project_name, wandb_group_name, wandb_run_name, wandb_config,update_on='all',rand_init=True,norm=np.inf,use_ytrue=True):
    is_log_wandb = not(wand_project_name is None)

    class_indx_to_visualize = [i for i in range(len(classes))]
    per_class_common_active_percentages = [None] * num_classes
    per_class_common_active_pixel_counts = [None] * num_classes
    per_class_common_total_pixel_counts = [None] * num_classes

    for c_indx in class_indx_to_visualize:
        class_label = classes[c_indx]
        print("************************************************************ Class:", class_label)
        if(is_log_wandb):
            wandb_config["class_label"] = class_label
            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                group=f"{wandb_group_name}",
                config=wandb_config,
            )

        per_class_dataset = PerClassDataset(
            true_input_data_list_per_class[c_indx], c_indx)

        coll_seed_gen = torch.Generator()
        coll_seed_gen.manual_seed(torch_seed)

        model.train(False)

        per_class_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=template_image_calculation_batch_size,
                                                            shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)

        tmp_gen = TemplateImageGenerator(
            model, None)

        print("$$$$$$$$$$$$$$$$$$$$$$$$ Collecting active pixels for original images $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # Sending original images for collection
        tmp_gen.collect_all_active_pixels_into_ymaps(
            per_class_data_loader, class_label, number_of_batch_to_collect, collect_threshold, is_save_original_image=False)
        active_maps_actual_images = tmp_gen.get_active_maps()
        tmp_gen.reset_collection_state()

        per_class_adv_dataset = generate_adv_examples(
            per_class_data_loader, model, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, number_of_batch_to_collect,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue)

        coll_seed_gen2 = torch.Generator()
        coll_seed_gen2.manual_seed(torch_seed)

        per_class_avg_data_loader = torch.utils.data.DataLoader(per_class_adv_dataset, batch_size=template_image_calculation_batch_size,
                                                                shuffle=True, generator=coll_seed_gen2, worker_init_fn=seed_worker)

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$ Collecting active pixels for adversarial images $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # Sending adversarial images for collection
        tmp_gen.collect_all_active_pixels_into_ymaps(
            per_class_avg_data_loader, class_label, number_of_batch_to_collect, collect_threshold, is_save_original_image=False)
        active_maps_adv_images = tmp_gen.get_active_maps()
        tmp_gen.reset_collection_state()

        common_active_percentage, common_active_pixels, total_pixels, active_pixels_actual_images, active_pixels_adv_images = calculate_common_among_two_activation_patterns(
            active_maps_actual_images, active_maps_adv_images)

        percentage_active_original_images = (
            100. * (active_pixels_actual_images/total_pixels))
        percentage_active_adversarial_images = (
            100. * (active_pixels_adv_images/total_pixels))

        per_class_common_active_percentages[c_indx] = common_active_percentage
        per_class_common_active_pixel_counts[c_indx] = common_active_pixels
        per_class_common_total_pixel_counts[c_indx] = total_pixels

        print("common_active_pixel_points", common_active_pixels)
        print("common_percent_active_pixels", common_active_percentage)
        print("total_pixel_points", total_pixels)

        if(is_log_wandb):
            wandb.log({"common_active_pixel_points": common_active_pixels, "total_pixel_points": total_pixels,
                       "percent_common_active_pixels_adv_orig": common_active_percentage, "percentage_active_original_images": percentage_active_original_images,
                       "percentage_active_adversarial_images": percentage_active_adversarial_images, "active_pixels_original_images": active_pixels_actual_images,
                       "active_pixels_adversarial_images": active_pixels_adv_images})
            wandb.finish()

    return per_class_common_active_percentages, per_class_common_active_pixel_counts, per_class_common_total_pixel_counts


def extract_common_activation_patterns_between_reconst_and_original(true_input_data_list_per_class, model, search_path,
                                                                    template_image_calculation_batch_size, number_of_batch_to_collect, collect_threshold, torch_seed, classes,
                                                                    wand_project_name, wandb_group_name, wandb_run_name, wandb_config):
    is_log_wandb = not(wand_project_name is None)

    class_indx_to_visualize = [i for i in range(len(classes))]
    per_class_common_active_percentages = [None] * num_classes
    per_class_common_active_pixel_counts = [None] * num_classes
    per_class_common_total_pixel_counts = [None] * num_classes

    for c_indx in class_indx_to_visualize:
        class_label = classes[c_indx]
        print("************************************************************ Class:", class_label)
        if(is_log_wandb):
            wandb_config["class_label"] = class_label
            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                group=f"{wandb_group_name}",
                config=wandb_config,
            )

        per_class_dataset = PerClassDataset(
            true_input_data_list_per_class[c_indx], c_indx)

        coll_seed_gen = torch.Generator()
        coll_seed_gen.manual_seed(torch_seed)

        model.train(False)

        per_class_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=template_image_calculation_batch_size,
                                                            shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)

        tmp_gen = TemplateImageGenerator(
            model, None)

        print("$$$$$$$$$$$$$$$$$$$$$$$$ Collecting active pixels for original images $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # Sending original images for collection
        tmp_gen.collect_all_active_pixels_into_ymaps(
            per_class_data_loader, class_label, number_of_batch_to_collect, collect_threshold, is_save_original_image=False)
        active_maps_actual_images = tmp_gen.get_active_maps()
        tmp_gen.reset_collection_state()

        per_class_reconst_dataset = get_reconstructed_template_images(
            search_path, class_label)

        coll_seed_gen2 = torch.Generator()
        coll_seed_gen2.manual_seed(torch_seed)

        per_class_reconst_data_loader = torch.utils.data.DataLoader(per_class_reconst_dataset, batch_size=template_image_calculation_batch_size,
                                                                    shuffle=True, generator=coll_seed_gen2, worker_init_fn=seed_worker)

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$ Collecting active pixels for reconstructed images $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # Sending reconstructed images for collection
        tmp_gen.collect_all_active_pixels_into_ymaps(
            per_class_reconst_data_loader, class_label, number_of_batch_to_collect, collect_threshold, is_save_original_image=False)
        active_maps_reconst_images = tmp_gen.get_active_maps()
        tmp_gen.reset_collection_state()

        common_active_percentage, common_active_pixels, total_pixels, active_pixels_actual_images, active_pixels_reconst_images = calculate_common_among_two_activation_patterns(
            active_maps_actual_images, active_maps_reconst_images)

        percentage_active_original_images = (
            100. * (active_pixels_actual_images/total_pixels))
        percentage_active_reconst_images = (
            100. * (active_pixels_reconst_images/total_pixels))

        per_class_common_active_percentages[c_indx] = common_active_percentage
        per_class_common_active_pixel_counts[c_indx] = common_active_pixels
        per_class_common_total_pixel_counts[c_indx] = total_pixels

        print("common_active_pixel_points", common_active_pixels)
        print("common_percent_active_pixels", common_active_percentage)
        print("total_pixel_points", total_pixels)

        if(is_log_wandb):
            wandb.log({"common_active_pixel_points": common_active_pixels, "total_pixel_points": total_pixels,
                       "percent_common_active_pixels_reconst_orig": common_active_percentage, "percentage_active_original_images": percentage_active_original_images,
                       "percentage_active_reconst_images": percentage_active_reconst_images, "active_pixels_original_images": active_pixels_actual_images,
                       "active_pixels_reconst_images": active_pixels_reconst_images})
            wandb.finish()

    return per_class_common_active_percentages, per_class_common_active_pixel_counts, per_class_common_total_pixel_counts

def get_model_from_path(dataset, model_arch_type, model_path, mask_percentage=40,custom_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_model = torch.load(model_path, map_location=device)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if(custom_model is None):
        custom_model = get_model_instance_from_dataset(
            dataset, model_arch_type)
        if device_str == 'cuda':
            if(torch.cuda.device_count() > 1):
                custom_model = torch.nn.DataParallel(custom_model)
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
        custom_model.load_state_dict(temp_model.state_dict(),strict=False)

    return custom_model

def project_to_eps_inf_ball(loader,eps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod_x = []
    with torch.no_grad():
        for X_adv, X_org in loader:
            X_adv ,X_org = X_adv.to(device),X_org.to(device)
            eta = X_adv - X_org
            eta = torch.clamp(eta, -eps, eps)
            tmp = (X_org + eta).cpu()
            mod_x.append(tmp)
    mod_x = torch.cat(mod_x)
    return mod_x


if __name__ == '__main__':
    # fashion_mnist , mnist, cifar10
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net , conv4_deep_gated_net_n16_small
    # fc_dnn , fc_dlgn , fc_dgn , dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias__, bc_fc_dnn ,  fc_sf_dlgn , madry_mnist_conv4_dnn , bc_fc_dnn
    model_arch_type = 'bc_fc_dnn'
    scheme_type = 'iterative_augmented_model_attack'
    # scheme_type = ''
    batch_size = 64

    torch_seed = 2022

    # None means that train on all classes
    list_of_classes_to_train_on = None
    list_of_classes_to_train_on = [3,8]

    # Percentage of information retention during PCA (values between 0-1)
    pca_exp_percent = None
    # pca_exp_percent = 0.85

    wandb_config_additional_dict = None
    # wandb_config_additional_dict = {
    #     "type_of_APR": "APRP","is_train_on_phase": True}
    # "is_train_on_phase": True
    # GATE_NET_FREEZE , VAL_NET_FREEZE
    # wandb_config_additional_dict = {
    #     "transfer_mode": "GATE_NET_FREEZE"}
    # wandb_config_additional_dict = {"type_of_APR": "APRS"}

    direct_model_path = None
    # direct_model_path = "root/model/save/mnist/CLEAN_TRAINING/TR_ON_3_8/ST_2022/bc_fc_sf_dlgn_W_16_D_4_dir.pt"
    direct_model_path = "root/model/save/mnist/adversarial_training/TR_ON_3_8/MT_bc_fc_dnn_W_16_D_4_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_FGSM/adv_type_PGD/EPS_0.3/OPT_Adam (Parameter Group 0    amsgrad: False    betas: (0.9, 0.999)    eps: 1e-08    lr: 0.0001    weight_decay: 0)/batch_size_64/eps_stp_size_0.01/adv_steps_40/update_on_all/R_init_True/norm_inf/use_ytrue_True/out_lossfn_BCEWithLogitsLoss()/inner_lossfn_Y_Logits_Binary_class_Loss()/adv_model_dir.pt"

    custom_dataset_path = None
    # custom_dataset_path = "data/custom_datasets/freq_band_dataset/mnist__MB.npy"

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
        fc_width = 16
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

    if(scheme_type == 'iterative_augmented_model_attack'):
        wand_project_name = None
        # wand_project_name = "cifar10_all_images_based_template_visualizations"
        # wand_project_name = "adv_attack_for_active_pixels_on_reconst_augmentation"
        # wand_project_name = "adv_attack_via_reconst_on_reconst_augmentation_with_orig"
        # wand_project_name = 'V2_adv_attack_on_reconst_augmentation_with_orig'
        # wand_project_name = "APR_experiments"
        # wand_project_name = "adv_attack_latest"
        # wand_project_name = "benchmarking_adv_exps"
        # wand_project_name = "NPK_reg"
        # wand_project_name = 'eval_model_band_frequency_experiments'
        # wand_project_name = "Part_training_for_robustness"
        # wand_project_name = "Pruning-exps"
        # wand_project_name = "Gatesat-exp"
        # wand_project_name = "minute_FC_dlgn"
        # wand_project_name = "L2RegCNNs"
        # wand_project_name = "adversarial_attacks_latest_madrys"
        # wand_project_name = "madry's_benchmarking"
        # wand_project_name = "SVM_loss_training"
        wand_project_name = "Cifar10_flamarion_replicate"

        torch_seed = 2022
        # FEATURE_FLIP , PGD , FGSM
        adv_attack_type = "PGD"
        adv_target = None
        # ACTIVATION_COMPARE , ADV_ATTACK , ACT_COMPARE_RECONST_ORIGINAL , ADV_ATTACK_EVAL_VIA_RECONST , ADV_ATTACK_PER_CLASS , FREQ_BAND_ADV_ATTACK_PER_CLASS
        # AFTER_ATT_FREQ_BAND_ADV_ATTACK_PER_CLASS
        exp_type = "ADV_ATTACK"
        is_adv_attack_on_train = False
        eps_step_size = 0.06

        # Best attack setting is always  update_on='all' rand_init=True norm=np.inf use_ytrue=True (False attacks slightly less actually contradictory to label leaking since it is multi-step) number_of_restarts=1(more restarts also end up in similar accuracies)
        update_on='all'
        rand_init=True
        norm=np.inf
        use_ytrue=True
        number_of_restarts = 1
        residue_vname = None
        # residue_vname = "all_tanh_gate_flip"

        model_and_data_save_prefix = "root/model/save/mnist/adversarial_training/MT_dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias___ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.06/batch_size_128/eps_stp_size_0.06/adv_steps_80/adv_model_dir.pt"

        number_of_augment_iterations = 1

        is_targetted = adv_target is not None
        is_log_wandb = not(wand_project_name is None)

        # eps_list = [0.03, 0.06, 0.1]
        if("mnist" in dataset):
            # 40 is a good sweet spot more than that doesn't help much typically
            number_of_adversarial_optimization_steps = 40
            eps_list = [0.3]
            eps_step_size = 0.01
        elif("cifar10" in dataset):
            number_of_adversarial_optimization_steps = 10
            eps_list = [8/255,6/255]
            eps_step_size = 2/255
        if(exp_type == "ACT_COMPARE_RECONST_ORIGINAL"):
            eps_list = [0]

        if(is_log_wandb):
            wandb.login()
        criterion= None
        if("bc_" in model_arch_type):
            # criterion = torch.nn.BCEWithLogitsLoss()
            criterion = Y_Logits_Binary_class_Loss()

        if(direct_model_path is not None):
            number_of_augment_iterations = 1

        for eps in eps_list:
            for current_aug_iter_num in range(1, number_of_augment_iterations+1):
                if(direct_model_path is None):
                    model_save_path = model_and_data_save_prefix+'aug_conv4_dlgn_iter_{}_dir.pt'.format(
                        current_aug_iter_num)
                    if not os.path.exists(model_and_data_save_prefix):
                        os.makedirs(model_and_data_save_prefix)
                else:
                    current_aug_iter_num = None
                    model_save_path = direct_model_path
                    if('CLEAN' in model_save_path or 'APR_TRAINING' in model_save_path):
                        model_and_data_save_prefix = model_save_path[0:model_save_path.rfind(
                            ".pt")]
                    else:
                        model_and_data_save_prefix = model_save_path[0:model_save_path.rfind(
                            "/")+1]

                isExist = os.path.exists(model_save_path)
                assert isExist == True, 'Model path does not have saved model'

                net = get_model_from_path(
                    dataset, model_arch_type, model_save_path,custom_model=net)

                if("cifar10" in dataset):
                    net.initialize_standardization_layer()
                
                net = net.to(device)
                device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
                if device_str == 'cuda':
                    cudnn.benchmark = True

                print("Loaded previously trained model for augment iteration:",
                      current_aug_iter_num)
                if(is_adv_attack_on_train == True):
                    eval_loader = trainloader
                else:
                    eval_loader = testloader

                print("Net:", net)
                if(direct_model_path is None):
                    final_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_adv_attack_on_train,update_on,rand_init,norm,use_ytrue,residue_vname,criterion)+"/aug_indx_{}/".format(current_aug_iter_num)
                    final_postfix_for_save = final_postfix_for_save.replace("RAW_ADV_SAVES","ADV_SAVES/exp_type_{}/".format(exp_type))
                else:
                    final_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,"",update_on,rand_init,norm,use_ytrue,residue_vname,criterion)
                    final_postfix_for_save = final_postfix_for_save.replace("RAW_ADV_SAVES","ADV_SAVES/exp_type_{}/".format(exp_type))
                    final_postfix_for_save = final_postfix_for_save.replace("on_train_","")
                    
                save_folder = model_and_data_save_prefix + final_postfix_for_save
                if(exp_type == "ADV_ATTACK"):
                    wandb_group_name = "DS_"+str(dataset_str) + \
                        "_adv_attack_over_aug_"+str(model_arch_type_str)

                    if(is_log_wandb):
                        wandb_run_name = str(
                            model_arch_type_str)+"_aug_it_"+str(current_aug_iter_num)+"adv_at_"+str(adv_attack_type)
                        wandb_config = get_wandb_config(exp_type, adv_attack_type, model_arch_type_str, dataset_str, is_adv_attack_on_train,
                                                        eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted, adv_target,update_on,rand_init,norm,use_ytrue)
                        wandb_config["residue_vname"]=residue_vname
                        wandb_config["aug_iter_num"] = current_aug_iter_num
                        wandb_config["number_of_restarts"] = number_of_restarts
                        wandb_config["inner_criterion"] = criterion
                        if(pca_exp_percent is not None):
                            wandb_config["pca_exp_percent"] = pca_exp_percent
                            wandb_config["num_comp_pca"] = number_of_components_for_pca
                        if(wandb_config_additional_dict is not None):
                            wandb_config.update(wandb_config_additional_dict)
                        wandb.init(
                            project=f"{wand_project_name}",
                            name=f"{wandb_run_name}",
                            group=f"{wandb_group_name}",
                            config=wandb_config,
                        )
                    if(direct_model_path is None):
                        each_save_postfix = "/aug_indx_{}".format(
                            current_aug_iter_num)
                        final_adv_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_adv_attack_on_train,update_on,rand_init,norm,use_ytrue,residue_vname,criterion)+str(each_save_postfix)
                    else:
                        final_adv_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_adv_attack_on_train,update_on,rand_init,norm,use_ytrue,residue_vname,criterion)
                    adv_save_path = model_and_data_save_prefix + \
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
                            eval_loader, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, is_save_adv=False, save_path=adv_save_path,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue,residue_vname=residue_vname,lossfn=criterion)

                    to_be_analysed_adversarial_dataloader = torch.utils.data.DataLoader(
                        adv_dataset, shuffle=False, batch_size=batch_size)

                    save_images_from_dataloader(to_be_analysed_adversarial_dataloader, classes,
                                                postfix_folder_for_save='/adver/', save_image_prefix=save_folder)

                    eval_adv_acc, adv_frequency_pred,adv_classwise_acc = plain_evaluate_model(
                        net, to_be_analysed_adversarial_dataloader,classes,True)
                    
                    if(number_of_restarts>1):
                        avg_adv_frequency_pred = adv_frequency_pred
                        avg_adv_classwise_acc = adv_classwise_acc
                        avg_adv_acc = eval_adv_acc
                        for cur_restart in range(number_of_restarts-1):
                            adv_dataset = generate_adv_examples(
                                eval_loader, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, is_save_adv=False, save_path=adv_save_path,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue,residue_vname=residue_vname,lossfn=criterion)

                            to_be_analysed_adversarial_dataloader = torch.utils.data.DataLoader(
                                adv_dataset, shuffle=False, batch_size=256)

                            eval_adv_acc, freq_pred,classwise_acc = plain_evaluate_model(
                                net, to_be_analysed_adversarial_dataloader,classes,True)
                            avg_adv_classwise_acc += classwise_acc
                            avg_adv_frequency_pred += freq_pred
                            avg_adv_acc += eval_adv_acc
                            print("Restart :{} adv acc:{}".format(cur_restart,eval_adv_acc))
                        eval_adv_acc = avg_adv_acc/number_of_restarts
                        adv_classwise_acc = avg_adv_classwise_acc/number_of_restarts
                        adv_frequency_pred = avg_adv_frequency_pred/number_of_restarts

                    eval_orig_acc, orig_frequency_pred,org_classwise_acc = plain_evaluate_model(
                        net, eval_loader,classes,True)

                    if(is_log_wandb):
                        wandb.log({"eval_orig_acc": eval_orig_acc,
                                  "eval_adv_acc": eval_adv_acc,
                                  "adv_frequency_pred":adv_frequency_pred,
                                  "orig_frequency_pred":orig_frequency_pred,
                                  "org_classwise_acc":org_classwise_acc,
                                  "adv_classwise_acc":adv_classwise_acc})
                        wandb.finish()
                elif(exp_type == "ADV_ATTACK_PER_CLASS"):
                    class_indx_to_visualize = [i for i in range(len(classes))]
                    if(direct_model_path is None):
                        each_save_postfix = "/aug_indx_{}".format(
                            current_aug_iter_num)
                        final_adv_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_adv_attack_on_train,update_on,rand_init,norm,use_ytrue,residue_vname,criterion)+str(each_save_postfix)
                    else:
                        final_adv_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_adv_attack_on_train,update_on,rand_init,norm,use_ytrue,residue_vname,criterion)
                    adv_save_path = model_and_data_save_prefix + \
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
                            eval_loader, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, is_save_adv=False, save_path=adv_save_path,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue,residue_vname=residue_vname,lossfn=criterion)

                    to_be_analysed_adversarial_dataloader = torch.utils.data.DataLoader(
                        adv_dataset, shuffle=False, batch_size=128)

                    true_eval_dataset_per_class = true_segregation(
                        eval_loader, num_classes_trained_on)
                    true_tobe_analysed_dataset_per_class = true_segregation(
                        to_be_analysed_adversarial_dataloader, num_classes_trained_on)

                    for c_indx in class_indx_to_visualize:
                        class_label = classes[c_indx]
                        print(
                            "************************************************************ Class:", class_label)
                        per_class_eval_dataset = PerClassDataset(
                            true_eval_dataset_per_class[c_indx], c_indx)
                        per_class_tobe_analysed_dataset = PerClassDataset(
                            true_tobe_analysed_dataset_per_class[c_indx], c_indx)
                        wandb_group_name = "DS_"+str(dataset_str) + \
                            "_adv_attack_over_aug_"+str(model_arch_type_str)

                        if(is_log_wandb):
                            wandb_run_name = str(
                                model_arch_type_str)+"_aug_it_"+str(current_aug_iter_num)+"adv_at_"+str(adv_attack_type)
                            wandb_config = get_wandb_config(exp_type, adv_attack_type, model_arch_type_str, dataset_str, is_adv_attack_on_train,
                                                            eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted, adv_target,update_on,rand_init,norm,use_ytrue)
                            wandb_config["residue_vname"] = residue_vname
                            wandb_config["aug_iter_num"] = current_aug_iter_num
                            wandb_config["class_label"] = class_label
                            wandb_config["c_indx"] = c_indx
                            wandb_config["inner_criterion"] = criterion
                            if(pca_exp_percent is not None):
                                wandb_config["pca_exp_percent"] = pca_exp_percent
                                wandb_config["num_comp_pca"] = number_of_components_for_pca
                            if(wandb_config_additional_dict is not None):
                                wandb_config.update(
                                    wandb_config_additional_dict)
                            wandb.init(
                                project=f"{wand_project_name}",
                                name=f"{wandb_run_name}",
                                group=f"{wandb_group_name}",
                                config=wandb_config,
                            )

                        coll_seed_gen = torch.Generator()
                        coll_seed_gen.manual_seed(torch_seed)

                        per_class_eval_data_loader = torch.utils.data.DataLoader(per_class_eval_dataset, batch_size=128,
                                                                                 shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)

                        coll_seed_gen = torch.Generator()
                        coll_seed_gen.manual_seed(torch_seed)

                        per_class_tobe_analysed_data_loader = torch.utils.data.DataLoader(per_class_tobe_analysed_dataset, batch_size=128,
                                                                                          shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)
                        
                        # save_bifurcated_images_from_dataloader(net,eval_loader, classes,
                                                    # postfix_folder_for_save='/det_orig/', save_image_prefix=save_folder)
                        # save_bifurcated_images_from_dataloader(net,to_be_analysed_adversarial_dataloader, classes,
                                                    # postfix_folder_for_save='/det_adver/', save_image_prefix=save_folder)
                        save_images_from_dataloader(to_be_analysed_adversarial_dataloader, classes,
                                                    postfix_folder_for_save='/adver/', save_image_prefix=save_folder)

                        eval_orig_acc, orig_frequency_pred = plain_evaluate_model(
                            net, per_class_eval_data_loader,classes)
                        eval_adv_acc, adv_frequency_pred = plain_evaluate_model(
                            net, per_class_tobe_analysed_data_loader, classes)
                        print("frequency_pred", adv_frequency_pred)

                        if(is_log_wandb):
                            wandb.log({"eval_orig_acc": eval_orig_acc,
                                       "eval_adv_acc": eval_adv_acc,
                                       "adv_frequency_pred": adv_frequency_pred,
                                       "orig_frequency_pred":orig_frequency_pred})
                            wandb.finish()
                elif(exp_type == "ADV_ATTACK_EVAL_VIA_RECONST"):
                    wandb_group_name = "DS_"+str(dataset_str) + \
                        "_adv_attack_over_aug_"+str(model_arch_type_str)
                    number_of_image_optimization_steps = 141
                    template_initial_image_type = 'zero_init_image'
                    template_loss_type = "TEMP_LOSS"

                    if(is_log_wandb):
                        wandb_run_name = str(
                            model_arch_type_str)+"_aug_it_"+str(current_aug_iter_num)+"adv_at_"+str(adv_attack_type)
                        wandb_config = get_wandb_config(exp_type, adv_attack_type, model_arch_type_str, dataset_str, is_adv_attack_on_train,
                                                        eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted, adv_target,update_on,rand_init,norm,use_ytrue)
                        wandb_config["aug_iter_num"] = current_aug_iter_num
                        wandb_config["number_of_image_optimization_steps"] = number_of_image_optimization_steps
                        wandb_config["template_initial_image_type"] = template_initial_image_type
                        wandb_config["template_loss_type"] = template_loss_type
                        wandb_config["inner_criterion"] = criterion
                        if(pca_exp_percent is not None):
                            wandb_config["pca_exp_percent"] = pca_exp_percent
                            wandb_config["num_comp_pca"] = number_of_components_for_pca
                        wandb.init(
                            project=f"{wand_project_name}",
                            name=f"{wandb_run_name}",
                            group=f"{wandb_group_name}",
                            config=wandb_config,
                        )

                    each_save_postfix = "/aug_indx_{}".format(
                        current_aug_iter_num)
                    final_adv_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_adv_attack_on_train,update_on,rand_init,norm,use_ytrue,criterion)+str(each_save_postfix)
                    adv_save_path = model_and_data_save_prefix + \
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
                            eval_loader, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, is_save_adv=True, save_path=adv_save_path,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue,lossfn=criterion)

                    to_be_analysed_adversarial_dataloader = torch.utils.data.DataLoader(
                        adv_dataset, shuffle=False, batch_size=128)

                    eval_orig_via_reconst_acc = plain_evaluate_model_via_reconstructed(
                        model_arch_type, net, eval_loader, classes, dataset, template_initial_image_type, number_of_image_optimization_steps, template_loss_type, adv_target)
                    eval_adv_via_reconst_acc = plain_evaluate_model_via_reconstructed(
                        model_arch_type, net, to_be_analysed_adversarial_dataloader, classes, dataset, template_initial_image_type, number_of_image_optimization_steps, template_loss_type, adv_target, save_image_prefix=save_folder, postfix_folder_for_save="/recons_adver/")

                    if(is_log_wandb):
                        wandb.log({"eval_orig_via_reconst_acc": eval_orig_via_reconst_acc,
                                  "eval_adv_via_reconst_acc": eval_adv_via_reconst_acc})
                        wandb.finish()

                elif(exp_type == "ACTIVATION_COMPARE"):
                    template_image_calculation_batch_size = 32
                    number_of_batch_to_collect = None
                    collect_threshold = 0.95
                    torch_seed = 2022

                    wandb_group_name = "DS_"+str(dataset_str) + "_EXP_"+str(exp_type) +\
                        "_adv_attack_over_aug_"+str(model_arch_type_str)
                    wandb_run_name = str(
                        model_arch_type_str)+"_aug_it_"+str(current_aug_iter_num)+"adv_at_"+str(adv_attack_type)
                    wandb_config = get_wandb_config(exp_type, adv_attack_type, model_arch_type_str, dataset_str, is_adv_attack_on_train,
                                                    eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted, adv_target,update_on,rand_init,norm,use_ytrue)
                    wandb_config["template_image_calculation_batch_size"] = template_image_calculation_batch_size
                    wandb_config["number_of_batch_to_collect"] = number_of_batch_to_collect
                    wandb_config["collect_threshold"] = collect_threshold
                    wandb_config["torch_seed"] = torch_seed
                    wandb_config["aug_iter_num"] = current_aug_iter_num

                    true_input_data_list_per_class = true_segregation(
                        eval_loader, num_classes_trained_on)

                    per_class_common_active_percentages, per_class_common_active_pixel_counts, per_class_common_total_pixel_counts = extract_common_activation_patterns_between_adv_and_normal(true_input_data_list_per_class, net,
                                                                                                                                                                                               template_image_calculation_batch_size, number_of_batch_to_collect, collect_threshold, torch_seed,
                                                                                                                                                                                               classes, eps, adv_attack_type,
                                                                                                                                                                                               number_of_adversarial_optimization_steps, eps_step_size, adv_target, wand_project_name, wandb_group_name, wandb_run_name, wandb_config)
                elif(exp_type == "ACT_COMPARE_RECONST_ORIGINAL"):
                    template_image_calculation_batch_size = 32
                    number_of_batch_to_collect = None
                    collect_threshold = 0.95
                    torch_seed = 2022

                    wandb_group_name = "DS_"+str(dataset_str) + "_EXP_"+str(exp_type) +\
                        "_adv_attack_over_aug_"+str(model_arch_type_str)
                    wandb_run_name = str(
                        model_arch_type_str)+"_aug_it_"+str(current_aug_iter_num)+"adv_at_"+str(adv_attack_type)
                    wandb_config = dict()
                    wandb_config["exp_type"] = exp_type
                    wandb_config["model_arch_type"] = model_arch_type_str
                    wandb_config["dataset"] = dataset_str
                    wandb_config["template_image_calculation_batch_size"] = template_image_calculation_batch_size
                    wandb_config["number_of_batch_to_collect"] = number_of_batch_to_collect
                    wandb_config["collect_threshold"] = collect_threshold
                    wandb_config["torch_seed"] = torch_seed
                    wandb_config["aug_iter_num"] = current_aug_iter_num
                    wandb_config["on_train"] = is_adv_attack_on_train
                    wandb_config["model_attacked_path"] = model_save_path

                    true_input_data_list_per_class = true_segregation(
                        eval_loader, num_classes_trained_on)

                    final_postfix_for_numpy_save = "aug_indx_{}/".format(
                        current_aug_iter_num)
                    search_path = model_and_data_save_prefix + final_postfix_for_numpy_save

                    per_class_common_active_percentages, per_class_common_active_pixel_counts, per_class_common_total_pixel_counts = extract_common_activation_patterns_between_reconst_and_original(true_input_data_list_per_class, net, search_path,
                                                                                                                                                                                                     template_image_calculation_batch_size, number_of_batch_to_collect, collect_threshold, torch_seed,
                                                                                                                                                                                                     classes, wand_project_name, wandb_group_name, wandb_run_name, wandb_config)
                elif(exp_type == "FREQ_BAND_ADV_ATTACK_PER_CLASS"):
                    allmodes=[["LB","MB","HB"],
                        ["MB","HB"],
                        ["LB","HB"],
                        ["LB","MB"],
                        ["LB"],
                        ["MB"],
                        ["HB"]]
                    for cur_mode in allmodes:
                        if("fashion_mnist" in dataset):
                            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
                        elif("mnist" in dataset):
                            (X_train, y_train), (X_test, y_test) = mnist.load_data()
                        loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), shuffle=False, batch_size=256)
                        (X_test, y_test) = modify_bandpass_freq_get_dataset(loader,cur_mode)
                        _, _, _, _, X_test, y_test = preprocess_mnist_fmnist(X_train,y_train,X_test,y_test,ds_config,model_arch_type,verbose=1, is_split_validation=False)
                        eval_loader = get_data_loader(
                            X_test, y_test, ds_config.batch_size, transforms=ds_config.test_transforms)

                        modestr = ""
                        for ee in cur_mode:
                            modestr += ee +"_"
                        modestr = modestr[0:len(modestr)-1]

                        class_indx_to_visualize = [i for i in range(len(classes))]
                        if(direct_model_path is None):
                            each_save_postfix = "/aug_indx_{}".format(
                                current_aug_iter_num)
                            final_adv_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_adv_attack_on_train,update_on,rand_init,norm,use_ytrue,criterion)+str(each_save_postfix)
                        else:
                            final_adv_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_adv_attack_on_train,update_on,rand_init,norm,use_ytrue,criterion)
                        # adv_save_path = model_and_data_save_prefix + \
                        #     final_adv_postfix_for_save+"/adv_dataset.npy"
                        # is_current_adv_aug_available = os.path.exists(
                        #     adv_save_path)
                        # if(is_current_adv_aug_available):
                        #     with open(adv_save_path, 'rb') as file:
                        #         npzfile = np.load(adv_save_path)
                        #         list_of_adv_images = npzfile['x']
                        #         list_of_labels = npzfile['y']
                        #         adv_dataset = CustomSimpleDataset(
                        #             list_of_adv_images, list_of_labels)
                        #         print("Loading adversarial examples from path:",
                        #               adv_save_path)
                        # else:
                        #     print("adv_save_path:", adv_save_path)
                        adv_dataset = generate_adv_examples(
                            eval_loader, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, is_save_adv=False, save_path=None,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue,lossfn=criterion)

                        to_be_analysed_adversarial_dataloader = torch.utils.data.DataLoader(
                            adv_dataset, shuffle=False, batch_size=128)

                        true_eval_dataset_per_class = true_segregation(
                            eval_loader, num_classes_trained_on)
                        true_tobe_analysed_dataset_per_class = true_segregation(
                            to_be_analysed_adversarial_dataloader, num_classes_trained_on)

                        for c_indx in class_indx_to_visualize:
                            class_label = classes[c_indx]
                            print(
                                "************************************************************ Class:", class_label)
                            per_class_eval_dataset = PerClassDataset(
                                true_eval_dataset_per_class[c_indx], c_indx)
                            per_class_tobe_analysed_dataset = PerClassDataset(
                                true_tobe_analysed_dataset_per_class[c_indx], c_indx)
                            wandb_group_name = "DS_"+str(dataset_str) + \
                                "_adv_attack_over_aug_"+str(model_arch_type_str)

                            if(is_log_wandb):
                                wandb_run_name = str(
                                    model_arch_type_str)+"_aug_it_"+str(current_aug_iter_num)+"adv_at_"+str(adv_attack_type)
                                wandb_config = get_wandb_config(exp_type, adv_attack_type, model_arch_type_str, dataset_str, is_adv_attack_on_train,
                                                                eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted, adv_target,update_on,rand_init,norm,use_ytrue)
                                wandb_config["aug_iter_num"] = current_aug_iter_num
                                wandb_config["class_label"] = class_label
                                wandb_config["c_indx"] = c_indx
                                wandb_config["freq_band"] = modestr
                                wandb_config["inner_criterion"] = criterion
                                if(pca_exp_percent is not None):
                                    wandb_config["pca_exp_percent"] = pca_exp_percent
                                    wandb_config["num_comp_pca"] = number_of_components_for_pca
                                if(wandb_config_additional_dict is not None):
                                    wandb_config.update(
                                        wandb_config_additional_dict)
                                wandb.init(
                                    project=f"{wand_project_name}",
                                    name=f"{wandb_run_name}",
                                    group=f"{wandb_group_name}",
                                    config=wandb_config,
                                )

                            coll_seed_gen = torch.Generator()
                            coll_seed_gen.manual_seed(torch_seed)

                            per_class_eval_data_loader = torch.utils.data.DataLoader(per_class_eval_dataset, batch_size=128,
                                                                                    shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)

                            coll_seed_gen = torch.Generator()
                            coll_seed_gen.manual_seed(torch_seed)

                            per_class_tobe_analysed_data_loader = torch.utils.data.DataLoader(per_class_tobe_analysed_dataset, batch_size=128,
                                                                                            shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)
                            save_images_from_dataloader(to_be_analysed_adversarial_dataloader, classes,
                                                        postfix_folder_for_save='/adver/'+str(modestr)+"/", save_image_prefix=save_folder)

                            eval_orig_acc, _ = plain_evaluate_model(
                                net, per_class_eval_data_loader)
                            eval_adv_acc, frequency_pred = plain_evaluate_model(
                                net, per_class_tobe_analysed_data_loader, classes)

                            frequency_pred = (
                                frequency_pred / len(per_class_tobe_analysed_dataset))*100
                            frequency_pred = torch.round(frequency_pred)
                            print("frequency_pred", frequency_pred)

                            if(is_log_wandb):
                                wandb.log({"eval_orig_acc": eval_orig_acc,
                                        "eval_adv_acc": eval_adv_acc,
                                        "frequency_pred": frequency_pred})
                                wandb.finish()
                elif(exp_type == "AFTER_ATT_FREQ_BAND_ADV_ATTACK_PER_CLASS"):
                    allmodes=[["LTB","MTB","HTB"],
                        ["MTB","HTB"],
                        ["LTB","HTB"],
                        ["LTB","MTB"],
                        ["LTB"],
                        ["MTB"],
                        ["HTB"]]
                    
                    for cur_mode in allmodes:
                        modestr = ""
                        for ee in cur_mode:
                            modestr += ee +"_"
                        modestr = modestr[0:len(modestr)-1]

                        class_indx_to_visualize = [i for i in range(len(classes))]
                        if(direct_model_path is None):
                            each_save_postfix = "/aug_indx_{}".format(
                                current_aug_iter_num)
                            final_adv_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_adv_attack_on_train,update_on,rand_init,norm,use_ytrue,criterion)+str(each_save_postfix)
                        else:
                            final_adv_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_adv_attack_on_train,update_on,rand_init,norm,use_ytrue,criterion)
                        adv_save_path = model_and_data_save_prefix + \
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
                            adv_dataset = generate_adv_examples(
                                eval_loader, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, is_save_adv=True, save_path=adv_save_path,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue,lossfn=criterion)
                            print("adv_save_path:", adv_save_path)
                        if("fashion_mnist" in dataset):
                            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
                        elif("mnist" in dataset):
                            (X_train, y_train), (X_test, y_test) = mnist.load_data()
                        
                        _, _, _, _, X_test, y_test = preprocess_mnist_fmnist(X_train,y_train,X_test,y_test,ds_config,model_arch_type,verbose=1, is_split_validation=False)
                        to_be_analysed_adversarial_dataloader = torch.utils.data.DataLoader(
                            adv_dataset, shuffle=False, batch_size=128)
                        
                        (X_filt_test, y_test) = modify_bandpass_freq_get_dataset(to_be_analysed_adversarial_dataloader,cur_mode)
                        X_test = np.stack(X_test,axis=0)
                        X_filt_test ,X_test = torch.from_numpy(X_filt_test),torch.from_numpy(X_test)
                        X_dataset = torch.utils.data.TensorDataset(X_filt_test,X_test)
                        to_be_analysed_adversarial_dataloader = torch.utils.data.DataLoader(
                            X_dataset, shuffle=False, batch_size=256)
                        X_test = project_to_eps_inf_ball(to_be_analysed_adversarial_dataloader,eps)
                        
                        to_be_analysed_adversarial_dataloader = get_data_loader(
                            X_test, y_test, ds_config.batch_size, transforms=ds_config.test_transforms)

                        true_eval_dataset_per_class = true_segregation(
                            eval_loader, num_classes_trained_on)
                        true_tobe_analysed_dataset_per_class = true_segregation(
                            to_be_analysed_adversarial_dataloader, num_classes_trained_on)

                        for c_indx in class_indx_to_visualize:
                            class_label = classes[c_indx]
                            print(
                                "************************************************************ Class:", class_label)
                            per_class_eval_dataset = PerClassDataset(
                                true_eval_dataset_per_class[c_indx], c_indx)
                            per_class_tobe_analysed_dataset = PerClassDataset(
                                true_tobe_analysed_dataset_per_class[c_indx], c_indx)
                            wandb_group_name = "DS_"+str(dataset_str) + \
                                "_adv_attack_over_aug_"+str(model_arch_type_str)

                            if(is_log_wandb):
                                wandb_run_name = str(
                                    model_arch_type_str)+"_aug_it_"+str(current_aug_iter_num)+"adv_at_"+str(adv_attack_type)
                                wandb_config = get_wandb_config(exp_type, adv_attack_type, model_arch_type_str, dataset_str, is_adv_attack_on_train,
                                                                eps, number_of_adversarial_optimization_steps, eps_step_size, model_save_path, is_targetted, adv_target,update_on,rand_init,norm,use_ytrue)
                                wandb_config["aug_iter_num"] = current_aug_iter_num
                                wandb_config["class_label"] = class_label
                                wandb_config["c_indx"] = c_indx
                                wandb_config["freq_band"] = modestr
                                wandb_config["inner_criterion"] = criterion
                                if(pca_exp_percent is not None):
                                    wandb_config["pca_exp_percent"] = pca_exp_percent
                                    wandb_config["num_comp_pca"] = number_of_components_for_pca
                                if(wandb_config_additional_dict is not None):
                                    wandb_config.update(
                                        wandb_config_additional_dict)
                                wandb.init(
                                    project=f"{wand_project_name}",
                                    name=f"{wandb_run_name}",
                                    group=f"{wandb_group_name}",
                                    config=wandb_config,
                                )

                            coll_seed_gen = torch.Generator()
                            coll_seed_gen.manual_seed(torch_seed)

                            per_class_eval_data_loader = torch.utils.data.DataLoader(per_class_eval_dataset, batch_size=128,
                                                                                    shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)

                            coll_seed_gen = torch.Generator()
                            coll_seed_gen.manual_seed(torch_seed)

                            per_class_tobe_analysed_data_loader = torch.utils.data.DataLoader(per_class_tobe_analysed_dataset, batch_size=128,
                                                                                            shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)
                            save_images_from_dataloader(to_be_analysed_adversarial_dataloader, classes,
                                                        postfix_folder_for_save='/adver/'+str(modestr)+"/", save_image_prefix=save_folder)

                            eval_orig_acc, _ = plain_evaluate_model(
                                net, per_class_eval_data_loader)
                            eval_adv_acc, frequency_pred = plain_evaluate_model(
                                net, per_class_tobe_analysed_data_loader, classes)

                            frequency_pred = (
                                frequency_pred / len(per_class_tobe_analysed_dataset))*100
                            frequency_pred = torch.round(frequency_pred)
                            print("frequency_pred", frequency_pred)

                            if(is_log_wandb):
                                wandb.log({"eval_orig_acc": eval_orig_acc,
                                        "eval_adv_acc": eval_adv_acc,
                                        "frequency_pred": frequency_pred})
                                wandb.finish()
                
    print("Finished execution!!!")

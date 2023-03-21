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
from structure.dlgn_conv_config_structure import DatasetConfig
from torchvision import transforms
from torch.autograd import Variable
from configs.dlgn_conv_config import HardRelu

from conv4_models import get_model_instance, get_model_save_path, get_model_instance_from_dataset, get_img_size
from visualization import run_visualization_on_config
from utils.weight_utils import get_gating_layer_weights
from raw_weight_analysis import convert_list_tensor_to_numpy
from utils.APR import APRecombination, mix_data
from utils.apr_evaluator import apr_evaluate_model


def evaluate_model(net, dataloader, num_classes_trained_on=None):
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    frequency_pred = None
    if(num_classes_trained_on is not None):
        frequency_pred = torch.zeros(num_classes_trained_on)
        frequency_pred = frequency_pred.to(device, non_blocking=True)
        all_classes = torch.arange(0, num_classes_trained_on)
        all_classes = all_classes.to(device, non_blocking=True)
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(
                device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if(num_classes_trained_on is not None):
                temp = torch.cat((predicted.float(), all_classes))
                temp = temp.to(device)
                frequency_pred += torch.histc(temp, num_classes_trained_on) - 1

    return 100. * correct / total, frequency_pred


def get_normalize(inp):
    if(inp.size()[1] == 3):
        normalize = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [
                                 0.2023, 0.1994, 0.2010]),
        ])
        return normalize(inp)

    return inp


def apr_train_model(net, type_of_APR, trainloader, testloader, epochs, criterion, optimizer, final_model_save_path, wand_project_name=None, apr_mix_prob=0.6, train_on_phase_labels=True):

    is_log_wandb = not(wand_project_name is None)
    net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        orig_correct = 1
        orig_total = 1
        overall_total = 1
        total_count = 1

        if("APRS" in type_of_APR):
            aprs_used_count = 1
            aprs_phase_aug_used = 1
        if(type_of_APR == "APRP" or type_of_APR == "APRSP"):
            ph_label_correct = 1
            am_label_correct = 1
            switch_total = 1

        running_loss = 0.0
        loader = tqdm.tqdm(trainloader, desc='Training')
        for batch_idx, data in enumerate(loader, 0):
            begin_time = time.time()
            loader.set_description(f"Epoch {epoch+1}")
            # get the inputs; data is a list of [inputs, labels]
            if("APRS" in type_of_APR):
                inputs, labels, phase_used_flag, amp_used_flag, x_aug, x_orig, orig_img = data
                with torch.no_grad():
                    aprs_used_count += torch.numel(phase_used_flag) - \
                        torch.count_nonzero(phase_used_flag).item()
                    # phase_used_flag is -1 when x is used and +1 when x_aug is used
                    aprs_phase_aug_used += torch.count_nonzero(
                        HardRelu()(phase_used_flag)).item()
            else:
                inputs, labels = data
            total_count += inputs.size(0)

            inputs, labels = inputs.to(
                device, non_blocking=True), labels.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            dev = inputs.get_device()

            if(type_of_APR == "APRP" or type_of_APR == "APRSP"):
                mix_train_labels = labels
                amp_labels = None
                phase_labels = None
                inputs_mix, amp_label_indx, _ = mix_data(
                    inputs, prob=apr_mix_prob)
                if(amp_label_indx is not None):
                    amp_labels = labels[amp_label_indx]
                    phase_labels = labels
                    if(not train_on_phase_labels):
                        mix_train_labels = amp_labels
                inputs_mix = Variable(inputs_mix)
                inputs, inputs_mix = get_normalize(
                    inputs), get_normalize(inputs_mix)
                inputs = torch.cat([inputs, inputs_mix], 0)
            else:
                inputs = get_normalize(inputs)

            inputs = inputs.to(device=dev, non_blocking=True)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            orig_size = batch_size
            if(type_of_APR == "APRP" or type_of_APR == "APRSP"):
                loss = criterion(outputs[:batch_size], labels) + \
                    criterion(outputs[batch_size:], mix_train_labels)
                if(amp_labels is None):
                    orig_size = 2*batch_size
                    labels = torch.cat([labels, labels], 0)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            overall_total += outputs.size(0)
            orig_total += labels.size(0)
            orig_correct += (predicted[:orig_size] == labels).sum().item()
            if(type_of_APR == "APRP" or type_of_APR == "APRSP"):
                if(amp_labels is not None):
                    am_label_correct += (predicted[batch_size:]
                                         == amp_labels).sum().item()
                    ph_label_correct += (predicted[batch_size:]
                                         == phase_labels).sum().item()
                    switch_total += phase_labels.size(0)

            running_loss += loss.item()

            cur_time = time.time()
            step_time = cur_time - begin_time
            if(type_of_APR == "APRP"):
                loader.set_postfix(total_samples=overall_total, train_loss=running_loss/(batch_idx+1), noswtch_per=100.*orig_total/overall_total, noswtch_ratio="{}/{}".format(orig_total, overall_total),
                                   orig_tr_acc=100.*orig_correct/orig_total, orgc_ratio="{}/{}".format(orig_correct, orig_total),
                                   ph_tr_acc=100.*ph_label_correct/switch_total, ph_acratio="{}/{}".format(ph_label_correct, switch_total),
                                   amp_tr_acc=100.*am_label_correct/switch_total, amp_acratio="{}/{}".format(am_label_correct, switch_total), stime=format_time(step_time))
            elif(type_of_APR == "APRSP"):
                loader.set_postfix(total_samples=overall_total, train_loss=running_loss/(batch_idx+1), aprs_use=100.*aprs_used_count/total_count, aprs_phase_use=100.*aprs_phase_aug_used/total_count,
                                   noswtch_per=100.*orig_total/overall_total, noswtch_ratio="{}/{}".format(orig_total, overall_total),
                                   orig_tr_acc=100.*orig_correct/orig_total, orgc_ratio="{}/{}".format(orig_correct, orig_total),
                                   ph_tr_acc=100.*ph_label_correct/switch_total, ph_acratio="{}/{}".format(ph_label_correct, switch_total),
                                   amp_tr_acc=100.*am_label_correct/switch_total, amp_acratio="{}/{}".format(am_label_correct, switch_total), stime=format_time(step_time))
            elif(type_of_APR == "APRS"):
                loader.set_postfix(total_samples=overall_total, train_loss=running_loss/(batch_idx+1), aprs_use=100.*aprs_used_count/total_count, aprs_phase_use=100.*aprs_phase_aug_used/total_count,
                                   orig_tr_acc=100.*orig_correct/orig_total, orgc_ratio="{}/{}".format(orig_correct, orig_total),
                                   stime=format_time(step_time))

        res_dict = apr_evaluate_model(net, testloader, type_of_APR)
        print("Test evaluation results:", res_dict)
        if(is_log_wandb):
            if(type_of_APR == "APRP"):
                wandb_update_dict = {"noswtch_per": 100.*orig_total/overall_total, "orig_samples_count": total_count, "total_samples": overall_total,
                                     "orig_tr_acc": 100.*orig_correct/orig_total, "ph_tr_acc": 100.*ph_label_correct/switch_total, "amp_tr_acc": 100.*am_label_correct/switch_total}
            elif(type_of_APR == "APRSP"):
                wandb_update_dict = {"aprs_use": 100.*aprs_used_count/total_count, "orig_samples_count": total_count, "aprs_phase_use": 100.*aprs_phase_aug_used/total_count,
                                     "noswtch_per": 100.*orig_total/overall_total, "total_samples": overall_total,
                                     "orig_tr_acc": 100.*orig_correct/orig_total, "ph_tr_acc": 100.*ph_label_correct/switch_total, "amp_tr_acc": 100.*am_label_correct/switch_total}
            elif(type_of_APR == "APRS"):
                wandb_update_dict = {"aprs_use": 100.*aprs_used_count/total_count, "orig_samples_count": total_count, "total_samples": overall_total, "aprs_phase_use": 100.*aprs_phase_aug_used/total_count,
                                     "orig_tr_acc": 100.*orig_correct/orig_total}
            wandb_update_dict.update(res_dict)
            wandb.log(wandb_update_dict)

        per_epoch_model_save_path = final_model_save_path.replace(
            "_dir.pt", "")
        if not os.path.exists(per_epoch_model_save_path):
            os.makedirs(per_epoch_model_save_path)
        per_epoch_model_save_path += "/epoch_{}_dir.pt".format(epoch)
        if(epoch % 7 == 0):
            torch.save(net, per_epoch_model_save_path)

    torch.save(net, final_model_save_path)
    return net


def train_model(net, trainloader, testloader, epochs, criterion, optimizer, final_model_save_path, wand_project_name=None):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_str == 'cuda':
        if(torch.cuda.device_count() > 1):
            print("Parallelizing model")
            net = torch.nn.DataParallel(net)

        cudnn.benchmark = True

    is_log_wandb = not(wand_project_name is None)
    best_test_acc = 0
    net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        correct = 0
        total = 0

        running_loss = 0.0
        loader = tqdm.tqdm(trainloader, desc='Training')
        for batch_idx, data in enumerate(loader, 0):
            begin_time = time.time()
            loader.set_description(f"Epoch {epoch+1}")
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(
                device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

            cur_time = time.time()
            step_time = cur_time - begin_time
            loader.set_postfix(train_loss=running_loss/(batch_idx+1),
                               train_acc=100.*correct/total, ratio="{}/{}".format(correct, total), stime=format_time(step_time))

        train_acc = 100. * correct/total
        test_acc, _ = evaluate_model(
            net, testloader)
        if(is_log_wandb):
            wandb.log({"train_acc": train_acc, "test_acc": test_acc})

        print("Test_acc: ", test_acc)
        per_epoch_model_save_path = final_model_save_path.replace(
            "_dir.pt", "")
        if not os.path.exists(per_epoch_model_save_path):
            os.makedirs(per_epoch_model_save_path)
        per_epoch_model_save_path += "/epoch_{}_dir.pt".format(epoch)
        if(epoch % 7 == 0):
            torch.save(net, per_epoch_model_save_path)
        if(test_acc >= best_test_acc):
            best_test_acc = test_acc

    torch.save(net, final_model_save_path)
    print('Finished Training: Best saved model test acc is:', best_test_acc)
    return best_test_acc, net


def get_wandb_config(exp_type, classes, model_arch_type, dataset, is_template_image_on_train,
                     is_class_segregation_on_ground_truth, template_initial_image_type,
                     template_image_calculation_batch_size, template_loss_type, torch_seed, number_of_image_optimization_steps,
                     plot_iteration_interval=None, number_of_batch_to_collect=None, collect_threshold=None):

    wandb_config = dict()
    wandb_config["classes"] = classes
    wandb_config["model_arch_type"] = model_arch_type
    wandb_config["dataset"] = dataset
    wandb_config["is_template_image_on_train"] = is_template_image_on_train
    wandb_config["is_class_segregation_on_ground_truth"] = is_class_segregation_on_ground_truth
    wandb_config["template_initial_image_type"] = template_initial_image_type
    wandb_config["template_image_calculation_batch_size"] = template_image_calculation_batch_size
    wandb_config["template_loss_type"] = template_loss_type
    wandb_config["torch_seed"] = torch_seed
    wandb_config["number_of_image_optimization_steps"] = number_of_image_optimization_steps
    wandb_config["exp_type"] = exp_type
    if(not(plot_iteration_interval is None)):
        wandb_config["plot_iteration_interval"] = plot_iteration_interval
    if(not(number_of_batch_to_collect is None)):
        wandb_config["number_of_batch_to_collect"] = number_of_batch_to_collect
    if(not(collect_threshold is None)):
        wandb_config["collect_threshold"] = collect_threshold

    return wandb_config


class CustomAugmentDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_x, list_of_y):
        self.list_of_x = list_of_x
        self.list_of_y = list_of_y

    def __len__(self):
        return len(self.list_of_x)

    def __getitem__(self, idx):
        x = self.list_of_x[idx]
        y = self.list_of_y[idx]

        return x, y


if __name__ == '__main__':
    # fashion_mnist , mnist , cifar10
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net , conv4_deep_gated_net_n16_small ,
    # conv4_deep_gated_net_with_actual_inp_in_wt_net , conv4_deep_gated_net_with_actual_inp_randomly_changed_in_wt_net
    # conv4_deep_gated_net_with_random_ones_in_wt_net , masked_conv4_dlgn , masked_conv4_dlgn_n16_small , fc_dnn , fc_dlgn , fc_dgn
    model_arch_type = 'conv4_dlgn_n16_small'
    # iterative_augmenting , nil , APR_exps
    scheme_type = 'APR_exps'
    # scheme_type = ''
    batch_size = 32

    # torch_seed = ""
    torch_seed = 2022

    wand_project_name = None
    wand_project_name = "APR_experiments"
    # wand_project_name = "common_model_init_exps"
    # wand_project_name = "V2_template_visualisation_augmentation"

    # Percentage of information retention during PCA (values between 0-1)
    pca_exp_percent = None
    # pca_exp_percent = 0.50

    # None means that train on all classes
    list_of_classes_to_train_on = None
    # list_of_classes_to_train_on = [3, 8]

    train_transforms = None
    is_normalize_data = True

    if(scheme_type == "APR_exps"):
        # APRP ,APRS, APRSP
        type_of_APR = "APRSP"
        aprp_mix_prob = 0.6
        train_on_phase_labels = True
        aprs_prob_threshold = 0.7
        aprs_mix_prob = 0.5

        if("APRS" in type_of_APR):
            img_size = get_img_size(dataset)[1]
            is_normalize_data = False
            train_transforms = transforms.RandomApply(
                [APRecombination(img_size=img_size, prob_threshold=aprs_prob_threshold)], p=1.0)

    if(dataset == "cifar10"):
        inp_channel = 3
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = len(classes)

        data_config = DatasetConfig(
            'cifar10', is_normalize_data=is_normalize_data, valid_split_size=0.1, batch_size=batch_size, list_of_classes=list_of_classes_to_train_on, train_transforms=train_transforms)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            data_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    elif(dataset == "mnist"):
        inp_channel = 1
        classes = [str(i) for i in range(0, 10)]
        num_classes = len(classes)

        data_config = DatasetConfig(
            'mnist', is_normalize_data=is_normalize_data, valid_split_size=0.1, batch_size=batch_size, list_of_classes=list_of_classes_to_train_on, train_transforms=train_transforms)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            data_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    elif(dataset == "fashion_mnist"):
        inp_channel = 1
        classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot')
        num_classes = len(classes)

        data_config = DatasetConfig(
            'fashion_mnist', is_normalize_data=is_normalize_data, valid_split_size=0.1, batch_size=batch_size, list_of_classes=list_of_classes_to_train_on, train_transforms=train_transforms)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            data_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    print("Training over "+dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes_trained_on = num_classes
    dataset_str = dataset

    list_of_classes_to_train_on_str = ""
    if(list_of_classes_to_train_on is not None):
        for each_class_to_train_on in list_of_classes_to_train_on:
            list_of_classes_to_train_on_str += \
                str(each_class_to_train_on)+"_"
        list_of_classes_to_train_on_str = list_of_classes_to_train_on_str[0:-1]
        dataset_str += "_"+str(list_of_classes_to_train_on_str)
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

    final_model_save_path = get_model_save_path(
        model_arch_type_str, dataset, torch_seed, list_of_classes_to_train_on_str)

    # list_of_weights, list_of_bias = get_gating_layer_weights(net)

    # list_of_weights = convert_list_tensor_to_numpy(list_of_weights)
    # for i in range(len(list_of_weights)):
    #     current_weight_np = list_of_weights[i]
    #     print("ind:{}=>{}".format(i, current_weight_np.shape))
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    if device_str == 'cuda':
        if(torch.cuda.device_count() > 1):
            print("Parallelizing model")
            net = torch.nn.DataParallel(net).cuda()
        else:
            net = net.to(device)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)
    lr = 3e-4
    optimizer = optim.Adam(net.parameters(), lr=lr)
    epochs = 32

    if(scheme_type == 'iterative_augmenting'):
        # If False, then on test
        is_template_image_on_train = True
        # If False, then segregation is over model prediction
        is_class_segregation_on_ground_truth = True
        template_initial_image_type = 'zero_init_image'
        template_image_calculation_batch_size = 1
        # MSE_LOSS , MSE_TEMP_LOSS_MIXED , ENTR_TEMP_LOSS , CCE_TEMP_LOSS_MIXED , TEMP_LOSS , CCE_ENTR_TEMP_LOSS_MIXED , TEMP_ACT_ONLY_LOSS
        # CCE_TEMP_ACT_ONLY_LOSS_MIXED
        template_loss_type = "TEMP_LOSS"
        number_of_batch_to_collect = 1
        # wand_project_name = "cifar10_all_images_based_template_visualizations"
        # wand_project_name = "template_images_visualization-test"
        wand_project_name = "V2_template_visualisation_augmentation"
        # wand_project_name = None
        wandb_group_name = "DS_"+str(dataset_str) + \
            "_template_vis_aug_"+str(model_arch_type_str)
        is_split_validation = False
        valid_split_size = 0.1

        number_of_image_optimization_steps = 161
        # GENERATE_ALL_FINAL_TEMPLATE_IMAGES
        exp_type = "GENERATE_ALL_FINAL_TEMPLATE_IMAGES"
        # Changing this parameter for the augmenting process acts merely as a different initialisation of model weight provided template_batch_size=1
        collect_threshold = 0.73
        entropy_calculation_batch_size = 64
        number_of_batches_to_calculate_entropy_on = None
        visualization_version = "V2"

        tmp_image_over_what_str = 'test'
        if(is_template_image_on_train):
            tmp_image_over_what_str = 'train'

        seg_over_what_str = 'MP'
        if(is_class_segregation_on_ground_truth):
            seg_over_what_str = 'GT'

        root_save_prefix = "root/" + \
            str(visualization_version)+"AUG_RECONS_SAVE/"
        model_and_data_save_prefix = "root/model/save/" + \
            str(dataset)+"/"+str(visualization_version)+"_iterative_augmenting/DS_"+str(dataset_str)+"/MT_"+str(model_arch_type_str)+"_ET_"+str(exp_type)+"/_COLL_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
                seg_over_what_str)+"/TMP_COLL_BS_"+str(template_image_calculation_batch_size)+"/TMP_LOSS_TP_"+str(template_loss_type)+"/TMP_INIT_"+str(template_initial_image_type)+"/_torch_seed_"+str(torch_seed)+"_c_thres_"+str(collect_threshold)+"/"

        # number_of_augment_iterations = 5
        # epochs_in_each_augment_iteration = [32, 10, 10, 10, 5]

        number_of_augment_iterations = 2
        epochs_in_each_augment_iteration = [32, 10]

        current_augmented_x_train = None
        current_augmented_y_train = None
        if(not(wand_project_name is None)):
            wandb.login()

        for i, inp_data in enumerate(trainloader):
            input_x, input_y = inp_data
            if(current_augmented_x_train is None):
                # print("input_x shape:",
                #       input_x.shape)
                current_augmented_x_train = input_x
            else:
                # print("input_x shape:",
                #       input_x.shape)
                current_augmented_x_train = np.vstack(
                    (current_augmented_x_train, input_x))

            # print("current_augmented_x_train shape:",
            #       current_augmented_x_train.shape)

            if(current_augmented_y_train is None):
                # print("input_y shape:",
                #       input_y.shape)
                current_augmented_y_train = input_y
            else:
                # print("input_y shape:",
                #       input_y.shape)
                current_augmented_y_train = np.concatenate(
                    (current_augmented_y_train, input_y))

            # print("current_augmented_y_train shape:",
            #       current_augmented_y_train.shape)

        augment_trainloader = trainloader
        is_log_wandb = not(wand_project_name is None)
        for current_aug_iter_num in range(1, number_of_augment_iterations+1):
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=3e-4)
            overall_output_template_list = None
            overall_y_label_list = None

            final_model_save_path = model_and_data_save_prefix+'aug_conv4_dlgn_iter_{}_dir.pt'.format(
                current_aug_iter_num)
            isExist = os.path.exists(final_model_save_path)
            if not os.path.exists(model_and_data_save_prefix):
                os.makedirs(model_and_data_save_prefix)
            current_epoch = epochs_in_each_augment_iteration[current_aug_iter_num-1]
            print("current_epoch", current_epoch)
            if(not(isExist)):
                print(
                    "Started training model for augment iteration:", current_aug_iter_num)
                print("net", net)
                if(is_log_wandb):
                    wandb_run_name = str(
                        model_arch_type_str)+"_aug_iteration_"+str(current_aug_iter_num)
                    experiment_type = 'TRAIN'+str(exp_type)
                    wandb_config = get_wandb_config(experiment_type, classes, model_arch_type_str, dataset_str, is_template_image_on_train,
                                                    is_class_segregation_on_ground_truth, template_initial_image_type,
                                                    template_image_calculation_batch_size, template_loss_type, torch_seed, number_of_image_optimization_steps,
                                                    collect_threshold=collect_threshold, number_of_batch_to_collect=number_of_batch_to_collect)
                    wandb_config["current_aug_iter_num"] = current_aug_iter_num
                    wandb_config["epochs"] = current_epoch
                    if(pca_exp_percent is not None):
                        wandb_config["pca_exp_percent"] = pca_exp_percent
                        wandb_config["num_comp_pca"] = number_of_components_for_pca
                    wandb.init(
                        project=f"{wand_project_name}",
                        name=f"{wandb_run_name}",
                        group=f"{wandb_group_name}",
                        config=wandb_config,
                    )

                optimizer = optim.Adam(net.parameters(), lr=3e-4)
                best_test_acc, net = train_model(net,
                                                 augment_trainloader, testloader, current_epoch, criterion, optimizer, final_model_save_path,
                                                 wand_project_name)
                net = torch.load(final_model_save_path)
                if(is_log_wandb):
                    wandb.log({"best_test_acc": best_test_acc})
                    wandb.finish()
                print(
                    "Completed training model for augment iteration:", current_aug_iter_num)
            else:
                net = torch.load(final_model_save_path)
                print(
                    "Loaded previously trained model for augment iteration:{} from path :{}".format(current_aug_iter_num, final_model_save_path))

            optimizer = optim.Adam(net.parameters(), lr=3e-4)

            final_postfix_for_save = "aug_indx_{}/".format(
                current_aug_iter_num)
            final_postfix_for_overall_save = "aug_indx_{}_perc_overall_template/".format(
                current_aug_iter_num)
            search_path = model_and_data_save_prefix + final_postfix_for_save

            class_indx_to_visualize = []

            output_template_list_per_class = [None] * num_classes
            y_label_list_per_class = [None] * num_classes
            for i in range(num_classes):
                output_template_list_per_class[i] = []
                y_label_list_per_class[i] = []

            for current_c_indx in range(num_classes):
                class_label = classes[current_c_indx]
                np_save_filename = search_path + \
                    '/class_'+str(class_label) + '.npy'
                is_current_aug_available = os.path.exists(np_save_filename)
                if(is_current_aug_available):
                    with open(np_save_filename, 'rb') as file:
                        npzfile = np.load(np_save_filename)
                        each_class_output_template_list = npzfile['x']
                        current_y_s = npzfile['y']
                        output_template_list_per_class[current_c_indx] = each_class_output_template_list
                        y_label_list_per_class[current_c_indx] = current_y_s
                else:
                    class_indx_to_visualize.append(current_c_indx)

            print("class_indx_to_visualize", class_indx_to_visualize)
            for current_c_indx in class_indx_to_visualize:
                current_class_indx_to_visualize = [current_c_indx]
                output_template_list = run_visualization_on_config(dataset, model_arch_type_str, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
                                                                   template_image_calculation_batch_size, template_loss_type, number_of_batch_to_collect, wand_project_name, is_split_validation,
                                                                   valid_split_size, torch_seed, number_of_image_optimization_steps, wandb_group_name, exp_type, collect_threshold,
                                                                   entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on, root_save_prefix, final_postfix_for_save,
                                                                   custom_model=net, custom_data_loader=(trainloader, testloader), class_indx_to_visualize=current_class_indx_to_visualize, vis_version=visualization_version)
                # TO get one template image per class
                run_visualization_on_config(dataset, model_arch_type_str, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
                                            template_image_calculation_batch_size=32, template_loss_type=template_loss_type, number_of_batch_to_collect=None,
                                            wand_project_name=wand_project_name, is_split_validation=is_split_validation, valid_split_size=valid_split_size,
                                            torch_seed=torch_seed, number_of_image_optimization_steps=number_of_image_optimization_steps, wandb_group_name=wandb_group_name,
                                            exp_type="GENERATE_TEMPLATE_IMAGES", collect_threshold=collect_threshold, entropy_calculation_batch_size=entropy_calculation_batch_size,
                                            number_of_batches_to_calculate_entropy_on=number_of_batches_to_calculate_entropy_on, root_save_prefix=root_save_prefix,
                                            final_postfix_for_save=final_postfix_for_overall_save, custom_model=net,
                                            custom_data_loader=(trainloader, testloader), class_indx_to_visualize=current_class_indx_to_visualize, vis_version=visualization_version)

                class_label = classes[current_c_indx]
                each_class_output_template_list = output_template_list[current_c_indx]
                if(not(each_class_output_template_list is None)):
                    current_y_s = np.full(
                        each_class_output_template_list.shape[0], current_c_indx)

                    if not os.path.exists(search_path):
                        os.makedirs(search_path)

                    np_save_filename = search_path + \
                        '/class_'+str(class_label) + '.npy'
                    with open(np_save_filename, 'wb') as file:
                        np.savez(
                            file, x=each_class_output_template_list, y=current_y_s)
                    output_template_list_per_class[current_c_indx] = each_class_output_template_list
                    y_label_list_per_class[current_c_indx] = current_y_s

            output_template_list_per_class = np.array(
                output_template_list_per_class)
            y_label_list_per_class = np.array(
                y_label_list_per_class)

            for indx in range(len(output_template_list_per_class)):
                current_temp_list = output_template_list_per_class[indx]
                current_y_label_list = y_label_list_per_class[indx]

                if(overall_output_template_list is None):
                    overall_output_template_list = current_temp_list
                else:
                    overall_output_template_list = np.vstack(
                        (overall_output_template_list, current_temp_list))

                if(overall_y_label_list is None):
                    overall_y_label_list = current_y_label_list
                else:
                    overall_y_label_list = np.concatenate(
                        (overall_y_label_list, current_y_label_list))

            current_augmented_x_train = np.vstack(
                (current_augmented_x_train, overall_output_template_list))
            current_augmented_y_train = np.concatenate(
                (current_augmented_y_train, overall_y_label_list))

            print("current_augmented_x_train shape:",
                  current_augmented_x_train.shape)
            print("current_augmented_y_train shape:",
                  current_augmented_y_train.shape)
            current_augment_dataset = CustomAugmentDataset(
                current_augmented_x_train, current_augmented_y_train)

            augment_trainloader = torch.utils.data.DataLoader(current_augment_dataset, batch_size=batch_size,
                                                              shuffle=True)

    elif(scheme_type == 'APR_exps'):
        aprp_postfix = ""
        if("APRS" in type_of_APR):
            aprp_postfix += "/ARPS_PROB_THRES_" + \
                str(aprs_prob_threshold)+"/ARPS_MPROB_"+str(aprs_mix_prob)+"/"
        if(type_of_APR == "APRP" or type_of_APR == "APRSP"):
            aprp_postfix += "/APRP_MPROB_" + \
                str(aprp_mix_prob)+"/TR_PHASE_"+str(train_on_phase_labels)+"/"

        final_model_save_path = final_model_save_path.replace(
            "CLEAN_TRAINING", "APR_TRAINING/TYP_"+str(type_of_APR)+"/"+aprp_postfix)
        print("final_model_save_path: ", final_model_save_path)
        is_log_wandb = not(wand_project_name is None)
        if(is_log_wandb):
            wandb_group_name = "DS_"+str(dataset_str) + \
                "_MT_"+str(model_arch_type_str)+"_SEED_" + \
                str(torch_seed)+"_APR_"+str(type_of_APR) + \
                aprp_postfix.replace("/", "_")
            wandb_run_name = "MT_" + \
                str(model_arch_type_str)+"/SEED_"+str(torch_seed)+"/EP_"+str(epochs)+"/LR_" + \
                str(lr)+"/OPT_"+str(optimizer)+"/LOSS_TYPE_" + \
                str(criterion)+"/BS_"+str(batch_size) + \
                "/SCH_TYP_"+str(scheme_type)
            wandb_run_name = wandb_run_name.replace("/", "")

            wandb_config = dict()
            wandb_config["dataset"] = dataset_str
            wandb_config["model_arch_type"] = model_arch_type_str
            wandb_config["torch_seed"] = torch_seed
            wandb_config["scheme_type"] = scheme_type
            wandb_config["final_model_save_path"] = final_model_save_path
            wandb_config["epochs"] = epochs
            wandb_config["optimizer"] = optimizer
            wandb_config["criterion"] = criterion
            wandb_config["type_of_APR"] = type_of_APR
            wandb_config["lr"] = lr
            wandb_config["batch_size"] = batch_size
            if(pca_exp_percent is not None):
                wandb_config["pca_exp_percent"] = pca_exp_percent
                wandb_config["num_comp_pca"] = number_of_components_for_pca
            if("APRS" in type_of_APR):
                wandb_config["aprs_prob_thres"] = aprs_prob_threshold
                wandb_config["aprs_mix_prob"] = aprs_mix_prob
            if(type_of_APR == "APRP" or type_of_APR == "APRSP"):
                wandb_config["aprp_mix_prob"] = aprp_mix_prob
                wandb_config["is_train_on_phase"] = train_on_phase_labels

            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                group=f"{wandb_group_name}",
                config=wandb_config,
            )

        model_save_folder = final_model_save_path[0:final_model_save_path.rfind(
            "/")+1]
        if not os.path.exists(model_save_folder):
            os.makedirs(model_save_folder)

        net = apr_train_model(net, type_of_APR, trainloader, testloader, epochs, criterion, optimizer,
                              final_model_save_path, wand_project_name, aprp_mix_prob, train_on_phase_labels)

        if(is_log_wandb):
            wandb.finish()

    else:
        is_log_wandb = not(wand_project_name is None)
        if(is_log_wandb):
            wandb_group_name = "DS_"+str(dataset_str) + \
                "_MT_"+str(model_arch_type_str)+"_SEED_"+str(torch_seed)
            wandb_run_name = "MT_" + \
                str(model_arch_type_str)+"/SEED_"+str(torch_seed)+"/EP_"+str(epochs)+"/LR_" + \
                str(lr)+"/OPT_"+str(optimizer)+"/LOSS_TYPE_" + \
                str(criterion)+"/BS_"+str(batch_size) + \
                "/SCH_TYP_"+str(scheme_type)
            wandb_run_name = wandb_run_name.replace("/", "")

            wandb_config = dict()
            wandb_config["dataset"] = dataset_str
            wandb_config["model_arch_type"] = model_arch_type_str
            wandb_config["torch_seed"] = torch_seed
            wandb_config["scheme_type"] = scheme_type
            wandb_config["final_model_save_path"] = final_model_save_path
            wandb_config["epochs"] = epochs
            wandb_config["optimizer"] = optimizer
            wandb_config["criterion"] = criterion
            wandb_config["lr"] = lr
            wandb_config["batch_size"] = batch_size
            if(pca_exp_percent is not None):
                wandb_config["pca_exp_percent"] = pca_exp_percent
                wandb_config["num_comp_pca"] = number_of_components_for_pca

            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                group=f"{wandb_group_name}",
                config=wandb_config,
            )

        model_save_folder = final_model_save_path[0:final_model_save_path.rfind(
            "/")+1]
        if not os.path.exists(model_save_folder):
            os.makedirs(model_save_folder)

        best_test_acc, net = train_model(net,
                                         trainloader, testloader, epochs, criterion, optimizer, final_model_save_path, wand_project_name)
        if(is_log_wandb):
            wandb.log({"best_test_acc": best_test_acc})
            wandb.finish()

    print("Finished execution!!!")

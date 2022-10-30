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
from structure.dlgn_conv_config_structure import DatasetConfig

from conv4_models import Plain_CONV4_Net, Conv4_DLGN_Net, Conv4_DLGN_Net_N16_Small
from visualization import run_visualization_on_config


def evaluate_model(net, dataloader):
    correct = 0
    total = 0
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

    return 100. * correct / total


def train_model(net, trainloader, testloader, epochs, criterion, optimizer, final_model_save_path, wand_project_name=None):
    is_log_wandb = not(wand_project_name is None)
    best_test_acc = 0
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
        test_acc = evaluate_model(net, testloader)
        if(is_log_wandb):
            wandb.log({"train_acc": train_acc, "test_acc": test_acc})

        print("Test_acc: ", test_acc)
        per_epoch_model_save_path = final_model_save_path.replace(
            "_dir.pt", "")
        if not os.path.exists(per_epoch_model_save_path):
            os.makedirs(per_epoch_model_save_path)
        per_epoch_model_save_path += "/epoch_{}_dir.pt".format(epoch)
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
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small
    model_arch_type = 'conv4_dlgn'
    scheme_type = 'iterative_augmenting'
    # scheme_type = ''
    batch_size = 32

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
    if(model_arch_type == 'plain_pure_conv4_dnn'):
        net = Plain_CONV4_Net(inp_channel)
        final_model_save_path = 'root/model/save/' + \
            str(dataset)+'/plain_pure_conv4_dnn_dir.pt'
    elif(model_arch_type == 'conv4_dlgn'):
        net = Conv4_DLGN_Net(inp_channel)
        final_model_save_path = 'root/model/save/' + \
            str(dataset)+'/conv4_dlgn_dir.pt'
    elif(model_arch_type == 'conv4_dlgn_n16_small'):
        net = Conv4_DLGN_Net_N16_Small(inp_channel)
        final_model_save_path = 'root/model/save/' + \
            str(dataset)+'/conv4_dlgn_n16_small_dir.pt'

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    epochs = 32

    if(scheme_type == 'iterative_augmenting'):
        dataset = 'mnist'
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
        # wand_project_name = "test_template_visualisation_augmentation"
        wand_project_name = None
        wandb_group_name = "DS_"+str(dataset) + \
            "_template_vis_aug_"+str(model_arch_type)
        is_split_validation = False
        valid_split_size = 0.1
        torch_seed = 2022
        number_of_image_optimization_steps = 161
        # GENERATE_ALL_FINAL_TEMPLATE_IMAGES
        exp_type = "GENERATE_ALL_FINAL_TEMPLATE_IMAGES"
        collect_threshold = 0.95
        entropy_calculation_batch_size = 64
        number_of_batches_to_calculate_entropy_on = None

        tmp_image_over_what_str = 'test'
        if(is_template_image_on_train):
            tmp_image_over_what_str = 'train'

        seg_over_what_str = 'MP'
        if(is_class_segregation_on_ground_truth):
            seg_over_what_str = 'GT'

        root_save_prefix = "root/AUG_RECONS_SAVE/"
        model_and_data_save_prefix = "root/model/save/" + \
            str(dataset)+"/iterative_augmenting/DS_"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/_COLL_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
                seg_over_what_str)+"/TMP_COLL_BS_"+str(template_image_calculation_batch_size)+"/TMP_LOSS_TP_"+str(template_loss_type)+"/TMP_INIT_"+str(template_initial_image_type)+"/_torch_seed_"+str(torch_seed)+"_c_thres_"+str(collect_threshold)+"/"

        number_of_augment_iterations = 5
        epochs_in_each_augment_iteration = [32, 10, 10, 10, 5]

        # number_of_augment_iterations = 3
        # epochs_in_each_augment_iteration = [5, 2, 2]

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
                        model_arch_type)+"_aug_iteration_"+str(current_aug_iter_num)
                    experiment_type = 'TRAIN'+str(exp_type)
                    wandb_config = get_wandb_config(experiment_type, classes, model_arch_type, dataset, is_template_image_on_train,
                                                    is_class_segregation_on_ground_truth, template_initial_image_type,
                                                    template_image_calculation_batch_size, template_loss_type, torch_seed, number_of_image_optimization_steps,
                                                    collect_threshold=collect_threshold, number_of_batch_to_collect=number_of_batch_to_collect)
                    wandb_config["current_aug_iter_num"] = current_aug_iter_num
                    wandb_config["epochs"] = current_epoch
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
            output_template_list = run_visualization_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
                                                               template_image_calculation_batch_size, template_loss_type, number_of_batch_to_collect, wand_project_name, is_split_validation,
                                                               valid_split_size, torch_seed, number_of_image_optimization_steps, wandb_group_name, exp_type, collect_threshold,
                                                               entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on, root_save_prefix, final_postfix_for_save,
                                                               custom_model=net, custom_data_loader=(trainloader, testloader), class_indx_to_visualize=class_indx_to_visualize)
            # TO get one template image per class
            run_visualization_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
                                        template_image_calculation_batch_size=32, template_loss_type=template_loss_type, number_of_batch_to_collect=None,
                                        wand_project_name=wand_project_name, is_split_validation=is_split_validation, valid_split_size=valid_split_size,
                                        torch_seed=torch_seed, number_of_image_optimization_steps=number_of_image_optimization_steps, wandb_group_name=wandb_group_name,
                                        exp_type="GENERATE_TEMPLATE_IMAGES", collect_threshold=collect_threshold, entropy_calculation_batch_size=entropy_calculation_batch_size,
                                        number_of_batches_to_calculate_entropy_on=number_of_batches_to_calculate_entropy_on, root_save_prefix=root_save_prefix,
                                        final_postfix_for_save=final_postfix_for_overall_save, custom_model=net,
                                        custom_data_loader=(trainloader, testloader), class_indx_to_visualize=class_indx_to_visualize)

            for current_c_indx in class_indx_to_visualize:
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

    else:
        best_test_acc, net = train_model(net,
                                         trainloader, testloader, epochs, criterion, optimizer, final_model_save_path)
    print("Finished execution!!!")

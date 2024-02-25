import numpy as np
import torch
import tqdm
import wandb
import torch.backends.cudnn as cudnn

from utils.data_preprocessing import preprocess_dataset_get_data_loader, generate_dataset_from_loader
from structure.dlgn_conv_config_structure import DatasetConfig
from torchvision import transforms

from conv4_models import get_model_instance, get_model_save_path, get_model_instance_from_dataset, get_img_size
from visualization import run_visualization_on_config
from utils.weight_utils import get_gating_layer_weights
from raw_weight_analysis import convert_list_tensor_to_numpy
from utils.APR import APRecombination, mix_data


def get_normalize(inp):
    if(inp.size()[1] == 3):
        normalize = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [
                                 0.2023, 0.1994, 0.2010]),
        ])
        return normalize(inp)

    return inp


def apr_evaluate_model(net, dataloader, num_classes_trained_on=None):
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frequency_pred = None
    if(num_classes_trained_on is not None):
        frequency_pred = torch.zeros(num_classes_trained_on)
        frequency_pred = frequency_pred.to(device, non_blocking=True)
        all_classes = torch.arange(0, num_classes_trained_on)
        all_classes = all_classes.to(device, non_blocking=True)

    orig_correct = 0
    orig_total = 0
    overall_total = 0
    total_count = 0

    ph_label_correct = 0
    am_label_correct = 0
    switch_total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        loader = tqdm.tqdm(dataloader, desc='Evaluating APR')
        for batch_idx, data in enumerate(loader, 0):
            inputs, labels = data

            total_count += inputs.size(0)

            inputs, labels = inputs.to(
                device, non_blocking=True), labels.to(device, non_blocking=True)
            dev = inputs.get_device()
            amp_labels = None
            phase_labels = None
            if(len(inputs.size())==3):
                inputs = torch.unsqueeze(inputs,1)
            mix1, amp_label_indx, mix2 = mix_data(inputs, prob=1.0)
            inputs_mix = torch.cat([mix1, mix2], 0)
            if(amp_label_indx is not None):
                temp = labels[amp_label_indx]
                amp_labels = torch.cat([temp, labels], 0)
                phase_labels = torch.cat([labels, temp], 0)

            batch_size = inputs.size(0)
            # inputs, inputs_mix = get_normalize(
            #     inputs), get_normalize(inputs_mix)
            inputs = torch.cat([inputs, inputs_mix], 0)

            inputs = inputs.to(device=dev, non_blocking=True)

            outputs = net(inputs)

            orig_size = batch_size
            if(amp_labels is None):
                orig_size = 3*batch_size
                labels = torch.cat([labels, labels, labels], 0)

            _, predicted = torch.max(outputs.data, 1)
            overall_total += outputs.size(0)
            orig_total += labels.size(0)
            orig_correct += (predicted[:orig_size] == labels).sum().item()
            if(amp_labels is not None):
                am_label_correct += (predicted[batch_size:]
                                     == amp_labels).sum().item()
                ph_label_correct += (predicted[batch_size:]
                                     == phase_labels).sum().item()
                switch_total += phase_labels.size(0)

            resdict = {"tst_noswtch_per": 100.*orig_total/overall_total, "orig_evl_count": total_count, "tst_total_samples": overall_total,
                       "orig_evl_acc": 100.*orig_correct/orig_total, "ph_evl_acc": 100.*ph_label_correct/switch_total, "amp_evl_acc": 100.*am_label_correct/switch_total}

            if(num_classes_trained_on is not None):
                temp = torch.cat((predicted.float(), all_classes))
                temp = temp.to(device)
                frequency_pred += torch.histc(temp, num_classes_trained_on) - 1

    return resdict, frequency_pred


if __name__ == '__main__':
    # fashion_mnist , mnist , cifar10
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net , conv4_deep_gated_net_n16_small ,
    # conv4_deep_gated_net_with_actual_inp_in_wt_net , conv4_deep_gated_net_with_actual_inp_randomly_changed_in_wt_net
    # conv4_deep_gated_net_with_random_ones_in_wt_net , masked_conv4_dlgn , masked_conv4_dlgn_n16_small , fc_dnn , fc_dlgn , fc_dgn
    model_arch_type = 'fc_dnn'

    scheme_type = 'APR_exps_eval'

    model_to_be_evaluated = "root/model/save/mnist/adversarial_training/MT_fc_dnn_W_128_D_4_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.3/batch_size_64/eps_stp_size_0.01/adv_steps_40/update_on_all/R_init_True/norm_inf/use_ytrue_True/adv_model_dir.pt"

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

    eval_on_test = True

    aprp_mix_prob = 0.6
    train_on_phase_labels = True
    aprs_prob_threshold = 0.7
    aprs_mix_prob = 0.5

    is_normalize_data = True
    train_transforms = None
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

    custom_temp_model = torch.load(model_to_be_evaluated)
    net.load_state_dict(custom_temp_model.state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    if device_str == 'cuda':
        if(torch.cuda.device_count() > 1):
            print("Parallelizing model")
            net = torch.nn.DataParallel(net).cuda()
        else:
            net = net.to(device)
        cudnn.benchmark = True

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

    if(eval_on_test):
        evalLoader = testloader
    else:
        evalLoader = trainloader

    is_log_wandb = not(wand_project_name is None)
    if(is_log_wandb):
        aprp_postfix = ""

        wandb_group_name = "DS_"+str(dataset_str) + \
            "_MT_"+str(model_arch_type_str)+"_SEED_" + \
            str(torch_seed) + \
            aprp_postfix.replace("/", "_")
        wandb_run_name = "MT_" + \
            str(model_arch_type_str)+"/SEED_"+str(torch_seed) +\
            "/BS_"+str(batch_size) + \
            "/SCH_TYP_"+str(scheme_type)
        wandb_run_name = wandb_run_name.replace("/", "")

        wandb_config = dict()
        wandb_config["dataset"] = dataset_str
        wandb_config["model_arch_type"] = model_arch_type_str
        wandb_config["torch_seed"] = torch_seed
        wandb_config["scheme_type"] = scheme_type
        wandb_config["model_to_be_evaluated_path"] = model_to_be_evaluated
        wandb_config["batch_size"] = batch_size
        wandb_config["eval_on_test"] = eval_on_test
        if(pca_exp_percent is not None):
            wandb_config["pca_exp_percent"] = pca_exp_percent
            wandb_config["num_comp_pca"] = number_of_components_for_pca

        wandb.init(
            project=f"{wand_project_name}",
            name=f"{wandb_run_name}",
            group=f"{wandb_group_name}",
            config=wandb_config,
        )

    res_dict, _ = apr_evaluate_model(net, evalLoader)
    print("res_dict:", res_dict)

    if(is_log_wandb):
        wandb.log(res_dict)
        wandb.finish()

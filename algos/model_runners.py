from structure.dlgn_conv_config_structure import DatasetConfig, AllParams
from algos.dlgn_conv_preprocess import preprocess_dataset_get_data_loader
from algos.dlgn_conv_train import construct_train_evaluate_DLGN_model
from utils.generic_utils import get_hash_for_string_of_length


def run_deep_dream(wandb_project_name, is_DLGN_all_ones, dataset_folder, version, gate_net_conv_info, weight_net_conv_info, list_of_hp_obj):
    hp_obj = list_of_hp_obj[0]

    cifar10_config = DatasetConfig(
        'cifar10', is_normalize_data=False, valid_split_size=0.1, batch_size=hp_obj.batch_size)

    trainloader, validloader, testloader = preprocess_dataset_get_data_loader(
        cifar10_config, verbose=1, dataset_folder=dataset_folder)

    all_param_hash = str(get_hash_for_string_of_length(str(AllParams(
        str(hp_obj), gate_net_conv_info, weight_net_conv_info, is_DLGN_all_ones))))
    conf_name = str(get_hash_for_string_of_length(
        str(weight_net_conv_info)+str(gate_net_conv_info), 20))

    opt_test_acc, opt_train_acc, opt_valid_acc, opt_test_loss, opt_train_loss, opt_valid_loss, ret_model = construct_train_evaluate_DLGN_model(
        gate_net_conv_info, weight_net_conv_info, trainloader, validloader, testloader, hp_obj, runs_out_dir='/root/model/runs', save_out_dir='/root/model/save', conf_name=conf_name, all_param_hash=all_param_hash, seed=2022, writer=None, is_DLGN_all_ones=is_DLGN_all_ones)

    return opt_test_acc, opt_train_acc, opt_valid_acc, opt_test_loss, opt_train_loss, opt_valid_loss, ret_model

from structure.dlgn_conv_config_structure import DatasetConfig, AllParams
from algos.dlgn_conv_preprocess import preprocess_dataset_get_data_loader
from algos.dlgn_conv_train import construct_train_evaluate_DLGN_model
from utils.generic_utils import get_hash_for_string_of_length, set_inputchannel_for_first_conv_layer
import copy
import wandb


def run_deep_dream(wandb_project_name, is_DLGN_all_ones, dataset_folder, version, gate_net_conv_info, weight_net_conv_info, list_of_hp_obj, verbose=1):
    hp_obj = list_of_hp_obj[0]

    # cifar10_config = DatasetConfig(
    #     'cifar10', is_normalize_data=False, valid_split_size=0.1, batch_size=hp_obj.batch_size)

    # trainloader, validloader, testloader = preprocess_dataset_get_data_loader(
    #     cifar10_config, verbose=1, dataset_folder=dataset_folder)

    mnist_config = DatasetConfig(
        'mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=hp_obj.batch_size)

    trainloader, validloader, testloader = preprocess_dataset_get_data_loader(
        mnist_config, verbose=verbose, dataset_folder=dataset_folder)

    trainBatch = next(iter(trainloader))
    input_image_size = (trainBatch[0].size()[-2], trainBatch[0].size()[-1])
    print("input_image_size", input_image_size)
    print("trainBatch [0]", trainBatch[0][0])
    gate_net_conv_info.input_image_size = input_image_size
    weight_net_conv_info.input_image_size = input_image_size

    input_image_channel = trainBatch[0].size()[-3]
    set_inputchannel_for_first_conv_layer(
        gate_net_conv_info, input_image_channel)
    set_inputchannel_for_first_conv_layer(
        weight_net_conv_info, input_image_channel)

    # print("gate_net_conv_info", gate_net_conv_info)
    # print("weight_net_conv_info", weight_net_conv_info)

    all_param_hash = None
    conf_name = str(get_hash_for_string_of_length(
        str(weight_net_conv_info)+str(gate_net_conv_info), 20))
    if(not(wandb_project_name is None)):
        all_param_hash = str(get_hash_for_string_of_length(str(AllParams(
            str(hp_obj), gate_net_conv_info, weight_net_conv_info, is_DLGN_all_ones))))
        wand_config_dnn = copy.deepcopy(hp_obj.__dict__)
        wand_config_dnn["is_DLGN_all_ones"] = is_DLGN_all_ones
        wand_config_dnn["all_param_hash"] = all_param_hash
        wand_config_dnn["version"] = version
        wand_config_dnn["gate_net_conv_info"] = gate_net_conv_info
        wand_config_dnn["weight_net_conv_info"] = weight_net_conv_info

        wandb.init(
            # Set the project where this run will be logged
            project=f"{wandb_project_name}",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            # Track hyperparameters and run metadata
            config=wand_config_dnn,
        )

    opt_test_acc, opt_train_acc, opt_valid_acc, opt_test_loss, opt_train_loss, opt_valid_loss, ret_model = construct_train_evaluate_DLGN_model(
        gate_net_conv_info, weight_net_conv_info, trainloader, validloader, testloader, hp_obj, runs_out_dir='root/model/runs', save_out_dir='root/model/save', conf_name=conf_name, all_param_hash=all_param_hash, seed=2022, writer=None, is_DLGN_all_ones=is_DLGN_all_ones, verbose=verbose)

    num_params_total = sum(p.numel() for p in ret_model.parameters())
    num_params_gate = sum(p.numel()
                          for p in ret_model.linear_conv_net.parameters())
    num_params_weight = sum(p.numel()
                            for p in ret_model.weight_conv_net.parameters())
    if(not(wandb_project_name is None)):
        wandb.log({
            "total_num_model_params": num_params_total, "num_params_gate_net": num_params_gate, "num_params_weight_net": num_params_weight,
            "train": {"acc": round(opt_train_acc, 5), 'loss': round(opt_train_loss, 5)},
            "val": {"acc": round(opt_valid_acc, 5), 'loss': round(opt_valid_loss, 5)},
            "test": {"acc": round(opt_test_acc, 5), 'loss': round(opt_test_loss, 5)}
        })
        wandb.finish()

    return opt_test_acc, opt_train_acc, opt_valid_acc, opt_test_loss, opt_train_loss, opt_valid_loss, ret_model

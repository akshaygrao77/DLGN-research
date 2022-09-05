import wandb
from structure.dlgn_conv_structure import convert_generic_object_list_to_All_Conv_info
from utils.generic_utils import get_object_from_json_file, convert_generic_object_list_to_HPParam_object_list
from algos.model_runners import run_deep_dream
import argparse
import torch


def print_all_config(conf_obj):
    print("List of experiments:", conf_obj.list_of_experiments)
    print("Wandb_project_name:", conf_obj.wandb_project_name)
    print("Folder containing datasets:", conf_obj.dataset_folder)
    print("List of hyper-parameters:", conf_obj.list_of_hp_obj)


def run_script_based_on_configuration(config_file_path):
    conf_obj = get_object_from_json_file(config_file_path)
    if(torch.cuda.is_available()):
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("CUDA is enabled")
        print("You have been assigned an " +
              str(torch.cuda.get_device_name(0)))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("**************************************************************************")
    print_all_config(conf_obj)
    print("**************************************************************************")
    if(not(conf_obj.wandb_project_name is None)):
        wandb.login()

    list_of_hp_obj = convert_generic_object_list_to_HPParam_object_list(
        conf_obj.list_of_hp_obj)
    print(str(list_of_hp_obj[0]))
    all_conv_info = convert_generic_object_list_to_All_Conv_info(
        conf_obj.list_of_layer_configs)

    run_deep_dream(wandb_project_name=conf_obj.wandb_project_name, is_DLGN_all_ones=bool(conf_obj.is_DLGN_all_ones == "True"),
                   dataset_folder=conf_obj.dataset_folder, version=conf_obj.version,all_conv_info=all_conv_info,list_of_hp_obj = list_of_hp_obj)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config_file_path", action="store", dest='config_file_path', required=True,
                        help="Specify path to configuration file",)

    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()

    if(not(args.config_file_path is None)):
        run_script_based_on_configuration(args.config_file_path)

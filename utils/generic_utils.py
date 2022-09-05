import math
from structure.dlgn_conv_config_structure import Configs, HPParams
import json
import hashlib
import os

def create_nested_dir_if_not_exists(directory):
    os.makedirs(directory, exist_ok=True)

def get_hash_for_string_of_length(inputstr, length=None):
    if(length is None):
        return int(hashlib.sha1(inputstr.encode("utf-8")).hexdigest(), 32)
    return int(hashlib.sha1(inputstr.encode("utf-8")).hexdigest(), 32) % (10 ** length)

def convert_from_integer_to_tuple_if_not(inp):
    if(type(inp) is tuple == True):
        return inp
    else:
        return (inp, inp)


def calculate_output_size_for_conv_per_dimension(input_dim, padding_dim, kernel_dim, stride_dim):
    return math.floor(((input_dim + 2 * padding_dim - kernel_dim) / stride_dim) + 1)


def calculate_output_size_for_conv(input_size, padding_size, kernel_size, stride_size):
    input_size = convert_from_integer_to_tuple_if_not(input_size)
    padding_size = convert_from_integer_to_tuple_if_not(padding_size)
    kernel_size = convert_from_integer_to_tuple_if_not(kernel_size)
    stride_size = convert_from_integer_to_tuple_if_not(stride_size)

    out_dim_1 = calculate_output_size_for_conv_per_dimension(
        input_size[0], padding_size[0], kernel_size[0], stride_size[0])
    out_dim_2 = calculate_output_size_for_conv_per_dimension(
        input_size[1], padding_size[1], kernel_size[1], stride_size[1])

    return out_dim_1, out_dim_2


def get_object_from_json_file(json_file_path):
    with open(json_file_path) as json_file:
        data = json.load(json_file)
        datastr = json.dumps(data)
        x = json.loads(datastr, object_hook=lambda d: Configs(**d))

    return x


def convert_generic_object_list_to_HPParam_object_list(list_of_gen_hp_obj):
    list_of_hp_obj = []
    for each_gen_hp_obj in list_of_gen_hp_obj:
        list_of_hp_obj.append(
            convert_generic_object_to_HPParam_object(each_gen_hp_obj))
    return list_of_hp_obj


def convert_generic_object_to_HPParam_object(gen_hp_obj):
    return HPParams(weight_init=gen_hp_obj.weight_init, output_activ_func=gen_hp_obj.output_activ_func, momentum=gen_hp_obj.momentum, lr=gen_hp_obj.lr, loss_fn=gen_hp_obj.loss_fn, epochs=gen_hp_obj.epochs, optimizer=gen_hp_obj.optimizer, activ_func=gen_hp_obj.activ_func)

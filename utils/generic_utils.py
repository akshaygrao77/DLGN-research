import math

import numpy as np
from structure.dlgn_conv_config_structure import Configs, HPParams
from structure.dlgn_conv_structure import All_Conv_info, Conv_info
import json
import hashlib
import os
import tqdm
import torch.nn as nn
import torch

class Y_True_Loss(nn.Module):
    def __init__(self):
        super(Y_True_Loss, self).__init__()

    def forward(self, inputs, targets):
        targets = torch.unsqueeze(targets,1).long()
        y_true_logits = torch.squeeze(torch.gather(inputs,1,targets))
        loss = torch.mean(y_true_logits)

        return -loss

def create_nested_dir_if_not_exists(directory):
    os.makedirs(directory, exist_ok=True)


def get_hash_for_string_of_length(inputstr, length=None):
    if(length is None):
        return int(hashlib.sha1(inputstr.encode("utf-8")).hexdigest(), 32)
    return int(hashlib.sha1(inputstr.encode("utf-8")).hexdigest(), 32) % (10 ** length)


def convert_from_integer_to_tuple_if_not(inp):
    if(inp is None):
        return inp

    if(isinstance(inp, tuple)):
        return inp
    else:
        return (inp, inp)


def convert_from_list_to_tuple_or_int(inp):
    if(inp is None):
        return inp

    if(isinstance(inp, list) == True):
        assert len(inp) == 2, 'Length of list input is not equal to two'

        return (inp[0], inp[1])
    else:
        return inp


def convert_generic_object_list_to_All_Conv_info(list_of_gen_all_conv_obj):
    list_of_conv_obj = []
    for each_gen_hp_obj in list_of_gen_all_conv_obj:
        list_of_conv_obj.append(
            convert_generic_object_to_Conv_info_object(each_gen_hp_obj))

    return All_Conv_info(list_of_conv_obj, None)


def convert_generic_object_to_Conv_info_object(gen_conv_obj):
    padding = convert_from_list_to_tuple_or_int(gen_conv_obj.padding)
    stride = convert_from_list_to_tuple_or_int(gen_conv_obj.stride)
    kernel_size = convert_from_list_to_tuple_or_int(gen_conv_obj.kernel_size)

    return Conv_info(gen_conv_obj.layer_type, gen_conv_obj.layer_sub_type, gen_conv_obj.in_ch, gen_conv_obj.number_of_filters, padding, stride, kernel_size, gen_conv_obj.num_nodes_in_fc, gen_conv_obj.weight_init_type)


def save_dataset_into_path_from_loader(dataloader, np_save_filename):
    ys = None
    xs = None

    data_loader = tqdm.tqdm(
        dataloader, desc='Saving dataset')
    for i, per_class_per_batch_data in enumerate(data_loader):
        images, labels = per_class_per_batch_data
        images = images.numpy()
        labels = labels.numpy()
        if(xs is None):
            xs = images
        else:
            xs = np.concatenate((xs, images), axis=0)

        if(ys is None):
            ys = labels
        else:
            ys = np.concatenate((ys, labels), axis=0)

    sfolder = np_save_filename[0:np_save_filename.rfind("/")+1]
    if not os.path.exists(sfolder):
        os.makedirs(sfolder)
    with open(np_save_filename, 'wb') as file:
        np.savez(file, x=xs, y=ys)


def calculate_output_size_for_conv_per_dimension(input_dim, padding_dim, kernel_dim, stride_dim):
    return math.floor(((input_dim + 2 * padding_dim - kernel_dim) / stride_dim) + 1)


def set_inputchannel_for_first_conv_layer(all_conv_info, input_channel):
    for each_conv_info in all_conv_info.list_of_each_conv_info:
        if(each_conv_info.layer_type == "CONV"):
            each_conv_info.in_ch = input_channel
            return


def calculate_output_size_for_conv(input_size, padding_size, kernel_size, stride_size):
    input_size = convert_from_integer_to_tuple_if_not(input_size)
    padding_size = convert_from_integer_to_tuple_if_not(padding_size)
    kernel_size = convert_from_integer_to_tuple_if_not(kernel_size)
    stride_size = convert_from_integer_to_tuple_if_not(stride_size)

    out_dim_1 = calculate_output_size_for_conv_per_dimension(
        input_size[0], padding_size[0], kernel_size[0], stride_size[0])
    out_dim_2 = calculate_output_size_for_conv_per_dimension(
        input_size[1], padding_size[1], kernel_size[1], stride_size[1])

    return (out_dim_1, out_dim_2)


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

import torch
import torch.nn as nn
import numpy as np
from structure.fc_models import DLGN_FC_Network, DNN_FC_Network, DGN_FC_Network
from utils.visualise_utils import determine_row_col_from_features
from sklearn.decomposition import PCA
from collections import OrderedDict
import torchvision.models as models
from torch._utils import _get_all_device_indices


def replace_percent_of_values(inp_np, const_value, percentage):
    inp_all_const = np.ones(inp_np.shape)*const_value
    mask_int = np.random.randint(101, size=inp_np.shape)
    mask = np.random.randint(2, size=inp_np.shape)
    mask[mask_int > percentage] = 0
    mask[mask_int <= percentage] = 1
    mask = mask.astype(np.bool)
    inp_np[mask] = inp_all_const[mask]
    return inp_np


def replace_percent_of_values_with_exact_percentages(inp_np, const_value, percentage):
    npr = np.arange(0, 100, step=100/inp_np.size, dtype=np.float32)
    np.random.shuffle(npr)
    npr = npr.reshape(inp_np.shape)
    ret = np.where(npr >= percentage, inp_np, const_value)
    return ret


def convert_relu_to_identity(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Identity())
        else:
            convert_relu_to_identity(child)


def get_last_relu(model, root_name=""):
    curr_last = None
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            curr_last = root_name + child_name
        else:
            c = get_last_relu(child, root_name+child_name+".")
            if(c is not None):
                curr_last = c
    return curr_last


def convert_layers_after_last_relu_to_identity(model, last_relu_layer_name, is_replace=False, root_name=""):
    for child_name, child in model.named_children():
        if (root_name + child_name) == last_relu_layer_name:
            is_replace = True
        elif(is_replace):
            print("Nodes replaced are", root_name + child_name)
            setattr(model, child_name, nn.Identity())
            # replace_leaf_with_identity(child,root_name+child_name)
        else:
            is_replace = is_replace or convert_layers_after_last_relu_to_identity(
                child, last_relu_layer_name, is_replace, root_name+child_name+".")

    return is_replace


class CONV_PCA_Layer(nn.Module):
    def __init__(self, input_channel, data, explained_var_required):
        super(CONV_PCA_Layer, self).__init__()
        self.input_channel = input_channel
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if(isinstance(data, torch.Tensor)):
            flattened_data = torch.flatten(data, 1)
        else:
            flattened_data = data.reshape(
                data.shape[0], data.shape[1]*data.shape[2])
        pca = PCA().fit(flattened_data)
        k = 0
        current_variance = 0
        while(current_variance < explained_var_required):
            current_variance = sum(pca.explained_variance_ratio_[:k])
            k = k + 1

        print("Number of PCA components used is:", k)
        self.k = k
        self.k_pca = PCA(n_components=k)
        self.k_pca.fit(flattened_data)

        d1, d2 = determine_row_col_from_features(self.k)
        self.input_size_list = [d1, d2]

        pc_mean = torch.from_numpy(self.k_pca.mean_)
        self.pc_mean = pc_mean.to(device)
        pc_comp = torch.from_numpy(self.k_pca.components_.T)
        self.pc_comp = pc_comp.to(device)

    def forward(self, inp):
        temp_size = inp.size()
        inp = torch.flatten(inp, 1)
        m_data = inp - self.pc_mean
        inp = torch.matmul(m_data, self.pc_comp).type(torch.float32)
        inp = torch.reshape(
            inp, (temp_size[0], self.input_channel, self.input_size_list[0], self.input_size_list[1]))

        return inp


class Mask_Conv4_DLGN_Net(nn.Module):
    def __init__(self, input_channel, random_inp_percent=40, beta=4, seed=2022, num_classes=10):
        super().__init__()
        torch.manual_seed(seed)
        self.input_channel = input_channel
        self.beta = beta
        self.seed = seed
        self.random_inp_percent = random_inp_percent
        self.prev_inp_size = None

        self.conv1_g = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_w = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_w = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(128, num_classes)

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = CONV_PCA_Layer(
            self.input_channel, data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        return self.pca_layer.k

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=device, non_blocking=True)

        if(self.prev_inp_size is None or self.prev_inp_size != inp.size()):
            self.prev_inp_size = inp.size()
            np.random.seed(self.seed)
            all_ones_np = np.ones(shape=inp.size()).astype(np.float32)
            self.mask_1 = replace_percent_of_values_with_exact_percentages(
                all_ones_np, 0, self.random_inp_percent)
            self.mask_1 = torch.from_numpy(self.mask_1)
            self.mask_1 = self.mask_1.to(device)

            self.mask_2 = replace_percent_of_values_with_exact_percentages(
                all_ones_np, 0, self.random_inp_percent)
            self.mask_2 = torch.from_numpy(self.mask_2)
            self.mask_2 = self.mask_2.to(device)

            self.mask_3 = replace_percent_of_values_with_exact_percentages(
                all_ones_np, 0, self.random_inp_percent)
            self.mask_3 = torch.from_numpy(self.mask_3)
            self.mask_3 = self.mask_3.to(device)

            self.mask_4 = replace_percent_of_values_with_exact_percentages(
                all_ones_np, 0, self.random_inp_percent)
            self.mask_4 = torch.from_numpy(self.mask_4)
            self.mask_4 = self.mask_4.to(device)

        conv_outs = []
        x_g1 = self.conv1_g(inp)
        conv_outs.append(x_g1)
        x_g1 = x_g1 * self.mask_1

        x_g2 = self.conv2_g(x_g1)
        conv_outs.append(x_g2)
        x_g2 = x_g2 * self.mask_2

        x_g3 = self.conv3_g(x_g2)
        conv_outs.append(x_g3)
        x_g3 = x_g3 * self.mask_3

        x_g4 = self.conv4_g(x_g3)
        conv_outs.append(x_g4)
        x_g4 = x_g4 * self.mask_4

        self.linear_conv_outputs = conv_outs

        g1 = nn.Sigmoid()(self.beta * x_g1)
        g2 = nn.Sigmoid()(self.beta * x_g2)
        g3 = nn.Sigmoid()(self.beta * x_g3)
        g4 = nn.Sigmoid()(self.beta * x_g4)

        inp_all_ones = torch.ones(inp.size(),
                                  requires_grad=True, device=device)

        x_w1 = self.conv1_w(inp_all_ones) * g1
        x_w2 = self.conv2_w(x_w1) * g2
        x_w3 = self.conv3_w(x_w2) * g3
        x_w4 = self.conv4_w(x_w3) * g4

        x_w5 = self.pool(x_w4)
        x_w5 = torch.flatten(x_w5, 1)
        x_w6 = self.fc1(x_w5)

        return x_w6

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            if(layer_num == 0):
                return self.conv1_g
            elif(layer_num == 1):
                return self.conv2_g
            elif(layer_num == 2):
                return self.conv3_g
            elif(layer_num == 3):
                return self.conv4_g
        elif(network_type == "WEIGHT_NET"):
            if(layer_num == 0):
                return self.conv1_w
            elif(layer_num == 1):
                return self.conv2_w
            elif(layer_num == 2):
                return self.conv3_w
            elif(layer_num == 3):
                return self.conv4_w
            elif(layer_num == 4):
                return self.pool
            elif(layer_num == 5):
                return self.fc1


class Mask_Conv4_DLGN_Net_N16_Small(nn.Module):
    def __init__(self, input_channel, random_inp_percent=40, beta=4, seed=2022, num_classes=10):
        super().__init__()
        self.beta = beta
        torch.manual_seed(seed)
        self.seed = seed
        self.random_inp_percent = random_inp_percent
        self.prev_inp_size = None
        self.input_channel = input_channel

        self.conv1_g = nn.Conv2d(input_channel, 16, 3, padding=1)
        self.conv2_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv1_w = nn.Conv2d(input_channel, 16, 3, padding=1)
        self.conv2_w = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3_w = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4_w = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(16, num_classes)

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = CONV_PCA_Layer(
            self.input_channel, data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        return self.pca_layer.k

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=device, non_blocking=True)

        if(self.prev_inp_size is None or self.prev_inp_size != inp.size()):
            self.prev_inp_size = inp.size()
            np.random.seed(self.seed)
            all_ones_np = np.ones(shape=inp.size()).astype(np.float32)
            self.mask_1 = replace_percent_of_values_with_exact_percentages(
                all_ones_np, 0, self.random_inp_percent)
            self.mask_1 = torch.from_numpy(self.mask_1)
            self.mask_1 = self.mask_1.to(device)

            self.mask_2 = replace_percent_of_values_with_exact_percentages(
                all_ones_np, 0, self.random_inp_percent)
            self.mask_2 = torch.from_numpy(self.mask_2)
            self.mask_2 = self.mask_2.to(device)

            self.mask_3 = replace_percent_of_values_with_exact_percentages(
                all_ones_np, 0, self.random_inp_percent)
            self.mask_3 = torch.from_numpy(self.mask_3)
            self.mask_3 = self.mask_3.to(device)

            self.mask_4 = replace_percent_of_values_with_exact_percentages(
                all_ones_np, 0, self.random_inp_percent)
            self.mask_4 = torch.from_numpy(self.mask_4)
            self.mask_4 = self.mask_4.to(device)

        conv_outs = []
        x_g1 = self.conv1_g(inp)
        conv_outs.append(x_g1)
        x_g1 = x_g1 * self.mask_1

        x_g2 = self.conv2_g(x_g1)
        conv_outs.append(x_g2)
        x_g2 = x_g2 * self.mask_2

        x_g3 = self.conv3_g(x_g2)
        conv_outs.append(x_g3)
        x_g3 = x_g3 * self.mask_3

        x_g4 = self.conv4_g(x_g3)
        conv_outs.append(x_g4)
        x_g4 = x_g4 * self.mask_4

        self.linear_conv_outputs = conv_outs

        g1 = nn.Sigmoid()(self.beta * x_g1)
        g2 = nn.Sigmoid()(self.beta * x_g2)
        g3 = nn.Sigmoid()(self.beta * x_g3)
        g4 = nn.Sigmoid()(self.beta * x_g4)

        inp_all_ones = torch.ones(inp.size(),
                                  requires_grad=True, device=device)

        x_w1 = self.conv1_w(inp_all_ones) * g1
        x_w2 = self.conv2_w(x_w1) * g2
        x_w3 = self.conv3_w(x_w2) * g3
        x_w4 = self.conv4_w(x_w3) * g4

        x_w5 = self.pool(x_w4)
        x_w5 = torch.flatten(x_w5, 1)
        x_w6 = self.fc1(x_w5)

        return x_w6

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            if(layer_num == 0):
                return self.conv1_g
            elif(layer_num == 1):
                return self.conv2_g
            elif(layer_num == 2):
                return self.conv3_g
            elif(layer_num == 3):
                return self.conv4_g
        elif(network_type == "WEIGHT_NET"):
            if(layer_num == 0):
                return self.conv1_w
            elif(layer_num == 1):
                return self.conv2_w
            elif(layer_num == 2):
                return self.conv3_w
            elif(layer_num == 3):
                return self.conv4_w
            elif(layer_num == 4):
                return self.pool
            elif(layer_num == 5):
                return self.fc1


class Plain_CONV4_Net(nn.Module):
    def __init__(self, input_channel, seed=2022, num_classes=10):
        super().__init__()
        torch.manual_seed(seed)
        self.input_channel = input_channel
        self.conv1_g = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=1)

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(128, num_classes)

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = CONV_PCA_Layer(
            self.input_channel, data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        return self.pca_layer.k

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=device, non_blocking=True)

        conv_outs = []
        x_g1 = self.conv1_g(inp)
        conv_outs.append(x_g1)
        x_g1 = self.relu(x_g1)
        x_g2 = self.conv2_g(x_g1)
        conv_outs.append(x_g2)
        x_g2 = self.relu(x_g2)
        x_g3 = self.conv3_g(x_g2)
        conv_outs.append(x_g3)
        x_g3 = self.relu(x_g3)
        x_g4 = self.conv4_g(x_g3)
        conv_outs.append(x_g4)

        self.linear_conv_outputs = conv_outs

        x_g4 = self.relu(x_g4)
        x_g5 = self.pool(x_g4)
        x_g5 = torch.flatten(x_g5, 1)
        x_g6 = self.fc1(x_g5)

        return x_g6

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            if(layer_num == 0):
                return self.conv1_g
            elif(layer_num == 1):
                return self.conv2_g
            elif(layer_num == 2):
                return self.conv3_g
            elif(layer_num == 3):
                return self.conv4_g
            elif(layer_num == 4):
                return self.pool
            elif(layer_num == 5):
                return self.fc1


class Plain_CONV4_Net_N16_Small(nn.Module):
    def __init__(self, input_channel, seed=2022, num_classes=10):
        super().__init__()
        torch.manual_seed(seed)
        self.input_channel = input_channel
        self.conv1_g = nn.Conv2d(input_channel, 16, 3, padding=1)
        self.conv2_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4_g = nn.Conv2d(16, 16, 3, padding=1)

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(16, num_classes)

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = CONV_PCA_Layer(
            self.input_channel, data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        return self.pca_layer.k

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=device, non_blocking=True)

        conv_outs = []
        x_g1 = self.conv1_g(inp)
        conv_outs.append(x_g1)
        x_g1 = self.relu(x_g1)
        x_g2 = self.conv2_g(x_g1)
        conv_outs.append(x_g2)
        x_g2 = self.relu(x_g2)
        x_g3 = self.conv3_g(x_g2)
        conv_outs.append(x_g3)
        x_g3 = self.relu(x_g3)
        x_g4 = self.conv4_g(x_g3)
        conv_outs.append(x_g4)

        self.linear_conv_outputs = conv_outs

        x_g4 = self.relu(x_g4)
        x_g5 = self.pool(x_g4)
        x_g5 = torch.flatten(x_g5, 1)
        x_g6 = self.fc1(x_g5)

        return x_g6

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            if(layer_num == 0):
                return self.conv1_g
            elif(layer_num == 1):
                return self.conv2_g
            elif(layer_num == 2):
                return self.conv3_g
            elif(layer_num == 3):
                return self.conv4_g
            elif(layer_num == 4):
                return self.pool
            elif(layer_num == 5):
                return self.fc1


class Conv4_DLGN_Net(nn.Module):
    def __init__(self, input_channel, beta=4, seed=2022, num_classes=10):
        super().__init__()
        torch.manual_seed(seed)
        self.input_channel = input_channel
        self.beta = beta

        self.conv1_g = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_w = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_w = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(128, num_classes)

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = CONV_PCA_Layer(
            self.input_channel, data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        return self.pca_layer.k

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=device, non_blocking=True)

        conv_outs = []
        x_g1 = self.conv1_g(inp)
        conv_outs.append(x_g1)
        x_g2 = self.conv2_g(x_g1)
        conv_outs.append(x_g2)
        x_g3 = self.conv3_g(x_g2)
        conv_outs.append(x_g3)
        x_g4 = self.conv4_g(x_g3)
        conv_outs.append(x_g4)

        self.linear_conv_outputs = conv_outs

        g1 = nn.Sigmoid()(self.beta * x_g1)
        g2 = nn.Sigmoid()(self.beta * x_g2)
        g3 = nn.Sigmoid()(self.beta * x_g3)
        g4 = nn.Sigmoid()(self.beta * x_g4)

        inp_all_ones = torch.ones(inp.size(),
                                  requires_grad=True, device=device)

        x_w1 = self.conv1_w(inp_all_ones) * g1
        x_w2 = self.conv2_w(x_w1) * g2
        x_w3 = self.conv3_w(x_w2) * g3
        x_w4 = self.conv4_w(x_w3) * g4

        x_w5 = self.pool(x_w4)
        x_w5 = torch.flatten(x_w5, 1)
        x_w6 = self.fc1(x_w5)

        return x_w6

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            if(layer_num == 0):
                return self.conv1_g
            elif(layer_num == 1):
                return self.conv2_g
            elif(layer_num == 2):
                return self.conv3_g
            elif(layer_num == 3):
                return self.conv4_g
        elif(network_type == "WEIGHT_NET"):
            if(layer_num == 0):
                return self.conv1_w
            elif(layer_num == 1):
                return self.conv2_w
            elif(layer_num == 2):
                return self.conv3_w
            elif(layer_num == 3):
                return self.conv4_w
            elif(layer_num == 4):
                return self.pool
            elif(layer_num == 5):
                return self.fc1


class Conv4_DeepGated_Net(nn.Module):
    def __init__(self, input_channel, beta=4, seed=2022, num_classes=10):
        super().__init__()
        self.beta = beta
        self.input_channel = input_channel
        torch.manual_seed(seed)

        self.conv1_g = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_w = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_w = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(128, num_classes)

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = CONV_PCA_Layer(
            self.input_channel, data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        return self.pca_layer.k

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=device, non_blocking=True)

        conv_outs = []
        x_g1 = self.conv1_g(inp)
        g1 = nn.Sigmoid()(self.beta * x_g1)
        conv_outs.append(x_g1)
        x_g1 = nn.ReLU()(x_g1)

        x_g2 = self.conv2_g(x_g1)
        g2 = nn.Sigmoid()(self.beta * x_g2)
        conv_outs.append(x_g2)
        x_g2 = nn.ReLU()(x_g2)

        x_g3 = self.conv3_g(x_g2)
        g3 = nn.Sigmoid()(self.beta * x_g3)
        conv_outs.append(x_g3)
        x_g3 = nn.ReLU()(x_g3)

        x_g4 = self.conv4_g(x_g3)
        g4 = nn.Sigmoid()(self.beta * x_g4)
        conv_outs.append(x_g4)
        x_g4 = nn.ReLU()(x_g4)

        self.linear_conv_outputs = conv_outs

        inp_all_ones = torch.ones(inp.size(),
                                  requires_grad=True, device=device)

        x_w1 = self.conv1_w(inp_all_ones) * g1
        x_w2 = self.conv2_w(x_w1) * g2
        x_w3 = self.conv3_w(x_w2) * g3
        x_w4 = self.conv4_w(x_w3) * g4

        x_w5 = self.pool(x_w4)
        x_w5 = torch.flatten(x_w5, 1)
        x_w6 = self.fc1(x_w5)

        return x_w6

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            if(layer_num == 0):
                return self.conv1_g
            elif(layer_num == 1):
                return self.conv2_g
            elif(layer_num == 2):
                return self.conv3_g
            elif(layer_num == 3):
                return self.conv4_g
        elif(network_type == "WEIGHT_NET"):
            if(layer_num == 0):
                return self.conv1_w
            elif(layer_num == 1):
                return self.conv2_w
            elif(layer_num == 2):
                return self.conv3_w
            elif(layer_num == 3):
                return self.conv4_w
            elif(layer_num == 4):
                return self.pool
            elif(layer_num == 5):
                return self.fc1


class Conv4_DeepGated_Net_N16_Small(nn.Module):
    def __init__(self, input_channel, beta=4, seed=2022, num_classes=10):
        super().__init__()
        self.beta = beta
        self.input_channel = input_channel
        torch.manual_seed(seed)

        self.conv1_g = nn.Conv2d(input_channel, 16, 3, padding=1)
        self.conv2_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv1_w = nn.Conv2d(input_channel, 16, 3, padding=1)
        self.conv2_w = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3_w = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4_w = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(16, num_classes)

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = CONV_PCA_Layer(
            self.input_channel, data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        return self.pca_layer.k

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=device, non_blocking=True)

        conv_outs = []
        x_g1 = self.conv1_g(inp)
        g1 = nn.Sigmoid()(self.beta * x_g1)
        conv_outs.append(x_g1)
        x_g1 = nn.ReLU()(x_g1)

        x_g2 = self.conv2_g(x_g1)
        g2 = nn.Sigmoid()(self.beta * x_g2)
        conv_outs.append(x_g2)
        x_g2 = nn.ReLU()(x_g2)

        x_g3 = self.conv3_g(x_g2)
        g3 = nn.Sigmoid()(self.beta * x_g3)
        conv_outs.append(x_g3)
        x_g3 = nn.ReLU()(x_g3)

        x_g4 = self.conv4_g(x_g3)
        g4 = nn.Sigmoid()(self.beta * x_g4)
        conv_outs.append(x_g4)
        x_g4 = nn.ReLU()(x_g4)

        self.linear_conv_outputs = conv_outs

        inp_all_ones = torch.ones(inp.size(),
                                  requires_grad=True, device=device)

        x_w1 = self.conv1_w(inp_all_ones) * g1
        x_w2 = self.conv2_w(x_w1) * g2
        x_w3 = self.conv3_w(x_w2) * g3
        x_w4 = self.conv4_w(x_w3) * g4

        x_w5 = self.pool(x_w4)
        x_w5 = torch.flatten(x_w5, 1)
        x_w6 = self.fc1(x_w5)

        return x_w6

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            if(layer_num == 0):
                return self.conv1_g
            elif(layer_num == 1):
                return self.conv2_g
            elif(layer_num == 2):
                return self.conv3_g
            elif(layer_num == 3):
                return self.conv4_g
        elif(network_type == "WEIGHT_NET"):
            if(layer_num == 0):
                return self.conv1_w
            elif(layer_num == 1):
                return self.conv2_w
            elif(layer_num == 2):
                return self.conv3_w
            elif(layer_num == 3):
                return self.conv4_w
            elif(layer_num == 4):
                return self.pool
            elif(layer_num == 5):
                return self.fc1


class Conv4_DeepGated_Net_With_Actual_Inp_Over_WeightNet(nn.Module):
    def __init__(self, input_channel, beta=4, seed=2022, num_classes=10):
        super().__init__()
        self.beta = beta
        torch.manual_seed(seed)

        self.conv1_g = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_w = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_w = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        conv_outs = []
        x_g1 = self.conv1_g(inp)
        g1 = nn.Sigmoid()(self.beta * x_g1)
        conv_outs.append(x_g1)
        x_g1 = nn.ReLU()(x_g1)

        x_g2 = self.conv2_g(x_g1)
        g2 = nn.Sigmoid()(self.beta * x_g2)
        conv_outs.append(x_g2)
        x_g2 = nn.ReLU()(x_g2)

        x_g3 = self.conv3_g(x_g2)
        g3 = nn.Sigmoid()(self.beta * x_g3)
        conv_outs.append(x_g3)
        x_g3 = nn.ReLU()(x_g3)

        x_g4 = self.conv4_g(x_g3)
        g4 = nn.Sigmoid()(self.beta * x_g4)
        conv_outs.append(x_g4)
        x_g4 = nn.ReLU()(x_g4)

        self.linear_conv_outputs = conv_outs

        x_w1 = self.conv1_w(inp) * g1
        x_w2 = self.conv2_w(x_w1) * g2
        x_w3 = self.conv3_w(x_w2) * g3
        x_w4 = self.conv4_w(x_w3) * g4

        x_w5 = self.pool(x_w4)
        x_w5 = torch.flatten(x_w5, 1)
        x_w6 = self.fc1(x_w5)

        return x_w6

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            if(layer_num == 0):
                return self.conv1_g
            elif(layer_num == 1):
                return self.conv2_g
            elif(layer_num == 2):
                return self.conv3_g
            elif(layer_num == 3):
                return self.conv4_g
        elif(network_type == "WEIGHT_NET"):
            if(layer_num == 0):
                return self.conv1_w
            elif(layer_num == 1):
                return self.conv2_w
            elif(layer_num == 2):
                return self.conv3_w
            elif(layer_num == 3):
                return self.conv4_w
            elif(layer_num == 4):
                return self.pool
            elif(layer_num == 5):
                return self.fc1


class Conv4_DeepGated_Net_With_Random_AllOnes_Over_WeightNet(nn.Module):
    def __init__(self, input_channel, random_inp_percent=4, beta=4, seed=2022, num_classes=10):
        super().__init__()
        self.beta = beta
        self.random_inp_percent = random_inp_percent
        torch.manual_seed(seed)

        self.conv1_g = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_w = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_w = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        conv_outs = []
        x_g1 = self.conv1_g(inp)
        g1 = nn.Sigmoid()(self.beta * x_g1)
        conv_outs.append(x_g1)
        x_g1 = nn.ReLU()(x_g1)

        x_g2 = self.conv2_g(x_g1)
        g2 = nn.Sigmoid()(self.beta * x_g2)
        conv_outs.append(x_g2)
        x_g2 = nn.ReLU()(x_g2)

        x_g3 = self.conv3_g(x_g2)
        g3 = nn.Sigmoid()(self.beta * x_g3)
        conv_outs.append(x_g3)
        x_g3 = nn.ReLU()(x_g3)

        x_g4 = self.conv4_g(x_g3)
        g4 = nn.Sigmoid()(self.beta * x_g4)
        conv_outs.append(x_g4)
        x_g4 = nn.ReLU()(x_g4)

        self.linear_conv_outputs = conv_outs

        all_zeros_np = np.zeros(shape=inp.size()).astype(np.float32)
        # print("inp_np shape:", inp_np.shape)
        # print("inp_np before:", all_zeros_np)
        inp_np = replace_percent_of_values(
            all_zeros_np, 1, self.random_inp_percent)
        # print("inp_np after:", inp_np)
        inp_np = torch.from_numpy(inp_np)
        inp_np = inp_np.to(device)
        inp_np.requires_grad_()

        x_w1 = self.conv1_w(inp_np) * g1
        x_w2 = self.conv2_w(x_w1) * g2
        x_w3 = self.conv3_w(x_w2) * g3
        x_w4 = self.conv4_w(x_w3) * g4

        x_w5 = self.pool(x_w4)
        x_w5 = torch.flatten(x_w5, 1)
        x_w6 = self.fc1(x_w5)

        return x_w6

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            if(layer_num == 0):
                return self.conv1_g
            elif(layer_num == 1):
                return self.conv2_g
            elif(layer_num == 2):
                return self.conv3_g
            elif(layer_num == 3):
                return self.conv4_g
        elif(network_type == "WEIGHT_NET"):
            if(layer_num == 0):
                return self.conv1_w
            elif(layer_num == 1):
                return self.conv2_w
            elif(layer_num == 2):
                return self.conv3_w
            elif(layer_num == 3):
                return self.conv4_w
            elif(layer_num == 4):
                return self.pool
            elif(layer_num == 5):
                return self.fc1


class Conv4_DeepGated_Net_With_Random_Actual_Inp_Over_WeightNet(nn.Module):
    def __init__(self, input_channel, random_inp_percent=4, beta=4, seed=2022, num_classes=10):
        super().__init__()
        self.beta = beta
        self.random_inp_percent = random_inp_percent
        torch.manual_seed(seed)

        self.conv1_g = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_w = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_w = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_w = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        conv_outs = []
        x_g1 = self.conv1_g(inp)
        g1 = nn.Sigmoid()(self.beta * x_g1)
        conv_outs.append(x_g1)
        x_g1 = nn.ReLU()(x_g1)

        x_g2 = self.conv2_g(x_g1)
        g2 = nn.Sigmoid()(self.beta * x_g2)
        conv_outs.append(x_g2)
        x_g2 = nn.ReLU()(x_g2)

        x_g3 = self.conv3_g(x_g2)
        g3 = nn.Sigmoid()(self.beta * x_g3)
        conv_outs.append(x_g3)
        x_g3 = nn.ReLU()(x_g3)

        x_g4 = self.conv4_g(x_g3)
        g4 = nn.Sigmoid()(self.beta * x_g4)
        conv_outs.append(x_g4)
        x_g4 = nn.ReLU()(x_g4)

        self.linear_conv_outputs = conv_outs

        inp_np = inp.cpu().clone().detach().numpy()
        # print("inp_np shape:", inp_np.shape)
        # print("inp_np before:", inp_np)
        inp_np = replace_percent_of_values(inp_np, 1, self.random_inp_percent)
        # print("inp_np after:", inp_np)
        inp_np = torch.from_numpy(inp_np)
        inp_np = inp_np.to(device)
        inp_np.requires_grad_()

        x_w1 = self.conv1_w(inp_np) * g1
        x_w2 = self.conv2_w(x_w1) * g2
        x_w3 = self.conv3_w(x_w2) * g3
        x_w4 = self.conv4_w(x_w3) * g4

        x_w5 = self.pool(x_w4)
        x_w5 = torch.flatten(x_w5, 1)
        x_w6 = self.fc1(x_w5)

        return x_w6

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            if(layer_num == 0):
                return self.conv1_g
            elif(layer_num == 1):
                return self.conv2_g
            elif(layer_num == 2):
                return self.conv3_g
            elif(layer_num == 3):
                return self.conv4_g
        elif(network_type == "WEIGHT_NET"):
            if(layer_num == 0):
                return self.conv1_w
            elif(layer_num == 1):
                return self.conv2_w
            elif(layer_num == 2):
                return self.conv3_w
            elif(layer_num == 3):
                return self.conv4_w
            elif(layer_num == 4):
                return self.pool
            elif(layer_num == 5):
                return self.fc1


class Conv4_DLGN_Net_N16_Small(nn.Module):
    def __init__(self, input_channel, beta=4, seed=2022, num_classes=10):
        super().__init__()
        self.beta = beta
        self.input_channel = input_channel
        torch.manual_seed(seed)

        self.conv1_g = nn.Conv2d(input_channel, 16, 3, padding=1)
        self.conv2_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv1_w = nn.Conv2d(input_channel, 16, 3, padding=1)
        self.conv2_w = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3_w = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4_w = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(16, num_classes)

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = CONV_PCA_Layer(
            self.input_channel, data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        return self.pca_layer.k

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=device, non_blocking=True)

        conv_outs = []
        x_g1 = self.conv1_g(inp)
        conv_outs.append(x_g1)
        x_g2 = self.conv2_g(x_g1)
        conv_outs.append(x_g2)
        x_g3 = self.conv3_g(x_g2)
        conv_outs.append(x_g3)
        x_g4 = self.conv4_g(x_g3)
        conv_outs.append(x_g4)

        self.linear_conv_outputs = conv_outs

        g1 = nn.Sigmoid()(self.beta * x_g1)
        g2 = nn.Sigmoid()(self.beta * x_g2)
        g3 = nn.Sigmoid()(self.beta * x_g3)
        g4 = nn.Sigmoid()(self.beta * x_g4)

        inp_all_ones = torch.ones(inp.size(),
                                  requires_grad=True, device=device)

        x_w1 = self.conv1_w(inp_all_ones) * g1
        x_w2 = self.conv2_w(x_w1) * g2
        x_w3 = self.conv3_w(x_w2) * g3
        x_w4 = self.conv4_w(x_w3) * g4

        x_w5 = self.pool(x_w4)
        x_w5 = torch.flatten(x_w5, 1)
        x_w6 = self.fc1(x_w5)

        return x_w6

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            if(layer_num == 0):
                return self.conv1_g
            elif(layer_num == 1):
                return self.conv2_g
            elif(layer_num == 2):
                return self.conv3_g
            elif(layer_num == 3):
                return self.conv4_g
        elif(network_type == "WEIGHT_NET"):
            if(layer_num == 0):
                return self.conv1_w
            elif(layer_num == 1):
                return self.conv2_w
            elif(layer_num == 2):
                return self.conv3_w
            elif(layer_num == 3):
                return self.conv4_w
            elif(layer_num == 4):
                return self.pool
            elif(layer_num == 5):
                return self.fc1


class ResNet_DLGN(nn.Module):
    def __init__(self, resnet_type, input_channel, beta=4, seed=2022, num_classes=1000, pretrained=False):
        super().__init__()
        torch.manual_seed(seed)
        self.num_classes = num_classes
        self.resnet_type = resnet_type
        self.input_channel = input_channel
        self.beta = beta
        self.seed = seed
        self.pretrained = pretrained

        self.initialize_network()

    def initialize_network(self):
        self.gating_network = ResNet_Gating_Network(
            self.resnet_type, self.input_channel, pretrained=self.pretrained)
        print("self.gating_network", self.gating_network)
        print("Gating net params:", sum(p.numel()
              for p in self.gating_network.parameters()))
        self.value_network = ALLONES_ResNet_Value_Network(
            self.resnet_type, self.input_channel, num_classes=self.num_classes, pretrained=self.pretrained)
        print("self.value_network", self.value_network)
        print("Value net params:", sum(p.numel()
              for p in self.value_network.parameters()))

    def initialize_hooks(self):
        self.gating_network.initialize_hooks()
        self.value_network.initialize_hooks()

    def clear_hooks(self):
        self.gating_network.clear_hooks()
        self.value_network.clear_hooks()

    def forward(self, inp, verbose=2):
        ip_device = inp.get_device()
        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=ip_device, non_blocking=True)

        inp_gating = torch.ones(inp.size(),
                                requires_grad=True, device=ip_device)

        before_relu_outputs = self.gating_network(inp, verbose=verbose)

        self.gating_node_outputs = OrderedDict()

        if(verbose > 3):
            print("before_relu_outputs keys")

        for key, value in before_relu_outputs.items():
            if(verbose > 3):
                print("key:{},value:{}".format(key, value.size()))

            self.gating_node_outputs[key] = nn.Sigmoid()(
                self.beta * value)

        if(verbose > 3):
            print("gating_node_outputs keys")
            for key, value in self.gating_node_outputs.items():
                print("key:{},value:{}".format(key, value.size()))

        final_layer_out = self.value_network(
            inp_gating, self.gating_node_outputs, verbose=verbose)

        return final_layer_out


class ResNet_Gating_Network(nn.Module):
    def __init__(self, resnet_type, input_channel, pretrained=False):
        super().__init__()
        self.resnet_type = resnet_type
        self.input_channel = input_channel
        self.pretrained = pretrained

        self.initialize_network()

    def initialize_network(self):
        self.f_id_hooks = []
        self.list_of_modules = []
        # Extracts the resnet type between "__"
        resnet_arch_type = self.resnet_type[self.resnet_type.index(
            "__")+2:self.resnet_type.rindex("__")]
        # Load the resnet model architecture
        self.resnet_instance = models.__dict__[
            resnet_arch_type](pretrained=self.pretrained)

        last_relu_name = get_last_relu(self.resnet_instance)
        convert_layers_after_last_relu_to_identity(
            self.resnet_instance, last_relu_name)
        # Replace relu activations with Identity functions
        convert_relu_to_identity(self.resnet_instance)

        self.list_of_modules.append(self.resnet_instance)

        self.list_of_modules = nn.ModuleList(self.list_of_modules)
        self.initialize_hooks()

    def initialize_hooks(self):
        self.clear_hooks()
        self.layer_outs = OrderedDict()
        prev_layer = None
        # Capture outputs of Identity module (earlier input to Relu module)
        for i, (name, layer) in enumerate(self.resnet_instance.named_modules()):
            if isinstance(layer, nn.Identity):
                self.f_id_hooks.append(prev_layer.register_forward_hook(
                    self.forward_identity_hook(name)))
            prev_layer = layer

    def forward_identity_hook(self, layer_name):
        def hook(module, input, output):
            self.layer_outs[layer_name] = output
        return hook

    def forward(self, inp, verbose=2):
        prev_out = inp
        for each_module in self.list_of_modules:
            prev_out = each_module(prev_out)

        return self.layer_outs

    def clear_hooks(self):
        for each_hook in self.f_id_hooks:
            each_hook.remove()
        self.f_id_hooks = []

    def __str__(self):
        ret = "Gate network pretrained?:" + \
            str(self.pretrained)+" \n module_list:"
        for each_module in self.list_of_modules:
            ret += str(each_module)+" \n Params in module is:" + \
                str(sum(p.numel() for p in each_module.parameters()))+"\n"

        return ret


class ALLONES_ResNet_Value_Network(nn.Module):
    def __init__(self, resnet_type, input_channel, num_classes=1000, pretrained=False):
        super(ALLONES_ResNet_Value_Network, self).__init__()
        self.pretrained = pretrained
        self.list_of_modules = []
        self.f_relu_hooks = []
        self.resnet_type = resnet_type
        # Extracts the resnet type between "__"
        resnet_arch_type = self.resnet_type[self.resnet_type.index(
            "__")+2:self.resnet_type.rindex("__")]
        # Load the resnet model architecture
        self.resnet_instance = models.__dict__[
            resnet_arch_type](pretrained=pretrained)

        # Replace relu activations with Identity functions
        convert_relu_to_identity(self.resnet_instance)

        num_ftrs = self.resnet_instance.fc.in_features
        self.resnet_instance.fc = nn.Linear(num_ftrs, num_classes)

        self.list_of_modules.append(self.resnet_instance)

        self.list_of_modules = nn.ModuleList(self.list_of_modules)
        self.initialize_hooks()

    def initialize_hooks(self):
        self.clear_hooks()
        self.gating_signals = None
        prev_layer = None

        all_devices = _get_all_device_indices()
        # Attaches hook to Identity and modify its inputs
        for i, (name, layer) in enumerate(self.resnet_instance.named_modules()):
            if isinstance(layer, nn.Identity):
                for each_device in all_devices:
                    buffer_name = "gating_signals" + \
                        str(each_device) + "__" + name
                    buffer_name = buffer_name.replace(".", "_")

                    self.register_buffer(buffer_name, torch.zeros(1))
                    print("Created ---buffer_name----", buffer_name)

                self.f_relu_hooks.append(
                    prev_layer.register_forward_hook(self.forward_hook(name)))
            prev_layer = layer

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            buffer_name = "gating_signals" + \
                str(torch.cuda.current_device()) + "__" + layer_name
            buffer_name = buffer_name.replace(".", "_")
            temp = self.get_buffer(buffer_name).to(device=output.get_device())
            return output * temp
        return hook

    def forward(self, inp, gating_signals, verbose=2):
        # iterating over the ordereddict
        for key, value in gating_signals.items():
            buffer_name = "gating_signals" + \
                str(torch.cuda.current_device()) + "__" + key
            buffer_name = buffer_name.replace(".", "_")
            setattr(self, buffer_name, value)

        prev_out = inp
        for each_module in self.list_of_modules:
            prev_out = each_module(prev_out)

        return prev_out

    def clear_hooks(self):
        for each_hook in self.f_relu_hooks:
            each_hook.remove()
        self.f_relu_hooks = []

    def __str__(self):
        ret = "Value network pretrained?" + \
            str(self.pretrained)+" \n module_list:"
        for each_module in self.list_of_modules:
            ret += str(each_module)+" \n Params in module is:" + \
                str(sum(p.numel() for p in each_module.parameters()))+"\n"

        return ret


def get_model_instance_from_dataset(dataset, model_arch_type, seed=2022, mask_percentage=40, num_classes=10, nodes_in_each_layer_list=[], pretrained=False):
    if(dataset == "cifar10"):
        inp_channel = 3
        input_size_list = [32, 32]
    elif(dataset == "mnist"):
        inp_channel = 1
        input_size_list = [28, 28]
    elif(dataset == "fashion_mnist"):
        inp_channel = 1
        input_size_list = [28, 28]
    elif(dataset == "imagenet_1000"):
        inp_channel = 3
        input_size_list = [224, 224]

    return get_model_instance(model_arch_type, inp_channel, seed=seed, mask_percentage=mask_percentage, num_classes=num_classes, input_size_list=input_size_list, nodes_in_each_layer_list=nodes_in_each_layer_list, pretrained=pretrained)


def get_model_instance(model_arch_type, inp_channel, seed=2022, mask_percentage=40, num_classes=10, nodes_in_each_layer_list=[], input_size_list=[], pretrained=False):
    if(seed == ""):
        seed = 2022

    net = None
    if(model_arch_type == 'plain_pure_conv4_dnn'):
        net = Plain_CONV4_Net(inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == 'conv4_dlgn'):
        net = Conv4_DLGN_Net(inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == 'conv4_dlgn_n16_small'):
        net = Conv4_DLGN_Net_N16_Small(
            inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == 'plain_pure_conv4_dnn_n16_small'):
        net = Plain_CONV4_Net_N16_Small(
            inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == 'conv4_deep_gated_net'):
        net = Conv4_DeepGated_Net(
            inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == 'conv4_deep_gated_net_n16_small'):
        net = Conv4_DeepGated_Net_N16_Small(
            inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == 'conv4_deep_gated_net_with_actual_inp_in_wt_net'):
        net = Conv4_DeepGated_Net_With_Actual_Inp_Over_WeightNet(
            inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == 'conv4_deep_gated_net_with_actual_inp_randomly_changed_in_wt_net'):
        net = Conv4_DeepGated_Net_With_Random_Actual_Inp_Over_WeightNet(
            inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == 'conv4_deep_gated_net_with_random_ones_in_wt_net'):
        net = Conv4_DeepGated_Net_With_Random_AllOnes_Over_WeightNet(
            inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == "masked_conv4_dlgn"):
        net = Mask_Conv4_DLGN_Net(
            inp_channel, random_inp_percent=mask_percentage, seed=seed, num_classes=num_classes)
    elif(model_arch_type == "masked_conv4_dlgn_n16_small"):
        net = Mask_Conv4_DLGN_Net_N16_Small(
            inp_channel, random_inp_percent=mask_percentage, seed=seed, num_classes=num_classes)
    elif(model_arch_type == "fc_dnn"):
        net = DNN_FC_Network(
            nodes_in_each_layer_list, seed=seed, input_size_list=input_size_list, num_classes=num_classes)
    elif(model_arch_type == "fc_dlgn"):
        net = DLGN_FC_Network(
            nodes_in_each_layer_list, seed=seed, input_size_list=input_size_list, num_classes=num_classes)
    elif(model_arch_type == "fc_dgn"):
        net = DGN_FC_Network(
            nodes_in_each_layer_list, seed=seed, input_size_list=input_size_list, num_classes=num_classes)
    elif('resnet' in model_arch_type and 'dlgn' in model_arch_type):
        net = ResNet_DLGN(
            model_arch_type, inp_channel, seed=seed, num_classes=num_classes, pretrained=pretrained)

    return net


def get_model_save_path(model_arch_type, dataset, seed="", list_of_classes_to_train_on_str=""):
    if(seed == ""):
        torch_seed_str = ""
    else:
        torch_seed_str = "/ST_"+str(seed)+"/"
    if(list_of_classes_to_train_on_str != ""):
        list_of_classes_to_train_on_str = "TR_ON_" + list_of_classes_to_train_on_str

    final_model_save_path = "root/model/save/" + \
        str(dataset)+"/CLEAN_TRAINING/"+str(list_of_classes_to_train_on_str)+"/" + \
        str(torch_seed_str)+str(model_arch_type)+"_dir.pt"

    return final_model_save_path

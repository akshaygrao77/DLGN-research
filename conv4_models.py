import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from structure.fc_models import DLGN_FC_Network, DNN_FC_Network, DGN_FC_Network
from utils.visualise_utils import determine_row_col_from_features
from sklearn.decomposition import PCA
from collections import OrderedDict
import torchvision.models as models
from torch._utils import _get_all_device_indices
from googlenet_custom import Custom_GoogLeNet
from utils.forward_visualization_helpers import merge_operations_in_modules, apply_input_on_conv_matrix, merge_layers_operations_in_modules


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


def convert_maxpool_to_avgpool(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.MaxPool2d):
            setattr(model, child_name, nn.AvgPool2d(kernel_size=child.kernel_size,
                    stride=child.stride, padding=child.padding, ceil_mode=child.ceil_mode))
        else:
            convert_maxpool_to_avgpool(child)


def get_first_layer_instance(model, layer, root_name=""):
    curr_last = None
    curr_last_layer = None
    for child_name, child in model.named_children():
        if isinstance(child, layer):
            curr_last = root_name + child_name
            curr_last_layer = child
            return curr_last, curr_last_layer
        else:
            c, cl = get_first_layer_instance(
                child, layer, root_name+child_name+".")
            if(c is not None):
                curr_last = c
                curr_last_layer = cl
                return curr_last, curr_last_layer
    return curr_last, curr_last_layer


def get_last_layer_instance(model, layer, root_name=""):
    curr_last = None
    curr_last_layer = None
    for child_name, child in model.named_children():
        if isinstance(child, layer):
            curr_last = root_name + child_name
            curr_last_layer = child
        else:
            c, cl = get_last_layer_instance(
                child, layer, root_name+child_name+".")
            if(c is not None):
                curr_last = c
                curr_last_layer = cl
    return curr_last, curr_last_layer


def replace_given_layer_name_with_layer(model, given_layer_name, replacement_layer_obj, root_name=""):
    for child_name, child in model.named_children():
        current_child_name = root_name + child_name
        if given_layer_name == current_child_name:
            setattr(model, child_name, replacement_layer_obj)
            return
        else:
            c = replace_given_layer_name_with_layer(
                child, given_layer_name, replacement_layer_obj, root_name+child_name+".")


def convert_identity_layers_after_first_linear_back_to_Relu(model, first_fc_lay_name, is_replace=False, root_name=""):
    for child_name, child in model.named_children():
        if (root_name + child_name) == first_fc_lay_name:
            is_replace = True
        elif(is_replace):
            if isinstance(child, nn.Identity):
                print("Nodes replaced back to relu are", root_name + child_name)
                setattr(model, child_name, nn.ReLU())
        else:
            is_replace = is_replace or convert_identity_layers_after_first_linear_back_to_Relu(
                child, first_fc_lay_name, is_replace, root_name+child_name+".")

    return is_replace


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


def convert_inplacerelu_to_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU())
        else:
            convert_inplacerelu_to_relu(child)


class dnn_st1_pad1_vgg16_bn_wo_bias(nn.Module):
    def __init__(self, init_weights: bool = True, num_classes: int = 10) -> None:
        super().__init__()

        self.conv1_v = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.conv2_v = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv3_v = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv4_v = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, bias=False)
        self.conv5_v = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, bias=False)
        self.conv6_v = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.conv7_v = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.conv8_v = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.conv9_v = nn.Conv2d(
            256, 512, kernel_size=3, padding=1, bias=False)
        self.conv10_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv11_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv12_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv13_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv14_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv15_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv16_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)

        self.bn1_v = nn.BatchNorm2d(64)
        self.bn2_v = nn.BatchNorm2d(64)
        self.bn3_v = nn.BatchNorm2d(128)
        self.bn4_v = nn.BatchNorm2d(128)

        self.bn5_v = nn.BatchNorm2d(256)
        self.bn6_v = nn.BatchNorm2d(256)
        self.bn7_v = nn.BatchNorm2d(256)
        self.bn8_v = nn.BatchNorm2d(256)

        self.bn9_v = nn.BatchNorm2d(512)
        self.bn10_v = nn.BatchNorm2d(512)
        self.bn11_v = nn.BatchNorm2d(512)
        self.bn12_v = nn.BatchNorm2d(512)

        self.bn13_v = nn.BatchNorm2d(512)
        self.bn14_v = nn.BatchNorm2d(512)
        self.bn15_v = nn.BatchNorm2d(512)
        self.bn16_v = nn.BatchNorm2d(512)

        self.fc1_v = nn.Linear(512*1*1, 4096)
        self.fc2_v = nn.Linear(4096, 4096)
        self.fc3_v = nn.Linear(4096, num_classes)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.globalpool = nn.AdaptiveAvgPool2d((1,1))

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        gating_net_layers_ordered["conv1_v"] = self.conv1_v
        gating_net_layers_ordered["bn1_v"] = self.bn1_v

        gating_net_layers_ordered["conv2_v"] = self.conv2_v
        gating_net_layers_ordered["bn2_v"] = self.bn2_v

        gating_net_layers_ordered["conv2_v"] = self.conv2_v
        gating_net_layers_ordered["bn2_v"] = self.bn2_v

        gating_net_layers_ordered["pool1"] = self.pool

        gating_net_layers_ordered["conv3_v"] = self.conv3_v
        gating_net_layers_ordered["bn3_v"] = self.bn3_v

        gating_net_layers_ordered["conv4_v"] = self.conv4_v
        gating_net_layers_ordered["bn4_v"] = self.bn4_v

        gating_net_layers_ordered["pool2"] = self.pool

        gating_net_layers_ordered["conv5_v"] = self.conv5_v
        gating_net_layers_ordered["bn5_v"] = self.bn5_v

        gating_net_layers_ordered["conv6_v"] = self.conv6_v
        gating_net_layers_ordered["bn6_v"] = self.bn6_v

        gating_net_layers_ordered["conv7_v"] = self.conv7_v
        gating_net_layers_ordered["bn7_v"] = self.bn7_v

        gating_net_layers_ordered["conv8_v"] = self.conv8_v
        gating_net_layers_ordered["bn8_v"] = self.bn8_v

        gating_net_layers_ordered["pool3"] = self.pool

        gating_net_layers_ordered["conv9_v"] = self.conv9_v
        gating_net_layers_ordered["bn9_v"] = self.bn9_v

        gating_net_layers_ordered["conv10_v"] = self.conv10_v
        gating_net_layers_ordered["bn10_v"] = self.bn10_v

        gating_net_layers_ordered["conv11_v"] = self.conv11_v
        gating_net_layers_ordered["bn11_v"] = self.bn11_v

        gating_net_layers_ordered["conv12_v"] = self.conv12_v
        gating_net_layers_ordered["bn12_v"] = self.bn12_v

        gating_net_layers_ordered["pool4"] = self.pool

        gating_net_layers_ordered["conv13_v"] = self.conv13_v
        gating_net_layers_ordered["bn13_v"] = self.bn13_v

        gating_net_layers_ordered["conv14_v"] = self.conv14_v
        gating_net_layers_ordered["bn14_v"] = self.bn14_v

        gating_net_layers_ordered["conv15_v"] = self.conv15_v
        gating_net_layers_ordered["bn15_v"] = self.bn15_v

        gating_net_layers_ordered["conv16_v"] = self.conv16_v
        gating_net_layers_ordered["bn16_v"] = self.bn16_v

        # gating_net_layers_ordered["avgpool"] = self.avgpool

        # gating_net_layers_ordered["fc1_g"] = self.fc1_g
        # gating_net_layers_ordered["fc2_g"] = self.fc2_g

        return gating_net_layers_ordered

    def forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        conv_matrix_operations_in_each_layer = OrderedDict()
        conv_bias_operations_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_matrix = None
        merged_conv_bias = None
        orig_out = x

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_matrix, merged_conv_bias, current_tensor_size = merge_operations_in_modules(
                    layer_obj, current_tensor_size, merged_conv_matrix, merged_conv_bias)
                conv_matrix_operations_in_each_layer[layer_name] = merged_conv_matrix
                conv_bias_operations_in_each_layer[layer_name] = merged_conv_bias

                orig_out = layer_obj(orig_out)

                convmatrix_output = apply_input_on_conv_matrix(
                    x, merged_conv_matrix, merged_conv_bias)
                convmatrix_output = torch.unsqueeze(torch.reshape(
                    convmatrix_output, current_tensor_size), 0)
                assert orig_out.size() == convmatrix_output.size(
                ), "Size of effective and actual output unequal"
                difference_in_output = (
                    orig_out - convmatrix_output).abs().sum()
                print("difference_in_output ", difference_in_output)

        return conv_matrix_operations_in_each_layer, conv_bias_operations_in_each_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature/Gating Network
        # Layer 1 : 64
        self.linear_conv_outputs = []

        # Layer 1 : 64
        x_v = self.conv1_v(x)
        x_v = self.bn1_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 2: 64
        x_v = self.conv2_v(x_v)
        x_v = self.bn2_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Max-Pool : 1
        x_v = self.pool(x_v)

        # Layer 3 : 128
        x_v = self.conv3_v(x_v)
        x_v = self.bn3_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 4 : 128
        x_v = self.conv4_v(x_v)
        x_v = self.bn4_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Max-Pool : 2
        x_v = self.pool(x_v)

        # Layer 5 : 256
        x_v = self.conv5_v(x_v)
        x_v = self.bn5_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 6 : 256
        x_v = self.conv6_v(x_v)
        x_v = self.bn6_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 7 : 256
        x_v = self.conv7_v(x_v)
        x_v = self.bn7_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 8 : 256
        x_v = self.conv8_v(x_v)
        x_v = self.bn8_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Max-Pool : 3
        x_v = self.pool(x_v)

        # Layer 9 : 512
        x_v = self.conv9_v(x_v)
        x_v = self.bn9_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 10 : 512
        x_v = self.conv10_v(x_v)
        x_v = self.bn10_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 11 : 512
        x_v = self.conv11_v(x_v)
        x_v = self.bn11_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 12 : 512
        x_v = self.conv12_v(x_v)
        x_v = self.bn12_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Max-Pool : 4
        x_v = self.pool(x_v)

        # Layer 13 : 512
        x_v = self.conv13_v(x_v)
        x_v = self.bn13_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 14 : 512
        x_v = self.conv14_v(x_v)
        x_v = self.bn14_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 15 : 512
        x_v = self.conv15_v(x_v)
        x_v = self.bn15_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 16 : 512
        x_v = self.conv16_v(x_v)
        x_v = self.bn16_v(x_v)
        x_v = F.relu(x_v)

        # Max-Pool : 5
        x_v = self.avgpool(x_v)

        x_v = torch.flatten(x_v, 1)
        x_v = self.fc1_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        #x_v = self.dropout(x_v)
        x_v = self.fc2_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        #x_v = self.dropout(x_v)
        x_v = self.fc3_v(x_v)

        return x_v


class dnn_vgg16_bn(nn.Module):
    def __init__(self, init_weights: bool = True, num_classes: int = 10) -> None:
        super().__init__()

        self.conv1_v = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2_v = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_v = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4_v = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5_v = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6_v = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7_v = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8_v = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv9_v = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv10_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv11_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv13_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv14_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv15_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv16_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.bn1_v = nn.BatchNorm2d(64)
        self.bn2_v = nn.BatchNorm2d(64)
        self.bn3_v = nn.BatchNorm2d(128)
        self.bn4_v = nn.BatchNorm2d(128)

        self.bn5_v = nn.BatchNorm2d(256)
        self.bn6_v = nn.BatchNorm2d(256)
        self.bn7_v = nn.BatchNorm2d(256)
        self.bn8_v = nn.BatchNorm2d(256)

        self.bn9_v = nn.BatchNorm2d(512)
        self.bn10_v = nn.BatchNorm2d(512)
        self.bn11_v = nn.BatchNorm2d(512)
        self.bn12_v = nn.BatchNorm2d(512)

        self.bn13_v = nn.BatchNorm2d(512)
        self.bn14_v = nn.BatchNorm2d(512)
        self.bn15_v = nn.BatchNorm2d(512)
        self.bn16_v = nn.BatchNorm2d(512)

        self.fc1_v = nn.Linear(512*1*1, 4096)
        self.fc2_v = nn.Linear(4096, 4096)
        self.fc3_v = nn.Linear(4096, num_classes)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.globalpool = nn.AdaptiveAvgPool2d((1,1))

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        gating_net_layers_ordered["conv1_v"] = self.conv1_v
        gating_net_layers_ordered["bn1_v"] = self.bn1_v

        gating_net_layers_ordered["conv2_v"] = self.conv2_v
        gating_net_layers_ordered["bn2_v"] = self.bn2_v

        gating_net_layers_ordered["conv2_v"] = self.conv2_v
        gating_net_layers_ordered["bn2_v"] = self.bn2_v

        gating_net_layers_ordered["pool1"] = self.pool

        gating_net_layers_ordered["conv3_v"] = self.conv3_v
        gating_net_layers_ordered["bn3_v"] = self.bn3_v

        gating_net_layers_ordered["conv4_v"] = self.conv4_v
        gating_net_layers_ordered["bn4_v"] = self.bn4_v

        gating_net_layers_ordered["pool2"] = self.pool

        gating_net_layers_ordered["conv5_v"] = self.conv5_v
        gating_net_layers_ordered["bn5_v"] = self.bn5_v

        gating_net_layers_ordered["conv6_v"] = self.conv6_v
        gating_net_layers_ordered["bn6_v"] = self.bn6_v

        gating_net_layers_ordered["conv7_v"] = self.conv7_v
        gating_net_layers_ordered["bn7_v"] = self.bn7_v

        gating_net_layers_ordered["conv8_v"] = self.conv8_v
        gating_net_layers_ordered["bn8_v"] = self.bn8_v

        gating_net_layers_ordered["pool3"] = self.pool

        gating_net_layers_ordered["conv9_v"] = self.conv9_v
        gating_net_layers_ordered["bn9_v"] = self.bn9_v

        gating_net_layers_ordered["conv10_v"] = self.conv10_v
        gating_net_layers_ordered["bn10_v"] = self.bn10_v

        gating_net_layers_ordered["conv11_v"] = self.conv11_v
        gating_net_layers_ordered["bn11_v"] = self.bn11_v

        gating_net_layers_ordered["conv12_v"] = self.conv12_v
        gating_net_layers_ordered["bn12_v"] = self.bn12_v

        gating_net_layers_ordered["pool4"] = self.pool

        gating_net_layers_ordered["conv13_v"] = self.conv13_v
        gating_net_layers_ordered["bn13_v"] = self.bn13_v

        gating_net_layers_ordered["conv14_v"] = self.conv14_v
        gating_net_layers_ordered["bn14_v"] = self.bn14_v

        gating_net_layers_ordered["conv15_v"] = self.conv15_v
        gating_net_layers_ordered["bn15_v"] = self.bn15_v

        gating_net_layers_ordered["conv16_v"] = self.conv16_v
        gating_net_layers_ordered["bn16_v"] = self.bn16_v

        # gating_net_layers_ordered["avgpool"] = self.avgpool

        # gating_net_layers_ordered["fc1_g"] = self.fc1_g
        # gating_net_layers_ordered["fc2_g"] = self.fc2_g

        return gating_net_layers_ordered

    def forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        conv_matrix_operations_in_each_layer = OrderedDict()
        conv_bias_operations_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_matrix = None
        merged_conv_bias = None
        orig_out = x

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_matrix, merged_conv_bias, current_tensor_size = merge_operations_in_modules(
                    layer_obj, current_tensor_size, merged_conv_matrix, merged_conv_bias)
                conv_matrix_operations_in_each_layer[layer_name] = merged_conv_matrix
                conv_bias_operations_in_each_layer[layer_name] = merged_conv_bias

                orig_out = layer_obj(orig_out)

                convmatrix_output = apply_input_on_conv_matrix(
                    x, merged_conv_matrix, merged_conv_bias)
                convmatrix_output = torch.unsqueeze(torch.reshape(
                    convmatrix_output, current_tensor_size), 0)
                assert orig_out.size() == convmatrix_output.size(
                ), "Size of effective and actual output unequal"
                difference_in_output = (
                    orig_out - convmatrix_output).abs().sum()
                print("difference_in_output ", difference_in_output)

        return conv_matrix_operations_in_each_layer, conv_bias_operations_in_each_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature/Gating Network
        # Layer 1 : 64
        self.linear_conv_outputs = []

        # Layer 1 : 64
        x_v = self.conv1_v(x)
        x_v = self.bn1_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 2: 64
        x_v = self.conv2_v(x_v)
        x_v = self.bn2_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Max-Pool : 1
        x_v = self.pool(x_v)

        # Layer 3 : 128
        x_v = self.conv3_v(x_v)
        x_v = self.bn3_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 4 : 128
        x_v = self.conv4_v(x_v)
        x_v = self.bn4_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Max-Pool : 2
        x_v = self.pool(x_v)

        # Layer 5 : 256
        x_v = self.conv5_v(x_v)
        x_v = self.bn5_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 6 : 256
        x_v = self.conv6_v(x_v)
        x_v = self.bn6_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 7 : 256
        x_v = self.conv7_v(x_v)
        x_v = self.bn7_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 8 : 256
        x_v = self.conv8_v(x_v)
        x_v = self.bn8_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Max-Pool : 3
        x_v = self.pool(x_v)

        # Layer 9 : 512
        x_v = self.conv9_v(x_v)
        x_v = self.bn9_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 10 : 512
        x_v = self.conv10_v(x_v)
        x_v = self.bn10_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 11 : 512
        x_v = self.conv11_v(x_v)
        x_v = self.bn11_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 12 : 512
        x_v = self.conv12_v(x_v)
        x_v = self.bn12_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Max-Pool : 4
        x_v = self.pool(x_v)

        # Layer 13 : 512
        x_v = self.conv13_v(x_v)
        x_v = self.bn13_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 14 : 512
        x_v = self.conv14_v(x_v)
        x_v = self.bn14_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 15 : 512
        x_v = self.conv15_v(x_v)
        x_v = self.bn15_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Layer 16 : 512
        x_v = self.conv16_v(x_v)
        x_v = self.bn16_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        # Max-Pool : 5
        x_v = self.avgpool(x_v)

        x_v = torch.flatten(x_v, 1)
        x_v = self.fc1_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        #x_v = self.dropout(x_v)
        x_v = self.fc2_v(x_v)
        self.linear_conv_outputs.append(x_v)
        x_v = F.relu(x_v)

        #x_v = self.dropout(x_v)
        x_v = self.fc3_v(x_v)

        return x_v


class vgg16_bn(nn.Module):
    def __init__(self, allones, init_weights: bool = True, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1_g = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2_g = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_g = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4_g = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5_g = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6_g = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7_g = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8_g = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv9_g = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv10_g = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv11_g = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12_g = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv13_g = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv14_g = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv15_g = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv16_g = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv1_v = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2_v = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_v = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4_v = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5_v = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6_v = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7_v = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8_v = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv9_v = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv10_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv11_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv13_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv14_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv15_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv16_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.bn1_g = nn.BatchNorm2d(64)
        self.bn2_g = nn.BatchNorm2d(64)
        self.bn3_g = nn.BatchNorm2d(128)
        self.bn4_g = nn.BatchNorm2d(128)

        self.bn5_g = nn.BatchNorm2d(256)
        self.bn6_g = nn.BatchNorm2d(256)
        self.bn7_g = nn.BatchNorm2d(256)
        self.bn8_g = nn.BatchNorm2d(256)

        self.bn9_g = nn.BatchNorm2d(512)
        self.bn10_g = nn.BatchNorm2d(512)
        self.bn11_g = nn.BatchNorm2d(512)
        self.bn12_g = nn.BatchNorm2d(512)

        self.bn13_g = nn.BatchNorm2d(512)
        self.bn14_g = nn.BatchNorm2d(512)
        self.bn15_g = nn.BatchNorm2d(512)
        self.bn16_g = nn.BatchNorm2d(512)

        self.bn1_v = nn.BatchNorm2d(64)
        self.bn2_v = nn.BatchNorm2d(64)
        self.bn3_v = nn.BatchNorm2d(128)
        self.bn4_v = nn.BatchNorm2d(128)

        self.bn5_v = nn.BatchNorm2d(256)
        self.bn6_v = nn.BatchNorm2d(256)
        self.bn7_v = nn.BatchNorm2d(256)
        self.bn8_v = nn.BatchNorm2d(256)

        self.bn9_v = nn.BatchNorm2d(512)
        self.bn10_v = nn.BatchNorm2d(512)
        self.bn11_v = nn.BatchNorm2d(512)
        self.bn12_v = nn.BatchNorm2d(512)

        self.bn13_v = nn.BatchNorm2d(512)
        self.bn14_v = nn.BatchNorm2d(512)
        self.bn15_v = nn.BatchNorm2d(512)
        self.bn16_v = nn.BatchNorm2d(512)

        self.fc1_g = nn.Linear(512*1*1, 4096)
        self.fc2_g = nn.Linear(4096, 4096)

        self.fc1_v = nn.Linear(512*1*1, 4096)
        self.fc2_v = nn.Linear(4096, 4096)
        self.fc3_v = nn.Linear(4096, num_classes)

        self.sig = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.allones = torch.ones(size=(1, 3, 32, 32))

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        gating_net_layers_ordered["conv1_g"] = self.conv1_g
        gating_net_layers_ordered["bn1_g"] = self.bn1_g

        gating_net_layers_ordered["conv2_g"] = self.conv2_g
        gating_net_layers_ordered["bn2_g"] = self.bn2_g

        gating_net_layers_ordered["conv2_g"] = self.conv2_g
        gating_net_layers_ordered["bn2_g"] = self.bn2_g

        gating_net_layers_ordered["pool1"] = self.pool

        gating_net_layers_ordered["conv3_g"] = self.conv3_g
        gating_net_layers_ordered["bn3_g"] = self.bn3_g

        gating_net_layers_ordered["conv4_g"] = self.conv4_g
        gating_net_layers_ordered["bn4_g"] = self.bn4_g

        gating_net_layers_ordered["pool2"] = self.pool

        gating_net_layers_ordered["conv5_g"] = self.conv5_g
        gating_net_layers_ordered["bn5_g"] = self.bn5_g

        gating_net_layers_ordered["conv6_g"] = self.conv6_g
        gating_net_layers_ordered["bn6_g"] = self.bn6_g

        gating_net_layers_ordered["conv7_g"] = self.conv7_g
        gating_net_layers_ordered["bn7_g"] = self.bn7_g

        gating_net_layers_ordered["conv8_g"] = self.conv8_g
        gating_net_layers_ordered["bn8_g"] = self.bn8_g

        gating_net_layers_ordered["pool3"] = self.pool

        gating_net_layers_ordered["conv9_g"] = self.conv9_g
        gating_net_layers_ordered["bn9_g"] = self.bn9_g

        gating_net_layers_ordered["conv10_g"] = self.conv10_g
        gating_net_layers_ordered["bn10_g"] = self.bn10_g

        gating_net_layers_ordered["conv11_g"] = self.conv11_g
        gating_net_layers_ordered["bn11_g"] = self.bn11_g

        gating_net_layers_ordered["conv12_g"] = self.conv12_g
        gating_net_layers_ordered["bn12_g"] = self.bn12_g

        gating_net_layers_ordered["pool4"] = self.pool

        gating_net_layers_ordered["conv13_g"] = self.conv13_g
        gating_net_layers_ordered["bn13_g"] = self.bn13_g

        gating_net_layers_ordered["conv14_g"] = self.conv14_g
        gating_net_layers_ordered["bn14_g"] = self.bn14_g

        gating_net_layers_ordered["conv15_g"] = self.conv15_g
        gating_net_layers_ordered["bn15_g"] = self.bn15_g

        gating_net_layers_ordered["conv16_g"] = self.conv16_g
        gating_net_layers_ordered["bn16_g"] = self.bn16_g

        # gating_net_layers_ordered["avgpool"] = self.avgpool

        # gating_net_layers_ordered["fc1_g"] = self.fc1_g
        # gating_net_layers_ordered["fc2_g"] = self.fc2_g

        return gating_net_layers_ordered

    def exact_forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        conv_matrix_operations_in_each_layer = OrderedDict()
        conv_bias_operations_in_each_layer = OrderedDict()
        channel_outs_size_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_matrix = None
        merged_conv_bias = None
        orig_out = x
        if(x.get_device() >= 0):
            x = x.type(torch.float16)

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_matrix, merged_conv_bias, current_tensor_size = merge_operations_in_modules(
                    layer_obj, current_tensor_size, merged_conv_matrix, merged_conv_bias)
                conv_matrix_operations_in_each_layer[layer_name] = merged_conv_matrix.cpu(
                )
                conv_bias_operations_in_each_layer[layer_name] = merged_conv_bias.cpu(
                )
                channel_outs_size_in_each_layer[layer_name] = current_tensor_size[0]

                orig_out = layer_obj(orig_out)

                convmatrix_output = apply_input_on_conv_matrix(
                    x, merged_conv_matrix, merged_conv_bias)
                convmatrix_output = torch.unsqueeze(torch.reshape(
                    convmatrix_output, current_tensor_size), 0)
                assert orig_out.size() == convmatrix_output.size(
                ), "Size of effective and actual output unequal"
                difference_in_output = (
                    orig_out - convmatrix_output).abs().sum()
                print("difference_in_output ", difference_in_output)

        return conv_matrix_operations_in_each_layer, conv_bias_operations_in_each_layer, channel_outs_size_in_each_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature/Gating Network
        # Layer 1 : 64
        self.linear_conv_outputs = []

        x_g = self.conv1_g(x)
        x_g = self.bn1_g(x_g)
        g_1 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x_g = F.relu(x_g)

        # Layer 2: 64
        x_g = self.conv2_g(x_g)
        x_g = self.bn2_g(x_g)
        g_2 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 1
        x_g = self.pool(x_g)

        # Layer 3 : 128
        x_g = self.conv3_g(x_g)
        x_g = self.bn3_g(x_g)
        g_3 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 4 : 128
        x_g = self.conv4_g(x_g)
        x_g = self.bn4_g(x_g)
        g_4 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 2
        x_g = self.pool(x_g)

        # Layer 5 : 256
        x_g = self.conv5_g(x_g)
        x_g = self.bn5_g(x_g)
        g_5 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 6 : 256
        x_g = self.conv6_g(x_g)
        x_g = self.bn6_g(x_g)
        g_6 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 7 : 256
        x_g = self.conv7_g(x_g)
        x_g = self.bn7_g(x_g)
        g_7 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 8 : 256
        x_g = self.conv8_g(x_g)
        x_g = self.bn8_g(x_g)
        g_8 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 3
        x_g = self.pool(x_g)

        # Layer 9 : 512
        x_g = self.conv9_g(x_g)
        x_g = self.bn9_g(x_g)
        g_9 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 10 : 512
        x_g = self.conv10_g(x_g)
        x_g = self.bn10_g(x_g)
        g_10 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 11 : 512
        x_g = self.conv11_g(x_g)
        x_g = self.bn11_g(x_g)
        g_11 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 12 : 512
        x_g = self.conv12_g(x_g)
        x_g = self.bn12_g(x_g)
        g_12 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 4
        x_g = self.pool(x_g)

        # Layer 13 : 512
        x_g = self.conv13_g(x_g)
        x_g = self.bn13_g(x_g)
        g_13 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 14 : 512
        x_g = self.conv14_g(x_g)
        x_g = self.bn14_g(x_g)
        g_14 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 15 : 512
        x_g = self.conv15_g(x_g)
        x_g = self.bn15_g(x_g)
        g_15 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 16 : 512
        x_g = self.conv16_g(x_g)
        x_g = self.bn16_g(x_g)
        g_16 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 5
        # x_g = self.pool(x_g)

        x_g = self.avgpool(x_g)
        x_g = torch.flatten(x_g, 1)
        x_g = self.fc1_g(x_g)
        g_17 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)
        #x_g = self.dropout(x_g)
        x_g = self.fc2_g(x_g)
        g_18 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Value Network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Layer 1 : 64
        x_v = self.conv1_v(self.allones.to(device))
        x_v = self.bn1_v(x_v)
        x_v = x_v*g_1
        #x_v = F.relu(x_v)

        # Layer 2: 64
        x_v = self.conv2_v(x_v)
        x_v = self.bn2_v(x_v)
        x_v = x_v*g_2
        #x = F.relu(x)

        # Max-Pool : 1
        x_v = self.pool(x_v)

        # Layer 3 : 128
        x_v = self.conv3_v(x_v)
        x_v = self.bn3_v(x_v)
        x_v = x_v*g_3
        #x = F.relu(x)

        # Layer 4 : 128
        x_v = self.conv4_v(x_v)
        x_v = self.bn4_v(x_v)
        x_v = x_v*g_4
        #x = F.relu(x)

        # Max-Pool : 2
        x_v = self.pool(x_v)

        # Layer 5 : 256
        x_v = self.conv5_v(x_v)
        x_v = self.bn5_v(x_v)
        x_v = x_v*g_5
        #x = F.relu(x)

        # Layer 6 : 256
        x_v = self.conv6_v(x_v)
        x_v = self.bn6_v(x_v)
        x_v = x_v*g_6
        #x = F.relu(x)

        # Layer 7 : 256
        x_v = self.conv7_v(x_v)
        x_v = self.bn7_v(x_v)
        x_v = x_v*g_7
        #x = F.relu(x)

        # Layer 8 : 256
        x_v = self.conv8_v(x_v)
        x_v = self.bn8_v(x_v)
        x_v = x_v*g_8
        #x = F.relu(x)

        # Max-Pool : 3
        x_v = self.pool(x_v)

        # Layer 9 : 512
        x_v = self.conv9_v(x_v)
        x_v = self.bn9_v(x_v)
        x_v = x_v*g_9
        #x = F.relu(x)

        # Layer 10 : 512
        x_v = self.conv10_v(x_v)
        x_v = self.bn10_v(x_v)
        x_v = x_v*g_10
        #x = F.relu(x)

        # Layer 11 : 512
        x_v = self.conv11_v(x_v)
        x_v = self.bn11_v(x_v)
        x_v = x_v*g_11
        #x = F.relu(x)

        # Layer 12 : 512
        x_v = self.conv12_v(x_v)
        x_v = self.bn12_v(x_v)
        x_v = x_v*g_12
        #x = F.relu(x)

        # Max-Pool : 4
        x_v = self.pool(x_v)

        # Layer 13 : 512
        x_v = self.conv13_v(x_v)
        x_v = self.bn13_v(x_v)
        x_v = x_v*g_13
        #x = F.relu(x)

        # Layer 14 : 512
        x_v = self.conv14_v(x_v)
        x_v = self.bn14_v(x_v)
        x_v = x_v*g_14
        #x = F.relu(x)

        # Layer 15 : 512
        x_v = self.conv15_v(x_v)
        x_v = self.bn15_v(x_v)
        x_v = x_v*g_15
        #x = F.relu(x)

        # Layer 16 : 512
        x_v = self.conv16_v(x_v)
        x_v = self.bn16_v(x_v)
        x_v = x_v*g_16
        #x = F.relu(x)

        # Max-Pool : 5
        x_v = self.avgpool(x_v)

        x_v = torch.flatten(x_v, 1)
        x_v = self.fc1_v(x_v)
        x_v = x_v*g_17

        #x_v = self.dropout(x_v)
        x_v = self.fc2_v(x_v)
        x_v = x_v*g_18
        #x = F.relu(x)
        #x_v = self.dropout(x_v)
        x_v = self.fc3_v(x_v)

        return x_v


class pad2_vgg16_bn(nn.Module):
    def __init__(self, init_weights: bool = True, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1_g = nn.Conv2d(3, 64, kernel_size=3, padding=2)
        self.conv2_g = nn.Conv2d(64, 64, kernel_size=3, padding=2)
        self.conv3_g = nn.Conv2d(64, 128, kernel_size=3, padding=2)
        self.conv4_g = nn.Conv2d(128, 128, kernel_size=3, padding=2)
        self.conv5_g = nn.Conv2d(128, 256, kernel_size=3, padding=2)
        self.conv6_g = nn.Conv2d(256, 256, kernel_size=3, padding=2)
        self.conv7_g = nn.Conv2d(256, 256, kernel_size=3, padding=2)
        self.conv8_g = nn.Conv2d(256, 256, kernel_size=3, padding=2)
        self.conv9_g = nn.Conv2d(256, 512, kernel_size=3, padding=2)
        self.conv10_g = nn.Conv2d(512, 512, kernel_size=3, padding=2)
        self.conv11_g = nn.Conv2d(512, 512, kernel_size=3, padding=2)
        self.conv12_g = nn.Conv2d(512, 512, kernel_size=3, padding=2)
        self.conv13_g = nn.Conv2d(512, 512, kernel_size=3, padding=2)
        self.conv14_g = nn.Conv2d(512, 512, kernel_size=3, padding=2)
        self.conv15_g = nn.Conv2d(512, 512, kernel_size=3, padding=2)
        self.conv16_g = nn.Conv2d(512, 512, kernel_size=3, padding=2)

        self.conv1_v = nn.Conv2d(3, 64, kernel_size=3, padding=2)
        self.conv2_v = nn.Conv2d(64, 64, kernel_size=3, padding=2)
        self.conv3_v = nn.Conv2d(64, 128, kernel_size=3, padding=2)
        self.conv4_v = nn.Conv2d(128, 128, kernel_size=3, padding=2)
        self.conv5_v = nn.Conv2d(128, 256, kernel_size=3, padding=2)
        self.conv6_v = nn.Conv2d(256, 256, kernel_size=3, padding=2)
        self.conv7_v = nn.Conv2d(256, 256, kernel_size=3, padding=2)
        self.conv8_v = nn.Conv2d(256, 256, kernel_size=3, padding=2)
        self.conv9_v = nn.Conv2d(256, 512, kernel_size=3, padding=2)
        self.conv10_v = nn.Conv2d(512, 512, kernel_size=3, padding=2)
        self.conv11_v = nn.Conv2d(512, 512, kernel_size=3, padding=2)
        self.conv12_v = nn.Conv2d(512, 512, kernel_size=3, padding=2)
        self.conv13_v = nn.Conv2d(512, 512, kernel_size=3, padding=2)
        self.conv14_v = nn.Conv2d(512, 512, kernel_size=3, padding=2)
        self.conv15_v = nn.Conv2d(512, 512, kernel_size=3, padding=2)
        self.conv16_v = nn.Conv2d(512, 512, kernel_size=3, padding=2)

        self.bn1_g = nn.BatchNorm2d(64)
        self.bn2_g = nn.BatchNorm2d(64)
        self.bn3_g = nn.BatchNorm2d(128)
        self.bn4_g = nn.BatchNorm2d(128)

        self.bn5_g = nn.BatchNorm2d(256)
        self.bn6_g = nn.BatchNorm2d(256)
        self.bn7_g = nn.BatchNorm2d(256)
        self.bn8_g = nn.BatchNorm2d(256)

        self.bn9_g = nn.BatchNorm2d(512)
        self.bn10_g = nn.BatchNorm2d(512)
        self.bn11_g = nn.BatchNorm2d(512)
        self.bn12_g = nn.BatchNorm2d(512)

        self.bn13_g = nn.BatchNorm2d(512)
        self.bn14_g = nn.BatchNorm2d(512)
        self.bn15_g = nn.BatchNorm2d(512)
        self.bn16_g = nn.BatchNorm2d(512)

        self.bn1_v = nn.BatchNorm2d(64)
        self.bn2_v = nn.BatchNorm2d(64)
        self.bn3_v = nn.BatchNorm2d(128)
        self.bn4_v = nn.BatchNorm2d(128)

        self.bn5_v = nn.BatchNorm2d(256)
        self.bn6_v = nn.BatchNorm2d(256)
        self.bn7_v = nn.BatchNorm2d(256)
        self.bn8_v = nn.BatchNorm2d(256)

        self.bn9_v = nn.BatchNorm2d(512)
        self.bn10_v = nn.BatchNorm2d(512)
        self.bn11_v = nn.BatchNorm2d(512)
        self.bn12_v = nn.BatchNorm2d(512)

        self.bn13_v = nn.BatchNorm2d(512)
        self.bn14_v = nn.BatchNorm2d(512)
        self.bn15_v = nn.BatchNorm2d(512)
        self.bn16_v = nn.BatchNorm2d(512)

        self.fc1_g = nn.Linear(512*1*1, 4096)
        self.fc2_g = nn.Linear(4096, 4096)

        self.fc1_v = nn.Linear(512*1*1, 4096)
        self.fc2_v = nn.Linear(4096, 4096)
        self.fc3_v = nn.Linear(4096, num_classes)

        self.sig = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.allones = torch.ones(size=(1, 3, 32, 32))

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        gating_net_layers_ordered["conv1_g"] = self.conv1_g
        gating_net_layers_ordered["bn1_g"] = self.bn1_g

        gating_net_layers_ordered["conv2_g"] = self.conv2_g
        gating_net_layers_ordered["bn2_g"] = self.bn2_g

        gating_net_layers_ordered["conv2_g"] = self.conv2_g
        gating_net_layers_ordered["bn2_g"] = self.bn2_g

        gating_net_layers_ordered["pool1"] = self.pool

        gating_net_layers_ordered["conv3_g"] = self.conv3_g
        gating_net_layers_ordered["bn3_g"] = self.bn3_g

        gating_net_layers_ordered["conv4_g"] = self.conv4_g
        gating_net_layers_ordered["bn4_g"] = self.bn4_g

        gating_net_layers_ordered["pool2"] = self.pool

        gating_net_layers_ordered["conv5_g"] = self.conv5_g
        gating_net_layers_ordered["bn5_g"] = self.bn5_g

        gating_net_layers_ordered["conv6_g"] = self.conv6_g
        gating_net_layers_ordered["bn6_g"] = self.bn6_g

        gating_net_layers_ordered["conv7_g"] = self.conv7_g
        gating_net_layers_ordered["bn7_g"] = self.bn7_g

        gating_net_layers_ordered["conv8_g"] = self.conv8_g
        gating_net_layers_ordered["bn8_g"] = self.bn8_g

        gating_net_layers_ordered["pool3"] = self.pool

        gating_net_layers_ordered["conv9_g"] = self.conv9_g
        gating_net_layers_ordered["bn9_g"] = self.bn9_g

        gating_net_layers_ordered["conv10_g"] = self.conv10_g
        gating_net_layers_ordered["bn10_g"] = self.bn10_g

        gating_net_layers_ordered["conv11_g"] = self.conv11_g
        gating_net_layers_ordered["bn11_g"] = self.bn11_g

        gating_net_layers_ordered["conv12_g"] = self.conv12_g
        gating_net_layers_ordered["bn12_g"] = self.bn12_g

        gating_net_layers_ordered["pool4"] = self.pool

        gating_net_layers_ordered["conv13_g"] = self.conv13_g
        gating_net_layers_ordered["bn13_g"] = self.bn13_g

        gating_net_layers_ordered["conv14_g"] = self.conv14_g
        gating_net_layers_ordered["bn14_g"] = self.bn14_g

        gating_net_layers_ordered["conv15_g"] = self.conv15_g
        gating_net_layers_ordered["bn15_g"] = self.bn15_g

        gating_net_layers_ordered["conv16_g"] = self.conv16_g
        gating_net_layers_ordered["bn16_g"] = self.bn16_g

        # gating_net_layers_ordered["avgpool"] = self.avgpool

        # gating_net_layers_ordered["fc1_g"] = self.fc1_g
        # gating_net_layers_ordered["fc2_g"] = self.fc2_g

        return gating_net_layers_ordered

    def forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        merged_conv_layer_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_layer = None
        orig_out = x
        if(self.conv1_g.weight.get_device() < 0):
            idevice = torch.device("cpu")
        else:
            idevice = self.conv1_g.weight.get_device()
        lay_type = self.conv1_g.weight.dtype

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_layer, _ = merge_layers_operations_in_modules(
                    layer_obj, current_tensor_size, lay_type, idevice, merged_conv_layer)
                merged_conv_layer_in_each_layer[layer_name] = merged_conv_layer

                orig_out = layer_obj(orig_out)

                merged_conv_output = merged_conv_layer(x)
                current_tensor_size = merged_conv_output.size()[1:]

                print("orig_out.size():{} merged_conv_output.size():{}".format(
                    orig_out.size(), merged_conv_output.size()))
                assert orig_out.size() == merged_conv_output.size(
                ), "Size of effective and actual output unequal"
                difference_in_output = (
                    orig_out - merged_conv_output).abs().sum()
                print("difference_in_output ", difference_in_output)

        return merged_conv_layer_in_each_layer, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature/Gating Network
        # Layer 1 : 64

        x_g = self.conv1_g(x)
        x_g = self.bn1_g(x_g)
        g_1 = self.sig(10*x_g)
        #x_g = F.relu(x_g)

        # Layer 2: 64
        x_g = self.conv2_g(x_g)
        x_g = self.bn2_g(x_g)
        g_2 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 1
        x_g = self.pool(x_g)

        # Layer 3 : 128
        x_g = self.conv3_g(x_g)
        x_g = self.bn3_g(x_g)
        g_3 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 4 : 128
        x_g = self.conv4_g(x_g)
        x_g = self.bn4_g(x_g)
        g_4 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 2
        x_g = self.pool(x_g)

        # Layer 5 : 256
        x_g = self.conv5_g(x_g)
        x_g = self.bn5_g(x_g)
        g_5 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 6 : 256
        x_g = self.conv6_g(x_g)
        x_g = self.bn6_g(x_g)
        g_6 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 7 : 256
        x_g = self.conv7_g(x_g)
        x_g = self.bn7_g(x_g)
        g_7 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 8 : 256
        x_g = self.conv8_g(x_g)
        x_g = self.bn8_g(x_g)
        g_8 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 3
        x_g = self.pool(x_g)

        # Layer 9 : 512
        x_g = self.conv9_g(x_g)
        x_g = self.bn9_g(x_g)
        g_9 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 10 : 512
        x_g = self.conv10_g(x_g)
        x_g = self.bn10_g(x_g)
        g_10 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 11 : 512
        x_g = self.conv11_g(x_g)
        x_g = self.bn11_g(x_g)
        g_11 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 12 : 512
        x_g = self.conv12_g(x_g)
        x_g = self.bn12_g(x_g)
        g_12 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 4
        x_g = self.pool(x_g)

        # Layer 13 : 512
        x_g = self.conv13_g(x_g)
        x_g = self.bn13_g(x_g)
        g_13 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 14 : 512
        x_g = self.conv14_g(x_g)
        x_g = self.bn14_g(x_g)
        g_14 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 15 : 512
        x_g = self.conv15_g(x_g)
        x_g = self.bn15_g(x_g)
        g_15 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 16 : 512
        x_g = self.conv16_g(x_g)
        x_g = self.bn16_g(x_g)
        g_16 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 5
        # x_g = self.pool(x_g)

        x_g = self.avgpool(x_g)
        x_g = torch.flatten(x_g, 1)
        x_g = self.fc1_g(x_g)
        g_17 = self.sig(10*x_g)
        #x = F.relu(x)
        #x_g = self.dropout(x_g)
        x_g = self.fc2_g(x_g)
        g_18 = self.sig(10*x_g)
        #x = F.relu(x)

        # Value Network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Layer 1 : 64
        x_v = self.conv1_v(self.allones.to(device))
        x_v = self.bn1_v(x_v)
        x_v = x_v*g_1
        #x_v = F.relu(x_v)

        # Layer 2: 64
        x_v = self.conv2_v(x_v)
        x_v = self.bn2_v(x_v)
        x_v = x_v*g_2
        #x = F.relu(x)

        # Max-Pool : 1
        x_v = self.pool(x_v)

        # Layer 3 : 128
        x_v = self.conv3_v(x_v)
        x_v = self.bn3_v(x_v)
        x_v = x_v*g_3
        #x = F.relu(x)

        # Layer 4 : 128
        x_v = self.conv4_v(x_v)
        x_v = self.bn4_v(x_v)
        x_v = x_v*g_4
        #x = F.relu(x)

        # Max-Pool : 2
        x_v = self.pool(x_v)

        # Layer 5 : 256
        x_v = self.conv5_v(x_v)
        x_v = self.bn5_v(x_v)
        x_v = x_v*g_5
        #x = F.relu(x)

        # Layer 6 : 256
        x_v = self.conv6_v(x_v)
        x_v = self.bn6_v(x_v)
        x_v = x_v*g_6
        #x = F.relu(x)

        # Layer 7 : 256
        x_v = self.conv7_v(x_v)
        x_v = self.bn7_v(x_v)
        x_v = x_v*g_7
        #x = F.relu(x)

        # Layer 8 : 256
        x_v = self.conv8_v(x_v)
        x_v = self.bn8_v(x_v)
        x_v = x_v*g_8
        #x = F.relu(x)

        # Max-Pool : 3
        x_v = self.pool(x_v)

        # Layer 9 : 512
        x_v = self.conv9_v(x_v)
        x_v = self.bn9_v(x_v)
        x_v = x_v*g_9
        #x = F.relu(x)

        # Layer 10 : 512
        x_v = self.conv10_v(x_v)
        x_v = self.bn10_v(x_v)
        x_v = x_v*g_10
        #x = F.relu(x)

        # Layer 11 : 512
        x_v = self.conv11_v(x_v)
        x_v = self.bn11_v(x_v)
        x_v = x_v*g_11
        #x = F.relu(x)

        # Layer 12 : 512
        x_v = self.conv12_v(x_v)
        x_v = self.bn12_v(x_v)
        x_v = x_v*g_12
        #x = F.relu(x)

        # Max-Pool : 4
        x_v = self.pool(x_v)

        # Layer 13 : 512
        x_v = self.conv13_v(x_v)
        x_v = self.bn13_v(x_v)
        x_v = x_v*g_13
        #x = F.relu(x)

        # Layer 14 : 512
        x_v = self.conv14_v(x_v)
        x_v = self.bn14_v(x_v)
        x_v = x_v*g_14
        #x = F.relu(x)

        # Layer 15 : 512
        x_v = self.conv15_v(x_v)
        x_v = self.bn15_v(x_v)
        x_v = x_v*g_15
        #x = F.relu(x)

        # Layer 16 : 512
        x_v = self.conv16_v(x_v)
        x_v = self.bn16_v(x_v)
        x_v = x_v*g_16
        #x = F.relu(x)

        # Max-Pool : 5
        x_v = self.avgpool(x_v)

        x_v = torch.flatten(x_v, 1)
        x_v = self.fc1_v(x_v)
        x_v = x_v*g_17

        #x_v = self.dropout(x_v)
        x_v = self.fc2_v(x_v)
        x_v = x_v*g_18
        #x = F.relu(x)
        #x_v = self.dropout(x_v)
        x_v = self.fc3_v(x_v)

        return x_v


class st1_pad2_vgg16_bn_wo_bias(nn.Module):
    def __init__(self, init_weights: bool = True, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1_g = nn.Conv2d(3, 64, kernel_size=3, padding=2, bias=False)
        self.conv2_g = nn.Conv2d(64, 64, kernel_size=3, padding=2, bias=False)
        self.conv3_g = nn.Conv2d(64, 128, kernel_size=3, padding=2, bias=False)
        self.conv4_g = nn.Conv2d(
            128, 128, kernel_size=3, padding=2, bias=False)
        self.conv5_g = nn.Conv2d(
            128, 256, kernel_size=3, padding=2, bias=False)
        self.conv6_g = nn.Conv2d(
            256, 256, kernel_size=3, padding=2, bias=False)
        self.conv7_g = nn.Conv2d(
            256, 256, kernel_size=3, padding=2, bias=False)
        self.conv8_g = nn.Conv2d(
            256, 256, kernel_size=3, padding=2, bias=False)
        self.conv9_g = nn.Conv2d(
            256, 512, kernel_size=3, padding=2, bias=False)
        self.conv10_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)
        self.conv11_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)
        self.conv12_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)
        self.conv13_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)
        self.conv14_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)
        self.conv15_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)
        self.conv16_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)

        self.conv1_v = nn.Conv2d(3, 64, kernel_size=3, padding=2, bias=False)
        self.conv2_v = nn.Conv2d(64, 64, kernel_size=3, padding=2, bias=False)
        self.conv3_v = nn.Conv2d(64, 128, kernel_size=3, padding=2, bias=False)
        self.conv4_v = nn.Conv2d(
            128, 128, kernel_size=3, padding=2, bias=False)
        self.conv5_v = nn.Conv2d(
            128, 256, kernel_size=3, padding=2, bias=False)
        self.conv6_v = nn.Conv2d(
            256, 256, kernel_size=3, padding=2, bias=False)
        self.conv7_v = nn.Conv2d(
            256, 256, kernel_size=3, padding=2, bias=False)
        self.conv8_v = nn.Conv2d(
            256, 256, kernel_size=3, padding=2, bias=False)
        self.conv9_v = nn.Conv2d(
            256, 512, kernel_size=3, padding=2, bias=False)
        self.conv10_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)
        self.conv11_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)
        self.conv12_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)
        self.conv13_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)
        self.conv14_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)
        self.conv15_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)
        self.conv16_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=2, bias=False)

        self.bn1_g = nn.BatchNorm2d(64)
        self.bn2_g = nn.BatchNorm2d(64)
        self.bn3_g = nn.BatchNorm2d(128)
        self.bn4_g = nn.BatchNorm2d(128)

        self.bn5_g = nn.BatchNorm2d(256)
        self.bn6_g = nn.BatchNorm2d(256)
        self.bn7_g = nn.BatchNorm2d(256)
        self.bn8_g = nn.BatchNorm2d(256)

        self.bn9_g = nn.BatchNorm2d(512)
        self.bn10_g = nn.BatchNorm2d(512)
        self.bn11_g = nn.BatchNorm2d(512)
        self.bn12_g = nn.BatchNorm2d(512)

        self.bn13_g = nn.BatchNorm2d(512)
        self.bn14_g = nn.BatchNorm2d(512)
        self.bn15_g = nn.BatchNorm2d(512)
        self.bn16_g = nn.BatchNorm2d(512)

        self.bn1_v = nn.BatchNorm2d(64)
        self.bn2_v = nn.BatchNorm2d(64)
        self.bn3_v = nn.BatchNorm2d(128)
        self.bn4_v = nn.BatchNorm2d(128)

        self.bn5_v = nn.BatchNorm2d(256)
        self.bn6_v = nn.BatchNorm2d(256)
        self.bn7_v = nn.BatchNorm2d(256)
        self.bn8_v = nn.BatchNorm2d(256)

        self.bn9_v = nn.BatchNorm2d(512)
        self.bn10_v = nn.BatchNorm2d(512)
        self.bn11_v = nn.BatchNorm2d(512)
        self.bn12_v = nn.BatchNorm2d(512)

        self.bn13_v = nn.BatchNorm2d(512)
        self.bn14_v = nn.BatchNorm2d(512)
        self.bn15_v = nn.BatchNorm2d(512)
        self.bn16_v = nn.BatchNorm2d(512)

        self.fc1_g = nn.Linear(512*1*1, 4096)
        self.fc2_g = nn.Linear(4096, 4096)

        self.fc1_v = nn.Linear(512*1*1, 4096)
        self.fc2_v = nn.Linear(4096, 4096)
        self.fc3_v = nn.Linear(4096, num_classes)

        self.sig = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.allones = torch.ones(size=(1, 3, 32, 32))

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        gating_net_layers_ordered["conv1_g"] = self.conv1_g
        gating_net_layers_ordered["bn1_g"] = self.bn1_g

        gating_net_layers_ordered["conv2_g"] = self.conv2_g
        gating_net_layers_ordered["bn2_g"] = self.bn2_g

        gating_net_layers_ordered["pool1"] = self.pool

        gating_net_layers_ordered["conv3_g"] = self.conv3_g
        gating_net_layers_ordered["bn3_g"] = self.bn3_g

        gating_net_layers_ordered["conv4_g"] = self.conv4_g
        gating_net_layers_ordered["bn4_g"] = self.bn4_g

        gating_net_layers_ordered["pool2"] = self.pool

        gating_net_layers_ordered["conv5_g"] = self.conv5_g
        gating_net_layers_ordered["bn5_g"] = self.bn5_g

        gating_net_layers_ordered["conv6_g"] = self.conv6_g
        gating_net_layers_ordered["bn6_g"] = self.bn6_g

        gating_net_layers_ordered["conv7_g"] = self.conv7_g
        gating_net_layers_ordered["bn7_g"] = self.bn7_g

        gating_net_layers_ordered["conv8_g"] = self.conv8_g
        gating_net_layers_ordered["bn8_g"] = self.bn8_g

        gating_net_layers_ordered["pool3"] = self.pool

        gating_net_layers_ordered["conv9_g"] = self.conv9_g
        gating_net_layers_ordered["bn9_g"] = self.bn9_g

        gating_net_layers_ordered["conv10_g"] = self.conv10_g
        gating_net_layers_ordered["bn10_g"] = self.bn10_g

        gating_net_layers_ordered["conv11_g"] = self.conv11_g
        gating_net_layers_ordered["bn11_g"] = self.bn11_g

        gating_net_layers_ordered["conv12_g"] = self.conv12_g
        gating_net_layers_ordered["bn12_g"] = self.bn12_g

        gating_net_layers_ordered["pool4"] = self.pool

        gating_net_layers_ordered["conv13_g"] = self.conv13_g
        gating_net_layers_ordered["bn13_g"] = self.bn13_g

        gating_net_layers_ordered["conv14_g"] = self.conv14_g
        gating_net_layers_ordered["bn14_g"] = self.bn14_g

        gating_net_layers_ordered["conv15_g"] = self.conv15_g
        gating_net_layers_ordered["bn15_g"] = self.bn15_g

        gating_net_layers_ordered["conv16_g"] = self.conv16_g
        gating_net_layers_ordered["bn16_g"] = self.bn16_g

        # gating_net_layers_ordered["avgpool"] = self.avgpool

        # gating_net_layers_ordered["fc1_g"] = self.fc1_g
        # gating_net_layers_ordered["fc2_g"] = self.fc2_g

        return gating_net_layers_ordered

    def forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        merged_conv_layer_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_layer = None
        orig_out = x
        if(self.conv1_g.weight.get_device() < 0):
            idevice = torch.device("cpu")
        else:
            idevice = self.conv1_g.weight.get_device()
        lay_type = self.conv1_g.weight.dtype

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_layer, _ = merge_layers_operations_in_modules(
                    layer_obj, current_tensor_size, lay_type, idevice, merged_conv_layer)
                merged_conv_layer_in_each_layer[layer_name] = merged_conv_layer

                orig_out = layer_obj(orig_out)

                merged_conv_output = merged_conv_layer(x)
                current_tensor_size = merged_conv_output.size()[1:]

                print("orig_out.size():{} merged_conv_output.size():{}".format(
                    orig_out.size(), merged_conv_output.size()))
                assert orig_out.size() == merged_conv_output.size(
                ), "Size of effective and actual output unequal"
                difference_in_output = (
                    orig_out - merged_conv_output).abs().sum()
                print("difference_in_output ", difference_in_output)

        return merged_conv_layer_in_each_layer, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature/Gating Network
        # Layer 1 : 64
        self.linear_conv_outputs = []
        x_g = self.conv1_g(x)
        x_g = self.bn1_g(x_g)
        g_1 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x_g = F.relu(x_g)

        # Layer 2: 64
        x_g = self.conv2_g(x_g)
        x_g = self.bn2_g(x_g)
        g_2 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 1
        x_g = self.pool(x_g)

        # Layer 3 : 128
        x_g = self.conv3_g(x_g)
        x_g = self.bn3_g(x_g)
        g_3 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 4 : 128
        x_g = self.conv4_g(x_g)
        x_g = self.bn4_g(x_g)
        g_4 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 2
        x_g = self.pool(x_g)

        # Layer 5 : 256
        x_g = self.conv5_g(x_g)
        x_g = self.bn5_g(x_g)
        g_5 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 6 : 256
        x_g = self.conv6_g(x_g)
        x_g = self.bn6_g(x_g)
        g_6 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 7 : 256
        x_g = self.conv7_g(x_g)
        x_g = self.bn7_g(x_g)
        g_7 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 8 : 256
        x_g = self.conv8_g(x_g)
        x_g = self.bn8_g(x_g)
        g_8 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 3
        x_g = self.pool(x_g)

        # Layer 9 : 512
        x_g = self.conv9_g(x_g)
        x_g = self.bn9_g(x_g)
        g_9 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 10 : 512
        x_g = self.conv10_g(x_g)
        x_g = self.bn10_g(x_g)
        g_10 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 11 : 512
        x_g = self.conv11_g(x_g)
        x_g = self.bn11_g(x_g)
        g_11 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 12 : 512
        x_g = self.conv12_g(x_g)
        x_g = self.bn12_g(x_g)
        g_12 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 4
        x_g = self.pool(x_g)

        # Layer 13 : 512
        x_g = self.conv13_g(x_g)
        x_g = self.bn13_g(x_g)
        g_13 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 14 : 512
        x_g = self.conv14_g(x_g)
        x_g = self.bn14_g(x_g)
        g_14 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 15 : 512
        x_g = self.conv15_g(x_g)
        x_g = self.bn15_g(x_g)
        g_15 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 16 : 512
        x_g = self.conv16_g(x_g)
        x_g = self.bn16_g(x_g)
        g_16 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 5
        # x_g = self.pool(x_g)

        x_g = self.avgpool(x_g)
        x_g = torch.flatten(x_g, 1)
        x_g = self.fc1_g(x_g)
        g_17 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)
        #x_g = self.dropout(x_g)
        x_g = self.fc2_g(x_g)
        g_18 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Value Network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Layer 1 : 64
        x_v = self.conv1_v(self.allones.to(device))
        x_v = self.bn1_v(x_v)
        x_v = x_v*g_1
        #x_v = F.relu(x_v)

        # Layer 2: 64
        x_v = self.conv2_v(x_v)
        x_v = self.bn2_v(x_v)
        x_v = x_v*g_2
        #x = F.relu(x)

        # Max-Pool : 1
        x_v = self.pool(x_v)

        # Layer 3 : 128
        x_v = self.conv3_v(x_v)
        x_v = self.bn3_v(x_v)
        x_v = x_v*g_3
        #x = F.relu(x)

        # Layer 4 : 128
        x_v = self.conv4_v(x_v)
        x_v = self.bn4_v(x_v)
        x_v = x_v*g_4
        #x = F.relu(x)

        # Max-Pool : 2
        x_v = self.pool(x_v)

        # Layer 5 : 256
        x_v = self.conv5_v(x_v)
        x_v = self.bn5_v(x_v)
        x_v = x_v*g_5
        #x = F.relu(x)

        # Layer 6 : 256
        x_v = self.conv6_v(x_v)
        x_v = self.bn6_v(x_v)
        x_v = x_v*g_6
        #x = F.relu(x)

        # Layer 7 : 256
        x_v = self.conv7_v(x_v)
        x_v = self.bn7_v(x_v)
        x_v = x_v*g_7
        #x = F.relu(x)

        # Layer 8 : 256
        x_v = self.conv8_v(x_v)
        x_v = self.bn8_v(x_v)
        x_v = x_v*g_8
        #x = F.relu(x)

        # Max-Pool : 3
        x_v = self.pool(x_v)

        # Layer 9 : 512
        x_v = self.conv9_v(x_v)
        x_v = self.bn9_v(x_v)
        x_v = x_v*g_9
        #x = F.relu(x)

        # Layer 10 : 512
        x_v = self.conv10_v(x_v)
        x_v = self.bn10_v(x_v)
        x_v = x_v*g_10
        #x = F.relu(x)

        # Layer 11 : 512
        x_v = self.conv11_v(x_v)
        x_v = self.bn11_v(x_v)
        x_v = x_v*g_11
        #x = F.relu(x)

        # Layer 12 : 512
        x_v = self.conv12_v(x_v)
        x_v = self.bn12_v(x_v)
        x_v = x_v*g_12
        #x = F.relu(x)

        # Max-Pool : 4
        x_v = self.pool(x_v)

        # Layer 13 : 512
        x_v = self.conv13_v(x_v)
        x_v = self.bn13_v(x_v)
        x_v = x_v*g_13
        #x = F.relu(x)

        # Layer 14 : 512
        x_v = self.conv14_v(x_v)
        x_v = self.bn14_v(x_v)
        x_v = x_v*g_14
        #x = F.relu(x)

        # Layer 15 : 512
        x_v = self.conv15_v(x_v)
        x_v = self.bn15_v(x_v)
        x_v = x_v*g_15
        #x = F.relu(x)

        # Layer 16 : 512
        x_v = self.conv16_v(x_v)
        x_v = self.bn16_v(x_v)
        x_v = x_v*g_16
        #x = F.relu(x)
        # print("x_v", x_v.size())

        # Max-Pool : 5
        x_v = self.avgpool(x_v)

        x_v = torch.flatten(x_v, 1)
        x_v = self.fc1_v(x_v)
        x_v = x_v*g_17

        #x_v = self.dropout(x_v)
        x_v = self.fc2_v(x_v)
        x_v = x_v*g_18
        #x = F.relu(x)
        #x_v = self.dropout(x_v)
        x_v = self.fc3_v(x_v)

        return x_v

class st1_pad0_vgg16_bn(nn.Module):
    def __init__(self, init_weights: bool = True, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1_g = nn.Conv2d(3, 64, kernel_size=3, padding=0, bias=True)
        self.conv2_g = nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=True)
        self.conv3_g = nn.Conv2d(64, 128, kernel_size=3, padding=0, bias=True)
        self.conv4_g = nn.Conv2d(
            128, 128, kernel_size=3, padding=0, bias=True)
        self.conv5_g = nn.Conv2d(
            128, 256, kernel_size=3, padding=0, bias=True)
        self.conv6_g = nn.Conv2d(
            256, 256, kernel_size=3, padding=0, bias=True)
        self.conv7_g = nn.Conv2d(
            256, 256, kernel_size=3, padding=0, bias=True)
        self.conv8_g = nn.Conv2d(
            256, 256, kernel_size=3, padding=0, bias=True)
        self.conv9_g = nn.Conv2d(
            256, 512, kernel_size=3, padding=0, bias=True)
        self.conv10_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)
        self.conv11_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)
        self.conv12_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)
        self.conv13_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)
        self.conv14_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)
        self.conv15_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)
        self.conv16_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)

        self.conv1_v = nn.Conv2d(3, 64, kernel_size=3, padding=0, bias=True)
        self.conv2_v = nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=True)
        self.conv3_v = nn.Conv2d(64, 128, kernel_size=3, padding=0, bias=True)
        self.conv4_v = nn.Conv2d(
            128, 128, kernel_size=3, padding=0, bias=True)
        self.conv5_v = nn.Conv2d(
            128, 256, kernel_size=3, padding=0, bias=True)
        self.conv6_v = nn.Conv2d(
            256, 256, kernel_size=3, padding=0, bias=True)
        self.conv7_v = nn.Conv2d(
            256, 256, kernel_size=3, padding=0, bias=True)
        self.conv8_v = nn.Conv2d(
            256, 256, kernel_size=3, padding=0, bias=True)
        self.conv9_v = nn.Conv2d(
            256, 512, kernel_size=3, padding=0, bias=True)
        self.conv10_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)
        self.conv11_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)
        self.conv12_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)
        self.conv13_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)
        self.conv14_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)
        self.conv15_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)
        self.conv16_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=0, bias=True)

        self.bn1_g = nn.BatchNorm2d(64)
        self.bn2_g = nn.BatchNorm2d(64)
        self.bn3_g = nn.BatchNorm2d(128)
        self.bn4_g = nn.BatchNorm2d(128)

        self.bn5_g = nn.BatchNorm2d(256)
        self.bn6_g = nn.BatchNorm2d(256)
        self.bn7_g = nn.BatchNorm2d(256)
        self.bn8_g = nn.BatchNorm2d(256)

        self.bn9_g = nn.BatchNorm2d(512)
        self.bn10_g = nn.BatchNorm2d(512)
        self.bn11_g = nn.BatchNorm2d(512)
        self.bn12_g = nn.BatchNorm2d(512)

        self.bn13_g = nn.BatchNorm2d(512)
        self.bn14_g = nn.BatchNorm2d(512)
        self.bn15_g = nn.BatchNorm2d(512)
        self.bn16_g = nn.BatchNorm2d(512)

        self.bn1_v = nn.BatchNorm2d(64)
        self.bn2_v = nn.BatchNorm2d(64)
        self.bn3_v = nn.BatchNorm2d(128)
        self.bn4_v = nn.BatchNorm2d(128)

        self.bn5_v = nn.BatchNorm2d(256)
        self.bn6_v = nn.BatchNorm2d(256)
        self.bn7_v = nn.BatchNorm2d(256)
        self.bn8_v = nn.BatchNorm2d(256)

        self.bn9_v = nn.BatchNorm2d(512)
        self.bn10_v = nn.BatchNorm2d(512)
        self.bn11_v = nn.BatchNorm2d(512)
        self.bn12_v = nn.BatchNorm2d(512)

        self.bn13_v = nn.BatchNorm2d(512)
        self.bn14_v = nn.BatchNorm2d(512)
        self.bn15_v = nn.BatchNorm2d(512)
        self.bn16_v = nn.BatchNorm2d(512)

        self.fc1_g = nn.Linear(512*1*1, 4096)
        self.fc2_g = nn.Linear(4096, 4096)

        self.fc1_v = nn.Linear(512*1*1, 4096)
        self.fc2_v = nn.Linear(4096, 4096)
        self.fc3_v = nn.Linear(4096, num_classes)

        self.sig = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.allones = torch.ones(size=(1, 3, 32, 32))

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        gating_net_layers_ordered["conv1_g"] = self.conv1_g
        gating_net_layers_ordered["bn1_g"] = self.bn1_g

        gating_net_layers_ordered["conv2_g"] = self.conv2_g
        gating_net_layers_ordered["bn2_g"] = self.bn2_g

        gating_net_layers_ordered["conv2_g"] = self.conv2_g
        gating_net_layers_ordered["bn2_g"] = self.bn2_g

        gating_net_layers_ordered["pool1"] = self.pool

        gating_net_layers_ordered["conv3_g"] = self.conv3_g
        gating_net_layers_ordered["bn3_g"] = self.bn3_g

        gating_net_layers_ordered["conv4_g"] = self.conv4_g
        gating_net_layers_ordered["bn4_g"] = self.bn4_g

        gating_net_layers_ordered["pool2"] = self.pool

        gating_net_layers_ordered["conv5_g"] = self.conv5_g
        gating_net_layers_ordered["bn5_g"] = self.bn5_g

        gating_net_layers_ordered["conv6_g"] = self.conv6_g
        gating_net_layers_ordered["bn6_g"] = self.bn6_g

        gating_net_layers_ordered["conv7_g"] = self.conv7_g
        gating_net_layers_ordered["bn7_g"] = self.bn7_g

        gating_net_layers_ordered["conv8_g"] = self.conv8_g
        gating_net_layers_ordered["bn8_g"] = self.bn8_g

        gating_net_layers_ordered["pool3"] = self.pool

        gating_net_layers_ordered["conv9_g"] = self.conv9_g
        gating_net_layers_ordered["bn9_g"] = self.bn9_g

        gating_net_layers_ordered["conv10_g"] = self.conv10_g
        gating_net_layers_ordered["bn10_g"] = self.bn10_g

        gating_net_layers_ordered["conv11_g"] = self.conv11_g
        gating_net_layers_ordered["bn11_g"] = self.bn11_g

        gating_net_layers_ordered["conv12_g"] = self.conv12_g
        gating_net_layers_ordered["bn12_g"] = self.bn12_g

        gating_net_layers_ordered["pool4"] = self.pool

        gating_net_layers_ordered["conv13_g"] = self.conv13_g
        gating_net_layers_ordered["bn13_g"] = self.bn13_g

        gating_net_layers_ordered["conv14_g"] = self.conv14_g
        gating_net_layers_ordered["bn14_g"] = self.bn14_g

        gating_net_layers_ordered["conv15_g"] = self.conv15_g
        gating_net_layers_ordered["bn15_g"] = self.bn15_g

        gating_net_layers_ordered["conv16_g"] = self.conv16_g
        gating_net_layers_ordered["bn16_g"] = self.bn16_g

        # gating_net_layers_ordered["avgpool"] = self.avgpool

        # gating_net_layers_ordered["fc1_g"] = self.fc1_g
        # gating_net_layers_ordered["fc2_g"] = self.fc2_g

        return gating_net_layers_ordered

    def forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        merged_conv_layer_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_layer = None
        orig_out = x
        if(self.conv1_g.weight.get_device() < 0):
            idevice = torch.device("cpu")
        else:
            idevice = self.conv1_g.weight.get_device()
        lay_type = self.conv1_g.weight.dtype

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_layer, _ = merge_layers_operations_in_modules(
                    layer_obj, current_tensor_size, lay_type, idevice, merged_conv_layer)
                merged_conv_layer.padding =0
                merged_conv_layer_in_each_layer[layer_name] = merged_conv_layer

                orig_out = layer_obj(orig_out)

                merged_conv_output = merged_conv_layer(x)
                current_tensor_size = merged_conv_output.size()[1:]

                print("orig_out.size():{} merged_conv_output.size():{}".format(
                    orig_out.size(), merged_conv_output.size()))
                assert orig_out.size() == merged_conv_output.size(
                ), "Size of effective and actual output unequal"
                difference_in_output = (
                    orig_out - merged_conv_output).abs().sum()
                print("difference_in_output ", difference_in_output)

        return merged_conv_layer_in_each_layer, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature/Gating Network
        # Layer 1 : 64
        self.linear_conv_outputs = []
        x_g = self.conv1_g(x)
        x_g = self.bn1_g(x_g)
        g_1 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x_g = F.relu(x_g)

        # Layer 2: 64
        x_g = self.conv2_g(x_g)
        x_g = self.bn2_g(x_g)
        g_2 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 1
        x_g = self.pool(x_g)

        # Layer 3 : 128
        x_g = self.conv3_g(x_g)
        x_g = self.bn3_g(x_g)
        g_3 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 4 : 128
        x_g = self.conv4_g(x_g)
        x_g = self.bn4_g(x_g)
        g_4 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 2
        x_g = self.pool(x_g)

        # Layer 5 : 256
        x_g = self.conv5_g(x_g)
        x_g = self.bn5_g(x_g)
        g_5 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 6 : 256
        x_g = self.conv6_g(x_g)
        x_g = self.bn6_g(x_g)
        g_6 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 7 : 256
        x_g = self.conv7_g(x_g)
        x_g = self.bn7_g(x_g)
        g_7 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 8 : 256
        x_g = self.conv8_g(x_g)
        x_g = self.bn8_g(x_g)
        g_8 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 3
        x_g = self.pool(x_g)

        # Layer 9 : 512
        x_g = self.conv9_g(x_g)
        x_g = self.bn9_g(x_g)
        g_9 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 10 : 512
        x_g = self.conv10_g(x_g)
        x_g = self.bn10_g(x_g)
        g_10 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 11 : 512
        x_g = self.conv11_g(x_g)
        x_g = self.bn11_g(x_g)
        g_11 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 12 : 512
        x_g = self.conv12_g(x_g)
        x_g = self.bn12_g(x_g)
        g_12 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 4
        x_g = self.pool(x_g)

        # Layer 13 : 512
        x_g = self.conv13_g(x_g)
        x_g = self.bn13_g(x_g)
        g_13 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 14 : 512
        x_g = self.conv14_g(x_g)
        x_g = self.bn14_g(x_g)
        g_14 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 15 : 512
        x_g = self.conv15_g(x_g)
        x_g = self.bn15_g(x_g)
        g_15 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Layer 16 : 512
        x_g = self.conv16_g(x_g)
        x_g = self.bn16_g(x_g)
        g_16 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Max-Pool : 5
        # x_g = self.pool(x_g)

        x_g = self.avgpool(x_g)
        x_g = torch.flatten(x_g, 1)
        x_g = self.fc1_g(x_g)
        g_17 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)
        #x_g = self.dropout(x_g)
        x_g = self.fc2_g(x_g)
        g_18 = self.sig(10*x_g)
        self.linear_conv_outputs.append(x_g)
        #x = F.relu(x)

        # Value Network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Layer 1 : 64
        x_v = self.conv1_v(self.allones.to(device))
        x_v = self.bn1_v(x_v)
        x_v = x_v*g_1
        #x_v = F.relu(x_v)

        # Layer 2: 64
        x_v = self.conv2_v(x_v)
        x_v = self.bn2_v(x_v)
        x_v = x_v*g_2
        #x = F.relu(x)

        # Max-Pool : 1
        x_v = self.pool(x_v)

        # Layer 3 : 128
        x_v = self.conv3_v(x_v)
        x_v = self.bn3_v(x_v)
        x_v = x_v*g_3
        #x = F.relu(x)

        # Layer 4 : 128
        x_v = self.conv4_v(x_v)
        x_v = self.bn4_v(x_v)
        x_v = x_v*g_4
        #x = F.relu(x)

        # Max-Pool : 2
        x_v = self.pool(x_v)

        # Layer 5 : 256
        x_v = self.conv5_v(x_v)
        x_v = self.bn5_v(x_v)
        x_v = x_v*g_5
        #x = F.relu(x)

        # Layer 6 : 256
        x_v = self.conv6_v(x_v)
        x_v = self.bn6_v(x_v)
        x_v = x_v*g_6
        #x = F.relu(x)

        # Layer 7 : 256
        x_v = self.conv7_v(x_v)
        x_v = self.bn7_v(x_v)
        x_v = x_v*g_7
        #x = F.relu(x)

        # Layer 8 : 256
        x_v = self.conv8_v(x_v)
        x_v = self.bn8_v(x_v)
        x_v = x_v*g_8
        #x = F.relu(x)

        # Max-Pool : 3
        x_v = self.pool(x_v)

        # Layer 9 : 512
        x_v = self.conv9_v(x_v)
        x_v = self.bn9_v(x_v)
        x_v = x_v*g_9
        #x = F.relu(x)

        # Layer 10 : 512
        x_v = self.conv10_v(x_v)
        x_v = self.bn10_v(x_v)
        x_v = x_v*g_10
        #x = F.relu(x)

        # Layer 11 : 512
        x_v = self.conv11_v(x_v)
        x_v = self.bn11_v(x_v)
        x_v = x_v*g_11
        #x = F.relu(x)

        # Layer 12 : 512
        x_v = self.conv12_v(x_v)
        x_v = self.bn12_v(x_v)
        x_v = x_v*g_12
        #x = F.relu(x)

        # Max-Pool : 4
        x_v = self.pool(x_v)

        # Layer 13 : 512
        x_v = self.conv13_v(x_v)
        x_v = self.bn13_v(x_v)
        x_v = x_v*g_13
        #x = F.relu(x)

        # Layer 14 : 512
        x_v = self.conv14_v(x_v)
        x_v = self.bn14_v(x_v)
        x_v = x_v*g_14
        #x = F.relu(x)

        # Layer 15 : 512
        x_v = self.conv15_v(x_v)
        x_v = self.bn15_v(x_v)
        x_v = x_v*g_15
        #x = F.relu(x)

        # Layer 16 : 512
        x_v = self.conv16_v(x_v)
        x_v = self.bn16_v(x_v)
        x_v = x_v*g_16
        #x = F.relu(x)
        # print("x_v", x_v.size())

        # Max-Pool : 5
        x_v = self.avgpool(x_v)

        x_v = torch.flatten(x_v, 1)
        x_v = self.fc1_v(x_v)
        x_v = x_v*g_17

        #x_v = self.dropout(x_v)
        x_v = self.fc2_v(x_v)
        x_v = x_v*g_18
        #x = F.relu(x)
        #x_v = self.dropout(x_v)
        x_v = self.fc3_v(x_v)

        return x_v


class st1_pad1_vgg16_bn_wo_bias(nn.Module):
    def __init__(self, init_weights: bool = True, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1_g = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.conv2_g = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv3_g = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv4_g = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, bias=False)
        self.conv5_g = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, bias=False)
        self.conv6_g = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.conv7_g = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.conv8_g = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.conv9_g = nn.Conv2d(
            256, 512, kernel_size=3, padding=1, bias=False)
        self.conv10_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv11_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv12_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv13_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv14_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv15_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv16_g = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)

        self.conv1_v = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.conv2_v = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv3_v = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv4_v = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, bias=False)
        self.conv5_v = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, bias=False)
        self.conv6_v = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.conv7_v = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.conv8_v = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.conv9_v = nn.Conv2d(
            256, 512, kernel_size=3, padding=1, bias=False)
        self.conv10_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv11_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv12_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv13_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv14_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv15_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.conv16_v = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)

        self.bn1_g = nn.BatchNorm2d(64)
        self.bn2_g = nn.BatchNorm2d(64)
        self.bn3_g = nn.BatchNorm2d(128)
        self.bn4_g = nn.BatchNorm2d(128)

        self.bn5_g = nn.BatchNorm2d(256)
        self.bn6_g = nn.BatchNorm2d(256)
        self.bn7_g = nn.BatchNorm2d(256)
        self.bn8_g = nn.BatchNorm2d(256)

        self.bn9_g = nn.BatchNorm2d(512)
        self.bn10_g = nn.BatchNorm2d(512)
        self.bn11_g = nn.BatchNorm2d(512)
        self.bn12_g = nn.BatchNorm2d(512)

        self.bn13_g = nn.BatchNorm2d(512)
        self.bn14_g = nn.BatchNorm2d(512)
        self.bn15_g = nn.BatchNorm2d(512)
        self.bn16_g = nn.BatchNorm2d(512)

        self.bn1_v = nn.BatchNorm2d(64)
        self.bn2_v = nn.BatchNorm2d(64)
        self.bn3_v = nn.BatchNorm2d(128)
        self.bn4_v = nn.BatchNorm2d(128)

        self.bn5_v = nn.BatchNorm2d(256)
        self.bn6_v = nn.BatchNorm2d(256)
        self.bn7_v = nn.BatchNorm2d(256)
        self.bn8_v = nn.BatchNorm2d(256)

        self.bn9_v = nn.BatchNorm2d(512)
        self.bn10_v = nn.BatchNorm2d(512)
        self.bn11_v = nn.BatchNorm2d(512)
        self.bn12_v = nn.BatchNorm2d(512)

        self.bn13_v = nn.BatchNorm2d(512)
        self.bn14_v = nn.BatchNorm2d(512)
        self.bn15_v = nn.BatchNorm2d(512)
        self.bn16_v = nn.BatchNorm2d(512)

        self.fc1_g = nn.Linear(512*1*1, 4096)
        self.fc2_g = nn.Linear(4096, 4096)

        self.fc1_v = nn.Linear(512*1*1, 4096)
        self.fc2_v = nn.Linear(4096, 4096)
        self.fc3_v = nn.Linear(4096, num_classes)

        self.sig = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.allones = torch.ones(size=(1, 3, 32, 32))

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        gating_net_layers_ordered["conv1_g"] = self.conv1_g
        gating_net_layers_ordered["bn1_g"] = self.bn1_g

        gating_net_layers_ordered["conv2_g"] = self.conv2_g
        gating_net_layers_ordered["bn2_g"] = self.bn2_g

        gating_net_layers_ordered["conv2_g"] = self.conv2_g
        gating_net_layers_ordered["bn2_g"] = self.bn2_g

        gating_net_layers_ordered["pool1"] = self.pool

        gating_net_layers_ordered["conv3_g"] = self.conv3_g
        gating_net_layers_ordered["bn3_g"] = self.bn3_g

        gating_net_layers_ordered["conv4_g"] = self.conv4_g
        gating_net_layers_ordered["bn4_g"] = self.bn4_g

        gating_net_layers_ordered["pool2"] = self.pool

        gating_net_layers_ordered["conv5_g"] = self.conv5_g
        gating_net_layers_ordered["bn5_g"] = self.bn5_g

        gating_net_layers_ordered["conv6_g"] = self.conv6_g
        gating_net_layers_ordered["bn6_g"] = self.bn6_g

        gating_net_layers_ordered["conv7_g"] = self.conv7_g
        gating_net_layers_ordered["bn7_g"] = self.bn7_g

        gating_net_layers_ordered["conv8_g"] = self.conv8_g
        gating_net_layers_ordered["bn8_g"] = self.bn8_g

        gating_net_layers_ordered["pool3"] = self.pool

        gating_net_layers_ordered["conv9_g"] = self.conv9_g
        gating_net_layers_ordered["bn9_g"] = self.bn9_g

        gating_net_layers_ordered["conv10_g"] = self.conv10_g
        gating_net_layers_ordered["bn10_g"] = self.bn10_g

        gating_net_layers_ordered["conv11_g"] = self.conv11_g
        gating_net_layers_ordered["bn11_g"] = self.bn11_g

        gating_net_layers_ordered["conv12_g"] = self.conv12_g
        gating_net_layers_ordered["bn12_g"] = self.bn12_g

        gating_net_layers_ordered["pool4"] = self.pool

        gating_net_layers_ordered["conv13_g"] = self.conv13_g
        gating_net_layers_ordered["bn13_g"] = self.bn13_g

        gating_net_layers_ordered["conv14_g"] = self.conv14_g
        gating_net_layers_ordered["bn14_g"] = self.bn14_g

        gating_net_layers_ordered["conv15_g"] = self.conv15_g
        gating_net_layers_ordered["bn15_g"] = self.bn15_g

        gating_net_layers_ordered["conv16_g"] = self.conv16_g
        gating_net_layers_ordered["bn16_g"] = self.bn16_g

        # gating_net_layers_ordered["avgpool"] = self.avgpool

        # gating_net_layers_ordered["fc1_g"] = self.fc1_g
        # gating_net_layers_ordered["fc2_g"] = self.fc2_g

        return gating_net_layers_ordered

    def forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        merged_conv_layer_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_layer = None
        orig_out = x
        if(self.conv1_g.weight.get_device() < 0):
            idevice = torch.device("cpu")
        else:
            idevice = self.conv1_g.weight.get_device()
        lay_type = self.conv1_g.weight.dtype

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_layer, _ = merge_layers_operations_in_modules(
                    layer_obj, current_tensor_size, lay_type, idevice, merged_conv_layer)
                merged_conv_layer_in_each_layer[layer_name] = merged_conv_layer

                orig_out = layer_obj(orig_out)

                merged_conv_output = merged_conv_layer(x)
                current_tensor_size = merged_conv_output.size()[1:]

                print("orig_out.size():{} merged_conv_output.size():{}".format(
                    orig_out.size(), merged_conv_output.size()))
                # assert orig_out.size() == merged_conv_output.size(
                # ), "Size of effective and actual output unequal"
                # difference_in_output = (
                #     orig_out - merged_conv_output).abs().sum()
                # print("difference_in_output ", difference_in_output)

        return merged_conv_layer_in_each_layer, None

    def exact_forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        conv_matrix_operations_in_each_layer = OrderedDict()
        conv_bias_operations_in_each_layer = OrderedDict()
        channel_outs_size_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_matrix = None
        merged_conv_bias = None
        orig_out = x
        if(x.get_device() >= 0):
            x = x.type(torch.float16)

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_matrix, merged_conv_bias, current_tensor_size = merge_operations_in_modules(
                    layer_obj, current_tensor_size, merged_conv_matrix, merged_conv_bias)
                conv_matrix_operations_in_each_layer[layer_name] = merged_conv_matrix.cpu(
                )
                conv_bias_operations_in_each_layer[layer_name] = merged_conv_bias.cpu(
                )
                channel_outs_size_in_each_layer[layer_name] = current_tensor_size[0]

                orig_out = layer_obj(orig_out)

                convmatrix_output = apply_input_on_conv_matrix(
                    x, merged_conv_matrix, merged_conv_bias)
                convmatrix_output = torch.unsqueeze(torch.reshape(
                    convmatrix_output, current_tensor_size), 0)
                assert orig_out.size() == convmatrix_output.size(
                ), "Size of effective and actual output unequal"
                difference_in_output = (
                    orig_out - convmatrix_output).abs().sum()
                print("difference_in_output ", difference_in_output)

        return conv_matrix_operations_in_each_layer, conv_bias_operations_in_each_layer, channel_outs_size_in_each_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature/Gating Network
        # Layer 1 : 64

        x_g = self.conv1_g(x)
        x_g = self.bn1_g(x_g)
        g_1 = self.sig(10*x_g)
        #x_g = F.relu(x_g)

        # Layer 2: 64
        x_g = self.conv2_g(x_g)
        x_g = self.bn2_g(x_g)
        g_2 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 1
        x_g = self.pool(x_g)

        # Layer 3 : 128
        x_g = self.conv3_g(x_g)
        x_g = self.bn3_g(x_g)
        g_3 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 4 : 128
        x_g = self.conv4_g(x_g)
        x_g = self.bn4_g(x_g)
        g_4 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 2
        x_g = self.pool(x_g)

        # Layer 5 : 256
        x_g = self.conv5_g(x_g)
        x_g = self.bn5_g(x_g)
        g_5 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 6 : 256
        x_g = self.conv6_g(x_g)
        x_g = self.bn6_g(x_g)
        g_6 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 7 : 256
        x_g = self.conv7_g(x_g)
        x_g = self.bn7_g(x_g)
        g_7 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 8 : 256
        x_g = self.conv8_g(x_g)
        x_g = self.bn8_g(x_g)
        g_8 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 3
        x_g = self.pool(x_g)

        # Layer 9 : 512
        x_g = self.conv9_g(x_g)
        x_g = self.bn9_g(x_g)
        g_9 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 10 : 512
        x_g = self.conv10_g(x_g)
        x_g = self.bn10_g(x_g)
        g_10 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 11 : 512
        x_g = self.conv11_g(x_g)
        x_g = self.bn11_g(x_g)
        g_11 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 12 : 512
        x_g = self.conv12_g(x_g)
        x_g = self.bn12_g(x_g)
        g_12 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 4
        x_g = self.pool(x_g)

        # Layer 13 : 512
        x_g = self.conv13_g(x_g)
        x_g = self.bn13_g(x_g)
        g_13 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 14 : 512
        x_g = self.conv14_g(x_g)
        x_g = self.bn14_g(x_g)
        g_14 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 15 : 512
        x_g = self.conv15_g(x_g)
        x_g = self.bn15_g(x_g)
        g_15 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 16 : 512
        x_g = self.conv16_g(x_g)
        x_g = self.bn16_g(x_g)
        g_16 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 5
        # x_g = self.pool(x_g)

        x_g = self.avgpool(x_g)
        x_g = torch.flatten(x_g, 1)
        x_g = self.fc1_g(x_g)
        g_17 = self.sig(10*x_g)
        #x = F.relu(x)
        #x_g = self.dropout(x_g)
        x_g = self.fc2_g(x_g)
        g_18 = self.sig(10*x_g)
        #x = F.relu(x)

        # Value Network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Layer 1 : 64
        x_v = self.conv1_v(self.allones.to(device))
        x_v = self.bn1_v(x_v)
        x_v = x_v*g_1
        #x_v = F.relu(x_v)

        # Layer 2: 64
        x_v = self.conv2_v(x_v)
        x_v = self.bn2_v(x_v)
        x_v = x_v*g_2
        #x = F.relu(x)

        # Max-Pool : 1
        x_v = self.pool(x_v)

        # Layer 3 : 128
        x_v = self.conv3_v(x_v)
        x_v = self.bn3_v(x_v)
        x_v = x_v*g_3
        #x = F.relu(x)

        # Layer 4 : 128
        x_v = self.conv4_v(x_v)
        x_v = self.bn4_v(x_v)
        x_v = x_v*g_4
        #x = F.relu(x)

        # Max-Pool : 2
        x_v = self.pool(x_v)

        # Layer 5 : 256
        x_v = self.conv5_v(x_v)
        x_v = self.bn5_v(x_v)
        x_v = x_v*g_5
        #x = F.relu(x)

        # Layer 6 : 256
        x_v = self.conv6_v(x_v)
        x_v = self.bn6_v(x_v)
        x_v = x_v*g_6
        #x = F.relu(x)

        # Layer 7 : 256
        x_v = self.conv7_v(x_v)
        x_v = self.bn7_v(x_v)
        x_v = x_v*g_7
        #x = F.relu(x)

        # Layer 8 : 256
        x_v = self.conv8_v(x_v)
        x_v = self.bn8_v(x_v)
        x_v = x_v*g_8
        #x = F.relu(x)

        # Max-Pool : 3
        x_v = self.pool(x_v)

        # Layer 9 : 512
        x_v = self.conv9_v(x_v)
        x_v = self.bn9_v(x_v)
        x_v = x_v*g_9
        #x = F.relu(x)

        # Layer 10 : 512
        x_v = self.conv10_v(x_v)
        x_v = self.bn10_v(x_v)
        x_v = x_v*g_10
        #x = F.relu(x)

        # Layer 11 : 512
        x_v = self.conv11_v(x_v)
        x_v = self.bn11_v(x_v)
        x_v = x_v*g_11
        #x = F.relu(x)

        # Layer 12 : 512
        x_v = self.conv12_v(x_v)
        x_v = self.bn12_v(x_v)
        x_v = x_v*g_12
        #x = F.relu(x)

        # Max-Pool : 4
        x_v = self.pool(x_v)

        # Layer 13 : 512
        x_v = self.conv13_v(x_v)
        x_v = self.bn13_v(x_v)
        x_v = x_v*g_13
        #x = F.relu(x)

        # Layer 14 : 512
        x_v = self.conv14_v(x_v)
        x_v = self.bn14_v(x_v)
        x_v = x_v*g_14
        #x = F.relu(x)

        # Layer 15 : 512
        x_v = self.conv15_v(x_v)
        x_v = self.bn15_v(x_v)
        x_v = x_v*g_15
        #x = F.relu(x)

        # Layer 16 : 512
        x_v = self.conv16_v(x_v)
        x_v = self.bn16_v(x_v)
        x_v = x_v*g_16
        #x = F.relu(x)
        # print("x_v", x_v.size())

        # Max-Pool : 5
        x_v = self.avgpool(x_v)

        x_v = torch.flatten(x_v, 1)
        x_v = self.fc1_v(x_v)
        x_v = x_v*g_17

        #x_v = self.dropout(x_v)
        x_v = self.fc2_v(x_v)
        x_v = x_v*g_18
        #x = F.relu(x)
        #x_v = self.dropout(x_v)
        x_v = self.fc3_v(x_v)

        return x_v


class CONV_PCA_Layer(nn.Module):
    def __init__(self, input_channel, data, explained_var_required):
        super(CONV_PCA_Layer, self).__init__()
        self.input_channel = input_channel
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
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
            "cuda" if torch.cuda.is_available() else "cpu")

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
            "cuda" if torch.cuda.is_available() else "cpu")

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
            "cuda" if torch.cuda.is_available() else "cpu")
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
            "cuda" if torch.cuda.is_available() else "cpu")
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
            "cuda" if torch.cuda.is_available() else "cpu")
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

class IM_Conv4_DLGN_Net_pad_k_1_wo_bn_wo_bias(nn.Module):
    def __init__(self, input_channel, beta=4, seed=2022, num_classes=10):
        super().__init__()
        torch.manual_seed(seed)
        self.input_channel = input_channel
        self.beta = beta

        self.conv1_g = nn.Conv2d(input_channel, 128, 3, padding=2,bias=False)
        self.conv2_g = nn.Conv2d(128, 128, 3, padding=2,bias=False)
        self.conv3_g = nn.Conv2d(128, 128, 3, padding=2,bias=False)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=2,bias=False)
        self.conv1_w = nn.Conv2d(input_channel, 128, 3, padding=2,bias=False)
        self.conv2_w = nn.Conv2d(128, 128, 3, padding=2,bias=False)
        self.conv3_w = nn.Conv2d(128, 128, 3, padding=2,bias=False)
        self.conv4_w = nn.Conv2d(128, 128, 3, padding=2,bias=False)
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
            "cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=device, non_blocking=True)
        
        if(self.conv1_g.weight.get_device() < 0):
            idevice = torch.device("cpu")
        else:
            idevice = self.conv1_g.weight.get_device()
        lay_type = self.conv1_g.weight.dtype
        
        inp = inp.flip(-1,-2)

        current_tensor_size = inp.size()[1:]
        merged_conv_layer = None
        conv_outs = []
        merged_conv_layer, _ = merge_layers_operations_in_modules(
                        self.conv1_g, current_tensor_size, lay_type, idevice, merged_conv_layer)
        x_g1 = merged_conv_layer(inp).flip(-1,-2)
        current_tensor_size = x_g1.size()[1:]
        conv_outs.append(x_g1)

        merged_conv_layer, _ = merge_layers_operations_in_modules(
                        self.conv2_g, current_tensor_size, lay_type, idevice, merged_conv_layer)
        x_g2 = merged_conv_layer(inp).flip(-1,-2)
        current_tensor_size = x_g2.size()[1:]
        conv_outs.append(x_g2)

        merged_conv_layer, _ = merge_layers_operations_in_modules(
                        self.conv3_g, current_tensor_size, lay_type, idevice, merged_conv_layer)
        x_g3 = merged_conv_layer(inp).flip(-1,-2)
        current_tensor_size = x_g3.size()[1:]
        conv_outs.append(x_g3)

        merged_conv_layer, _ = merge_layers_operations_in_modules(
                        self.conv4_g, current_tensor_size, lay_type, idevice, merged_conv_layer)
        x_g4 = merged_conv_layer(inp).flip(-1,-2)
        current_tensor_size = x_g4.size()[1:]
        conv_outs.append(x_g4)

        self.linear_conv_outputs = conv_outs

        g1 = nn.Sigmoid()(self.beta * x_g1)
        g2 = nn.Sigmoid()(self.beta * x_g2)
        g3 = nn.Sigmoid()(self.beta * x_g3)
        g4 = nn.Sigmoid()(self.beta * x_g4)

        inp_all_ones = torch.ones(inp.size(),
                                  requires_grad=True, device=device,dtype=inp.dtype)

        x_w1 = self.conv1_w(inp_all_ones) * g1
        x_w2 = self.conv2_w(x_w1) * g2
        x_w3 = self.conv3_w(x_w2) * g3
        x_w4 = self.conv4_w(x_w3) * g4
        x_w4 = x_w4.flip(-1,-2)

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
    
    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        gating_net_layers_ordered["conv1_g"] = self.conv1_g
        gating_net_layers_ordered["conv2_g"] = self.conv2_g
        gating_net_layers_ordered["conv3_g"] = self.conv3_g
        gating_net_layers_ordered["conv4_g"] = self.conv4_g
        gating_net_layers_ordered["pool1"] = self.pool

        return gating_net_layers_ordered

    
    def exact_forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        conv_matrix_operations_in_each_layer = OrderedDict()
        conv_bias_operations_in_each_layer = OrderedDict()
        channel_outs_size_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_matrix = None
        merged_conv_bias = None
        orig_out = x
        if(x.get_device() >= 0):
            x = x.type(torch.float16)

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_matrix, merged_conv_bias, current_tensor_size = merge_operations_in_modules(
                    layer_obj, current_tensor_size, merged_conv_matrix, merged_conv_bias)
                conv_matrix_operations_in_each_layer[layer_name] = merged_conv_matrix.cpu(
                )
                conv_bias_operations_in_each_layer[layer_name] = merged_conv_bias.cpu(
                )
                channel_outs_size_in_each_layer[layer_name] = current_tensor_size[0]

                orig_out = layer_obj(orig_out)

                convmatrix_output = apply_input_on_conv_matrix(
                    x, merged_conv_matrix, merged_conv_bias)
                convmatrix_output = torch.unsqueeze(torch.reshape(
                    convmatrix_output, current_tensor_size), 0)
                assert orig_out.size() == convmatrix_output.size(
                ), "Size of effective and actual output unequal"
                difference_in_output = (
                    orig_out - convmatrix_output).abs().sum()
                print("difference_in_output ", difference_in_output)

        return conv_matrix_operations_in_each_layer, conv_bias_operations_in_each_layer, channel_outs_size_in_each_layer


    def forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        merged_conv_layer_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_layer = None
        x = x.flip(-1,-2)
        orig_out = x
        if(self.conv1_g.weight.get_device() < 0):
            idevice = torch.device("cpu")
        else:
            idevice = self.conv1_g.weight.get_device()
        lay_type = self.conv1_g.weight.dtype

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                if(not isinstance(layer_obj, nn.AdaptiveAvgPool2d) and not isinstance(layer_obj,nn.Linear)):
                    merged_conv_layer, _ = merge_layers_operations_in_modules(
                        layer_obj, current_tensor_size, lay_type, idevice, merged_conv_layer)
                    merged_conv_layer_in_each_layer[layer_name] = merged_conv_layer

                    orig_out = layer_obj(orig_out).flip(-1,-2)

                    merged_conv_output = merged_conv_layer(x).flip(-1,-2)
                    current_tensor_size = merged_conv_output.size()[1:]

                    print("orig_out.size():{} merged_conv_output.size():{}".format(
                        orig_out.size(), merged_conv_output.size()))
                    assert orig_out.size() == merged_conv_output.size(
                    ), "Size of effective and actual output unequal"
                    difference_in_output = (
                        orig_out - merged_conv_output).abs().sum()
                    print("difference_in_output ", difference_in_output)
                    orig_out = orig_out.flip(-1,-2)

        return merged_conv_layer_in_each_layer, None


class Conv4_DLGN_Net_pad_k_1_wo_bn_wo_bias(nn.Module):
    def __init__(self, input_channel, beta=4, seed=2022, num_classes=10):
        super().__init__()
        torch.manual_seed(seed)
        self.input_channel = input_channel
        self.beta = beta

        self.conv1_g = nn.Conv2d(input_channel, 128, 3, padding=2,bias=False)
        self.conv2_g = nn.Conv2d(128, 128, 3, padding=2,bias=False)
        self.conv3_g = nn.Conv2d(128, 128, 3, padding=2,bias=False)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=2,bias=False)
        self.conv1_w = nn.Conv2d(input_channel, 128, 3, padding=2,bias=False)
        self.conv2_w = nn.Conv2d(128, 128, 3, padding=2,bias=False)
        self.conv3_w = nn.Conv2d(128, 128, 3, padding=2,bias=False)
        self.conv4_w = nn.Conv2d(128, 128, 3, padding=2,bias=False)
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
            "cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=device, non_blocking=True)
        
        inp = inp.flip(-1,-2)
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
                                  requires_grad=True, device=device,dtype=inp.dtype)

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
    
    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        gating_net_layers_ordered["conv1_g"] = self.conv1_g
        gating_net_layers_ordered["conv2_g"] = self.conv2_g
        gating_net_layers_ordered["conv3_g"] = self.conv3_g
        gating_net_layers_ordered["conv4_g"] = self.conv4_g
        gating_net_layers_ordered["pool1"] = self.pool

        return gating_net_layers_ordered

    
    def exact_forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        conv_matrix_operations_in_each_layer = OrderedDict()
        conv_bias_operations_in_each_layer = OrderedDict()
        channel_outs_size_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_matrix = None
        merged_conv_bias = None
        orig_out = x
        if(x.get_device() >= 0):
            x = x.type(torch.float16)

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_matrix, merged_conv_bias, current_tensor_size = merge_operations_in_modules(
                    layer_obj, current_tensor_size, merged_conv_matrix, merged_conv_bias)
                conv_matrix_operations_in_each_layer[layer_name] = merged_conv_matrix.cpu(
                )
                conv_bias_operations_in_each_layer[layer_name] = merged_conv_bias.cpu(
                )
                channel_outs_size_in_each_layer[layer_name] = current_tensor_size[0]

                orig_out = layer_obj(orig_out)

                convmatrix_output = apply_input_on_conv_matrix(
                    x, merged_conv_matrix, merged_conv_bias)
                convmatrix_output = torch.unsqueeze(torch.reshape(
                    convmatrix_output, current_tensor_size), 0)
                assert orig_out.size() == convmatrix_output.size(
                ), "Size of effective and actual output unequal"
                difference_in_output = (
                    orig_out - convmatrix_output).abs().sum()
                print("difference_in_output ", difference_in_output)

        return conv_matrix_operations_in_each_layer, conv_bias_operations_in_each_layer, channel_outs_size_in_each_layer


    def forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        merged_conv_layer_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_layer = None
        x = x.flip(-1,-2)
        orig_out = x
        if(self.conv1_g.weight.get_device() < 0):
            idevice = torch.device("cpu")
        else:
            idevice = self.conv1_g.weight.get_device()
        lay_type = self.conv1_g.weight.dtype

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                if(not isinstance(layer_obj, nn.AdaptiveAvgPool2d) and not isinstance(layer_obj,nn.Linear)):
                    merged_conv_layer, _ = merge_layers_operations_in_modules(
                        layer_obj, current_tensor_size, lay_type, idevice, merged_conv_layer)
                    merged_conv_layer_in_each_layer[layer_name] = merged_conv_layer

                    orig_out = layer_obj(orig_out).flip(-1,-2)

                    merged_conv_output = merged_conv_layer(x).flip(-1,-2)
                    current_tensor_size = merged_conv_output.size()[1:]

                    print("orig_out.size():{} merged_conv_output.size():{}".format(
                        orig_out.size(), merged_conv_output.size()))
                    assert orig_out.size() == merged_conv_output.size(
                    ), "Size of effective and actual output unequal"
                    difference_in_output = (
                        orig_out - merged_conv_output).abs().sum()
                    print("difference_in_output ", difference_in_output)
                    orig_out = orig_out.flip(-1,-2)

        return merged_conv_layer_in_each_layer, None
    
class Conv4_DLGN_Net_pad0_wo_bn(nn.Module):
    def __init__(self, input_channel, beta=4, seed=2022, num_classes=10):
        super().__init__()
        torch.manual_seed(seed)
        self.input_channel = input_channel
        self.beta = beta

        self.conv1_g = nn.Conv2d(input_channel, 128, 3, padding=0)
        self.conv2_g = nn.Conv2d(128, 128, 3, padding=0)
        self.conv3_g = nn.Conv2d(128, 128, 3, padding=0)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=0)
        self.conv1_w = nn.Conv2d(input_channel, 128, 3, padding=0)
        self.conv2_w = nn.Conv2d(128, 128, 3, padding=0)
        self.conv3_w = nn.Conv2d(128, 128, 3, padding=0)
        self.conv4_w = nn.Conv2d(128, 128, 3, padding=0)
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
            "cuda" if torch.cuda.is_available() else "cpu")
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
    
    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        gating_net_layers_ordered["conv1_g"] = self.conv1_g
        gating_net_layers_ordered["conv2_g"] = self.conv2_g
        gating_net_layers_ordered["conv3_g"] = self.conv3_g
        gating_net_layers_ordered["conv4_g"] = self.conv4_g
        gating_net_layers_ordered["pool1"] = self.pool

        return gating_net_layers_ordered

    def forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        merged_conv_layer_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_layer = None
        orig_out = x
        if(self.conv1_g.weight.get_device() < 0):
            idevice = torch.device("cpu")
        else:
            idevice = self.conv1_g.weight.get_device()
        lay_type = self.conv1_g.weight.dtype

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                if(not isinstance(layer_obj, nn.AdaptiveAvgPool2d) and not isinstance(layer_obj,nn.Linear)):
                    merged_conv_layer, _ = merge_layers_operations_in_modules(
                        layer_obj, current_tensor_size, lay_type, idevice, merged_conv_layer)
                    merged_conv_layer_in_each_layer[layer_name] = merged_conv_layer
                    merged_conv_layer.padding = 0

                    orig_out = layer_obj(orig_out)

                    merged_conv_output = merged_conv_layer(x)
                    current_tensor_size = merged_conv_output.size()[1:]

                    print("orig_out.size():{} merged_conv_output.size():{}".format(
                        orig_out.size(), merged_conv_output.size()))
                    assert orig_out.size() == merged_conv_output.size(
                    ), "Size of effective and actual output unequal"
                    difference_in_output = (
                        orig_out - merged_conv_output).abs().sum()
                    print("difference_in_output ", difference_in_output)

        return merged_conv_layer_in_each_layer, None

    def exact_forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        conv_matrix_operations_in_each_layer = OrderedDict()
        conv_bias_operations_in_each_layer = OrderedDict()
        channel_outs_size_in_each_layer = OrderedDict()
        current_tensor_size = x.size()[1:]
        print("current_tensor_size ", current_tensor_size)
        merged_conv_matrix = None
        merged_conv_bias = None
        orig_out = x
        if(x.get_device() >= 0):
            x = x.type(torch.float16)

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_matrix, merged_conv_bias, current_tensor_size = merge_operations_in_modules(
                    layer_obj, current_tensor_size, merged_conv_matrix, merged_conv_bias)
                conv_matrix_operations_in_each_layer[layer_name] = merged_conv_matrix.cpu(
                )
                conv_bias_operations_in_each_layer[layer_name] = merged_conv_bias.cpu(
                )
                channel_outs_size_in_each_layer[layer_name] = current_tensor_size[0]

                orig_out = layer_obj(orig_out)

                convmatrix_output = apply_input_on_conv_matrix(
                    x, merged_conv_matrix, merged_conv_bias)
                convmatrix_output = torch.unsqueeze(torch.reshape(
                    convmatrix_output, current_tensor_size), 0)
                assert orig_out.size() == convmatrix_output.size(
                ), "Size of effective and actual output unequal"
                difference_in_output = (
                    orig_out - convmatrix_output).abs().sum()
                print("difference_in_output ", difference_in_output)

        return conv_matrix_operations_in_each_layer, conv_bias_operations_in_each_layer, channel_outs_size_in_each_layer



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
            "cuda" if torch.cuda.is_available() else "cpu")

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
            "cuda" if torch.cuda.is_available() else "cpu")

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
            "cuda" if torch.cuda.is_available() else "cpu")
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
            "cuda" if torch.cuda.is_available() else "cpu")
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
            "cuda" if torch.cuda.is_available() else "cpu")
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
            "cuda" if torch.cuda.is_available() else "cpu")
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


class TorchVision_DeepGatedNet(nn.Module):
    def __init__(self, arch_type, input_channel, beta=4, seed=2022, num_classes=1000, pretrained=False):
        super().__init__()
        torch.manual_seed(seed)
        self.num_classes = num_classes
        self.arch_type = arch_type
        self.input_channel = input_channel
        self.beta = beta
        self.seed = seed
        self.pretrained = pretrained

        self.initialize_network()

    def initialize_network(self):
        self.gating_node_outputs = dict()
        all_devices = _get_all_device_indices()
        for each_device in all_devices:
            self.gating_node_outputs[str(each_device)] = OrderedDict()

        self.gating_network = TorchVision_Deep_Gating_Network(
            self.arch_type, self.input_channel, pretrained=self.pretrained)
        print("self.gating_network", self.gating_network)
        print("Gating net params:", sum(p.numel()
              for p in self.gating_network.parameters()))
        self.value_network = ALLONES_TorchVision_Value_Network(
            self.arch_type, self.input_channel, num_classes=self.num_classes, pretrained=self.pretrained, gating_node_outputs=self.gating_node_outputs)
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
        idevice = inp.get_device()
        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=idevice, non_blocking=True)

        before_relu_outputs = self.gating_network(inp, verbose=verbose)

        if(verbose > 3):
            print("before_relu_outputs keys")

        for key, value in before_relu_outputs[str(idevice)].items():
            ip_device = value.get_device()
            if(verbose > 3):
                print("key:{},value:{}".format(key, value.size()))

            temp = nn.Sigmoid()(
                self.beta * value)
            self.gating_node_outputs[str(ip_device)][key] = temp
            if(key == 'relu' and verbose > 3):
                print("Generated Gate signal Dev:{},input device:{},layer_name:{},gs.size():{}, gs:{}".format(
                    str(ip_device), str(idevice), key, temp.size(), temp))
            assert self.gating_node_outputs[str(ip_device)][key].get_device(
            ) == ip_device, 'Gating signal generated moved to different device'

        if(verbose > 3):
            print("gating_node_outputs keys")
            for key, value in self.gating_node_outputs[str(ip_device)].items():
                print("key:{},value:{}".format(key, value.size()))

        inp_gating = torch.ones(inp.size(),
                                requires_grad=True, device=ip_device)

        final_layer_out = self.value_network(
            inp_gating, self.gating_node_outputs, verbose=verbose)

        return final_layer_out


class TorchVision_Deep_Gating_Network(nn.Module):
    def __init__(self, arch_type, input_channel, pretrained=False):
        super().__init__()
        self.arch_type = arch_type
        self.input_channel = input_channel
        self.pretrained = pretrained

        self.layer_outs = dict()
        all_devices = _get_all_device_indices()
        for each_device in all_devices:
            self.layer_outs[str(each_device)] = OrderedDict()

        self.initialize_network()

    def initialize_network(self):
        self.f_id_hooks = []
        self.list_of_modules = []
        # Extracts the arch type between "__"
        arch_type = self.arch_type[self.arch_type.index(
            "__")+2:self.arch_type.rindex("__")]
        # Load the model architecture
        self.model_instance = models.__dict__[
            arch_type](pretrained=self.pretrained)

        last_relu_name, _ = get_last_layer_instance(
            self.model_instance, layer=nn.ReLU)
        convert_layers_after_last_relu_to_identity(
            self.model_instance, last_relu_name)
        convert_inplacerelu_to_relu(self.model_instance)
        convert_maxpool_to_avgpool(self.model_instance)

        self.list_of_modules.append(self.model_instance)

        self.list_of_modules = nn.ModuleList(self.list_of_modules)
        self.initialize_hooks()

    def initialize_hooks(self):
        self.clear_hooks()

        prev_layer = None
        prev_layer_name = None
        # Capture outputs of Identity module (earlier input to Relu module)
        for i, (name, layer) in enumerate(self.model_instance.named_modules()):
            if (isinstance(layer, nn.ReLU)):
                print("Gating signals hooked at layer:", prev_layer_name)
                self.f_id_hooks.append(prev_layer.register_forward_hook(
                    self.forward_relu_hook(name)))
            prev_layer = layer
            prev_layer_name = name

    def forward_relu_hook(self, layer_name):
        def hook(module, input, output):
            self.layer_outs[str(output.get_device())][layer_name] = output
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


class TorchVision_DLGN(nn.Module):
    def __init__(self, arch_type, input_channel, beta=4, seed=2022, num_classes=1000, pretrained=False, is_convert_maxpool_to_avgpool=True):
        super().__init__()
        torch.manual_seed(seed)
        self.num_classes = num_classes
        self.arch_type = arch_type
        self.input_channel = input_channel
        self.beta = beta
        self.seed = seed
        self.pretrained = pretrained
        self.is_convert_maxpool_to_avgpool = is_convert_maxpool_to_avgpool

        self.initialize_network()

    def initialize_network(self):
        self.gating_node_outputs = dict()
        all_devices = _get_all_device_indices()
        for each_device in all_devices:
            self.gating_node_outputs[str(each_device)] = OrderedDict()

        self.gating_network = TorchVision_Gating_Network(
            self.arch_type, self.input_channel, pretrained=self.pretrained, is_convert_maxpool_to_avgpool=self.is_convert_maxpool_to_avgpool)
        print("self.gating_network", self.gating_network)
        print("Gating net params:", sum(p.numel()
              for p in self.gating_network.parameters()))
        self.value_network = ALLONES_TorchVision_Value_Network(
            self.arch_type, self.input_channel, num_classes=self.num_classes, pretrained=self.pretrained, gating_node_outputs=self.gating_node_outputs)
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
        idevice = inp.get_device()
        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=idevice, non_blocking=True)

        before_relu_outputs = self.gating_network(inp, verbose=verbose)

        if(verbose > 3):
            print("before_relu_outputs keys")

        for key, value in before_relu_outputs[str(idevice)].items():
            ip_device = value.get_device()
            if(verbose > 3):
                print("key:{},value:{}".format(key, value.size()))

            temp = nn.Sigmoid()(
                self.beta * value)
            self.gating_node_outputs[str(ip_device)][key] = temp
            if(key == 'relu' and verbose > 3):
                print("Generated Gate signal Dev:{},input device:{},layer_name:{},gs.size():{}, gs:{}".format(
                    str(ip_device), str(idevice), key, temp.size(), temp))
            assert self.gating_node_outputs[str(ip_device)][key].get_device(
            ) == ip_device, 'Gating signal generated moved to different device'

        if(verbose > 3):
            print("gating_node_outputs keys")
            for key, value in self.gating_node_outputs[str(ip_device)].items():
                print("key:{},value:{}".format(key, value.size()))

        inp_gating = torch.ones(inp.size(),
                                requires_grad=True, device=ip_device)

        final_layer_out = self.value_network(
            inp_gating, self.gating_node_outputs, verbose=verbose)

        return final_layer_out


class TorchVision_Gating_Network(nn.Module):
    def __init__(self, arch_type, input_channel, pretrained=False, is_convert_maxpool_to_avgpool=True):
        super().__init__()
        self.is_convert_maxpool_to_avgpool = is_convert_maxpool_to_avgpool
        self.arch_type = arch_type
        self.input_channel = input_channel
        self.pretrained = pretrained

        self.layer_outs = dict()
        all_devices = _get_all_device_indices()
        for each_device in all_devices:
            self.layer_outs[str(each_device)] = OrderedDict()

        self.initialize_network()

    def initialize_network(self):
        self.f_id_hooks = []
        self.list_of_modules = []
        # Extracts the arch type between "__"
        arch_type = self.arch_type[self.arch_type.index(
            "__")+2:self.arch_type.rindex("__")]
        # Load the model architecture
        self.model_instance = models.__dict__[
            arch_type](pretrained=self.pretrained)

        last_relu_name, _ = get_last_layer_instance(
            self.model_instance, layer=nn.ReLU)
        convert_layers_after_last_relu_to_identity(
            self.model_instance, last_relu_name)
        # Replace relu activations with Identity functions
        convert_relu_to_identity(self.model_instance)
        if(self.is_convert_maxpool_to_avgpool):
            convert_maxpool_to_avgpool(self.model_instance)

        self.list_of_modules.append(self.model_instance)

        self.list_of_modules = nn.ModuleList(self.list_of_modules)
        self.initialize_hooks()

    def initialize_hooks(self):
        self.clear_hooks()

        prev_layer = None
        # Capture outputs of Identity module (earlier input to Relu module)
        for i, (name, layer) in enumerate(self.model_instance.named_modules()):
            if (isinstance(layer, nn.Identity)):
                self.f_id_hooks.append(prev_layer.register_forward_hook(
                    self.forward_identity_hook(name)))
            prev_layer = layer

    def forward_identity_hook(self, layer_name):
        def hook(module, input, output):
            self.layer_outs[str(output.get_device())][layer_name] = output
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


class ALLONES_TorchVision_Value_Network(nn.Module):
    def __init__(self, arch_type, input_channel, num_classes=1000, pretrained=False, gating_node_outputs=None):
        super(ALLONES_TorchVision_Value_Network, self).__init__()
        self.gating_node_outputs = gating_node_outputs
        self.pretrained = pretrained
        self.list_of_modules = []
        self.f_relu_hooks = []
        self.arch_type = arch_type
        # Extracts the arch type between "__"
        arch_type = self.arch_type[self.arch_type.index(
            "__")+2:self.arch_type.rindex("__")]
        # Load the model architecture
        self.model_instance = models.__dict__[
            arch_type](pretrained=pretrained)

        # Replace relu activations with Identity functions
        convert_relu_to_identity(self.model_instance)
        convert_maxpool_to_avgpool(self.model_instance)

        last_linear_layer_name, last_linear_layer = get_last_layer_instance(
            self.model_instance, nn.Linear)
        num_ftrs = last_linear_layer.in_features
        replacement_layer_obj = nn.Linear(num_ftrs, num_classes)
        replace_given_layer_name_with_layer(
            self.model_instance, last_linear_layer_name, replacement_layer_obj)

        self.list_of_modules.append(self.model_instance)

        self.list_of_modules = nn.ModuleList(self.list_of_modules)
        self.initialize_hooks()

    def initialize_hooks(self):
        self.clear_hooks()
        self.gating_signals = None
        prev_layer = None
        prev_layer_name = None

        all_devices = _get_all_device_indices()
        # Attaches hook to Identity and modify its inputs
        for i, (name, layer) in enumerate(self.model_instance.named_modules()):
            if (isinstance(layer, nn.Identity)):
                print("Hook added to layer name", prev_layer_name)
                self.f_relu_hooks.append(
                    prev_layer.register_forward_hook(self.forward_hook(name)))
            prev_layer = layer
            prev_layer_name = name

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            temp = self.gating_node_outputs[str(
                output.get_device())][layer_name]
            # if(layer_name == 'relu'):
            #     print("Forward Gate signal Dev:{},output device:{},layer_name:{},input[0].size():{},temp.size():{}, temp:{}".format(
            #         str(temp.get_device()), str(input[0].get_device()), layer_name, input[0].size(), temp.size(), temp))
            del self.gating_node_outputs[str(output.get_device())][layer_name]
            return output * temp
        return hook

    def forward(self, inp, gating_signals, verbose=2):
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


def get_img_size(dataset):
    if(dataset == "cifar10"):
        return [3, 32, 32]
    elif(dataset == "mnist"):
        return [1, 28, 28]
    elif(dataset == "fashion_mnist"):
        return [1, 28, 28]
    elif(dataset == "imagenet_1000"):
        return [3, 224, 224]
    elif("mnist_" in dataset):
        return [1, 28, 28]
    elif("fashion_mnist_" in dataset):
        return [1, 28, 28]


def get_model_instance_from_dataset(dataset, model_arch_type, seed=2022, mask_percentage=40, num_classes=10, nodes_in_each_layer_list=[], pretrained=False, aux_logits=True):
    temp = get_img_size(dataset)
    inp_channel = temp[0]
    input_size_list = temp[1:]

    return get_model_instance(model_arch_type, inp_channel, seed=seed, mask_percentage=mask_percentage, num_classes=num_classes, input_size_list=input_size_list, nodes_in_each_layer_list=nodes_in_each_layer_list, pretrained=pretrained, aux_logits=aux_logits)


def get_model_instance(model_arch_type, inp_channel, seed=2022, mask_percentage=40, num_classes=10, nodes_in_each_layer_list=[], input_size_list=[], pretrained=False, aux_logits=True):
    if(seed == ""):
        seed = 2022

    torchvision_model_names = sorted(name for name in models.__dict__
                                     if name.islower() and not name.startswith("__")
                                     and callable(models.__dict__[name]))

    net = None
    if(model_arch_type == 'plain_pure_conv4_dnn'):
        net = Plain_CONV4_Net(inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == 'conv4_dlgn'):
        net = Conv4_DLGN_Net(inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == 'dlgn__conv4_dlgn_pad0_st1_bn__'):
        net = Conv4_DLGN_Net_pad0_wo_bn(inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == 'dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias__'):
        net = Conv4_DLGN_Net_pad_k_1_wo_bn_wo_bias(inp_channel, seed=seed, num_classes=num_classes)
    elif(model_arch_type == 'dlgn__im_conv4_dlgn_pad_k_1_st1_bn_wo_bias__'):
        net = IM_Conv4_DLGN_Net_pad_k_1_wo_bn_wo_bias(inp_channel, seed=seed, num_classes=num_classes)
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
    elif(model_arch_type == "dlgn__vgg16_bn__"):
        allones = np.ones((1, 3, 32, 32)).astype(np.float32)
        net = vgg16_bn(allones, num_classes=num_classes)
    elif(model_arch_type == "dlgn__pad2_vgg16_bn__"):
        allones = np.ones((1, 3, 32, 32)).astype(np.float32)
        net = pad2_vgg16_bn()
    elif(model_arch_type == "dlgn__st1_pad2_vgg16_bn_wo_bias__"):
        allones = np.ones((1, 3, 32, 32)).astype(np.float32)
        net = st1_pad2_vgg16_bn_wo_bias()
    elif(model_arch_type == "dnn__st1_pad1_vgg16_bn_wo_bias__"):
        net = dnn_st1_pad1_vgg16_bn_wo_bias()
    elif(model_arch_type == "dlgn__st1_pad0_vgg16_bn__"):    
        net = st1_pad0_vgg16_bn()
    elif(model_arch_type == "dlgn__st1_pad1_vgg16_bn_wo_bias__"):
        net = st1_pad1_vgg16_bn_wo_bias()
    elif(model_arch_type == "dnn__cvgg16_bn__"):
        net = dnn_vgg16_bn()
    elif(model_arch_type == "dlgn__googlenet__"):
        net = Custom_GoogLeNet(
            "dlgn", num_classes, aux_logits)
    elif(model_arch_type == "dgn__googlenet__"):
        net = Custom_GoogLeNet(
            "dgn", num_classes, aux_logits)
    elif(model_arch_type == "dlgn__gatempool_resnet18__"):
        model_arch_type = model_arch_type.replace("gatempool_", "")
        net = TorchVision_DLGN(
            model_arch_type, inp_channel, seed=seed, num_classes=num_classes, pretrained=pretrained, is_convert_maxpool_to_avgpool=False)

    # If no specific implementation was found for model arch type, then try to instantiate from torchvision
    if(net is None):
        arch_type_extracted_from_model_arch = model_arch_type[model_arch_type.index(
            "__")+2:model_arch_type.rindex("__")]
        if(arch_type_extracted_from_model_arch in torchvision_model_names and 'dlgn' in model_arch_type):
            print("Instantiating torchvision architecture")
            net = TorchVision_DLGN(
                model_arch_type, inp_channel, seed=seed, num_classes=num_classes, pretrained=pretrained)
        elif(arch_type_extracted_from_model_arch in torchvision_model_names and 'dgn' in model_arch_type):
            print("Instantiating torchvision architecture")
            net = TorchVision_DeepGatedNet(
                model_arch_type, inp_channel, seed=seed, num_classes=num_classes, pretrained=pretrained)
        elif(arch_type_extracted_from_model_arch in torchvision_model_names and 'dnn' in model_arch_type):
            print("Instantiating torchvision architecture")
            arch_type = model_arch_type[model_arch_type.index(
                "__")+2:model_arch_type.rindex("__")]
            net = models.__dict__[
                arch_type](pretrained=pretrained)
            last_linear_layer_name, last_linear_layer = get_last_layer_instance(
                net, nn.Linear)
            num_ftrs = last_linear_layer.in_features
            replacement_layer_obj = nn.Linear(num_ftrs, num_classes)
            replace_given_layer_name_with_layer(
                net, last_linear_layer_name, replacement_layer_obj)
            convert_maxpool_to_avgpool(net)
            print("net", net)

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

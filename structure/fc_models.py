import torch
import torch.nn as nn
import numpy as np


class DLGN_FC_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, beta=4, input_size_list=[28, 28], seed=2022, num_classes=10):
        super(DLGN_FC_Network, self).__init__()
        torch.manual_seed(seed)
        self.beta = beta
        input_size = input_size_list[0]
        for ind in range(1, len(input_size_list)):
            input_size *= input_size_list[ind]

        self.gating_network = DLGN_FC_Gating_Network(
            nodes_in_each_layer_list, input_size, seed=seed)
        print("self.gating_network", self.gating_network)
        print("Gating net params:", sum(p.numel()
              for p in self.gating_network.parameters()))
        self.value_network = ALLONES_FC_Value_Network(
            nodes_in_each_layer_list, input_size, seed=seed, num_classes=num_classes)
        print("self.value_network", self.value_network)
        print("Value net params:", sum(p.numel()
              for p in self.value_network.parameters()))

    def forward(self, inp, verbose=2):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        inp_gating = torch.ones(inp.size(),
                                requires_grad=True, device=device)

        linear_conv_outputs, _ = self.gating_network(inp, verbose=verbose)
        self.linear_conv_outputs = linear_conv_outputs

        # for indx in range(len(linear_conv_outputs)):
        #     each_conv_out = linear_conv_outputs[indx]
        #     print("each_conv_out: {} => size {}".format(
        #         indx, each_conv_out.size()))

        self.gating_node_outputs = [None] * len(linear_conv_outputs)

        for indx in range(len(linear_conv_outputs)):
            each_linear_conv_output = linear_conv_outputs[indx]
            self.gating_node_outputs[indx] = nn.Sigmoid()(
                self.beta * each_linear_conv_output)

        final_layer_out = self.value_network(
            inp_gating, self.gating_node_outputs, verbose=verbose)

        return final_layer_out
    
    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            return self.gating_network.list_of_modules[layer_num]
        elif(network_type == "WEIGHT_NET"):
            return self.value_network.list_of_modules[layer_num]


class DLGN_FC_Gating_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, input_size, seed=2022):
        super(DLGN_FC_Gating_Network, self).__init__()
        torch.manual_seed(seed)

        list_of_modules = []

        previous_layer_size = input_size
        for each_current_layer_size in nodes_in_each_layer_list:
            list_of_modules.append(nn.Linear(
                previous_layer_size, each_current_layer_size))
            previous_layer_size = each_current_layer_size

        self.list_of_modules = nn.ModuleList(list_of_modules)

    def forward(self, inp, verbose=2):
        prev_out = torch.flatten(inp, 1)
        layer_outs = []
        for each_module in self.list_of_modules:
            prev_out = each_module(prev_out)
            layer_outs.append(prev_out)

        return layer_outs, prev_out

    def __str__(self):
        ret = "Gating network "+" \n module_list:"
        for each_module in self.list_of_modules:
            ret += str(each_module)+" \n Params in module is:" + \
                str(sum(p.numel() for p in each_module.parameters()))+"\n"

        return ret


class ALLONES_FC_Value_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, input_size, seed=2022, num_classes=10):
        super(ALLONES_FC_Value_Network, self).__init__()
        torch.manual_seed(seed)
        list_of_modules = []

        previous_layer_size = input_size
        for each_current_layer_size in nodes_in_each_layer_list:
            list_of_modules.append(nn.Linear(
                previous_layer_size, each_current_layer_size))
            previous_layer_size = each_current_layer_size

        self.list_of_modules = nn.ModuleList(list_of_modules)
        self.output_layer = nn.Linear(previous_layer_size, num_classes)

    def forward(self, inp, gating_signals, verbose=2):
        prev_out = torch.flatten(inp, 1)
        for indx in range(len(self.list_of_modules)):
            each_module = self.list_of_modules[indx]
            prev_out = each_module(prev_out)
            prev_out *= gating_signals[indx]

        prev_out = self.output_layer(prev_out)
        return prev_out

    def __str__(self):
        ret = "Value network "+" \n module_list:"
        for each_module in self.list_of_modules:
            ret += str(each_module)+" \n Params in module is:" + \
                str(sum(p.numel() for p in each_module.parameters()))+"\n"

        return ret


class DGN_FC_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, beta=4, input_size_list=[28, 28], seed=2022, num_classes=10):
        super(DGN_FC_Network, self).__init__()
        torch.manual_seed(seed)
        self.beta = beta
        input_size = input_size_list[0]
        for ind in range(1, len(input_size_list)):
            input_size *= input_size_list[ind]

        self.gating_network = DGN_FC_Gating_Network(
            nodes_in_each_layer_list, input_size, seed=seed)
        print("self.gating_network", self.gating_network)
        print("Gating net params:", sum(p.numel()
              for p in self.gating_network.parameters()))
        self.value_network = ALLONES_FC_Value_Network(
            nodes_in_each_layer_list, input_size, seed=seed, num_classes=num_classes)
        print("self.value_network", self.value_network)
        print("Value net params:", sum(p.numel()
              for p in self.value_network.parameters()))

    def forward(self, inp, verbose=2):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        inp_gating = torch.ones(inp.size(),
                                requires_grad=True, device=device)

        linear_conv_outputs, _ = self.gating_network(inp, verbose=verbose)
        self.linear_conv_outputs = linear_conv_outputs

        # for indx in range(len(linear_conv_outputs)):
        #     each_conv_out = linear_conv_outputs[indx]
        #     print("each_conv_out: {} => size {}".format(
        #         indx, each_conv_out.size()))

        self.gating_node_outputs = [None] * len(linear_conv_outputs)

        for indx in range(len(linear_conv_outputs)):
            each_linear_conv_output = linear_conv_outputs[indx]
            self.gating_node_outputs[indx] = nn.Sigmoid()(
                self.beta * each_linear_conv_output)

        final_layer_out = self.value_network(
            inp_gating, self.gating_node_outputs, verbose=verbose)

        return final_layer_out

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            return self.gating_network.list_of_modules[layer_num]
        elif(network_type == "WEIGHT_NET"):
            return self.value_network.list_of_modules[layer_num]


class DGN_FC_Gating_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, input_size, seed=2022):
        super(DGN_FC_Gating_Network, self).__init__()
        torch.manual_seed(seed)

        list_of_modules = []

        previous_layer_size = input_size
        for each_current_layer_size in nodes_in_each_layer_list:
            list_of_modules.append(nn.Linear(
                previous_layer_size, each_current_layer_size))
            previous_layer_size = each_current_layer_size

        self.list_of_modules = nn.ModuleList(list_of_modules)
        self.relu = nn.ReLU()

    def forward(self, inp, verbose=2):
        prev_out = torch.flatten(inp, 1)
        layer_outs = []
        for each_module in self.list_of_modules:
            prev_out = each_module(prev_out)
            layer_outs.append(prev_out)
            prev_out = self.relu(prev_out)

        return layer_outs, prev_out

    def __str__(self):
        ret = "Gating network "+" \n module_list:"
        for each_module in self.list_of_modules:
            ret += str(each_module)+" \n Params in module is:" + \
                str(sum(p.numel() for p in each_module.parameters()))+"\n"

        return ret


class DNN_FC_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, input_size_list=[28, 28], seed=2022, num_classes=10):
        super(DNN_FC_Network, self).__init__()
        torch.manual_seed(seed)
        list_of_modules = []

        input_size = input_size_list[0]
        for ind in range(1, len(input_size_list)):
            input_size *= input_size_list[ind]

        previous_layer_size = input_size
        for indx in range(len(nodes_in_each_layer_list)):
            each_current_layer_size = nodes_in_each_layer_list[indx]
            list_of_modules.append(nn.Linear(
                previous_layer_size, each_current_layer_size))
            previous_layer_size = each_current_layer_size

        self.list_of_modules = nn.ModuleList(list_of_modules)
        self.output_layer = nn.Linear(
            previous_layer_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, inp, verbose=2):
        prev_out = torch.flatten(inp, 1)
        num_layers = len(self.list_of_modules)
        layer_outs = []
        for indx in range(num_layers):
            each_module = self.list_of_modules[indx]
            prev_out = each_module(prev_out)
            layer_outs.append(prev_out)
            prev_out = self.relu(prev_out)

        self.linear_conv_outputs = layer_outs
        prev_out = self.output_layer(prev_out)
        return prev_out

    def __str__(self):
        ret = "Value network "+" \n module_list:"
        for each_module in self.list_of_modules:
            ret += str(each_module)+" \n Params in module is:" + \
                str(sum(p.numel() for p in each_module.parameters()))+"\n"

        return ret

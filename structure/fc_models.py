import torch
import torch.nn as nn
import numpy as np
from utils.visualise_utils import determine_row_col_from_features
from sklearn.decomposition import PCA
from collections import OrderedDict
from utils.forward_visualization_helpers import merge_operations_in_modules, apply_input_on_conv_matrix, merge_layers_operations_in_modules


class PCA_Layer(nn.Module):
    def __init__(self, data, explained_var_required):
        super(PCA_Layer, self).__init__()
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
        if hasattr(super, 'input_channel'):
            inp = torch.reshape(
                inp, (temp_size[0], super.input_channel, self.input_size_list[0], self.input_size_list[1]))
        else:
            inp = torch.reshape(
                inp, (temp_size[0], self.input_size_list[0], self.input_size_list[1]))

        return inp

class FC_Standardize(nn.Module):
    def __init__(self, mu=None, std=None):
        super(FC_Standardize, self).__init__()
        if(mu is None):
            mu = torch.tensor([0.4914, 0.4822, 0.4465])
        if(std is None):
            std = torch.tensor([0.2023, 0.1994, 0.2010])
        mu = mu[None,:,None,None]
        std = std[None,:,None,None]
        self.mu, self.std = mu, std

    def forward(self, x,idevice=None):
        if(idevice is not None):
            self.mu = self.mu.to(device=idevice)
            self.std = self.std.to(device=idevice)
        return (x - self.mu) / self.std

class BC_SF_DLGN_FC_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, beta=4, input_size_list=[28, 28], seed=2022, num_classes=2):
        super(BC_SF_DLGN_FC_Network, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.nodes_in_each_layer_list = nodes_in_each_layer_list
        self.seed = seed
        self.beta = beta
        self.num_classes = num_classes
        self.input_size_list = input_size_list
        self.initialize_network()

    def init_gate_net(self):
        self.gating_network = SF_DLGN_FC_Gating_Network(
            self.nodes_in_each_layer_list, self.input_size, seed=self.seed)
    
    
    def init_value_net(self):
        self.value_network = ALLONES_FC_Value_Network(
            self.nodes_in_each_layer_list, self.input_size, seed=self.seed,num_classes=1)
        self.sigmoid = nn.Sigmoid()

    def initialize_network(self):
        self.input_size = self.input_size_list[0]
        for ind in range(1, len(self.input_size_list)):
            self.input_size *= self.input_size_list[ind]
        
        self.init_gate_net()        
        print("self.gating_network", self.gating_network)
        print("Gating net params:", sum(p.numel()
              for p in self.gating_network.parameters()))
        self.init_value_net()
        print("self.value_network", self.value_network)
        print("Value net params:", sum(p.numel()
              for p in self.value_network.parameters()))

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = PCA_Layer(data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        self.initialize_network()
        return self.pca_layer.k

    def initialize_standardization_layer(self,mu=None,std=None):
        self.standardize_layer = FC_Standardize(mu,std)
        self.standardize_layer.mu = self.standardize_layer.mu.to(device=self.device,non_blocking=True)
        self.standardize_layer.std = self.standardize_layer.std.to(device=self.device,non_blocking=True)

    def forward(self, inp, verbose=2):
        if hasattr(self, 'standardize_layer'):
            inp = self.standardize_layer(inp)
        inp = torch.flatten(inp, 1)

        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=self.device, non_blocking=True)

        inp_gating = torch.ones(inp.size(),
                                requires_grad=True, device=self.device)

        linear_conv_outputs, _ = self.gating_network(inp, verbose=verbose)
        self.linear_conv_outputs = linear_conv_outputs

        # for indx in range(len(linear_conv_outputs)):
        #     each_conv_out = linear_conv_outputs[indx]
        #     print("each_conv_out: {} => size {}".format(
        #         indx, each_conv_out.size()))

        self.gating_node_outputs = [None] * len(linear_conv_outputs)

        for indx in range(len(linear_conv_outputs)):
            each_linear_conv_output = linear_conv_outputs[indx]
            self.gating_node_outputs[indx] = self.sigmoid(
                self.beta * each_linear_conv_output)
            # print("indx:{} self.gating_node_outputs[indx] size:{}----->{}".format(indx,self.gating_node_outputs[indx].size(),self.gating_node_outputs[indx]))

        self.output_logits = self.value_network(
            inp_gating, self.gating_node_outputs, verbose=verbose)
        final_layer_out = self.output_logits
        
        return torch.squeeze(final_layer_out,-1)

    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        for layer_num in range(len(self.gating_network.list_of_modules)):
          gating_net_layers_ordered["fc"+str(layer_num)+"_g"] = self.gating_network.list_of_modules[layer_num]

        return gating_net_layers_ordered

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            return self.gating_network.list_of_modules[layer_num]
        elif(network_type == "WEIGHT_NET"):
            return self.value_network.list_of_modules[layer_num]

    def exact_forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        x = torch.flatten(x,1)
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        conv_matrix_operations_in_each_layer = OrderedDict()
        conv_bias_operations_in_each_layer = OrderedDict()
        channel_outs_size_in_each_layer = OrderedDict()
        current_tensor_size = x.size()
        print("current_tensor_size ", current_tensor_size)
        merged_conv_matrix = None
        merged_conv_bias = None
        orig_out = x

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_matrix, merged_conv_bias, current_tensor_size = layer_obj.weight, layer_obj.bias, (layer_obj.out_features,)
                conv_matrix_operations_in_each_layer[layer_name] = merged_conv_matrix
                conv_bias_operations_in_each_layer[layer_name] = merged_conv_bias
                channel_outs_size_in_each_layer[layer_name] = current_tensor_size

                orig_out = layer_obj(x)

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


class BC_DLGN_FC_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, beta=4, input_size_list=[28, 28], seed=2022, num_classes=2):
        super(BC_DLGN_FC_Network, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.nodes_in_each_layer_list = nodes_in_each_layer_list
        self.seed = seed
        self.beta = beta
        self.num_classes = num_classes
        self.input_size_list = input_size_list
        self.initialize_network()

    def init_gate_net(self):
        self.gating_network = DLGN_FC_Gating_Network(
            self.nodes_in_each_layer_list, self.input_size, seed=self.seed)
    
    
    def init_value_net(self):
        self.value_network = ALLONES_FC_Value_Network(
            self.nodes_in_each_layer_list, self.input_size, seed=self.seed,num_classes=1)
        self.sigmoid = nn.Sigmoid()

    def initialize_network(self):
        self.input_size = self.input_size_list[0]
        for ind in range(1, len(self.input_size_list)):
            self.input_size *= self.input_size_list[ind]
        
        self.init_gate_net()        
        print("self.gating_network", self.gating_network)
        print("Gating net params:", sum(p.numel()
              for p in self.gating_network.parameters()))
        self.init_value_net()
        print("self.value_network", self.value_network)
        print("Value net params:", sum(p.numel()
              for p in self.value_network.parameters()))

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = PCA_Layer(data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        self.initialize_network()
        return self.pca_layer.k

    def initialize_standardization_layer(self,mu=None,std=None):
        self.standardize_layer = FC_Standardize(mu,std)
        self.standardize_layer.mu = self.standardize_layer.mu.to(device=self.device,non_blocking=True)
        self.standardize_layer.std = self.standardize_layer.std.to(device=self.device,non_blocking=True)

    def forward(self, inp, verbose=2):
        if hasattr(self, 'standardize_layer'):
            inp = self.standardize_layer(inp)
        inp = torch.flatten(inp, 1)

        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=self.device, non_blocking=True)

        inp_gating = torch.ones(inp.size(),
                                requires_grad=True, device=self.device)

        linear_conv_outputs, _ = self.gating_network(inp, verbose=verbose)
        self.linear_conv_outputs = linear_conv_outputs

        # for indx in range(len(linear_conv_outputs)):
        #     each_conv_out = linear_conv_outputs[indx]
        #     print("each_conv_out: {} => size {}".format(
        #         indx, each_conv_out.size()))

        self.gating_node_outputs = [None] * len(linear_conv_outputs)

        for indx in range(len(linear_conv_outputs)):
            each_linear_conv_output = linear_conv_outputs[indx]
            self.gating_node_outputs[indx] = self.sigmoid(
                self.beta * each_linear_conv_output)
            # print("indx:{} self.gating_node_outputs[indx] size:{}----->{}".format(indx,self.gating_node_outputs[indx].size(),self.gating_node_outputs[indx]))

        self.output_logits = self.value_network(
            inp_gating, self.gating_node_outputs, verbose=verbose)
        final_layer_out = self.output_logits
        
        return torch.squeeze(final_layer_out,-1)

    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        for layer_num in range(len(self.gating_network.list_of_modules)):
          gating_net_layers_ordered["fc"+str(layer_num)+"_g"] = self.gating_network.list_of_modules[layer_num]

        return gating_net_layers_ordered

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            return self.gating_network.list_of_modules[layer_num]
        elif(network_type == "WEIGHT_NET"):
            return self.value_network.list_of_modules[layer_num]

    def exact_forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        x = torch.flatten(x,1)
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        conv_matrix_operations_in_each_layer = OrderedDict()
        conv_bias_operations_in_each_layer = OrderedDict()
        channel_outs_size_in_each_layer = OrderedDict()
        current_tensor_size = x.size()
        print("current_tensor_size ", current_tensor_size)
        merged_conv_matrix = None
        merged_conv_bias = None
        orig_out = x

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_matrix, merged_conv_bias, current_tensor_size = layer_obj.weight, layer_obj.bias, (layer_obj.out_features,)
                conv_matrix_operations_in_each_layer[layer_name] = merged_conv_matrix
                conv_bias_operations_in_each_layer[layer_name] = merged_conv_bias
                channel_outs_size_in_each_layer[layer_name] = current_tensor_size

                orig_out = layer_obj(x)

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


class SF_DLGN_FC_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, beta=4, input_size_list=[28, 28], seed=2022, num_classes=10):
        super(SF_DLGN_FC_Network, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.nodes_in_each_layer_list = nodes_in_each_layer_list
        self.seed = seed
        self.beta = beta
        self.num_classes = num_classes
        self.input_size_list = input_size_list
        self.initialize_network()

    def init_gate_net(self):
        self.gating_network = SF_DLGN_FC_Gating_Network(
            self.nodes_in_each_layer_list, self.input_size, seed=self.seed)
    
    
    def init_value_net(self):
        self.value_network = ALLONES_FC_Value_Network(
            self.nodes_in_each_layer_list, self.input_size, seed=self.seed, num_classes=self.num_classes)

    def initialize_network(self):
        self.input_size = self.input_size_list[0]
        for ind in range(1, len(self.input_size_list)):
            self.input_size *= self.input_size_list[ind]
        
        self.init_gate_net()        
        print("self.gating_network", self.gating_network)
        print("Gating net params:", sum(p.numel()
              for p in self.gating_network.parameters()))
        self.init_value_net()
        print("self.value_network", self.value_network)
        print("Value net params:", sum(p.numel()
              for p in self.value_network.parameters()))

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = PCA_Layer(data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        self.initialize_network()
        return self.pca_layer.k

    def initialize_standardization_layer(self,mu=None,std=None):
        self.standardize_layer = FC_Standardize(mu,std)
        self.standardize_layer.mu = self.standardize_layer.mu.to(device=self.device,non_blocking=True)
        self.standardize_layer.std = self.standardize_layer.std.to(device=self.device,non_blocking=True)

    def forward(self, inp, verbose=2):
        if hasattr(self, 'standardize_layer'):
            inp = self.standardize_layer(inp)
        inp = torch.flatten(inp, 1)

        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=self.device, non_blocking=True)

        inp_gating = torch.ones(inp.size(),
                                requires_grad=True, device=self.device)

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

    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        for layer_num in range(len(self.gating_network.list_of_modules)):
          gating_net_layers_ordered["fc"+str(layer_num)+"_g"] = self.gating_network.list_of_modules[layer_num]

        return gating_net_layers_ordered
    
    def get_value_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        for layer_num in range(len(self.gating_network.list_of_modules)):
          gating_net_layers_ordered["fc"+str(layer_num)+"_g"] = self.value_network.list_of_modules[layer_num]

        return gating_net_layers_ordered

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            return self.gating_network.list_of_modules[layer_num]
        elif(network_type == "WEIGHT_NET"):
            return self.value_network.list_of_modules[layer_num]

    def exact_forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        x = torch.flatten(x,1)
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        conv_matrix_operations_in_each_layer = OrderedDict()
        conv_bias_operations_in_each_layer = OrderedDict()
        channel_outs_size_in_each_layer = OrderedDict()
        current_tensor_size = x.size()
        print("current_tensor_size ", current_tensor_size)
        merged_conv_matrix = None
        merged_conv_bias = None
        orig_out = x

        with torch.no_grad():
            for layer_name, layer_obj in gating_net_layers_ordered.items():
                merged_conv_matrix, merged_conv_bias, current_tensor_size = layer_obj.weight, layer_obj.bias, (layer_obj.out_features,)
                conv_matrix_operations_in_each_layer[layer_name] = merged_conv_matrix
                conv_bias_operations_in_each_layer[layer_name] = merged_conv_bias
                channel_outs_size_in_each_layer[layer_name] = current_tensor_size

                orig_out = layer_obj(x)

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

class SF_DLGN_FC_Gating_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, input_size, seed=2022):
        super(SF_DLGN_FC_Gating_Network, self).__init__()
        torch.manual_seed(seed)

        list_of_modules = []

        for each_current_layer_size in nodes_in_each_layer_list:
            list_of_modules.append(nn.Linear(
                input_size, each_current_layer_size))

        self.list_of_modules = nn.ModuleList(list_of_modules)

    def forward(self, inp, verbose=2):
        inp = torch.flatten(inp, 1)
        prev_out = inp
        layer_outs = []
        for each_module in self.list_of_modules:
            prev_out = each_module(inp)
            layer_outs.append(prev_out)

        return layer_outs, prev_out

    def __str__(self):
        ret = "Gating network "+" \n module_list:"
        for each_module in self.list_of_modules:
            ret += str(each_module)+" \n Params in module is:" + \
                str(sum(p.numel() for p in each_module.parameters()))+"\n"

        return ret

class DLGN_FC_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, beta=4, input_size_list=[28, 28], seed=2022, num_classes=10):
        super(DLGN_FC_Network, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.nodes_in_each_layer_list = nodes_in_each_layer_list
        self.seed = seed
        self.beta = beta
        self.num_classes = num_classes
        self.input_size_list = input_size_list
        self.initialize_network()

    def initialize_network(self):
        input_size = self.input_size_list[0]
        for ind in range(1, len(self.input_size_list)):
            input_size *= self.input_size_list[ind]
        self.input_size = input_size
        self.init_gate_net()
        print("self.gating_network", self.gating_network)
        print("Gating net params:", sum(p.numel()
              for p in self.gating_network.parameters()))
        self.init_value_net()
        print("self.value_network", self.value_network)
        print("Value net params:", sum(p.numel()
              for p in self.value_network.parameters()))
    
    def init_gate_net(self):
        self.gating_network = DLGN_FC_Gating_Network(
            self.nodes_in_each_layer_list, self.input_size, seed=self.seed)

    def init_value_net(self):
        self.value_network = ALLONES_FC_Value_Network(
            self.nodes_in_each_layer_list, self.input_size, seed=self.seed, num_classes=self.num_classes)

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = PCA_Layer(data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        self.initialize_network()
        return self.pca_layer.k

    def initialize_standardization_layer(self,mu=None,std=None):
        self.standardize_layer = FC_Standardize(mu,std)
        self.standardize_layer.mu = self.standardize_layer.mu.to(device=self.device,non_blocking=True)
        self.standardize_layer.std = self.standardize_layer.std.to(device=self.device,non_blocking=True)

    def forward(self, inp, verbose=2):
        if hasattr(self, 'standardize_layer'):
            inp = self.standardize_layer(inp)

        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=self.device, non_blocking=True)

        inp_gating = torch.ones(inp.size(),
                                requires_grad=True, device=self.device)

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

    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        for layer_num in range(len(self.gating_network.list_of_modules)):
          gating_net_layers_ordered["fc"+str(layer_num)+"_g"] = self.gating_network.list_of_modules[layer_num]

        return gating_net_layers_ordered

    def get_value_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        for layer_num in range(len(self.gating_network.list_of_modules)):
          gating_net_layers_ordered["fc"+str(layer_num)+"_g"] = self.value_network.list_of_modules[layer_num]

        return gating_net_layers_ordered

    def get_layer_object(self, network_type, layer_num):
        if(network_type == "GATE_NET"):
            return self.gating_network.list_of_modules[layer_num]
        elif(network_type == "WEIGHT_NET"):
            return self.value_network.list_of_modules[layer_num]
    
    def exact_forward_vis(self, x) -> torch.Tensor:
        """
        x - Dummy input with batch size =1 to generate linear transformations
        """
        self.eval()
        x = torch.flatten(x,1)
        gating_net_layers_ordered = self.get_gate_layers_ordered_dict()
        conv_matrix_operations_in_each_layer = OrderedDict()
        conv_bias_operations_in_each_layer = OrderedDict()
        channel_outs_size_in_each_layer = OrderedDict()
        current_tensor_size = x.size()
        print("current_tensor_size ", current_tensor_size)
        merged_conv_matrix = None
        merged_conv_bias = None
        orig_out = x

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
            gs_len = len(gating_signals[indx].shape)
            pout_len = len(prev_out.shape)
            
            if(gs_len > pout_len):
                prev_out = torch.unsqueeze(prev_out,-1)
            elif(gs_len < pout_len):
                gating_signals[indx] = torch.unsqueeze(gating_signals[indx],-1)
            if(verbose > 2):
                print("size:{} prev_out_{}-----:{} ".format(prev_out.size(),indx,prev_out))
            prev_out = prev_out * gating_signals[indx]
            if(verbose > 2):
                print("size:{} prev_out_{}-----:{} ".format(prev_out.size(),indx,prev_out))

        prev_out = self.output_layer(prev_out)
        if(verbose > 2):
            print("size:{} final prev_out-----:{} ".format(prev_out.size(),prev_out))
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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.nodes_in_each_layer_list = nodes_in_each_layer_list
        self.seed = seed
        self.beta = beta
        self.num_classes = num_classes
        self.input_size_list = input_size_list
        self.initialize_network()

    def initialize_network(self):
        input_size = self.input_size_list[0]
        for ind in range(1, len(self.input_size_list)):
            input_size *= self.input_size_list[ind]

        self.gating_network = DGN_FC_Gating_Network(
            self.nodes_in_each_layer_list, input_size, seed=self.seed)
        print("self.gating_network", self.gating_network)
        print("Gating net params:", sum(p.numel()
              for p in self.gating_network.parameters()))
        self.value_network = ALLONES_FC_Value_Network(
            self.nodes_in_each_layer_list, input_size, seed=self.seed, num_classes=self.num_classes)
        print("self.value_network", self.value_network)
        print("Value net params:", sum(p.numel()
              for p in self.value_network.parameters()))

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = PCA_Layer(data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        self.initialize_network()
        return self.pca_layer.k

    def initialize_standardization_layer(self,mu=None,std=None):
        self.standardize_layer = FC_Standardize(mu,std)
        self.standardize_layer.mu = self.standardize_layer.mu.to(device=self.device,non_blocking=True)
        self.standardize_layer.std = self.standardize_layer.std.to(device=self.device,non_blocking=True)

    def forward(self, inp, verbose=2):
        if hasattr(self, 'standardize_layer'):
            inp = self.standardize_layer(inp)

        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=self.device, non_blocking=True)

        inp_gating = torch.ones(inp.size(),
                                requires_grad=True, device=self.device)

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


class GALU_DNN_FC_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, input_size_list=[28, 28], seed=2022, num_classes=10):
        super(GALU_DNN_FC_Network, self).__init__()
        torch.manual_seed(seed)
        self.nodes_in_each_layer_list = nodes_in_each_layer_list
        self.seed = seed
        self.num_classes = num_classes
        self.input_size_list = input_size_list
        self.initialize_network()

    def initialize_network(self):
        list_of_modules = []
        input_size = self.input_size_list[0]
        for ind in range(1, len(self.input_size_list)):
            input_size *= self.input_size_list[ind]

        previous_layer_size = input_size
        for indx in range(len(self.nodes_in_each_layer_list)):
            each_current_layer_size = self.nodes_in_each_layer_list[indx]
            list_of_modules.append(nn.Linear(
                previous_layer_size, each_current_layer_size))
            previous_layer_size = each_current_layer_size

        self.list_of_modules = nn.ModuleList(list_of_modules)
        self.output_layer = nn.Linear(
            previous_layer_size, self.num_classes)

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = PCA_Layer(data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        self.initialize_network()
        return self.pca_layer.k

    def forward(self, inp, verbose=2):
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=device, non_blocking=True)
        prev_out = torch.flatten(inp, 1)
        num_layers = len(self.list_of_modules)
        layer_outs = []
        for indx in range(num_layers):
            each_module = self.list_of_modules[indx]
            prev_out = each_module(prev_out)
            layer_outs.append(prev_out)
            prev_out = prev_out*nn.Sigmoid()(prev_out)

        self.linear_conv_outputs = layer_outs
        self.prev_out = self.output_layer(prev_out)
        return self.prev_out

    def __str__(self):
        ret = "Value network "+" \n module_list:"
        for each_module in self.list_of_modules:
            ret += str(each_module)+" \n Params in module is:" + \
                str(sum(p.numel() for p in each_module.parameters()))+"\n"

        return ret


class DNN_FC_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, input_size_list=[28, 28], seed=2022, num_classes=10):
        super(DNN_FC_Network, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.nodes_in_each_layer_list = nodes_in_each_layer_list
        self.seed = seed
        self.num_classes = num_classes
        self.input_size_list = input_size_list
        self.initialize_network()

    def initialize_network(self):
        list_of_modules = []
        input_size = self.input_size_list[0]
        for ind in range(1, len(self.input_size_list)):
            input_size *= self.input_size_list[ind]

        previous_layer_size = input_size
        for indx in range(len(self.nodes_in_each_layer_list)):
            each_current_layer_size = self.nodes_in_each_layer_list[indx]
            list_of_modules.append(nn.Linear(
                previous_layer_size, each_current_layer_size))
            previous_layer_size = each_current_layer_size

        self.list_of_modules = nn.ModuleList(list_of_modules)
        self.output_layer = nn.Linear(
            previous_layer_size, self.num_classes)
        self.relu = nn.ReLU()

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = PCA_Layer(data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        self.initialize_network()
        return self.pca_layer.k

    def initialize_standardization_layer(self,mu=None,std=None):
        self.standardize_layer = FC_Standardize(mu,std)
        self.standardize_layer.mu = self.standardize_layer.mu.to(device=self.device,non_blocking=True)
        self.standardize_layer.std = self.standardize_layer.std.to(device=self.device,non_blocking=True)

    def forward(self, inp, verbose=2):
        if hasattr(self, 'standardize_layer'):
            inp = self.standardize_layer(inp)
        
        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=self.device, non_blocking=True)
        prev_out = torch.flatten(inp, 1)
        num_layers = len(self.list_of_modules)
        layer_outs = []
        for indx in range(num_layers):
            each_module = self.list_of_modules[indx]
            prev_out = each_module(prev_out)
            layer_outs.append(prev_out)
            prev_out = self.relu(prev_out)

        self.linear_conv_outputs = layer_outs
        self.prev_out = self.output_layer(prev_out)
        return self.prev_out

    def __str__(self):
        ret = "Value network "+" \n module_list:"
        for each_module in self.list_of_modules:
            ret += str(each_module)+" \n Params in module is:" + \
                str(sum(p.numel() for p in each_module.parameters()))+"\n"

        return ret

class BC_DNN_FC_Network(nn.Module):
    def __init__(self, nodes_in_each_layer_list, input_size_list=[28, 28], seed=2022):
        super(BC_DNN_FC_Network, self).__init__()
        torch.manual_seed(seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.nodes_in_each_layer_list = nodes_in_each_layer_list
        self.seed = seed
        self.input_size_list = input_size_list
        self.initialize_network()

    def initialize_network(self):
        list_of_modules = []
        input_size = self.input_size_list[0]
        for ind in range(1, len(self.input_size_list)):
            input_size *= self.input_size_list[ind]

        previous_layer_size = input_size
        for indx in range(len(self.nodes_in_each_layer_list)):
            each_current_layer_size = self.nodes_in_each_layer_list[indx]
            list_of_modules.append(nn.Linear(
                previous_layer_size, each_current_layer_size))
            previous_layer_size = each_current_layer_size

        self.list_of_modules = nn.ModuleList(list_of_modules)
        self.output_layer = nn.Linear(previous_layer_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def initialize_PCA_transformation(self, data, explained_var_required):
        self.pca_layer = PCA_Layer(data, explained_var_required)
        d1, d2 = determine_row_col_from_features(self.pca_layer.k)
        self.input_size_list = [d1, d2]
        self.initialize_network()
        return self.pca_layer.k

    def initialize_standardization_layer(self,mu=None,std=None):
        self.standardize_layer = FC_Standardize(mu,std)
        self.standardize_layer.mu = self.standardize_layer.mu.to(device=self.device,non_blocking=True)
        self.standardize_layer.std = self.standardize_layer.std.to(device=self.device,non_blocking=True)

    def forward(self, inp, verbose=2):
        if hasattr(self, 'standardize_layer'):
            inp = self.standardize_layer(inp)
        
        if hasattr(self, 'pca_layer'):
            inp = self.pca_layer(inp)
            inp = inp.to(device=self.device, non_blocking=True)
        prev_out = torch.flatten(inp, 1)
        num_layers = len(self.list_of_modules)
        layer_outs = []
        for indx in range(num_layers):
            each_module = self.list_of_modules[indx]
            prev_out = each_module(prev_out)
            layer_outs.append(prev_out)
            prev_out = self.relu(prev_out)

        self.linear_conv_outputs = layer_outs
        self.prev_out = self.output_layer(prev_out)
        return torch.squeeze(self.prev_out,-1)

    def __str__(self):
        ret = "Value network "+" \n module_list:"
        for each_module in self.list_of_modules:
            ret += str(each_module)+" \n Params in module is:" + \
                str(sum(p.numel() for p in each_module.parameters()))+"\n"

        return ret
    
    def get_gate_layers_ordered_dict(self):
        gating_net_layers_ordered = OrderedDict()
        for layer_num in range(len(self.list_of_modules)):
          gating_net_layers_ordered["fc"+str(layer_num)+"_g"] = self.list_of_modules[layer_num]

        return gating_net_layers_ordered

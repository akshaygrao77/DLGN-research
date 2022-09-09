import torch
import torch.nn as nn
from utils.generic_utils import calculate_output_size_for_conv
from configs.dlgn_conv_config import get_activation_function_from_key


class Basic_Network(nn.Module):
    def __init__(self):
        super(Basic_Network, self).__init__()

    def initialize_weights(self, init_type, mod_obj):
        if(init_type == "XAVIER_NORMAL"):
            nn.init.xavier_normal_(mod_obj.weight)
        elif(init_type == "XAVIER_UNIFORM"):
            nn.init.xavier_uniform_(mod_obj.weight)
        elif(init_type == "KAIMING_UNIFORM_FIN"):
            nn.init.kaiming_uniform_(mod_obj.weight, mode='fan_in')
        elif(init_type == "KAIMING_UNIFORM_FOUT"):
            nn.init.kaiming_uniform_(mod_obj.weight, mode='fan_out')
        elif(init_type == "KAIMING_NORMAL_FIN"):
            nn.init.kaiming_normal_(mod_obj.weight, mode='fan_in')
        elif(init_type == "KAIMING_NORMAL_FOUT"):
            nn.init.kaiming_normal_(mod_obj.weight, mode='fan_out')

    def initialize_layers(self, all_conv_info):
        input_size = all_conv_info.input_image_size
        self.module_list = []
        conv_count = 0
        just_count = 0
        pool_count = 0
        last_out_ch_size = None
        last_output_size = None
        last_fc_nodes = None
        for each_conv_info in all_conv_info.list_of_each_conv_info:
            just_count += 1
            if(each_conv_info.layer_type == "CONV"):
                if(each_conv_info.layer_sub_type == "2D"):
                    if(last_out_ch_size is None):
                        in_ch = each_conv_info.in_ch
                    else:
                        in_ch = last_out_ch_size

                    mod_obj = nn.Conv2d(in_ch, each_conv_info.number_of_filters, kernel_size=each_conv_info.kernel_size,
                                        padding=each_conv_info.padding, stride=each_conv_info.stride)
                    self.module_list.append(mod_obj)
                    self.initialize_weights(
                        each_conv_info.weight_init_type, mod_obj)
                    last_out_ch_size = each_conv_info.number_of_filters
                    last_output_size = calculate_output_size_for_conv(
                        input_size, each_conv_info.padding, each_conv_info.kernel_size, each_conv_info.stride)
                    input_size = last_output_size

            elif(each_conv_info.layer_type == "POOL"):
                if(each_conv_info.layer_sub_type == "GLOBAL_AVERAGE"):
                    output_size = (1, 1)
                    self.module_list.append(
                        nn.AdaptiveAvgPool2d(output_size=output_size))

                    last_output_size = output_size
                    input_size = last_output_size

            elif(each_conv_info.layer_type == "ACTIVATION"):
                each_module = get_activation_function_from_key(
                    each_conv_info.layer_sub_type)
                self.module_list.append(each_module)

            elif(each_conv_info.layer_type == "FULLY-CONNECTED"):
                if(each_conv_info.layer_sub_type == "LINEAR"):
                    if(last_fc_nodes is None):
                        inp_size = last_output_size[0] * \
                            last_output_size[1] * last_out_ch_size
                    else:
                        inp_size = last_fc_nodes

                    mod_obj = nn.Linear(
                        inp_size, each_conv_info.num_nodes_in_fc)
                    self.module_list.append(mod_obj)
                    self.initialize_weights(
                        each_conv_info.weight_init_type, mod_obj)
                    last_fc_nodes = each_conv_info.num_nodes_in_fc

        self.module_list = nn.ModuleList(self.module_list)


class DLGN_CONV_Network(nn.Module):
    def __init__(self, gate_net_conv_info, weight_net_conv_info, gating_activation_func, is_weight_net_all_ones=True, seed=2022, is_enable_weight_net_weight_restore=False, is_enable_gate_net_weight_restore=False):
        super(DLGN_CONV_Network, self).__init__()
        self.gate_net_conv_info = gate_net_conv_info
        self.weight_net_conv_info = weight_net_conv_info
        self.is_weight_net_all_ones = is_weight_net_all_ones
        self.seed = seed
        self.gating_activation_func = gating_activation_func
        self.is_enable_weight_net_weight_restore = is_enable_weight_net_weight_restore

        self.linear_conv_net = DLGN_CONV_LinearNetwork(
            gate_net_conv_info, seed, is_enable_gate_net_weight_restore)
        print("self.linear_conv_net ", self.linear_conv_net)

        self.weight_conv_net = DLGN_CONV_WeightNetwork(
            weight_net_conv_info, seed, is_enable_weight_net_weight_restore)

        print("self.weight_conv_net ", self.weight_conv_net)

    def forward(self, inp, verbose=2):
        linear_conv_outputs, _ = self.linear_conv_net(inp, verbose=verbose)
        self.linear_conv_outputs = linear_conv_outputs

        self.gating_node_outputs = [None] * len(linear_conv_outputs)
        for indx in range(len(linear_conv_outputs)):
            each_linear_conv_output = linear_conv_outputs[indx]
            self.gating_node_outputs[indx] = self.gating_activation_func(4 *
                                                                         each_linear_conv_output)
        # print("gating_node_outputs[0]", self.gating_node_outputs[0])
        if(self.is_weight_net_all_ones == True):
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            # inp = torch.ones((2,5),dtype=torch.double, requires_grad=True,device=device)
            inp = torch.ones(inp.size(),
                             requires_grad=True, device=device)

        final_layer_out = self.weight_conv_net(
            inp, self.gating_node_outputs, verbose=verbose)
        # self.final_outs = final_outs

        return final_layer_out


class DLGN_CONV_WeightNetwork(Basic_Network):
    def __init__(self, weight_net_conv_info, seed, is_enable_weight_restore=False):
        super(DLGN_CONV_WeightNetwork, self).__init__()
        self.weight_net_conv_info = weight_net_conv_info
        self.is_enable_weight_restore = is_enable_weight_restore
        self.seed = seed
        torch.manual_seed(seed)
        self.initialize_layers(self.weight_net_conv_info)

    def forward(self, inp, gating_signal, verbose=2):
        # each_mod_outputs = [None] * len(self.module_list)
        previous_output = inp
        first_fc = True
        if(verbose > 2):
            print("WeightNetwork Inp size", inp.size())
        if(verbose > 4):
            print("WeightNetwork Input:: ", inp)

        for indx in range(len(self.module_list)):
            each_module = self.module_list[indx]
            if(self.weight_net_conv_info.list_of_each_conv_info[indx].layer_type == "FULLY-CONNECTED" and first_fc):
                previous_output = torch.flatten(previous_output, 1)
                first_fc = False

            previous_output = each_module(previous_output)
            if(verbose > 2):
                print("WeightNetwork Module {} immediate output size is:: {}".format(
                    indx, previous_output.size()))
            if(verbose > 4):
                print("WeightNetwork Module {} immediate output is:: {}".format(
                    indx, previous_output))
            if(not(gating_signal is None) and len(gating_signal) > indx):
                previous_output = previous_output * gating_signal[indx]
                if(verbose > 2):
                    print("WeightNetwork Module {} after gating output size is:: {}".format(
                        indx, previous_output.size()))
                if(verbose > 4):
                    print("WeightNetwork Module {} after gating output is:: {}".format(
                        indx, previous_output))

            # each_mod_outputs[indx] = previous_output

        # self.each_mod_outputs = each_mod_outputs
        return previous_output

    def __str__(self):
        ret = "weight_net_conv_info: " + \
            str(self.weight_net_conv_info)+" \n module_list:"
        for each_module in self.module_list:
            ret += str(each_module)+" \n Params in module is:" + \
                str(sum(p.numel() for p in each_module.parameters()))+"\n"

        return ret


class DLGN_CONV_LinearNetwork(Basic_Network):
    def __init__(self, gate_net_conv_info, seed, is_enable_weight_restore=False):
        super(DLGN_CONV_LinearNetwork, self).__init__()
        self.gate_net_conv_info = gate_net_conv_info
        self.seed = seed
        self.is_enable_weight_restore = is_enable_weight_restore
        torch.manual_seed(seed)
        self.initialize_layers(self.gate_net_conv_info)

    def forward(self, inp, verbose=2):
        each_mod_outputs = [None] * len(self.module_list)
        previous_output = inp
        if(verbose > 2):
            print("LinearNetwork Inp size", inp.size())
        if(verbose > 4):
            print("LinearNetwork Input:: ", inp)

        for indx in range(len(self.module_list)):
            each_module = self.module_list[indx]
            previous_output = each_module(previous_output)
            if(verbose > 2):
                print("LinearNetwork Module {} immediate output size is:: {}".format(
                    indx, previous_output.size()))
            if(verbose > 4):
                print("LinearNetwork Module {} immediate output is:: {}".format(
                    indx, previous_output))
            each_mod_outputs[indx] = previous_output

        return each_mod_outputs, previous_output

    def __str__(self):
        ret = "gate_net_conv_info: " + \
            str(self.gate_net_conv_info)+" \n module_list:"
        for each_module in self.module_list:
            ret += str(each_module)+" \n Params in module is:" + \
                str(sum(p.numel() for p in each_module.parameters()))+"\n"
        return ret

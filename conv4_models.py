import torch
import torch.nn as nn
import numpy as np


def replace_percent_of_values(inp_np, const_value, percentage):
    inp_all_const = np.ones(inp_np.shape)*const_value
    mask_int = np.random.randint(101, size=inp_np.shape)
    mask = np.random.randint(2, size=inp_np.shape)
    mask[mask_int > percentage] = 0
    mask[mask_int <= percentage] = 1
    mask = mask.astype(np.bool)
    inp_np[mask] = inp_all_const[mask]
    return inp_np


class Plain_CONV4_Net(nn.Module):
    def __init__(self, input_channel, seed=2022):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1_g = nn.Conv2d(input_channel, 128, 3, padding=1)
        self.conv2_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_g = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_g = nn.Conv2d(128, 128, 3, padding=1)

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(128, 10)

    def forward(self, inp):

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
    def __init__(self, input_channel, seed=2022):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1_g = nn.Conv2d(input_channel, 16, 3, padding=1)
        self.conv2_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3_g = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4_g = nn.Conv2d(16, 16, 3, padding=1)

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(16, 10)

    def forward(self, inp):

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
    def __init__(self, input_channel, beta=4, seed=2022):
        super().__init__()
        torch.manual_seed(seed)
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
        self.fc1 = nn.Linear(128, 10)

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
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
    def __init__(self, input_channel, beta=4, seed=2022):
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
        self.fc1 = nn.Linear(128, 10)

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
    def __init__(self, input_channel, beta=4, seed=2022):
        super().__init__()
        self.beta = beta
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
        self.fc1 = nn.Linear(16, 10)

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
    def __init__(self, input_channel, beta=4, seed=2022):
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
        self.fc1 = nn.Linear(128, 10)

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
    def __init__(self, input_channel, random_inp_percent=4, beta=4, seed=2022):
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
        self.fc1 = nn.Linear(128, 10)

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
    def __init__(self, input_channel, random_inp_percent=4, beta=4, seed=2022):
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
        self.fc1 = nn.Linear(128, 10)

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
    def __init__(self, input_channel, beta=4, seed=2022):
        super().__init__()
        self.beta = beta
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
        self.fc1 = nn.Linear(16, 10)

    def forward(self, inp):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
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


def get_model_instance_from_dataset(dataset, model_arch_type, seed=2022):
    if(dataset == "cifar10"):
        inp_channel = 3
    elif(dataset == "mnist"):
        inp_channel = 1
    elif(dataset == "fashion_mnist"):
        inp_channel = 1

    return get_model_instance(model_arch_type, inp_channel, seed=seed)


def get_model_instance(model_arch_type, inp_channel, seed=2022):
    if(seed == ""):
        seed = 2022

    net = None
    if(model_arch_type == 'plain_pure_conv4_dnn'):
        net = Plain_CONV4_Net(inp_channel, seed=seed)
    elif(model_arch_type == 'conv4_dlgn'):
        net = Conv4_DLGN_Net(inp_channel, seed=seed)
    elif(model_arch_type == 'conv4_dlgn_n16_small'):
        net = Conv4_DLGN_Net_N16_Small(inp_channel, seed=seed)
    elif(model_arch_type == 'plain_pure_conv4_dnn_n16_small'):
        net = Plain_CONV4_Net_N16_Small(inp_channel, seed=seed)
    elif(model_arch_type == 'conv4_deep_gated_net'):
        net = Conv4_DeepGated_Net(inp_channel, seed=seed)
    elif(model_arch_type == 'conv4_deep_gated_net_n16_small'):
        net = Conv4_DeepGated_Net_N16_Small(inp_channel, seed=seed)
    elif(model_arch_type == 'conv4_deep_gated_net_with_actual_inp_in_wt_net'):
        net = Conv4_DeepGated_Net_With_Actual_Inp_Over_WeightNet(
            inp_channel, seed=seed)
    elif(model_arch_type == 'conv4_deep_gated_net_with_actual_inp_randomly_changed_in_wt_net'):
        net = Conv4_DeepGated_Net_With_Random_Actual_Inp_Over_WeightNet(
            inp_channel, seed=seed)
    elif(model_arch_type == 'conv4_deep_gated_net_with_random_ones_in_wt_net'):
        net = Conv4_DeepGated_Net_With_Random_AllOnes_Over_WeightNet(
            inp_channel, seed=seed)

    return net


def get_model_save_path(model_arch_type, dataset, seed=""):
    if(seed == ""):
        torch_seed_str = ""
    else:
        torch_seed_str = "/ST_"+str(seed)+"/"
    final_model_save_path = "root/model/save/" + \
        str(dataset)+"/CLEAN_TRAINING/" + \
        str(torch_seed_str)+str(model_arch_type)+"_dir.pt"

    return final_model_save_path

import torch
import torch.nn as nn


class Plain_CONV4_Net(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
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

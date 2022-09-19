import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class vgg19(nn.Module):
    def __init__(self, allones, init_weights: bool = True) -> None:
        super().__init__()
        self.conv1_g = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2_g = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_g = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4_g = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5_g = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6_g = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7_g = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8_g = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv9_g = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv10_g = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv11_g = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12_g = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv13_g = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv1_v = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2_v = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_v = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4_v = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5_v = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6_v = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7_v = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8_v = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv9_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv10_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv11_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv13_v = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.bn1_g = nn.BatchNorm2d(64)
        self.bn2_g = nn.BatchNorm2d(64)
        self.bn3_g = nn.BatchNorm2d(128)
        self.bn4_g = nn.BatchNorm2d(128)

        self.bn5_g = nn.BatchNorm2d(256)
        self.bn6_g = nn.BatchNorm2d(256)
        self.bn7_g = nn.BatchNorm2d(256)

        self.bn8_g = nn.BatchNorm2d(512)
        self.bn9_g = nn.BatchNorm2d(512)
        self.bn10_g = nn.BatchNorm2d(512)

        self.bn11_g = nn.BatchNorm2d(512)
        self.bn12_g = nn.BatchNorm2d(512)
        self.bn13_g = nn.BatchNorm2d(512)

        self.bn1_v = nn.BatchNorm2d(64)
        self.bn2_v = nn.BatchNorm2d(64)
        self.bn3_v = nn.BatchNorm2d(128)
        self.bn4_v = nn.BatchNorm2d(128)

        self.bn5_v = nn.BatchNorm2d(256)
        self.bn6_v = nn.BatchNorm2d(256)
        self.bn7_v = nn.BatchNorm2d(256)

        self.bn8_v = nn.BatchNorm2d(512)
        self.bn9_v = nn.BatchNorm2d(512)
        self.bn10_v = nn.BatchNorm2d(512)

        self.bn11_v = nn.BatchNorm2d(512)
        self.bn12_v = nn.BatchNorm2d(512)
        self.bn13_v = nn.BatchNorm2d(512)

        self.fc1_v = nn.Linear(512, 10)

        self.sig = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        # (2*(torch.randint(low=0, high=2, size=(1, 3,32,32))-0.5))
        self.allones = allones
        # torch.ones(1, 3, 32, 32).half().cuda()
        self.allones_half = self.allones.half().cuda()
        # torch.ones(1, 3, 32, 32).half().cuda()
        self.allones_float = self.allones.cuda()
        #self.allones = 2*(torch.randint(low=0, high=2, size=(1, 3,32,32)).to('torch.half').cuda()-0.5)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature/Gating Network
        # Layer 1 : 64
        conv_g_outs = []

        x_g = self.conv1_g(x)
        x_g = self.bn1_g(x_g)
        conv_g_outs.append(x_g)
        g_1 = self.sig(10*x_g)
        #x_g = F.relu(x_g)

        # Layer 2: 64
        x_g = self.conv2_g(x_g)
        x_g = self.bn2_g(x_g)
        conv_g_outs.append(x_g)
        g_2 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 1
        x_g = self.avgpool(x_g)

        # Layer 3 : 128
        x_g = self.conv3_g(x_g)
        x_g = self.bn3_g(x_g)
        conv_g_outs.append(x_g)
        g_3 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 4 : 128
        x_g = self.conv4_g(x_g)
        x_g = self.bn4_g(x_g)
        conv_g_outs.append(x_g)
        g_4 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 2
        x_g = self.avgpool(x_g)

        # Layer 5 : 256
        x_g = self.conv5_g(x_g)
        x_g = self.bn5_g(x_g)
        conv_g_outs.append(x_g)
        g_5 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 6 : 256
        x_g = self.conv6_g(x_g)
        x_g = self.bn6_g(x_g)
        conv_g_outs.append(x_g)
        g_6 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 7 : 256
        x_g = self.conv7_g(x_g)
        x_g = self.bn7_g(x_g)
        conv_g_outs.append(x_g)
        g_7 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 3
        x_g = self.avgpool(x_g)

        # Layer 8 : 512
        x_g = self.conv8_g(x_g)
        x_g = self.bn8_g(x_g)
        conv_g_outs.append(x_g)
        g_8 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 9 : 512
        x_g = self.conv9_g(x_g)
        x_g = self.bn9_g(x_g)
        conv_g_outs.append(x_g)
        g_9 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 10 : 512
        x_g = self.conv10_g(x_g)
        x_g = self.bn10_g(x_g)
        conv_g_outs.append(x_g)
        g_10 = self.sig(10*x_g)
        #x = F.relu(x)

        # Max-Pool : 4
        x_g = self.avgpool(x_g)

        # Layer 11 : 512
        x_g = self.conv11_g(x_g)
        x_g = self.bn11_g(x_g)
        conv_g_outs.append(x_g)
        g_11 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 12 : 512
        x_g = self.conv12_g(x_g)
        x_g = self.bn12_g(x_g)
        conv_g_outs.append(x_g)
        g_12 = self.sig(10*x_g)
        #x = F.relu(x)

        # Layer 13 : 512
        x_g = self.conv13_g(x_g)
        x_g = self.bn13_g(x_g)
        conv_g_outs.append(x_g)
        g_13 = self.sig(10*x_g)
        #x = F.relu(x)

        self.linear_conv_outputs = conv_g_outs

        # Value Network

        # Layer 1 : 64
        x_v = self.conv1_v(self.allones.cuda())
        x_v = self.bn1_v(x_v)
        x_v = x_v*g_1
        #x_v = F.relu(x_v)

        # Layer 2: 64
        x_v = self.conv2_v(x_v)
        x_v = self.bn2_v(x_v)
        x_v = x_v*g_2
        #x = F.relu(x)

        # Max-Pool : 1
        x_v = self.avgpool(x_v)

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
        x_v = self.avgpool(x_v)

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

        # Max-Pool : 3
        x_v = self.avgpool(x_v)

        # Layer 8 : 512
        x_v = self.conv8_v(x_v)
        x_v = self.bn8_v(x_v)
        x_v = x_v*g_8
        #x = F.relu(x)

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
        # Max-Pool : 4
        x_v = self.avgpool(x_v)

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

        # Layer 13 : 512
        x_v = self.conv13_v(x_v)
        x_v = self.bn13_v(x_v)
        x_v = x_v*g_13
        #x = F.relu(x)

        # Max-Pool : 5
        x_v = self.avgpool(x_v)

        x_v = self.globalpool(x_v)
        x_v = torch.flatten(x_v, 1)
        x_v = self.fc1_v(x_v)
        #x_v = x_v*g_17
        #x = F.relu(x)
        #x_v = self.dropout(x_v)
        #x_v = self.fc2_v(x_v)
        #x_v = x_v*g_18
        #x = F.relu(x)
        #x_v = self.dropout(x_v)
        #x_v = self.fc3_v(x_v)

        return x_v

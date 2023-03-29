import warnings
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Callable, Any

__all__ = ['GoogLeNet', 'googlenet', "GoogLeNetOutputs", "_GoogLeNetOutputs"]

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

GoogLeNetOutputs = namedtuple(
    'GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


def custom_googlenet(arch_mode: str = True, **kwargs: Any) -> "Custom_GoogLeNet":
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    The required minimum input size of the model is 15x15.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """

    return Custom_GoogLeNet(arch_mode, **kwargs)


class Custom_GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(
        self,
        arch_mode: str,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None
    ) -> None:
        super(Custom_GoogLeNet, self).__init__()
        self.arch_mode = arch_mode
        self.is_apply_relu = False
        if(arch_mode == "dgn"):
            self.is_apply_relu = True
        if blocks is None:
            blocks = [Custom_BasicConv2d,
                      Custom_Inception, Custom_InceptionAux]
        if init_weights is None:
            warnings.warn('The default weight initialization of GoogleNet will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(self.is_apply_relu, 3,
                                64, kernel_size=7, stride=2, padding=3)
        self.g_avgpool1 = nn.AvgPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(self.is_apply_relu, 64, 64, kernel_size=1)
        self.conv3 = conv_block(self.is_apply_relu, 64,
                                192, kernel_size=3, padding=1)
        self.g_avgpool2 = nn.AvgPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(
            self.is_apply_relu, 192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(
            self.is_apply_relu, 256, 128, 128, 192, 32, 96, 64)
        self.g_avgpool3 = nn.AvgPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(
            self.is_apply_relu, 480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(
            self.is_apply_relu, 512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(
            self.is_apply_relu, 512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(
            self.is_apply_relu, 512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(
            self.is_apply_relu, 528, 256, 160, 320, 32, 128, 128)
        self.g_avgpool4 = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(
            self.is_apply_relu, 832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(
            self.is_apply_relu, 832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(
                self.is_apply_relu, 512, num_classes)
            self.aux2 = inception_aux_block(
                self.is_apply_relu, 528, num_classes)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.w_avgpool1 = nn.AvgPool2d(3, stride=2, ceil_mode=True)
        self.w_avgpool2 = nn.AvgPool2d(3, stride=2, ceil_mode=True)
        self.w_avgpool3 = nn.AvgPool2d(3, stride=2, ceil_mode=True)
        self.w_avgpool4 = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(
                    m.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * \
                (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * \
                (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * \
                (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, inp: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        allones = torch.ones(inp.size(), requires_grad=True,
                             device=inp.device)
        # N x 3 x 224 x 224
        x_g, x_w = self.conv1(inp, allones)

        # N x 64 x 112 x 112
        x_g = self.g_avgpool1(x_g)
        x_w = self.w_avgpool1(x_w)

        # N x 64 x 56 x 56
        x_g, x_w = self.conv2(x_g, x_w)
        # N x 64 x 56 x 56
        x_g, x_w = self.conv3(x_g, x_w)
        # N x 192 x 56 x 56
        x_g = self.g_avgpool2(x_g)
        x_w = self.w_avgpool2(x_w)

        # N x 192 x 28 x 28
        x_g, x_w = self.inception3a(x_g, x_w)
        # N x 256 x 28 x 28
        x_g, x_w = self.inception3b(x_g, x_w)

        # N x 480 x 28 x 28
        x_g = self.g_avgpool3(x_g)
        x_w = self.w_avgpool3(x_w)

        # N x 480 x 14 x 14
        x_g, x_w = self.inception4a(x_g, x_w)
        # N x 512 x 14 x 14
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x_g, x_w)

        x_g, x_w = self.inception4b(x_g, x_w)
        # N x 512 x 14 x 14
        x_g, x_w = self.inception4c(x_g, x_w)
        # N x 512 x 14 x 14
        x_g, x_w = self.inception4d(x_g, x_w)
        # N x 528 x 14 x 14
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x_g, x_w)

        x_g, x_w = self.inception4e(x_g, x_w)
        # N x 832 x 14 x 14
        x_g = self.g_avgpool4(x_g)
        x_w = self.w_avgpool4(x_w)
        # N x 832 x 7 x 7
        x_g, x_w = self.inception5a(x_g, x_w)
        # N x 832 x 7 x 7
        x_g, x_w = self.inception5b(x_g, x_w)
        # N x 1024 x 7 x 7

        x_w = self.avgpool(x_w)
        # N x 1024 x 1 x 1
        x_w = torch.flatten(x_w, 1)
        # N x 1024
        x_w = self.dropout(x_w)
        x_w = self.fc(x_w)
        # N x 1000 (num_classes)
        return x_w, aux2, aux1

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs:
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x   # type: ignore[return-value]

    def forward(self, x: Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn(
                    "Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)


class Custom_Inception(nn.Module):

    def __init__(
        self,
        is_apply_relu,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Custom_Inception, self).__init__()
        if conv_block is None:
            conv_block = Custom_BasicConv2d
        self.is_apply_relu = is_apply_relu

        self.branch1 = conv_block(
            self.is_apply_relu, in_channels, ch1x1, kernel_size=1)

        self.branch2_conv1 = conv_block(
            self.is_apply_relu, in_channels, ch3x3red, kernel_size=1)
        self.branch2_conv2 = conv_block(self.is_apply_relu,
                                        ch3x3red, ch3x3, kernel_size=3, padding=1)

        self.branch3_conv1 = conv_block(
            self.is_apply_relu, in_channels, ch5x5red, kernel_size=1)
        self.branch3_conv2 = conv_block(self.is_apply_relu,
                                        ch5x5red, ch5x5, kernel_size=3, padding=1)

        self.g_branch4_avg = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1, ceil_mode=True)
        self.w_branch4_avg = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1, ceil_mode=True)

        self.branch4_conv = conv_block(
            self.is_apply_relu, in_channels, pool_proj, kernel_size=1)

    def _forward(self, x_g: Tensor, x_w: Tensor) -> List[Tensor]:
        g_branch1, w_branch1 = self.branch1(x_g, x_w)

        g_branch2, w_branch2 = self.branch2_conv1(x_g, x_w)
        g_branch2, w_branch2 = self.branch2_conv2(g_branch2, w_branch2)

        g_branch3, w_branch3 = self.branch3_conv1(x_g, x_w)
        g_branch3, w_branch3 = self.branch3_conv2(g_branch3, w_branch3)

        g_branch4 = self.g_branch4_avg(x_g)
        w_branch4 = self.w_branch4_avg(x_w)

        g_branch4, w_branch4 = self.branch4_conv(g_branch4, w_branch4)

        g_outputs = [g_branch1, g_branch2, g_branch3, g_branch4]
        w_outputs = [w_branch1, w_branch2, w_branch3, w_branch4]
        return g_outputs, w_outputs

    def forward(self, x_g: Tensor, x_w: Tensor) -> Tensor:
        g_outputs, w_outputs2 = self._forward(x_g, x_w)
        return torch.cat(g_outputs, 1), torch.cat(w_outputs2, 1)


class Custom_InceptionAux(nn.Module):

    def __init__(
        self,
        is_apply_relu: bool,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Custom_InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = Custom_BasicConv2d
        self.is_apply_relu = is_apply_relu
        self.conv = conv_block(
            self.is_apply_relu, in_channels, 128, kernel_size=1)

        self.g_fc1 = nn.Linear(2048, 1024)
        self.w_fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x_g: Tensor, x_w: Tensor) -> Tensor:
        beta = 10
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x_g = F.adaptive_avg_pool2d(x_g, (4, 4))
        x_w = F.adaptive_avg_pool2d(x_w, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x_g, x_w = self.conv(x_g, x_w)
        # N x 128 x 4 x 4
        x_g = torch.flatten(x_g, 1)
        x_w = torch.flatten(x_w, 1)
        # N x 2048
        x_g = self.g_fc1(x_g)
        x_w = self.w_fc1(x_w)
        x_w = x_w * torch.sigmoid(beta * x_g)
        if(self.is_apply_relu):
            x_g = F.relu(x_g)
        # N x 1024
        x_w = F.dropout(x_w, 0.7, training=self.training)
        # N x 1024
        x_w = self.fc2(x_w)
        # N x 1000 (num_classes)

        return x_w


class Custom_BasicConv2d(nn.Module):

    def __init__(
        self,
        is_apply_relu,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(Custom_BasicConv2d, self).__init__()
        self.is_apply_relu = is_apply_relu
        self.g_conv = nn.Conv2d(
            in_channels, out_channels, bias=False, **kwargs)
        self.w_conv = nn.Conv2d(
            in_channels, out_channels, bias=False, **kwargs)
        self.g_bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.w_bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x_g: Tensor, x_w: Tensor) -> Tensor:
        beta = 10

        x_g = self.g_conv(x_g)
        x_g = self.g_bn(x_g)
        x_w = self.w_conv(x_w)
        x_w = self.w_bn(x_w)
        x_w = x_w * torch.sigmoid(beta * x_g)
        if(self.is_apply_relu):
            x_g = F.relu(x_g)
        return x_g, x_w

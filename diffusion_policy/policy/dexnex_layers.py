import math
import torch
import torch.nn as nn

import robomimic.models.base_nets as rmbn
from torchvision import models as vision_models

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

from torch import Tensor

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

# copy/pasted from here because it's not exported: https://github.com/pytorch/vision/blob/d84aa8936db75d73a6b8085316f4e2a802fe0b99/torchvision/models/resnet.py#L59
def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, padding: int = 1, dilation: int = 1) -> nn.Conv2d:
    """5x5 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
    
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class WideBlock(BasicBlock):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(inplanes, planes, stride=stride, downsample=downsample, groups=groups, base_width=base_width, dilation=dilation, norm_layer=norm_layer)
        
        self.conv1 = conv5x5(inplanes, planes, stride, padding=2)
        self.conv2 = conv5x5(planes, planes, padding=2)

class WideResNet(vision_models.ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, WideBlock]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer)
        
        # compat
        self.inplanes = 128
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        # replace first layers with more channels
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        
        # replace layer1 with a lot more channels
        self.layer1 = self._make_layer(block, 128, layers[0])
        
        # must remake layer2 as well for layer1 compat
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    

# custom module, based off robomimic.models.base_nets::ResNet18Conv
class ResNetSlice(rmbn.ConvBase):
    def __init__(
        self,
        input_channel=3,
        input_coord_conv=False,
    ):
        super(ResNetSlice, self).__init__()
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        net = WideResNet(WideBlock, [16, 1, 1, 1])
        # net = vision_models.resnet18()

        if input_coord_conv:
            net.conv1 = CoordConv2d(
                    input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(
                    input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer, and last two convs
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        
        # resnet layers. 0:5 will keep only the layer1's. 0:6 will keep the layer2's too
        resnet_layers = list(net.children())[0:5]
        # resnet_layers = list(net.children())[0:6]
        
        # replace the first conv2d to be compat with 128 channels
        
        self.nets = torch.nn.Sequential(*resnet_layers)
        pass

    def output_shape(self, input_shape):
        assert (len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 4.))
        out_w = int(math.ceil(input_shape[2] / 4.))
        # out_h = int(math.ceil(input_shape[1] / 8.))
        # out_w = int(math.ceil(input_shape[2] / 8.))

        # quick dirty replace 512 with 128
        return [128, out_h, out_w]
        # return [128, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)

class ImageModule(rmbn.Module):
    def __init__(self):
        super().__init__()

        self.nets = nn.Sequential(
            ResNetSlice(),
            nn.Conv2d(128, 3, kernel_size=1),
            # nn.Conv2d(64, 16, kernel_size=1),
            # nn.Conv2d(16, 4, kernel_size=1),
            # nn.Conv2d(4, 1, kernel_size=1),
            nn.Flatten(),
            # nn.Linear(1728, 128)
        )
        
        print("ResNet params: %e" % sum(p.numel() for p in self.nets[0].parameters()))
        
        pass

    def output_shape(self, input_shape=None):
        # (crop size / 4)^2
        # return [2116]
        # return [1728]
        return [6348] # 3 * 46 * 46

    def forward(self, inputs):
        return self.nets(inputs)
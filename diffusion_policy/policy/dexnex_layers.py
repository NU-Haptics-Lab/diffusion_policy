import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.base_nets as rmbn
from torchvision import models as vision_models

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

from torch import Tensor

from clip import model as clip_model

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
        channels=128,
    ) -> None:
        super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer)
        
        # compat
        self.inplanes = channels
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        # replace first layers with more channels
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        
        # replace layer1 with a lot more channels
        self.layer1 = self._make_layer(block, channels, layers[0])
        
        # must remake layer2 as well for layer1 compat
        self.layer2 = self._make_layer(block, channels, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    

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
    
class ViT(clip_model.VisionTransformer):
    """
    Same as theirs, but allows nb input channels to be variable
    """
    def __init__(self, in_channels, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__(input_resolution, patch_size, width, layers, heads, output_dim)
        
        # overwrite conv1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        # reset proj to none
        self.proj = None
        self.grid = input_resolution // patch_size
        
        scale = width ** -0.5
        
        # remove the class embedding from the positional embedding
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid ** 2, width))
        
    # copy paste the forward function, remove the class embedding since we don't have classes, and keep the entire output in the ln_post
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj
            
        # revert the 1st permutation
        x = x.permute(0, 2, 1)  # shape = [*, width, grid ** 2]
        
        # revert the reshape
        x = x.reshape(x.shape[0], x.shape[1], self.grid, self.grid)  # shape = [*, width, grid, grid]
        
        return x
    
class SpatialSoftmaxWithValue(rmbn.SpatialSoftmax):
    """
    robomimic Spatial Softmax returns the xy and variance but not the expected feature values, which I think would be useful. Obviously don't normalize the input features otherwise the feature value would be meaningless.
    
    I think it's useful because just expected XY doesn't tell me how strongly the layer thinks there's something interesting at that XY. The expected value would indicate that strength.
    """
    def __init__(self, input_shape, num_kp=None, temperature=1, learnable_temperature=False, output_variance=False, noise_std=0):
        super().__init__(input_shape, num_kp, temperature, learnable_temperature, output_variance, noise_std)
        
        self.output_feature_value = True
        
    def forward(self, feature):
        
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise
            
        if self.output_feature_value:
            expected_value = torch.sum(feature * attention, dim=1, keepdim=True)
            
            # reshape to [B, K, 1]
            expected_value = expected_value.reshape([-1, self._num_kp, 1])
            
            # output [B, K, 3]
            feature_keypoints = torch.cat([feature_keypoints, expected_value], 2)

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints
        
class CascadingCNNSpatialSoftmax(rmbn.ConvBase):
    """
    Similar to ResNet, but bleeds off spatial softmax arrays at each layer
    """
    def __init__(self):
        super().__init__()
        
        # params
        self.nb_keypoints = 32
        input_shape = 184
        
        # make the resnet
        resnet = vision_models.resnet18()
        
        children = list(resnet.children())
        self.resnet_layers = nn.ModuleList([
            nn.Sequential(*children[0:3]), # conv1, bn1, relu
            nn.Sequential(*children[3:5]), # maxpool, layer1
            nn.Sequential(*children[5:6]), # layer2
            nn.Sequential(*children[6:7]), # layer3
            nn.Sequential(*children[7:8]), # layer4
        ])
        resnet_channels = [64, 64, 128, 256, 512]
        shapes = [92, 46, 23, 12, 6]
        
        spatial_softmaxs = []
        
        for idx in range(len(self.resnet_layers)):
            
            spatial_softmaxs.append(SpatialSoftmaxWithValue((resnet_channels[idx], shapes[idx], shapes[idx]), num_kp=self.nb_keypoints))
            
        self.spatial_softmaxs = nn.ModuleList(spatial_softmaxs)
            
        pass
        
        
    def forward(self, inputs):
        # setup
        x = inputs
        z = []
        
        # iterate over the resnet layers
        for count, net in enumerate(self.resnet_layers):
            # calculate the output
            x = net(x)
            
            # calculate the spatial softmax
            y = self.spatial_softmaxs[count](x)
            
            # append to outputs
            z.append(y)
            
        # output is [B, total K, 3]
        z = torch.cat(z, dim=1)
        
        return z
    
    def output_shape(self, input_shape):
        return [len(self.resnet_layers) * self.nb_keypoints, 3]
            
            
    
class HybridCNNViT(rmbn.ConvBase):
    """
    Hybrid CNN + ViT
    using CLIP
    """
    def __init__(self):
        super().__init__()
        
        # from the CLIP paper: https://arxiv.org/pdf/2103.00020 
        heads = 12
        layers = 12
        width = 504 # i.e. nb "channels", must be divisible by nb heads. The input "image" will be upscaled in nb of channels to the width value using a Conv2d. Width slows down training less than sequence length
        
        #
        
        # patch size of 1 means we don't lose any spatial resolution
        patch_size = 1
        
        # make a wide ResNet
        resnet = WideResNet(BasicBlock, [8, 1, 1, 1], channels=feature_channels)
        
        # only keep layer 1's
        resnet_layers = list(resnet.children())[0:5]
        
        # calculated params
        input_res = cnn.output_shape(None)[0]
        feature_res = input_res // 4 # // 4 comes from the first couple layers of resnet
        # feature_channels = 384 # from WideResNet.inplanes
        feature_channels = cnn.output_shape(None)[1]
        
        # make the ViT
        if layers > 0:
            vit = ViT(feature_channels, feature_res, patch_size, width, layers, heads, width)
        else:
            vit = nn.Identity()
            
        if layers > 0:
            final_conv_channels = width
        else:
            final_conv_channels = feature_channels
        
        # assemble
        self.nets = nn.Sequential(
            *resnet_layers,
            vit,
            nn.Conv2d(final_conv_channels, 4, kernel_size=1), # reduce nb channels to something manageable
            nn.Flatten(), # final flatten so the output can be fed into the 1d-UNet
        )
        pass
        
        
    def forward(self, inputs):
        return self.nets(inputs)
    
    def output_shape(self, input_shape):
        # output is same shape as the input to the transformer
        return [8464] # 4x160
    
## from clip, with minor changes
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        
        # change: use our custom ResidualAttentionBlock
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = clip_model.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", clip_model.QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = clip_model.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # change: I want to do layer norm before and after the permutation, because if not, then it's actually batch norm, because the sequence dimension and the batch dimension get swapped, and batch norm can lead to unstable training with small batch sizes, which is what we have
        x = self.ln_1(x)
        
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
        # must re-arrange so that sequence dimension is 1st, batch dimension is 2nd, embed (aka feature) dimension is 3rd
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = x + self.attention(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = x + self.mlp(self.ln_2(x))
        return x
    
## end from clip
    
class CNNSpatialSoftmaxTransformer(rmbn.ConvBase):
    def __init__(self):
        super().__init__()
        
        # from the CLIP paper: https://arxiv.org/pdf/2103.00020 
        heads = 1
        layers = 12
        width = 3 # i.e. nb "channels", must be divisible by nb heads. The input "image" will be upscaled in nb of channels to the width value using a Conv2d. Width slows down training less than sequence length
        
        # make the spatial cnn
        self.spatial_cnn = CascadingCNNSpatialSoftmax()
        
        # make the transformer, our custom one
        self.transformer = Transformer(width, layers, heads)
        
        self.fl = nn.Flatten()
        
    def forward(self, inputs):
        # get the spatial features
        x = self.spatial_cnn(inputs) # output: [B, K, 3]
        
        x = x + self.transformer(x) # residual style
        
        # flatten
        x = self.fl(x)
        
        return x
    
    def output_shape(self, input_shape):
        return [480] # 3 * 160 from the spatial_cnn
    


def test():
    m = CNNSpatialSoftmaxTransformer()
    print("nb params: %e" % sum(p.numel() for p in m.parameters()))
    # m = CascadingCNNSpatialSoftmax()
    m(torch.zeros(1,3,184,184))
    
    pass


if __name__ == "__main__":
    test()
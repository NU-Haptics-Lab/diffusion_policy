




"""
just like the Unet, but the shape is more like a slash, aka the first half of the unet. So, only down_dims, no up_dims. The output is a single value.

Used for critics and the like.
"""













from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalResidualBlock1D

logger = logging.getLogger(__name__)


class ConditionalSlashnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        num_mid_module_repeats = 4,
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        cond_dim = 0
            
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        
        # TODO: put in config
        num_repeats = num_mid_module_repeats

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ) for i in range(num_repeats)
            ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ] + [nn.ModuleList([ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale) for i in range(num_repeats)])] + 
                [Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            
        # final dense layers
        horizon = 8 # should come from the config...
        nb_input_features = all_dims[-1] * (horizon / 2**2)
        nb_input_features = int(nb_input_features)
        self.final_layers = nn.Sequential(
            nn.Linear(nb_input_features, 1),
        )

        self.local_cond_encoder = local_cond_encoder
        self.down_modules = down_modules

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            task_ids = None,
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # assertions
        
        
        sample = einops.rearrange(sample, 'b h t -> b t h')
        

        if global_cond is not None:
            global_feature = global_cond
        else:
            global_feature = None
            
        # encode local features
        h_local = list()
        if local_cond is not None:
            assert(self.local_cond_encoder is not None)
            
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2s, downsample) in enumerate(self.down_modules): #type:ignore
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
                
            for resnet2 in resnet2s:
                x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # final dense layers
        x = x.flatten(1) # flatten all but batch dimension
        
        x = self.final_layers(x)

        # doesn't play well with vmap(grad())
        # assert(not x.isnan().any())
        return x

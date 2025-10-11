from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

import torchsummary
from torchvision import models as vision_models

import diffusion_policy.model.components.dexnex_layers as dexnex_layers
from diffusion_policy.model.obs_encoder import ObsEncoderMaker

import diffusion_policy.globals as globals
from diffusion_policy.utils import print_nb_params


class DiffusionModel(BaseImagePolicy):
    def __init__(self, 
            action_shape: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder_maker: ObsEncoderMaker,
            n_obs_steps, # input time-length
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        
        # save from global config
        self.action_rel_indices = globals.CONFIG.action_rel_indices

        # parse shape_meta
        assert len(action_shape) == 1
        action_dim = action_shape[0]
            
        # get the obs encoder object
        self.obs_encoder = obs_encoder_maker.get()
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=self.obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: 
                    nn.GroupNorm(
                    num_groups=x.num_features//16,
                    num_channels=x.num_features)
            )
        
        if eval_fixed_crop:
            replace_submodules(
                root_module=self.obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )
            
        # print 

        # create diffusion model. Get first/only element from list
        obs_feature_dim = self.obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.horizon = len(self.action_rel_indices)
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        print_nb_params(self.model, "Diffusion params")
        print_nb_params(self.obs_encoder, "Vision params")
        
        ### TEST
        self.random_noise = None
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, 
            condition_mask,
            scheduler,
            local_cond=None, 
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        
        if False: # self.random_noise is None:
            self.random_noise = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        
        

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, nobs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.predict_action_impl(
            nobs_dict,
            self.noise_scheduler,
            )
        
    def denoise(self, 
            nobs_dict,
            noise_scheduler,
            ):
        """ alias for predict_action_impl """
        return self.predict_action_impl(nobs_dict, noise_scheduler)

    def predict_action_impl(self, 
            nobs_dict: Dict[str, torch.Tensor],
            noise_scheduler,
            ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        nobs = nobs_dict
        value = next(iter(nobs.values()))
        B = value.shape[0] # batch
        T = self.horizon # trajectory length
        Da = self.action_dim # action dimension
        Do = self.obs_feature_dim # output length of the obs encoder
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype
        
        # DEBUGGING
        if False:
            # https://stackoverflow.com/questions/52796121/how-to-get-the-output-from-a-specific-layer-from-a-pytorch-model
            spatial_softmax_output = None
            
            def hook(m, i, o):
                nonlocal spatial_softmax_output
                spatial_softmax_output = o
                
            self.obs_encoder.obs_nets['image'].nets[1].register_forward_hook(hook)

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            
            # # I'm not sure why this line was included...
            # this_nobs = dict_apply(nobs, lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))
            this_nobs = nobs
            
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,-To:,Da:] = nobs_features
            cond_mask[:,-To:,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            noise_scheduler,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize elsewhere
        naction_pred = nsample[...,:Da]

        # get action
        start = np.argmax(np.array(self.action_rel_indices) >= 0)
        
        naction = naction_pred[:, start:]
        
        nresult = {
            'naction': naction,
            'naction_pred': naction_pred
        }
        
        # DEBUGGING
        
            
        # if printing spatial softmax output
        if False:
            img_xy = spatial_softmax_output[0].cpu()
            x = img_xy[:,0]
            y = img_xy[:,1]
            
            # # unnormalize for plotting
            # img = obs['image']
            
            import matplotlib.pyplot as plt
            
            plt.plot(x, y, '*')
            
        return nresult
    
    def loss(self, nbatch, task_id):
        loss2 = self.compute_loss(nbatch)
        
        # logging
        dd = {task_id + ": bc_actor_loss": loss2}
        globals.LOGGER.log(dd)
        
        return loss2
            
    def get_val_action_mse_error(self, nbatch):
        # ground truth action
        nobs = nbatch['obs']
        gt_action = nbatch['action']
        
        # denoise
        nresult = self.predict_action(nobs)
        
        # extract the predicted action
        pred_action = nresult['naction_pred']
        
        # calc mse
        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        
        # move to cpu
        action_mse_error = mse.item()
        
        return action_mse_error

    # ========= training  ============
    def compute_loss(self, nbatch):
        # normalize input
        assert 'valid_mask' not in nbatch
        
        # for cotraining, we normalize when we construct the batch
        nobs = nbatch['obs']
        nactions = nbatch['action']
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

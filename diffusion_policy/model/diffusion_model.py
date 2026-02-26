from typing import Dict
from typing import Union


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

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
from diffusion_policy.common import pytorch_util
from diffusion_policy import utils

import torchsummary
from torchvision import models as vision_models

import diffusion_policy.model.components.dexnex_layers as dexnex_layers
from diffusion_policy.model.obs_encoder import ObsEncoderMaker

import diffusion_policy.globals as globals
from diffusion_policy.utils import print_nb_params

from diffusion_policy.model.components.tree import Tree

import logging
logger = logging.getLogger(__name__)

class Leaf(nn.Module):
    def __init__(self, nb_action, horizon):
        super().__init__()
        
        self.model = rmbn.MLP(
            input_dim = nb_action * horizon,
            output_dim = nb_action * horizon,   
        )
        
        print_nb_params(self.model, "Leaf params")
        
    def forward(self, inputs: torch.Tensor):
        sh = inputs.shape
        
        # flatten, skip the batch axis
        x = torch.flatten(inputs, start_dim=1)
        
        self.model(x)
        
        out = x.reshape(sh)
        return out
    
def ForNoiseStep(noise_scheduler,
               pred,
               timesteps,
               noisy_trajectory,
               ):
    
    out = []
    for i in range(pred.shape[0]):
        f = noise_scheduler.step(
                pred[i], timesteps[i], noisy_trajectory[i]
                ).pred_original_sample
        f2 = torch.unsqueeze(f, 0)
        out.append(f2)
        
    outt = torch.concat(out)
    return outt

def VectorNoiseStep(noise_scheduler,
               pred,
               timesteps,
               noisy_trajectory,
               ):
    """
    doesn't work with vmap
    """
    def func(p, t, n):
        a = noise_scheduler.step(
                p, t, n
                ).pred_original_sample
        return a
        
    batched_func = torch.vmap(func)  
    out = batched_func(pred, timesteps, noisy_trajectory)
    return out

class SimpleModel(nn.Module):
    def __init__(self, input_dim, action_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = rmbn.MLP(
            input_dim = input_dim,
            output_dim = action_dim,   
            layer_dims = [64, 64, 64]
        )
        
    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        
        # flatten the traj, keep the batch dim
        s = torch.reshape(sample, [sample.shape[0], -1])
        
        if isinstance(timestep, torch.Tensor):
            # this happens during eval, and timestep is on cpu
            if len(timestep.shape) == 0:
                ts = torch.unsqueeze(timestep, dim=0)
                ts = torch.tile(torch.Tensor(timestep), [sample.shape[0], 1])
                ts = ts.to(globals.CONFIG.device)
            else:
                ts = torch.reshape(timestep, [sample.shape[0], 1])
        else:
            # tile it
            ts = torch.tile(torch.Tensor(timestep), [sample.shape[0], 1])
        
        # concat them all along the not-batch dim
        x = torch.concat([s, ts], dim=1)
        
        if local_cond is not None:
            x = torch.concat([x, local_cond], dim=1)
            
        if global_cond is not None:
            x = torch.concat([x, global_cond], dim=1)
            
        # model
        x = self.model(x)
        
        # reshape
        x = torch.reshape(x, sample.shape)
        
        return x
            

class DiffusionModel(BaseImagePolicy):
    def __init__(self, 
            action_shape: dict,
            noise_scheduler: DDPMScheduler | DDIMScheduler,
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
            action_relative_to_state = False,
            use_tree = False,
            use_simple_model = False,
            num_mid_module_repeats = 4,
            # parameters passed to step
            **kwargs):
        super().__init__()
        self.action_shape = action_shape
        self.obs_encoder_maker = obs_encoder_maker
        self.obs_encoder_group_norm = obs_encoder_group_norm
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.down_dims = down_dims
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.cond_predict_scale = cond_predict_scale
        self.eval_fixed_crop = eval_fixed_crop
        
        self.action_relative_to_state = action_relative_to_state
        self.use_tree = use_tree
        self.use_simple_model = use_simple_model
        
        self.noise_scheduler = noise_scheduler
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.num_mid_module_repeats = num_mid_module_repeats
        
        
    def setup(self):
        self.obs_encoder_maker.setup()
        
        # save from global config
        self.action_rel_indices = globals.CONFIG.action_rel_indices #type:ignore
        self.horizon = len(self.action_rel_indices)

        # parse shape_meta
        assert len(self.action_shape) == 1
        self.action_dim = self.action_shape[0]
            
        # get the obs encoder object
        self.obs_encoder = self.obs_encoder_maker.get()
        
        if self.obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=self.obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: 
                    nn.GroupNorm(
                    num_groups=x.num_features//16,
                    num_channels=x.num_features)
            )
        
        if self.eval_fixed_crop:
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

        #### trunk
        # create diffusion model. Get first/only element from list
        obs_feature_dim = self.obs_encoder.output_shape()[0]
        input_dim = self.action_dim + obs_feature_dim
        global_cond_dim = None
        if self.obs_as_global_cond:
            input_dim = self.action_dim
            global_cond_dim = obs_feature_dim * self.n_obs_steps

        if self.use_simple_model:
            nb_inps = self.action_dim * self.horizon + obs_feature_dim + 1 # trajectory, obs, timestep
            model = SimpleModel(nb_inps, self.action_dim * self.horizon)
            pass
            
        else:
            model = ConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=self.diffusion_step_embed_dim,
                down_dims=self.down_dims,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_predict_scale=self.cond_predict_scale,
                num_mid_module_repeats = self.num_mid_module_repeats,
            )
        #### end trunk
        
        if self.use_tree:
            ### leafs
            leafs = nn.ModuleDict()
            for key, val in globals.REPLAY_BUFFER_LOADER.rbs.items():
                logger.info("Making Unet leaf for task: {}".format(key))
                # hard-coded leafs for now. We'd expect the input to be the same shape as the output of `model`, which is action_dim (the horizon dim is taken care of implicitly)
                # leafs[key] = ConditionalUnet1D(
                #             input_dim = action_dim,
                #             local_cond_dim=None,
                #             global_cond_dim=global_cond_dim,
                #             down_dims=[16,32,64],
                #             diffusion_step_embed_dim=diffusion_step_embed_dim,
                #             kernel_size=kernel_size,
                #             n_groups=n_groups,
                #             cond_predict_scale=cond_predict_scale
                #         )
                leafs[key] = Leaf(self.action_dim, self.horizon)
                
            ### end leafs

            # tree
            self.tree = Tree(
                            leafs=leafs
                            )
            
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0 if self.obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=self.n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.obs_feature_dim = obs_feature_dim

        print_nb_params(self.model, "Diffusion params")
        print_nb_params(self.obs_encoder, "Vision params")
        
        ### TEST
        self.random_noise = None
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, 
            condition_mask,
            scheduler: DDPMScheduler,
            local_cond=None, 
            global_cond=None,
            generator=None,
            task_id = None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        
        if False: # self.random_noise is None:
            self.random_noise = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        trajectory = self.make_noise(condition_data, generator)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output (could be noise or trajectory depending on pred_type)
            model_output = self.model(trajectory, 
                                      t, 
                                      task_id,
                local_cond=local_cond, global_cond=global_cond)
            
            # tree?
            if self.use_tree:
                assert(task_id is not None)
                # model_output = self.tree.leafs[task_id](model_output, t, global_cond=global_cond) # hack
                model_output = self.tree.forward(model_output, leaf=task_id)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        assert(not torch.isnan(trajectory).any())
        return trajectory
    
    @torch.no_grad()
    def infer(self, nobs_dict: dict, task_id=None, noise_scheduler=None):
        """
        setup for predict_action. Squeeze all tensors, then add the appropriate dimensions
        """
        # for key, val in nobs_dict.items():
        #     val = torch.squeeze(val)
            
        #     # predict_action expects a batch dimension, and a history dimension, so add two axes
        #     val = torch.reshape(val, [1, 1] + list(val.shape))
            
        #     nobs_dict[key] = val
        
        if noise_scheduler is None:
            noise_scheduler = self.noise_scheduler
            
        self.eval()
        nresult = self.predict_action_impl(nobs_dict, task_id=task_id, noise_scheduler=noise_scheduler)
        self.train()
        
        # naction doesn't include past actions
        naction = nresult["naction"]
        naction_og = naction.clone() # for debugging
        all_nactions = nresult['naction_pred']
        
        if self.action_relative_to_state:
            # requires action and obs
            nbatch = {
                'action': naction,
                'obs': nobs_dict
                    }
            
            nbatch = self.ActionRelativeToState(nbatch, inference=True)
            
            naction = nbatch['action']
            
        # done
        return naction, all_nactions

    def predict_action(self, # type:ignore
                       nobs_dict: Dict[str, torch.Tensor],
                        task_id = None,
                       ) -> Dict[str, torch.Tensor]: 
        return self.predict_action_impl(
            nobs_dict,
            self.noise_scheduler,
            task_id=task_id,
            )
        
    def denoise(self, 
            nobs_dict,
            noise_scheduler,
            task_id = None,
            ):
        """ alias for predict_action_impl """
        return self.predict_action_impl(nobs_dict, noise_scheduler , task_id=task_id)
    
    def get_future_actions(self, all_actions):
        start = np.argmax(np.array(self.action_rel_indices) >= 0)
        
        future_actions = all_actions[:, start:]
        return future_actions

    def predict_action_impl(self, 
            nobs_dict: Dict[str, torch.Tensor],
            noise_scheduler,
            task_id = None,
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
            
            # # I'm not sure why this line was included... required during training ... I think it's because batches during training are [Batch-dim, Traj-dim, data] but during inference we often only have the data-dim
            this_nobs = dict_apply(nobs, lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))
            # this_nobs = nobs
            
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            
        # not available rn
        else:
            raise

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            noise_scheduler,
            local_cond=local_cond,
            global_cond=global_cond,
            task_id = task_id,
            **self.kwargs)
        
        # unnormalize elsewhere
        naction_pred = nsample[...,:Da]
        
        # should be used with DQL ... but I don't use DQL directly anymore
        if False:
            act = nn.Tanh() # map [-inf, inf] to [-1, 1]. It's like a soft version of clamping. Do this because I clamp actions before execution so the actor should be aware of that somehow.
            
            naction_pred = act(naction_pred)

        # get action
        naction = self.get_future_actions(naction_pred)
        
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
        
        assert(not torch.isnan(naction_pred).any())
        return nresult
    
    def loss(self, nbatch, task_id):
        loss2, a0, timesteps = self.compute_loss(nbatch,task_id)
        
        # # logging
        # dd = {task_id + ": bc_actor_loss": loss2}
        # globals.LOGGER.log(dd)
        
        return loss2, a0, timesteps
            
    def get_val_action_mse_error(self, nbatch, task_id=None):
        if self.action_relative_to_state:
            nbatch = self.ActionRelativeToState(nbatch)
            
        # ground truth action
        nobs = nbatch['obs']
        gt_action = nbatch['action']
        
        # denoise
        nresult = self.predict_action(nobs, task_id=task_id)
        
        # extract the predicted action
        pred_action = nresult['naction_pred']
        
        # calc mse
        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        
        # move to cpu
        action_mse_error = mse.item()
        
        return action_mse_error
    
    def ActionRelativeToState(self, nbatch0, inference=False):
        """
        subtract the state value off the action values
        """
        # copy to prevent modifying the upstream object
        nbatch = pytorch_util.dict_of_tensor_copy(nbatch0)
        
        nobs = nbatch['obs']
        nactions = nbatch['action']
        
        nb_actions = nactions.shape[-1]
        
        nstate = nobs['state']
        
        # assume the first n state values correspond to action values
        nstate_actions = nstate[..., 0:nb_actions]
        
        if inference:
            # if inferring, action = state + rel_action
            nactions += nstate_actions
        else:
            # subtract off, using broadcasting in the trajectory-dimension
            nactions -= nstate_actions
        
        # save in nbatch
        nbatch['action'] = nactions
        
        return nbatch
    
    def encode_one_nobs(self, nobs):
        """
        nobs keys example: state, img, img2

        NOT FINISHED
        """
        pass

        # if self.obs_as_global_cond:
        #     # reshape B, T, ... to B*T
        #     this_nobs = dict_apply(nobs, 
        #         lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))
        #     nobs_features = self.obs_encoder(this_nobs)
        #     # reshape back to B, Do
        #     global_cond = nobs_features.reshape(batch_size, -1)
        # else:
        #     # reshape B, T, ... to B*T
        #     this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        #     nobs_features = self.obs_encoder(this_nobs)
        #     # reshape back to B, T, Do
        #     nobs_features = nobs_features.reshape(batch_size, horizon, -1)
        #     cond_data = torch.cat([nactions, nobs_features], dim=-1)
        #     trajectory = cond_data.detach()

        # return nobs_features
        
    def make_noise(self, x, generator=None):
        """
        make noise. If we're clamping model I/O, then clamp the noise too (since we'd never expect to see an action outside of [-1, 1])
        
        
        torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        """
        n = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        
        must_clamp = globals.CONFIG.clamp #type:ignore
        
        if must_clamp:
            n = torch.clamp(n, -1.0, 1.0)
            
        return n
        
        

    # ========= training  ============
    def compute_loss(self, nbatch, task_id_UNUSED):
        # normalize input
        assert 'valid_mask' not in nbatch
        
        # i don't like this here anymore, should be handled in the sampler instead
        # if self.action_relative_to_state:
        #     nbatch = self.ActionRelativeToState(nbatch)
        
        # for cotraining, we normalize when we construct the batch
        task_id = nbatch['task_id'] if 'task_id' in nbatch else None
        nobs = nbatch['obs']
        nactions = nbatch['action']
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps
        
        # testing / troubleshooting
        if False:
            with torch.no_grad():
                s = nobs['state'][:, 0, 5:6]
                a = nactions[:, 0, 5:6]
                j = utils.compute_jerk_waypoints(s, a)
                pass

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T ...
            # could use einops.rearrange(global_cond, 'b h t -> b (h t)') instead
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        # else:
        #     # reshape B, T, ... to B*T
        #     this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        #     nobs_features = self.obs_encoder(this_nobs)
        #     # reshape back to B, T, Do
        #     nobs_features = nobs_features.reshape(batch_size, horizon, -1)
        #     cond_data = torch.cat([nactions, nobs_features], dim=-1)
        #     trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll interpolate the input to
        noise = self.make_noise(trajectory)
        bsz = trajectory.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, #type:ignore
            (bsz,), device=trajectory.device
        ).long()
        
        # (this is the forward diffusion process)
        # basically does t' = sqrt(alpha) * t + sqrt((1-alpha)) * noise
        # but it's not exactly linear interpolation because of the sqrt
        # from the original DDPM paper, this is adding noise
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, task_id, 
            local_cond=local_cond, global_cond=global_cond)

        # tree?
        if self.use_tree:
            # pred = self.tree.leafs[task_id](pred, timesteps, global_cond=global_cond) # hack
            pred = self.tree.forward(pred, leaf=task_id)

        pred_type = self.noise_scheduler.config.prediction_type #type:ignore
        
        # model prediction is the noise that's been added to the original trajectory of actions
        if pred_type == 'epsilon':
            target = noise
            
            # must use the noise scheduler to compute the original sample. TODO: only do if DQL is used.
            # NOTE: THIS IS QUITE SLOW
            if False:
                a0 = ForNoiseStep(self.noise_scheduler, pred, timesteps, noisy_trajectory)
            else:
                a0 = None
            
        # model prediction is the original trajectory of actions
        elif pred_type == 'sample':
            target = trajectory
            a0 = pred
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # calculate the loss
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        
        # testing weighing the gofa joints more than hand joints since their link lengths are larger -> ehh... idk.. didn't really work
        if False:
            loss[:, :, 0:6] = loss[:, :, 0:6] * 10.0
        
        # testing weighing the earlier waypoints higher because those are the ones we tend to execute before replanning
        if False:
            loss[:, 0:8, :] = loss[:, 0:8, :] * 2.5
            
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss.mean()
        
            
        return loss, a0, timesteps

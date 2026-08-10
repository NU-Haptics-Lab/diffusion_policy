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
from diffusion_policy.model.diffusion.dexnex_transformer_for_diffusion import DexNexTransformerForDiffusion
from diffusion_policy.model.pellet_localizer import PelletLocalizer
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
from diffusion_policy.model.obs_encoder import ObsEncoderMaker, per_key_output_dims, encode_obs_per_key

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

class DexNexTransformerAdapter(nn.Module):
    """
    Adapts DexNexTransformerForDiffusion's (sample, timestep, cond) forward
    to match ConditionalUnet1D's (sample, timestep, task_ids, local_cond, global_cond)
    call signature, so DiffusionModel's call sites don't need to branch on model type.

    `global_cond` is expected to already be a dict of {obs_key: (B, D_k)} tensors,
    built upstream (see DiffusionModel._encode_obs_cond) -- one token per obs key.
    """
    def __init__(self, transformer: DexNexTransformerForDiffusion):
        super().__init__()
        self.transformer = transformer

    def forward(self, sample, timestep, task_ids=None, local_cond=None, global_cond=None, patches=None, data_source=None, log_attn=False, **kwargs):
        return self.transformer(sample, timestep, cond=global_cond, patches=patches, task_ids=task_ids, data_source=data_source, log_attn=log_attn)

    def get_optim_groups(self, weight_decay: float=1e-3):
        return self.transformer.get_optim_groups(weight_decay=weight_decay)

    def configure_optimizers(self, learning_rate: float=1e-4, weight_decay: float=1e-3, betas=(0.9,0.95)):
        return self.transformer.configure_optimizers(learning_rate=learning_rate, weight_decay=weight_decay, betas=betas)


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
            embed_task_id = False,
            num_tasks = 0, # size of the task-id embedding table (dexnex_transformer only)
            model_type = 'unet', # 'unet' | 'dexnex_transformer'
            transformer_n_layer = 8,
            transformer_n_head = 8,
            transformer_n_emb = 256,
            transformer_p_drop_emb = 0.1,
            transformer_p_drop_attn = 0.1,
            transformer_n_cond_layers = 4,
            transformer_p_drop_token = 0.0, # whole-token (modality) dropout probability
            transformer_droppable_token_keys = None, # obs/task_id keys eligible for whole-token dropout; None = all except timestep
            patch_keys_to_use = None, # obs keys that are unpooled patch-token grids (e.g. DINOv3 patches); dexnex_transformer only. Read directly from nbatch['obs'], bypassing obs_encoder, since ObservationEncoder always flattens per-key output and would destroy the patch grid.
            spatial_softmax_patch_keys = None, # subset of patch_keys_to_use to reduce via spatial softmax (one token per group) instead of keeping every patch as its own token
            spatial_softmax_num_keypoints = 16, # per spatial_softmax_patch_keys group
            spatial_softmax_temperature_init = 1.0,
            sim_only_token_keys = None, # token keys hard-zeroed for real (data_source==0) samples, always (train + eval)
            real_only_token_keys = None, # token keys hard-zeroed for sim (data_source==1) samples, always (train + eval)
            diagnostics_every_n_steps = 50, # gate for the pricier periodic diagnostics (attn entropy, task embedding stats)
            # pellet localizer (aiet_alignment_sim): an upstream module that predicts a
            # pellet xyz position from a patch-token grid + lowdim state, whose (detached)
            # output is injected as an extra lowdim obs key for the main transformer --
            # see _maybe_run_pellet_localizer. None (default) disables it entirely.
            pellet_localizer_patch_key = None, # obs key holding the (H,W,patch_dim) patch grid, e.g. "wrist_camera_patch_features"
            pellet_localizer_lowdim_keys = None, # list of lowdim obs keys concatenated as the localizer's non-vision input, e.g. ["joint_positions", "wrist_camera_pose"]
            pellet_localizer_pred_obs_key = "pellet_xyz_pred", # obs key the (detached) xyz prediction is injected under -- must also be registered in common_obs_encoder.lowdims + obs_keys_to_use
            pellet_localizer_has_pellet_obs_key = "has_pellet_pred", # obs key the (detached) has-pellet probability (sigmoid of the classifier logit) is injected under -- same registration requirement as pellet_localizer_pred_obs_key
            pellet_localizer_num_keypoints = 16,
            pellet_localizer_hidden_dim = 128,
            pellet_localizer_temperature_init = 1.0,
            pellet_localizer_output_dim = 3,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert model_type in ('unet', 'dexnex_transformer')
        if patch_keys_to_use:
            assert n_obs_steps == 1, "patch-token history is not supported yet; only the most recent obs step is used"
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

        self.embed_task_id = embed_task_id
        self.num_tasks = num_tasks

        self.model_type = model_type
        self.transformer_n_layer = transformer_n_layer
        self.transformer_n_head = transformer_n_head
        self.transformer_n_emb = transformer_n_emb
        self.transformer_p_drop_emb = transformer_p_drop_emb
        self.transformer_p_drop_attn = transformer_p_drop_attn
        self.transformer_n_cond_layers = transformer_n_cond_layers
        self.transformer_p_drop_token = transformer_p_drop_token
        self.transformer_droppable_token_keys = transformer_droppable_token_keys
        self.patch_keys_to_use = patch_keys_to_use or []
        self.spatial_softmax_patch_keys = spatial_softmax_patch_keys or []
        self.spatial_softmax_num_keypoints = spatial_softmax_num_keypoints
        self.spatial_softmax_temperature_init = spatial_softmax_temperature_init
        self.sim_only_token_keys = sim_only_token_keys
        self.real_only_token_keys = real_only_token_keys
        self.diagnostics_every_n_steps = diagnostics_every_n_steps

        self.pellet_localizer_patch_key = pellet_localizer_patch_key
        self.pellet_localizer_lowdim_keys = pellet_localizer_lowdim_keys or []
        self.pellet_localizer_pred_obs_key = pellet_localizer_pred_obs_key
        self.pellet_localizer_has_pellet_obs_key = pellet_localizer_has_pellet_obs_key
        self.pellet_localizer_num_keypoints = pellet_localizer_num_keypoints
        self.pellet_localizer_hidden_dim = pellet_localizer_hidden_dim
        self.pellet_localizer_temperature_init = pellet_localizer_temperature_init
        self.pellet_localizer_output_dim = pellet_localizer_output_dim

        # Nones
        self.obs_encoder = None
        self.pellet_localizer = None
        self._last_pellet_xyz_pred = None
        self._last_pellet_has_pellet_logit = None
        self._last_pellet_keypoints = None
        
        
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

        if self.pellet_localizer_patch_key is not None:
            H, W, patch_dim = globals.CONFIG.shape_meta[self.pellet_localizer_patch_key].shape  # type:ignore
            lowdim_dim = sum(
                globals.CONFIG.shape_meta[key].shape[0]  # type:ignore
                for key in self.pellet_localizer_lowdim_keys
            )
            self.pellet_localizer = PelletLocalizer(
                patch_h=H,
                patch_w=W,
                patch_dim=patch_dim,
                lowdim_dim=lowdim_dim,
                num_keypoints=self.pellet_localizer_num_keypoints,
                hidden_dim=self.pellet_localizer_hidden_dim,
                spatial_softmax_temperature_init=self.pellet_localizer_temperature_init,
                output_dim=self.pellet_localizer_output_dim,
            )

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

        elif self.model_type == 'dexnex_transformer':
            assert self.obs_as_global_cond, "dexnex_transformer only supports obs_as_global_cond=True"
            # one token per obs key; each key's n_obs_steps history is flattened into its token
            cond_dims = {
                key: dim * self.n_obs_steps
                for key, dim in per_key_output_dims(self.obs_encoder).items()
            }
            # patch-token groups (e.g. DINOv3 patches per camera) bypass obs_encoder entirely,
            # since ObservationEncoder always flattens per-key output and would destroy the
            # patch grid. Shapes come straight from shape_meta, same source ObsEncoderMaker uses.
            # shape_meta stores the raw (H, W, patch_dim) spatial grid (e.g. 14x14x1024 for
            # DINOv3). Keys in spatial_softmax_patch_keys get reduced to one token via a
            # learned spatial softmax; the rest keep every patch as its own token (flatten
            # H,W into a single num_patches axis, since DexNexTransformerForDiffusion's
            # patch_group_dims expects (num_patches, patch_dim)).
            patch_group_dims = {}
            spatial_softmax_group_dims = {}
            for key in self.patch_keys_to_use:
                H, W, patch_dim = globals.CONFIG.shape_meta[key].shape #type:ignore
                if key in self.spatial_softmax_patch_keys:
                    spatial_softmax_group_dims[key] = (H, W, patch_dim, self.spatial_softmax_num_keypoints)
                else:
                    patch_group_dims[key] = (H * W, patch_dim)
            transformer = DexNexTransformerForDiffusion(
                input_dim=self.action_dim,
                output_dim=self.action_dim,
                horizon=self.horizon,
                n_obs_steps=self.n_obs_steps,
                cond_dims=cond_dims,
                patch_group_dims=patch_group_dims if patch_group_dims else None,
                spatial_softmax_group_dims=spatial_softmax_group_dims if spatial_softmax_group_dims else None,
                spatial_softmax_temperature_init=self.spatial_softmax_temperature_init,
                num_tasks=self.num_tasks if self.embed_task_id else 0,
                n_layer=self.transformer_n_layer,
                n_head=self.transformer_n_head,
                n_emb=self.transformer_n_emb,
                p_drop_emb=self.transformer_p_drop_emb,
                p_drop_attn=self.transformer_p_drop_attn,
                n_cond_layers=self.transformer_n_cond_layers,
                p_drop_token=self.transformer_p_drop_token,
                droppable_token_keys=self.transformer_droppable_token_keys,
                sim_only_token_keys=self.sim_only_token_keys,
                real_only_token_keys=self.real_only_token_keys,
            )
            model = DexNexTransformerAdapter(transformer)

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
                embed_task_id = self.embed_task_id,
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

    def _maybe_run_pellet_localizer(self, nobs):
        """
        If configured (pellet_localizer_patch_key is not None), runs
        self.pellet_localizer on the current nobs (most recent obs step) and
        injects its DETACHED xyz prediction AND has-pellet probability
        (sigmoid of the classifier logit) into nobs under
        pellet_localizer_pred_obs_key / pellet_localizer_has_pellet_obs_key,
        as if they were normal lowdim obs keys -- so both flow through the
        SAME per-key obs_encoder tokenization as everything else in
        _encode_obs_cond. The transformer needs has-pellet as an explicit
        input because pellet_xyz_pred alone can't distinguish "confidently
        localized" from "no pellet visible, this is just whatever the MLP
        extrapolated" -- without it, the policy has no way to react
        differently when the xyz prediction shouldn't be trusted. No-op if
        not configured.

        Also caches the prediction, has-pellet logit, and intermediate
        keypoints (all still gradient-attached, pre-detach) for BatchLoss's
        aux losses to read afterward -- same pattern as
        DexNexTransformerForDiffusion's spatial-softmax/cross-attention
        caching. Must be called before _encode_obs_cond, and before any aux
        loss in the same step tries to read the cache.

        Detached deliberately before injection: the localizer is trained
        only by its own losses (pixel keypoint + xyz regression + has-pellet
        classifier), not by the diffusion BC loss, so a bad early prediction
        doesn't get "corrected" by distorting the vision pathway to fit the
        BC objective instead of the real pellet location.
        """
        self._last_pellet_xyz_pred = None
        self._last_pellet_has_pellet_logit = None
        self._last_pellet_keypoints = None
        if self.pellet_localizer is None:
            return

        x = nobs[self.pellet_localizer_patch_key][:, -1, ...]  # (B, H, W, patch_dim)
        B, patch_dim = x.shape[0], x.shape[-1]
        patches = x.reshape(B, -1, patch_dim)

        lowdim = torch.cat(
            [nobs[key][:, -1, :] for key in self.pellet_localizer_lowdim_keys], dim=-1
        )

        pred_xyz, has_pellet_logit, keypoints = self.pellet_localizer(patches, lowdim)
        self._last_pellet_xyz_pred = pred_xyz
        self._last_pellet_has_pellet_logit = has_pellet_logit
        self._last_pellet_keypoints = keypoints

        nobs[self.pellet_localizer_pred_obs_key] = pred_xyz.detach().unsqueeze(1)  # (B, 1, output_dim)
        has_pellet_prob = torch.sigmoid(has_pellet_logit.detach())
        nobs[self.pellet_localizer_has_pellet_obs_key] = has_pellet_prob.unsqueeze(-1).unsqueeze(1)  # (B, 1, 1)

    def get_last_pellet_prediction(self):
        """
        Returns (pred_xyz, has_pellet_logit, keypoints) from the most recent
        _maybe_run_pellet_localizer call, all gradient-attached (unlike the
        detached copy injected into nobs) -- for BatchLoss's aux losses.
        All None if the pellet localizer isn't configured, or this is called
        before any forward pass.
        """
        return self._last_pellet_xyz_pred, self._last_pellet_has_pellet_logit, self._last_pellet_keypoints

    def _encode_obs_cond(self, nobs, To, B):
        """
        Flatten obs history (B,To,...) -> (B*To,...), encode, and reshape back.

        For the unet, returns one fused vector (B, obs_feature_dim*To).
        For the dexnex_transformer, returns a dict of {obs_key: (B, D_k*To)}
        -- one token per obs key, so obs keys are never fused together.
        """
        assert self.obs_encoder is not None
        this_nobs = dict_apply(nobs, lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))

        if self.model_type == 'dexnex_transformer':
            per_key_features = encode_obs_per_key(self.obs_encoder, this_nobs)
            return {key: feat.reshape(B, -1) for key, feat in per_key_features.items()}
        else:
            nobs_features = self.obs_encoder(this_nobs)
            return nobs_features.reshape(B, -1)

    def _encode_patch_groups(self, nobs):
        """
        Read patch-token groups (e.g. DINOv3 patches per camera) straight out of
        nobs, bypassing self.obs_encoder entirely -- ObservationEncoder always
        flattens per-key output to (B,D), which would destroy the (num_patches,
        patch_dim) grid. Only the most recent obs step is used (n_obs_steps==1
        is asserted in __init__ whenever patch_keys_to_use is non-empty).

        nobs[key] arrives as (B, To, H, W, patch_dim) -- the raw spatial grid
        (e.g. 14x14x1024 for DINOv3); flatten H,W into a single num_patches
        axis to match DexNexTransformerForDiffusion's (B, num_patches, patch_dim).
        """
        if not self.patch_keys_to_use:
            return None
        patches = {}
        for key in self.patch_keys_to_use:
            x = nobs[key][:, -1, ...]  # (B, H, W, patch_dim)
            B = x.shape[0]
            patch_dim = x.shape[-1]
            patches[key] = x.reshape(B, -1, patch_dim)  # (B, num_patches, patch_dim)
        return patches

    # ========= inference  ============
    def conditional_sample(self,
            condition_data,
            condition_mask,
            scheduler: DDPMScheduler,
            local_cond=None,
            global_cond=None,
            patches=None,
            data_source=None,
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
                local_cond=local_cond, global_cond=global_cond, patches=patches, data_source=data_source)
            
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
    def infer(self, nobs_dict: dict, task_id=None, data_source=None, noise_scheduler=None):
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
        nresult = self.predict_action_impl(nobs_dict, task_id=task_id, data_source=data_source, noise_scheduler=noise_scheduler)
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
                        data_source = None,
                       ) -> Dict[str, torch.Tensor]:
        return self.predict_action_impl(
            nobs_dict,
            self.noise_scheduler,
            task_id=task_id,
            data_source=data_source,
            )

    def denoise(self,
            nobs_dict,
            noise_scheduler,
            task_id = None,
            data_source = None,
            ):
        """ alias for predict_action_impl """
        return self.predict_action_impl(nobs_dict, noise_scheduler, task_id=task_id, data_source=data_source)
    
    def get_future_actions(self, all_actions):
        start = np.argmax(np.array(self.action_rel_indices) >= 0)
        
        future_actions = all_actions[:, start:]
        return future_actions

    def predict_action_impl(self,
            nobs_dict: Dict[str, torch.Tensor],
            noise_scheduler,
            task_id = None,
            data_source = None,
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
            self._maybe_run_pellet_localizer(nobs)
            global_cond = self._encode_obs_cond(nobs, To, B)
            patches = self._encode_patch_groups(nobs)
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
            patches=patches,
            data_source=data_source,
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

        # if task_id is a per-sample tensor (as opposed to a plain rb_id string),
        # also log each task's validation MSE separately -- training loss can look
        # fine on an underrepresented task purely because it's rarely sampled, while
        # actual policy quality for that task stays bad; this is the more trustworthy signal.
        if isinstance(task_id, torch.Tensor):
            with torch.no_grad():
                per_sample_mse = torch.nn.functional.mse_loss(pred_action, gt_action, reduction='none')
                per_sample_mse = per_sample_mse.mean(dim=tuple(range(1, per_sample_mse.dim())))
                task_id_flat = torch.reshape(task_id, [-1])
                unique_task_ids = torch.unique(task_id_flat)
                if unique_task_ids.numel() > 1:
                    for tid in unique_task_ids.tolist():
                        task_mask = task_id_flat == tid
                        if task_mask.any():
                            globals.LOGGER.log_one(f"val_action_mse/task_{tid}", per_sample_mse[task_mask].mean())

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
        data_source = nbatch['data_source'] if 'data_source' in nbatch else None
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
        patches = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T ...
            self._maybe_run_pellet_localizer(nobs)
            global_cond = self._encode_obs_cond(nobs, To, batch_size)
            patches = self._encode_patch_groups(nobs)
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

        # periodically capture extra diagnostics (attention entropy, task embedding
        # drift) -- gated so the (small) extra cost of need_weights=True isn't paid every step
        do_diagnostics = self.model_type == 'dexnex_transformer' and utils.StepFreqTrigger(self.diagnostics_every_n_steps)

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, task_id,
            local_cond=local_cond, global_cond=global_cond, patches=patches, data_source=data_source, log_attn=do_diagnostics)

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

        # log total loss, and if task_id/timestep info is available per-sample, break it down further
        with torch.no_grad():
            per_sample_loss = loss.mean(dim=tuple(range(1, loss.dim())))
            globals.LOGGER.log_one("diffusion_loss/total", per_sample_loss.mean())

            if task_id is not None:
                task_id_flat = torch.reshape(task_id, [-1])
                unique_task_ids = torch.unique(task_id_flat)
                # skip the per-task breakdown (and its .tolist() CPU sync) when
                # the whole batch is one task -- it'd just duplicate "total"
                if unique_task_ids.numel() > 1:
                    for tid in unique_task_ids.tolist():
                        task_mask = task_id_flat == tid
                        n = task_mask.sum()
                        if n > 0:
                            globals.LOGGER.log_one(f"diffusion_loss/task_{tid}", per_sample_loss[task_mask].mean())
                            globals.LOGGER.log_one(f"task_counts/task_{tid}", n)

            # bucket loss by diffusion timestep (low/mid/high noise) -- a flat average can
            # hide whether the model has learned the easy (low-noise) end but is still
            # stuck on the hard (high-noise) end, or vice versa
            num_train_timesteps = self.noise_scheduler.config.num_train_timesteps #type:ignore
            num_buckets = 4
            bucket_idx = torch.clamp(
                (timesteps.float() / num_train_timesteps * num_buckets).long(),
                max=num_buckets - 1)
            for b in range(num_buckets):
                bucket_mask = bucket_idx == b
                if bucket_mask.any():
                    globals.LOGGER.log_one(f"diffusion_loss/timestep_bucket_{b}", per_sample_loss[bucket_mask].mean())

            # bucket loss by waypoint position in the horizon (early vs. late in the
            # trajectory) -- helps check whether later waypoints are predicted as
            # accurately as near-term ones, e.g. if displacement between waypoints
            # looks too small in rollouts, check whether late-horizon buckets have
            # disproportionately higher loss (or just near-flat/low loss everywhere,
            # which would instead suggest the model is under-predicting motion overall)
            num_waypoint_buckets = 4
            horizon_len = loss.shape[1]
            bucket_bounds = torch.linspace(0, horizon_len, num_waypoint_buckets + 1).long()
            for b in range(num_waypoint_buckets):
                start, end = bucket_bounds[b].item(), bucket_bounds[b + 1].item()
                if end > start:
                    globals.LOGGER.log_one(f"diffusion_loss/waypoint_bucket_{b}", loss[:, start:end, :].mean())

            if do_diagnostics:
                transformer = self.model.transformer #type:ignore
                if getattr(transformer, 'last_cross_attn_entropy', None) is not None:
                    globals.LOGGER.log_one("diagnostics/cross_attn_entropy", transformer.last_cross_attn_entropy.mean())
                    # a name (e.g. a patch-group) can appear for many tokens -- average
                    # them together before logging, rather than overwriting per-name
                    weight_by_name = {}
                    for name, w in zip(transformer.last_cross_attn_token_names,
                                        transformer.last_cross_attn_weight_by_token.mean(dim=0)):
                        weight_by_name.setdefault(name, []).append(w)
                    for name, ws in weight_by_name.items():
                        globals.LOGGER.log_one(f"diagnostics/cross_attn_weight_{name}", torch.stack(ws).mean())

                task_stats = transformer.task_embedding_stats()
                if task_stats is not None:
                    globals.LOGGER.log_one("diagnostics/task_emb_mean_row_norm", task_stats['mean_row_norm'])
                    globals.LOGGER.log_one("diagnostics/task_emb_mean_pairwise_distance", task_stats['mean_pairwise_distance'])

        return loss, a0, timesteps



"""
implicit model means that the model takes (s, a) as its input and produces a single value. During inference, we can produce actions by doing gradient ascent on the input action to maximize the output. 

These classes, however, are not responsible for defining the loss, nor what the output means. That should be done by a parent class.
"""

import torch
import torch as th
import torch.nn.functional as F

# reference:
from diffusion_policy.model.diffusion_model import (
    DiffusionModel
)

from diffusion_policy.model.obs_encoder import (
    ObsEncoderMaker
)

from diffusion_policy.model.conditional_slashnet1d import (
    ConditionalSlashnet1D,
)

from diffusion_policy.utils import (
    print_nb_params,
)

from diffusion_policy.common.pytorch_util import (
    dict_apply,
)

from torch import (
    nn
)

from diffusion_policy.policy.base_image_policy import (
    BaseImagePolicy
)

from diffusers.schedulers.scheduling_ddpm import (
    DDPMScheduler
)

import diffusion_policy.globals as globals

from torch._functorch.apis import vmap, grad

from tqdm import (
    tqdm
)

class ImplicitPolicy(nn.Module):
    def __init__(self,
            action_shape: list,
            obs_encoder_maker: ObsEncoderMaker,
            n_obs_steps, # input time-length
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            num_mid_module_repeats = 4,
            ):
        super().__init__()
        self.action_shape = action_shape
        self.obs_encoder_maker = obs_encoder_maker
        self.n_obs_steps = n_obs_steps
        self.down_dims = down_dims
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.cond_predict_scale = cond_predict_scale
        self.num_mid_module_repeats = num_mid_module_repeats

        # my members
        self.obs_encoder = self.obs_encoder_maker.get()

        # parse shape_meta
        assert len(self.action_shape) == 1
        self.action_dim = action_shape[0]
        obs_feature_dim = self.obs_encoder.output_shape()[0]
        global_cond_dim = obs_feature_dim * n_obs_steps

        self.model = ConditionalSlashnet1D(
                        input_dim=self.action_dim,
                        global_cond_dim=global_cond_dim,
                        down_dims=down_dims,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                        num_mid_module_repeats=num_mid_module_repeats,
                    )
        
        

        print_nb_params(self.model, "Dense params")
        print_nb_params(self.obs_encoder, "Vision params")

    def forward_obs_encoder(self, *args, **kwargs):
        return self.obs_encoder(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def copy(self):
        other = ImplicitPolicy(
            self.action_shape,
            self.obs_encoder_maker.copy(),
            n_obs_steps = self.n_obs_steps,
            down_dims=self.down_dims,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            cond_predict_scale=self.cond_predict_scale,
            num_mid_module_repeats=self.num_mid_module_repeats,
        )
        
        return other

class ImplicitAlgorithm(BaseImagePolicy):
    def __init__(self,
            policy: ImplicitPolicy,
            num_inference_steps = 8,
            step_size = 1e-1,
    ):
        super().__init__()
        
        self.num_inference_steps = num_inference_steps
        self.step_size = step_size

        self.policy = policy
        self.noise_scheduler = DDPMScheduler()


        # save from global config
        self.action_rel_indices = globals.CONFIG.action_rel_indices #type:ignore
        self.horizon = len(self.action_rel_indices)
        
    def per_sample_backward(self, btrajectory, timestep, bglobal_cond):
        def fcn(trajectory, global_cond):
            return self.policy(trajectory.unsqueeze(0), timestep, global_cond = global_cond.unsqueeze(0)).squeeze()
        
        # 2. Vectorize the gradient calculation across the batch dimension
        # (0, 0) tells vmap to map over the 0th dimension of x and y
        vmap_fcn = vmap(grad(fcn), in_dims=(0, 0))
        per_sample_grads = vmap_fcn(btrajectory, bglobal_cond)
        
        assert(per_sample_grads.shape == btrajectory.shape)
        return per_sample_grads

    # alias
    def predict_action(self, nobs): #type:ignore
        return self.inference(nobs)
    
    # alias
    def infer(self, nobs_dict: dict, action = None, task_id=None, noise_scheduler=None):
        # return 3 things for backwards compat
        action = self.inference(nobs_dict)
        return action
    
    @torch.enable_grad()
    def inference(self, nobs: dict):
        To = 1
        value = next(iter(nobs.values()))
        B = value.shape[0] # batch
        T = self.horizon # trajectory length
        Da = self.policy.action_dim # action dimension
        device = self.device
        dtype = self.dtype
        step_size = self.step_size # might come from scheduler instead
        timestep = 0.0

        # reshape obs: B, T, ... to B*T ...
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))

        # get encoded obs
        nobs_features = self.policy.forward_obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1)

        # dummy trajectory
        dummy_trajectory = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # randomly initialize a
        trajectory = self.make_noise(dummy_trajectory)
        trajectory.requires_grad = True

        # do gradient ascent
        for _ in range(self.num_inference_steps):
        # for _ in tqdm(range(self.num_inference_steps), desc="Inference steps"):
            # forward & backward
            a_grad = self.per_sample_backward(trajectory, timestep, global_cond)

            # grad-free update step, to make a new trajectory object
            with torch.no_grad():
                assert(a_grad.shape == trajectory.shape)
                trajectory = trajectory + step_size * a_grad
            
        return trajectory

        
    def make_noise(self, x, generator=None):
        """
        straight from DiffusionModel
        """
        n = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        
        must_clamp = globals.CONFIG.clamp #type:ignore
        
        if must_clamp:
            n = torch.clamp(n, -1.0, 1.0)
            
        return n


    
    def forward(self, state_dict, action, options = None):
        this_nobs = state_dict
        
        batch_size = next(iter(this_nobs.values())).shape[0]
        
        # get encoded obs
        nobs_features = self.policy.forward_obs_encoder(this_nobs)
        
        global_cond = nobs_features.reshape(batch_size, -1)

        # predict
        x = self.policy.forward(action, global_cond=global_cond)

        return x


    def get_val_action_mse_error(self, nbatch, task_id=None):
        # ground truth action
        nobs = nbatch['obs']
        gt_action = nbatch['action']
        
        # denoise
        ntrajectory = self.predict_action(nobs)
        
        # extract the predicted action
        pred_action = ntrajectory
        
        # calc mse
        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        
        # move to cpu
        action_mse_error = mse.item()
        
        return action_mse_error

    
    def copy(self):
        """
        Make a copy with the same parameters, but different weights
        """
        # other = copy.deepcopy(self.q1_model)
        
        # # re-init for new weight values
        # with torch.no_grad():
        #     nn.Module.__init__(other)
        
        
        other = ImplicitAlgorithm(
                        self.policy.copy(),
                        self.num_inference_steps,
                        self.step_size
        )
        
        return other


"""
implicit bc meaning that a model takes (s, a) as its input and produces a score. test-time actions are produced via argmin_a(pi(s, a)), aka gradient ascent on the policy. 

in plain english: given the state s, the action which is most like the dataset will maximize the score.

So then we can formulate our score such that the max value is when the data-point is from the dataset.

one method: very similar to training DDPM's:
1. draw (s, a_gt) from the dataset
2. draw alpha from uniform([0, 1])
3. draw random noise e from N(0, 1)
4. compute a = alpha * a_gt + (1 - alpha) * e
5. compute score = pi(s, a)
6. compute loss = MSE(alpha, score)
7. backprop
8. step the policy weights
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

from diffusion_policy.model.implicit_model import (
    ImplicitPolicy,
    ImplicitAlgorithm
)

class ImplicitBehaviorCloningAlgorithm(ImplicitAlgorithm):
    def __init__(self,
            policy: ImplicitPolicy,
            use_compiled_policy: bool = False,
            use_only_for_inference: bool = False
    ):
        super().__init__(policy, use_compiled_policy, use_only_for_inference)
        

        self.policy = policy
        self.use_compiled_policy = use_compiled_policy
        self.use_only_for_inference = use_only_for_inference
        
    def setup(self):
        self.policy.setup()
        self.noise_scheduler = DDPMScheduler()


        # save from global config
        self.action_rel_indices = globals.CONFIG.action_rel_indices #type:ignore
        self.horizon = len(self.action_rel_indices)

    def make_alphas(self, trajectory):
        # batch size of the traj
        bsz = trajectory.shape[0]
        
        # make the shape the same length as trajectory
        shape = (bsz,) + (1,) * (len(trajectory.shape) - 1)

        # alphas
        alphas = torch.rand(shape, device=trajectory.device
        )

        return alphas
    
    def linear_interp(self, x1, x2, alpha):
        """
        alpha = 1 means y = x1
        """
        y = alpha * x1 + (1.0 - alpha) * x2
        return y

    
    # alias
    def loss(self, nbatch, task_id=None):
        loss2, a0, timesteps = self.compute_loss(nbatch)
        
        # # logging
        # dd = {task_id + ": bc_actor_loss": loss2}
        # globals.LOGGER.log(dd)
        
        return loss2, a0, timesteps
    
    # alias
    def compute_loss(self, nbatch):
        """
        no impainting, no local conds, no task id, linear scheduling
        """
        To = 1

        # extract nobs and nactions
        nobs = nbatch['obs']
        nactions = nbatch['action']
        batch_size = nactions.shape[0]

        trajectory = nactions
        cond_data = trajectory
        timesteps = 0.0

        # draw alpha
        alpha = self.make_alphas(trajectory)

        # draw noise
        noise = self.make_noise(trajectory)

        # corrupt a_gt. Simple linear interp for now, can always switch to a more clever noise scheduler (like DDPM) later
        noisy_trajectory = self.linear_interp(trajectory, noise, alpha)

        # reshape obs: B, T, ... to B*T ...
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))

        # predict the score
        score = super().forward(this_nobs, noisy_trajectory)

        # loss -- we want our score to match the alpha value
        loss = F.mse_loss(score.squeeze(), alpha.squeeze())
        
        # backwards compat, add dims 1 & 2
        loss = loss.unsqueeze(0).unsqueeze(1).unsqueeze(2)

        # we're done
        return loss, None, timesteps


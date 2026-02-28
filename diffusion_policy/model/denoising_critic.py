

"""
denoising critic meaning that the critic is trained to predict the noised q-val as a function of the noised trajectory.

pretty similar to ImplicitBehaviorCloningAlgorithm, also similar to CriticAlgorithm
"""

import torch
import torch as th
import torch.nn.functional as F


from diffusion_policy.common import pytorch_util
from diffusion_policy import utils

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

from diffusion_policy.model.implicit_behavior_cloning import (
    ImplicitBehaviorCloningAlgorithm,
)

from diffusion_policy.model.critic_algorithm import (
    CriticAlgorithm,
)

class DenoisingCritic(CriticAlgorithm):
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
        
    def make_noise(self, x, generator=None):
        """
        straight from DiffusionModel
        """
        n = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        
        must_clamp = globals.CONFIG.clamp #type:ignore
        
        if must_clamp:
            n = torch.clamp(n, -1.0, 1.0)
            
        return n

    def loss(self, nbatch_dict):
        """
        combination of CriticAlgorithm and ImplicitBehaviorCloningAlgorithm
        """
        # Sample replay buffer / batch
        state = nbatch_dict['obs']
        next_state = nbatch_dict['obs_next']
        action = nbatch_dict['action']
        next_action = nbatch_dict['action_next']
        reward = nbatch_dict['reward']
        not_done = nbatch_dict['not_done']
        
        # can't train when batch length is 1
        if True:
            if len(action) <= 1:
                critic_loss = utils.InitZeroTensorOnDevice()
                return critic_loss
            
        #### noise the trajectory
        # draw alpha, only once
        alpha = self.make_alphas(action)

        # draw noise
        noise = self.make_noise(action)

        # corrupt a_gt. Simple linear interp for now, can always switch to a more clever noise scheduler (like DDPM) later
        action = self.linear_interp(action, noise, alpha)

        # and now the next action, same alpha
        # draw noise
        noise = self.make_noise(next_action)

        # corrupt a_gt. Simple linear interp for now, can always switch to a more clever noise scheduler (like DDPM) later
        next_action = self.linear_interp(next_action, noise, alpha)
        ####


        """ Q Training """
        # predicted q-values
        current_q1, current_q2 = self.critic(state, action)
        
        # if using a target network
        if self.use_target_network:
            target_network = self.critic_target
        else:
            target_network = self.critic
            
        # get the next q-value, Q(s', a')
        target_q1, target_q2 = target_network(next_state, next_action)
        
        # if using double-q learning
        if self.use_double_q:
            target_qm = torch.min(target_q1, target_q2)
        else:
            target_qm = target_q1

        ##################
        """
        hmm, now I'm unsure if I should noise the target_q, or the reward and then compute target_q like normal. Either noise the next_action and then target_qm will be w.r.t. noise, then noise the reward and compute the bellman eq.
        - or, don't noise the reward nor next action, compute the bellman eq, and then noise the target_q
        """
        # start with noising the reward. In this setup, target_qm should be w.r.t. noisy next action
        if True:
            # assumes that the reward is linearly [0, reward] w.r.t. alpha
            # must squeeze 1 dimension out for proper broadcasting.
            alpha2 = alpha.squeeze(1)
            
            assert(alpha2.shape == reward.shape)
            noisy_reward = alpha2 * reward

            # compute the bellman equation, this is what we want our prediction to match
            target_q = (noisy_reward + not_done * self.discount * target_qm).detach()


        ##################
        # compute the loss
        critic_loss = F.mse_loss(current_q1, target_q, reduction="none")
        
        # if using double-q learning
        if self.use_double_q:
            critic_loss = critic_loss + F.mse_loss(current_q2, target_q, reduction="none")
        
        # logging
        dd = {}
        dd["qval/" + self.get_mode_string() + ": avg q-value"] = current_q1.mean()
        if globals.LOGGER is not None:
            globals.LOGGER.log(dd)
        
        if reward.mean() > 0.0:
            pass # debugging
        
        return critic_loss

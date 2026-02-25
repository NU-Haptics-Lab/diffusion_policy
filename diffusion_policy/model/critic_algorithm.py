


"""
includes common critic features like a target network, and the option to use double-q learning.

NOTE: the target-network is basically the same as an EMA network, and this class contains its own optimizer, so for now you don't need to use the ModelEmaOptim wrapper.
"""








import torch

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.diffusion_ql.critic_arch import DoubleCritic
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.common import pytorch_util
from diffusion_policy import utils

import diffusion_policy.globals as globals

"""
Based on this paper: https://arxiv.org/pdf/2208.06193
and this associated repo: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL 
"""

class CriticAlgorithm(nn.Module):
    """
    default values taken from the diffusion-ql repo.
    https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL/blob/d871f5c6b4a3a3a19a10c662a54f32d5819dfcdb/agents/ql_diffusion.py#L49 
    """
    def __init__(self,
                 critic: DoubleCritic,
                 discount=0.99,
                 tau=0.005,
                 max_q_backup=False,
                 lr=3e-4,
                 use_lr_decay=False,
                 lr_min = 1e-5,
                 lr_maxt=10000, # nb steps (NOT EPOCHS)
                 grad_norm=1.0,
                 use_target_network = True,
                 use_double_q = True,
                 action_relative_to_state = False,
                 use_denoise = False,
                 ):
        nn.Module.__init__(self)
        
        self.critic = critic
        self.discount = discount
        self.tau = tau
        self.max_q_backup = max_q_backup
        self.lr = lr
        self.use_lr_decay = use_lr_decay
        self.lr_min = lr_min
        self.lr_maxt = lr_maxt
        self.grad_norm = grad_norm
        self.use_target_network = use_target_network
        self.use_double_q = use_double_q
        self.action_relative_to_state = action_relative_to_state
        self.use_denoise = use_denoise

        
    def setup(self):
        self.critic.setup()
        
        if self.use_target_network:
            self.critic_target = copy.deepcopy(self.critic)
            
        # set up the optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=1.0e-06)
        
        # device transfer members, since I own them
        device = torch.device(globals.CONFIG.device) #type:ignore
        self.critic.to(device)
        
        if self.use_target_network:
            self.critic_target.to(device)
        optimizer_to(self.critic_optimizer, device)
        
        # EMA
        ema_model = copy.deepcopy(self.critic)
        self.ema = EMAModel(ema_model)
        
        # device transfer of the ema, since I own it
        ema_model.to(device)

        if self.use_lr_decay:
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=self.lr_maxt, eta_min=self.lr_min)

        
    def MakeOptions(self, task_id):
        options = {}
        
        return options
    
    def ForwardCritic(self, *args, **kwargs):
        if self.training:
            return self.critic(*args, **kwargs)
        else:
            return self.ema.averaged_model(*args, **kwargs)
        
    def LossCritic(self, nbatch_dict):
        """
        We need the g.t. state, action, reward, and next_state for QL training. 
        
        We then get the predicted next_action as a fcn of next_state, and use that (next_action, next_state) tuple to get the predicted next_Q. 
        
        We then add reward and next_Q to get the target (standard q-learning), and we compare it to the Q-value to get a loss. This loss is used to train the Q-network.
        
        use next_action from the dataset to make it off-policy
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
        
        # hack TODO: fix. squeeze the dims
        # state = pytorch_util.dict_of_tensor_copy(state)
        # for key, val in state.items():
        #     state[key] = torch.squeeze(val)
        # action = torch.squeeze_copy(action)

        """ Q Training """
        current_q1, current_q2 = self.critic(state, action)

        """ max-q-backup not yet integrated, Kumar et al. 2020 """
        # if self.max_q_backup:
        #     next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
        #     next_action_rpt = self.ema_model(next_state_rpt)
        #     target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
        #     target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
        #     target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
        #     target_q = torch.min(target_q1, target_q2)
        # else:
        
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

        # compute the bellman equation
        target_q = (reward + not_done * self.discount * target_qm).detach()

        # compute the loss
        critic_loss = F.mse_loss(current_q1, target_q)
        
        # if using double-q learning
        if self.use_double_q:
            critic_loss = critic_loss + F.mse_loss(current_q2, target_q)
        
        # logging
        dd = {}
        dd["qval/" + self.get_mode_string() + ": avg q-value"] = current_q1.mean()
        globals.LOGGER.log(dd)
        
        if reward.mean() > 0.0:
            pass
        
        return critic_loss

    def step(self):
        """
        steps the critic network
        """
        metric = {}
        
        # done in step_trainer now
        # if self.grad_norm > 0:
        #     # clips the gradients. the trailing _ indicates an in-place operation
        #     critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        #     metric['Critic Grad Norm'] = critic_grad_norms.max().item()
            
        # update critic weights
        self.critic_optimizer.step()

        # update critic target weights
        if self.use_target_network:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # step the lr scheduler
        if self.use_lr_decay: 
            self.critic_lr_scheduler.step()

        return metric
        
    def loss(self, nbatch_dict):
        """
        next_action - grad-free denoised next observation using the actor
        task_id - task identifier
        a0 - grad-full diffusion policy outputs, replaces new_action
        
        next_action should be used to compute the critic loss
        """
        dd = {}
                
        # training the critic
        # calc loss for the critic, using (s, a, r, s') & a'
        critic_loss = self.LossCritic(nbatch_dict)
        
        # critic logging
        dd[self.get_mode_string() + ": critic_loss"] = critic_loss
                
        # logging
        globals.LOGGER.log(dd)
        
        return critic_loss
    
    def get_mode_string(self):
        mode = "Training" if self.training else "Validation"
        return mode

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))
        
    def reset(self):
        self.critic_optimizer.zero_grad()
        
    def get_model(self, want_target_network=False):
        if want_target_network and self.use_target_network:
            return self.critic_target
        else:
            return self.critic
        
    def inference(self, nobs):
        return self.ema.averaged_model.infer(nobs)
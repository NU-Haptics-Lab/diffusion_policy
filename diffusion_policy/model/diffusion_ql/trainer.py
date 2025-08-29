
import torch

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.diffusion_ql.critic import DoubleCritic
from diffusion_policy.common.pytorch_util import optimizer_to

import diffusion_policy.globals as globals

"""
Based on this paper: https://arxiv.org/pdf/2208.06193
and this associated repo: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL 
"""

class DiffusionQL(object):
    """
    default values taken from the diffusion-ql repo.
    https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL/blob/d871f5c6b4a3a3a19a10c662a54f32d5819dfcdb/agents/ql_diffusion.py#L49 
    """
    def __init__(self,
                 critic: DoubleCritic, # should be on device
                 discount=0.99,
                 tau=0.005,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):
        
        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # device transfer members, since I own it
        device = torch.device(globals.CONFIG.device)
        self.critic.to(device)
        self.critic_target.to(device)
        optimizer_to(self.critic_optimizer, device)
        
        # EMA
        ema_model = copy.deepcopy(self.critic)
        self.ema = EMAModel(ema_model)
        
        # device transfer of the ema, since I own it
        ema_model.to(device)

        if lr_decay:
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.max_q_backup = max_q_backup
        
    def MakeOptions(self, task_id):
        options = {}
        options['leaf'] = task_id
        return options
        
    def LossCritic(self, nbatch_dict, next_action, options):
        """
        samples - must be the same noised trajectories that were passed through the actor model (fcn: policy.compute_loss)
        new_action - should be the action calculated by using the predicted noise from the actor model.
        
        We need the g.t. state, action, reward, and next_state for QL training. We then get the predicted next_action as a fcn of next_state, and use that (next_action, next_state) tuple to get the predicted next_Q. We then add reward and next_Q to get the target, and we compare it to the Q-value to get a loss. This loss is used to train the Q-network.
        """

        metric = {}

        # Sample replay buffer / batch
        state = nbatch_dict['obs']
        next_state = nbatch_dict['obs_next']
        action = nbatch_dict['action']
        reward = nbatch_dict['reward']
        not_done = nbatch_dict['not_done']

        """ Q Training """
        current_q1, current_q2 = self.critic(state, action, options)

        """ max-q-backup not yet integrated, Kumar et al. 2020 """
        # if self.max_q_backup:
        #     next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
        #     next_action_rpt = self.ema_model(next_state_rpt)
        #     target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
        #     target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
        #     target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
        #     target_q = torch.min(target_q1, target_q2)
        # else:
        
        # TODO [tobyb] small optimization: only calc target_qm if not_done is True
        target_q1, target_q2 = self.critic_target(next_state, next_action, options)
        target_qm = torch.min(target_q1, target_q2)

        target_q = (reward + not_done * self.discount * target_qm).detach()

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        metric['critic_loss'] = critic_loss.item()
        metric['Target_Q Mean'] = target_q.mean().item()
        
        return critic_loss

    def step(self):
        """
        steps the critic network
        """
        metric = {}
        
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            metric['Critic Grad Norm'] = critic_grad_norms.max().item()
            
        # update critic weights
        self.critic_optimizer.step()

        # update critic target weights
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # step the lr scheduler
        if self.lr_decay: 
            self.critic_lr_scheduler.step()

        return metric
    
    def LossActor(self, state, new_action, options):
        """
        Use the uncorrupted (a.k.a. ground truth) state and the denoised action from the actor for that state to obtain a predicted cumulative reward, convert it into a loss, and use it update the actor
        """
        q1_new_action, q2_new_action = self.critic(state, new_action, options)
        
        # flip a coin, randomly use q1 or q2
        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        loss = self.eta * q_loss
        
        return loss
        
    def Loss(self, nbatch_dict, new_action, next_action, task_id):
        
        options = self.MakeOptions(task_id)
        
        # train the critic, using (s, a, r, s') & a'
        critic_loss = self.LossCritic(nbatch_dict, next_action, options)
        
        # extract the state
        state = nbatch_dict['obs']
        
        # get the actor loss using (s, a)
        actor_loss = self.LossActor(state, new_action, options)
                
        # logging
        dd = {
            "dql_critic_loss": critic_loss,
            "dql_actor_loss": actor_loss  
              }
        globals.LOGGER.log(dd)
        
        return actor_loss, critic_loss

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
            
    def eval(self):
        self.critic.eval()
        
    def train(self):
        self.critic.train()
        
    def reset(self):
        self.critic_optimizer.zero_grad()
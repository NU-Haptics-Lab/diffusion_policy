
import torch
import diffusion_ql

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.diffusion.ema_model import EMAModel

"""
Based on this paper: https://arxiv.org/pdf/2208.06193
"""

class DiffusionQL(object):
    def __init__(self,
                 critic, # should be on device
                 discount,
                 tau,
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

        self.step = 0
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # EMA
        self.ema = EMAModel(self.critic)

        if lr_decay:
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.max_q_backup = max_q_backup
        
    def StepCritic(self, nbatch_dict, next_action):
        """
        samples - must be the same noised trajectories that were passed through the actor model (fcn: policy.compute_loss)
        new_action - should be the action calculated by using the predicted noise from the actor model.
        
        We need the g.t. state, action, reward, and next_state for QL training. We then get the predicted next_action as a fcn of next_state, and use that (next_action, next_state) tuple to get the predicted next_Q. We then add reward and next_Q to get the target, and we compare it to the Q-value to get a loss. This loss is used to train the Q-network.
        """

        metric = {}

        # Sample replay buffer / batch
        state = nbatch_dict['nobs']
        next_state = nbatch_dict['nobs_next']
        action = nbatch_dict['naction']
        reward = nbatch_dict['nreward']
        not_done = nbatch_dict['not_done']

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
        target_q1, target_q2 = self.critic_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)

        target_q = (reward + not_done * self.discount * target_q).detach()

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            metric['Critic Grad Norm'] = critic_grad_norms.max().item()
        self.critic_optimizer.step()

        """ Step Target network """
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.step += 1

        metric['critic_loss'] = critic_loss.item()
        metric['Target_Q Mean'] = target_q.mean().item()

        if self.lr_decay: 
            self.critic_lr_scheduler.step()

        return metric
    
    def StepActor(self, state, new_action):
        """
        
        Use the uncorrupted state and the denoise action from the actor for that state to obtain a predicted cumulative reward, convert it into a loss, and use it update the actor
        """
        metric = {}

        q1_new_action, q2_new_action = self.critic(state, new_action)
        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        loss = self.eta * q_loss
        
        # metrics
        metric['ql_loss'] = q_loss.item()
        
        return loss, metric
        
    def Step(self, nbatch_dict, new_action, next_action):
        # train the critic, using (s, a, r, s') & a'
        metric = self.StepCritic(nbatch_dict, next_action)
        
        # extract the state
        state = nbatch_dict['nobs']
        
        # train the actor using (s, a)
        loss, ametric = self.StepActor(state, new_action)
        
        metric.update(ametric)
        
        return loss, metric

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
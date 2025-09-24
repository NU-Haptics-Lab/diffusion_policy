
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
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):
        
        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # device transfer members, since I own them
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
        self.max_q_backup = max_q_backup
        
    def MakeOptions(self, task_id):
        options = {}
        options['leaf'] = task_id
        return options
        
    def LossCritic(self, nbatch_dict, next_action_TEST_UNUSED, options):
        """
        We need the g.t. state, action, reward, and next_state for QL training. 
        
        We then get the predicted next_action as a fcn of next_state, and use that (next_action, next_state) tuple to get the predicted next_Q. 
        
        We then add reward and next_Q to get the target (standard q-learning), and we compare it to the Q-value to get a loss. This loss is used to train the Q-network.
        """
        # Sample replay buffer / batch
        state = nbatch_dict['obs']
        next_state = nbatch_dict['obs_next']
        action = nbatch_dict['action']
        next_action = nbatch_dict['action_next']
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
        
        # TODO [tobyb] small optimization: only calc target_qm if not_done is True -- nvm because recall, these are all tensors of vectors (recall the batch dimension)
        target_q1, target_q2 = self.critic_target(next_state, next_action, options)
        target_qm = torch.min(target_q1, target_q2)

        target_q = (reward + not_done * self.discount * target_qm).detach()

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # TESTING
        # if reward.detach().to('cpu').numpy() > 0.0:
        #     pass
        
        return critic_loss
    
    def step_ema(self):
        # NOT NEEDED BECAUSE THIS IS ONLY FOR THE ACTOR, WHICH WE DO ELSEWHERE
        
        # if self.step % self.update_ema_every == 0 and self.step >= self.step_start_ema:
        #     self.ema.update_model_average(self.ema_model, self.actor)
        pass

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
            
        # weighted loss
        loss = q_loss
        
        return loss
        
    def Loss(self, nbatch_dict, new_action, next_action, task_id):
        """
        new_action - grad-full denoised observation using the actor
        next_action - grad-free denoised next observation using the actor
        task_id - task identifier
        
        new_action should be used to compute the DQL actor loss
        next_action should be used to compute the DQL critic loss
        """
        
        options = self.MakeOptions(task_id)
        
        # calc loss for the critic, using (s, a, r, s') & a'
        critic_loss = self.LossCritic(nbatch_dict, next_action, options)
        
        # extract the state
        state = nbatch_dict['obs']
        
        # get the actor loss using (s, a)
        actor_loss = 0.0 # self.LossActor(state, new_action, options)
                
        # logging
        dd = {
            task_id + ": dql_critic_loss": critic_loss,
            task_id + ": dql_actor_loss": actor_loss  
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
        
    def get_model(self):
        return self.critic
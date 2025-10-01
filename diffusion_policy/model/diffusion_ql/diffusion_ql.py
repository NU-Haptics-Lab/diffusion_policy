
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

class DiffusionQL(nn.Module):
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
                 use_actor = True
                 ):
        nn.Module.__init__(self)
        
        self.use_lr_decay = use_lr_decay
        self.grad_norm = grad_norm
        self.use_target_network = use_target_network
        self.use_double_q = use_double_q
        self.use_actor = use_actor

        self.critic = critic
        
        if self.use_target_network:
            self.critic_target = copy.deepcopy(self.critic)
            
        # set up the optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # device transfer members, since I own them
        device = torch.device(globals.CONFIG.device)
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
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr_min)

        self.discount = discount
        self.tau = tau
        self.max_q_backup = max_q_backup
        
    def MakeOptions(self, task_id):
        options = {}
        options['leaf'] = task_id
        return options
    
    def ForwardCritic(self, *args, **kwargs):
        if self.training:
            return self.critic(*args, **kwargs)
        else:
            return self.ema.averaged_model(*args, **kwargs)
        
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
        # if using a target network
        if self.use_target_network:
            target_network = self.critic_target
        else:
            target_network = self.critic
            
        # get the next q-value, Q(s', a')
        target_q1, target_q2 = target_network(next_state, next_action, options)
        
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
        dd[self.get_mode_string() + " mode. " + options['leaf'] + ": avg q-value"] = current_q1.mean()
        globals.LOGGER.log(dd)
        
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
        if self.use_target_network:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # step the lr scheduler
        if self.use_lr_decay: 
            self.critic_lr_scheduler.step()

        return metric
    
    def LossActor(self, state, new_action, options):
        """
        Use the uncorrupted (a.k.a. ground truth) state and the denoised action from the actor for that state to obtain a predicted cumulative reward, convert it into a loss, and use it update the actor
        """
        q1_new_action, q2_new_action = self.critic(state, new_action, options)
        
        # TODO: implement use_double_q flag
        
        # flip a coin, randomly use q1 or q2, and negate it to convert from a cumulative discounted reward to a loss
        # Fujimoto & Gu (2021), alpha = eta / E(s,a)∼D [ |Qϕ(s,a)| ]
        # https://arxiv.org/pdf/2106.06860 
        # the denominator is supposed to be a normalization term and NOT differentiated over
        # tensor.detach() excludes that term from the gradient calculation
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
        dd = {}
        
        options = self.MakeOptions(task_id)
        
        # training the critic
        if "critic" in globals.CONFIG.models_to_train:
            # calc loss for the critic, using (s, a, r, s') & a'
            critic_loss = self.LossCritic(nbatch_dict, next_action, options)
            
            # critic logging
            dd[self.get_mode_string() + " mode. " + task_id + ": dql_critic_loss"] = critic_loss
        else:
            critic_loss = None
        
        # extract the state
        state = nbatch_dict['obs']
        
        # training the actor
        if "actor" in globals.CONFIG.models_to_train:
            # get the actor loss using (s, a)
            actor_loss = self.LossActor(state, new_action, options)
            
            # actor logging
            dd[self.get_mode_string() + " mode. " + task_id + ": dql_actor_loss"] = actor_loss
        else:
            actor_loss = None
                
        # logging
        globals.LOGGER.log(dd)
        
        return actor_loss, critic_loss
    
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
        
    def get_model(self):
        return self.critic
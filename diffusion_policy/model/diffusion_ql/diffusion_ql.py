
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
                 use_actor = True,
                 use_tree = False,
                 action_relative_to_state = False,
                 use_denoise = False,
                 ):
        nn.Module.__init__(self)
        
        self.use_lr_decay = use_lr_decay
        self.grad_norm = grad_norm
        self.use_target_network = use_target_network
        self.use_double_q = use_double_q
        self.use_actor = use_actor
        self.use_tree = use_tree
        self.action_relative_to_state = action_relative_to_state
        self.use_denoise = use_denoise
        self.discount = discount
        self.tau = tau
        self.max_q_backup = max_q_backup
        self.lr = lr
        self.lr_min = lr_min
        self.lr_maxt = lr_maxt

        self.critic = critic
        
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
        
        if self.use_tree:
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
        
        # if using a target network
        if self.use_target_network:
            target_network = self.critic_target
        else:
            target_network = self.critic
            
        # get the next q-value, Q(s', a')
        target_q1, target_q2 = target_network(next_state, next_action, options)
        
        # if using double-q learning
        if self.use_double_q:
            # this will pose issues when q-values are negative ?
            # actually maybe that's intended. it's a pessimistic q-value
            target_qm = torch.min(target_q1, target_q2)
        else:
            target_qm = target_q1

        # compute the bellman equation
        target_q = (reward + not_done * self.discount * target_qm).detach()

        # compute the loss
        critic_loss = F.mse_loss(current_q1, target_q, reduction="none")
        
        # if using double-q learning
        if self.use_double_q:
            critic_loss = critic_loss + F.mse_loss(current_q2, target_q, reduction="none")
        
        # logging
        dd = {}
        if self.use_tree:
            dd["qval/" + self.get_mode_string() + " mode. " + options['leaf'] + ": avg q-value"] = current_q1.mean()
        else:
            dd["qval/" + self.get_mode_string() + ": avg q-value"] = current_q1.mean()
        if globals.LOGGER is not None:
            globals.LOGGER.log(dd)
        
        if reward.mean() > 0.0:
            pass
        
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
    
    def LossActor(self, state, new_action, options,
                  a0 = None,
                  timesteps = None,
                  ):
        """
        Use the uncorrupted (a.k.a. ground truth) state and the denoised action from the actor for that state to obtain a predicted cumulative reward, convert it into a loss, and use it update the actor
        """
        if self.use_denoise:
            actions = new_action
            assert(actions is not None)
        else:
            assert(a0 is not None)
            actions = a0
        
        
        if self.use_target_network:
            q1_new_action, q2_new_action = self.critic_target(state, actions, options)
        else:
            q1_new_action, q2_new_action = self.critic(state, actions, options)
        
        # if not using denoising, and using the BC a0 instead, then we should scale the actor loss as a function of the timesteps. That way, samples closer to real actions have higher weighting since we'd expect them to be more accurate, and samples closer to the noise have less weighting as we'd expect them to be less accurate.
        if False:
            if not self.use_denoise:
                assert(timesteps is not None)
                # reshape
                t = torch.reshape(timesteps, q1_new_action.shape)
                
                # add 1 to t to protect from division by zero
                q1_new_action = q1_new_action / (t + 1)
                
                if self.use_double_q:
                    q2_new_action = q2_new_action / (t + 1)
        
        
        # TODO: implement use_double_q flag
        
        # flip a coin, randomly use q1 or q2, and negate it to convert from a cumulative discounted reward to a loss
        # Fujimoto & Gu (2021), alpha = eta / E(s,a)∼D [ |Qϕ(s,a)| ]
        # https://arxiv.org/pdf/2106.06860 
        # the denominator is supposed to be a normalization term and NOT differentiated over
        # tensor.detach() excludes that term from the gradient calculation
        if self.use_double_q:
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        else:
            q_loss = - q1_new_action.mean()
            
        # weighted loss
        loss = q_loss
        
        return loss
    
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
        
    def Loss(self, nbatch_dict, new_action, next_action, task_id,
             a0 = None, timesteps = None
             ):
        """
        new_action - grad-full denoised observation using the actor
        next_action - grad-free denoised next observation using the actor
        task_id - task identifier
        a0 - grad-full diffusion policy outputs, replaces new_action
        
        new_action should be used to compute the DQL actor loss
        next_action should be used to compute the DQL critic loss
        """
        dd = {}
        if self.action_relative_to_state:
            nbatch_dict = self.ActionRelativeToState(nbatch_dict)
            # a0?
        
        options = self.MakeOptions(task_id)
        models_to_train: list = globals.CONFIG.models_to_train #type:ignore
        
        # training the critic
        if "critic" in models_to_train:
            # calc loss for the critic, using (s, a, r, s') & a'
            critic_loss = self.LossCritic(nbatch_dict, next_action, options)
            
            # critic logging
            dd[self.get_mode_string() + " mode. " + task_id + ": dql_critic_loss"] = critic_loss
        else:
            critic_loss = None
        
        # extract the state
        state = nbatch_dict['obs']
        
        # training the actor
        if (new_action is not None or a0 is not None) and utils.GlobalStepFreqTrigger('sac_actor'):
            # get the actor loss using (s, a)
            actor_loss = self.LossActor(state, new_action, options, a0, timesteps)
            
            # actor logging
            dd[self.get_mode_string() + " mode. " + task_id + ": dql_actor_loss"] = actor_loss
        else:
            actor_loss = None
                
        # logging
        if globals.LOGGER is not None:
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
        
    def get_model(self, want_target_network=False):
        if want_target_network and self.use_target_network:
            return self.critic_target
        else:
            return self.critic
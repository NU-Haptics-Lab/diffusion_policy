import numpy as np

import torch
from torch import nn
from typing import Any
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import hydra
from diffusion_policy.model.diffusion_ql.diffusion_ql import DiffusionQL
import diffusion_policy.globals as globals
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


import diffusion_policy.globals as globals
from diffusion_policy.model.model import ModelEmaOptim
from diffusion_policy.model.diffusion_model import DiffusionModel
from diffusion_policy import utils

"""
reference paper: https://arxiv.org/pdf/2208.06193

modes:
a0 - as described in "DIFFUSION POLICIES AS AN EXPRESSIVE POLICY CLASS FOR OFFLINE REINFORCEMENT LEARNING", "the end sample of the reverse chain, a0, is the action used for RL evaluation". So a0 is the iteratively denoised actions

a0_single - single step diffusion, so a0_single = a + policy(a, step)
"""

"""
in policy.compute_loss, an input batch of trajectories is noised (different noise level per sample), the noisy trajectories are passed through the model, and then the output, 'pred', is the predicted noise which is compared against the actual noise that was added to the original non-noisy batch.
So for us, 'a0_single' simply takes the predicted noise and adds it to the noised trajectories, and that is a0 from a single step. 'a0_single' is actually the case where you use DDIM to denoise with a step value of 1 I think, so maybe I'll just include that instead
"""

"""
another decision to make: which action to start with. We pass partially noised actions into the diffusion policy and get out 1-pass noise values, which are then compared against the true noise and used to calculate a BC loss. But for the QL loss we don't need to use those same noised actions. We could just start from nothing and generate the actions from scratch using predict_action. Easiest to implement. This is also what the repo for "DIFFUSION POLICIES AS AN EXPRESSIVE POLICY CLASS FOR OFFLINE REINFORCEMENT LEARNING" does.

"""

class CriticLoss(nn.Module):
    def __init__(self,
                 critic: DiffusionQL,
                 noise_scheduler: DDIMScheduler,
                 num_inference_steps: int,
                 use_random_num_inference_steps = False,
                 randomizer_fcn = False,
                 use_denoise = True, # whether to denoise using a scheduler, or to use the BC noise vector (many more samples)
                 ) -> None:
        nn.Module.__init__(self)
        """
        actor - an actor policy
        critic - a critic policy
        """
        self.critic = critic
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps
        self.use_random_num_inference_steps = use_random_num_inference_steps
        self.randomizer_fcn = lambda : np.random.randint(int(0.5*self.num_inference_steps), int(2.0*self.num_inference_steps))
        self.use_denoise = use_denoise
        
        # the same as a 1-step denoise
        if not self.use_denoise:
            self.num_inference_steps = 1
        
        # set the noise scheduler time steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        
    def Denoise(self, nobs_dict, use_ema=False, task_id=None):
        """
        Should use the regular model if doing training
        Should use the ema model if doing inference, or doing critic training
        """
        # save handles to nodes
        actor: ModelEmaOptim = globals.MODELS["actor"] #type:ignore
        
        # setup inference steps
        if self.use_random_num_inference_steps:
            steps = self.randomizer_fcn()
            self.noise_scheduler.set_timesteps(steps)
        
        m: DiffusionModel
        if use_ema:
            m = actor.get_ema_model()
        else:
            m = actor.get_model()

        # None protection, also for pylance
        assert(m is not None)
        
        nresult = m.denoise(
            nobs_dict, 
            self.noise_scheduler, 
            task_id=task_id
            )
        naction_pred = nresult['naction_pred']
        return naction_pred
        
    def loss(self, nbatch, task_id, 
             a0 = None
             ):
        """
        nbatch - normalized batch dictionary with keys: nobs, naction, nreward, <...>
        a0 - diffusion policy outputs, [B, noisy-samples?]
        
        Take the current state, run it through the actor to get actions, then run the (state, action) tuple through the critic to get a predicted cumulative reward, then compute a loss. This method returns that loss.
        
        [TODO]
        This is exactly the same as the Diffusion-QL repo, the downside however is that we have to do denoising twice per step, once on this state and once on the next state. If this proves to be very slow to train, then a potential optimization is to do critic training on the previous state & this state so we only have to denoise once per step.
        """
        # TODO: add feature for training the critic separately, so no denoising is needed
        models_to_train: list = globals.CONFIG.models_to_train #type:ignore
        
        # training the actor, so we need to use the actor to denoise an observation
        # also a0 must be none so we must denoise
        if "actor" in models_to_train and utils.StepFreqTrigger(globals.CONFIG.step_freqs['actor']) and a0 is None:
            # want gradients on this one so we can back-prop from the critic through to the actor
            new_action = self.Denoise(nbatch['obs'], task_id=task_id)
        else:
            new_action = None
            
        # only need to denoise obs_next if we're using a SARS setup instead of SARSA
        if False:
            # no grad because this is only used in the dql critic update which won't be back-propagated to the actor
            with torch.no_grad():
                next_action = self.Denoise(nbatch['obs_next'], use_ema=True)
                    
        # returns loss, metric
        dql_actor_loss, dql_critic_loss = self.critic.Loss(nbatch, new_action, None, task_id, a0=a0)
        
        return dql_actor_loss, dql_critic_loss
        
    def step(self):
        # I'm only in charge of the critic
        self.critic.step()
        
    def reset(self):
        self.critic.reset()
        
    def get_model(self):
        return self.critic.get_model()
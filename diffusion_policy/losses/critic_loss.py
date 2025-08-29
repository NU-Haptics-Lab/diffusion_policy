import torch
from typing import Any
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import hydra
from diffusion_policy.model.diffusion_ql.trainer import DiffusionQL
import diffusion_policy.globals as globals
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


import diffusion_policy.globals as globals

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

class CriticLoss:
    def __init__(self,
                 critic: DiffusionQL,
                 noise_scheduler: DDIMScheduler,
                 num_inference_steps: int
                 ) -> None:
        """
        actor - an actor policy
        critic - a critic policy
        """
        self.critic = critic
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps
        
        # set the noise scheduler time steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        
    def Denoise(self, nobs_dict):
        # save handles to nodes
        actor = globals.MODELS["actor"]
        
        # use actor or ema actor? idk
        nresult = actor.denoise(
            nobs_dict, 
            self.noise_scheduler, 
            )
        naction_pred = nresult['naction_pred']
        return naction_pred
        
    def loss(self, nbatch, task_id):
        """
        nbatch - normalized batch dictionary with keys: nobs, naction, nreward, <...>
        
        Take the current state, run it through the actor to get actions, then run the (state, action) tuple through the critic to get a predicted cumulative reward, then compute a loss. This method returns that loss.
        
        [TODO]
        This is exactly the same as the Diffusion-QL repo, the downside however is that we have to do denoising twice per step, once on this state and once on the next state. If this proves to be very slow to train, then a potential optimization is to do critic training on the previous state & this state so we only have to denoise once per step.
        """
        action = self.Denoise(nbatch['obs'])
        
        # TODO: get the next action from the ema model, same as the Diffusion-QL repo
        # no grad because this is only used in the dql critic update
        with torch.no_grad():
            action_next = self.Denoise(nbatch['obs_next'])
        
        # returns loss, metric
        dql_actor_loss, dql_critic_loss = self.critic.Loss(nbatch, action, action_next, task_id)
        
        return dql_actor_loss, dql_critic_loss
        
    def step(self):
        # I'm only in charge of the critic
        self.critic.step()
        
    def reset(self):
        self.critic.reset()
        
    def eval(self):
        self.critic.eval()
        
    def train(self):
        self.critic.train()
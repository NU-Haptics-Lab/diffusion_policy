from typing import Any
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import hydra
import diffusion_ql

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

class QLTrainer:
    def __init__(self,
                 actor,
                 ema_actor,
                 critic: diffusion_ql.DiffusionQL,
                 cfg
                 ) -> None:
        """
        actor - an actor policy
        critic - a critic policy
        """
        self.actor = actor
        self.ema_actor = ema_actor
        self.critic = critic
        
        # setup the noise scheduler
        cls = hydra.utils.get_class(cfg.qloss.noise_scheduler._target_)
        scheduler = cls(cfg.qloss.noise_scheduler)
        
        scheduler.set_timesteps(cfg.qloss.noise_scheduler.num_inference_steps)
        
        self.noise_scheduler = scheduler
        
    def Denoise(self, nobs_dict):
        nresult = self.actor.predict_action_impl(
            nobs_dict, 
            self.noise_scheduler, 
            self.noise_scheduler.timesteps
            )
        naction_pred = nresult['naction_pred']
        return naction_pred
        
    def __call__(self, nbatch):
        """
        nbatch - normalized batch dictionary with keys: nobs, naction, nreward, <...>
        
        Take the current state, run it through the actor to get actions, then run the (state, action) tuple through the critic to get a predicted cumulative reward, then compute a loss. This method returns that loss.
        
        [TODO]
        This is exactly the same as the Diffusion-QL repo, the downside however is that we have to do denoising twice per step, once on this state and once on the next state. If this proves very slow to train, then a potential optimization is to do critic training on the previous state & this state so we only have to denoise once per step.
        """
        new_action = self.Denoise(nbatch['nobs'])
        
        # get the next action from the ema model, same as the Diffusion-QL repo
        next_action = self.Denoise(nbatch['nobs_next'])
        
        # returns loss, metric
        return self.critic.Step(nbatch, new_action, next_action)
        
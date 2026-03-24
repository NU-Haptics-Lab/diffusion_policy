"""
Core algorithm taken from sb3.PPO. Default values for parameters also from sb3.PPO

Use the sb3 ActorCriticPolicy class, which contains all the necessary methods and an optimizer


"""
import stable_baselines3 as sb3

from stable_baselines3.common.utils import explained_variance

from stable_baselines3.common.policies import ActorCriticPolicy

import numpy as np

import torch as th
import torch.nn.functional as F

import diffusion_policy.globals as globals


from diffusion_policy.model.model_and_optim import ModelandOptim

class PPO:
    def __init__(self,
                 policy: ActorCriticPolicy,
                 batch_size,
                 ent_coef = 0.0,
                 vf_coef = 0.5,
                 target_kl = None,
                 verbose = 0,
                 clip_range_vf = None,
                 normalize_advantage = True,
                 clip_range = 0.2,
                 
    ):
        self.policy = policy
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.target_kl = target_kl
        self.verbose = verbose
        self.clip_range_vf = clip_range_vf
        self.batch_size = batch_size
        self.normalize_advantage = normalize_advantage
        self.clip_range = clip_range
        
    # def get_current_progress_remaining(self):
    #     """
    #     Get the current training progress remaining (from 1 to 0)
    #     """
    #     return 1.0 - (globals.EPOCH / globals.CONFIG.total_num_epochs)
        
    def loss(self, nbatch: dict):
        """
        should be called by a batch loss class
        """
        # Compute current clip range
        clip_range = self.clip_range
        
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf
        else:
            clip_range_vf = th.inf
            
        # extract batch data
        nobs = nbatch['obs']
        nactions = nbatch['action']
        advantages = nbatch['advantages']
        old_log_prob = nbatch['old_log_prob']
        old_values = nbatch['old_values']
        returns = nbatch['returns']
        rb_values = nbatch['rb_values']
        
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]


        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        approx_kl_divs = []


        values, log_prob, entropy = self.policy.evaluate_actions(nobs, nactions)
        values = values.flatten()


        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(log_prob - old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # Logging
        pg_losses.append(policy_loss.item())
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        clip_fractions.append(clip_fraction)

        if self.clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the difference between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = old_values + th.clamp(
                values - old_values, -clip_range_vf, clip_range_vf
            )
        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(returns, values_pred)
        value_losses.append(value_loss.item())

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        entropy_losses.append(entropy_loss.item())

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with th.no_grad():
            log_ratio = log_prob - old_log_prob
            approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
            approx_kl_divs.append(approx_kl_div)

        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
            if self.verbose >= 1:
                print(f"Early stopping at step {globals.EPOCH} due to reaching max kl: {approx_kl_div:.2f}")
                
            # reset loss to nothing, grad = True for compat
            loss = th.tensor(0.0).requires_grad_(True)


        explained_var = explained_variance(rb_values.flatten(), returns.flatten())
        
        globals.log_one_if_exists("ppo/entropy_loss", np.mean(entropy_losses))
        globals.log_one_if_exists("ppo/policy_gradient_loss", np.mean(pg_losses))
        globals.log_one_if_exists("ppo/value_loss", np.mean(value_losses))
        globals.log_one_if_exists("ppo/approx_kl", np.mean(approx_kl_divs))
        globals.log_one_if_exists("ppo/clip_fraction", np.mean(clip_fractions))
        globals.log_one_if_exists("ppo/loss", loss.item())

        globals.log_one_if_exists("ppo/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            globals.log_one_if_exists("ppo/std", th.exp(self.policy.log_std).mean().item())
        globals.log_one_if_exists("ppo/clip_range", clip_range)
        if self.clip_range_vf is not None:
            globals.log_one_if_exists("ppo/clip_range_vf", clip_range_vf)
            
        # we're done
        return loss
"""
SimpleInference extracted from eval.py to avoid circular imports
caused by eval.py's heavy top-level dependencies (moveit, BaseImagePolicy, etc.).
"""

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.common import pytorch_util
from diffusion_policy.model.diffusion_model import DiffusionModel
from diffusion_policy import globals
from diffusion_policy import utils


class SimpleInference:
    """
    no state history
    """
    def __init__(self,
                 policy: DiffusionModel,
                 task_id=None,
                 ) -> None:
        self.policy = policy
        self.task_id = task_id

        # my members
        self.current_obs: dict = None #type:ignore


    def setup(self, skip_policy_setup: bool = False):
        if self.policy is not None and not skip_policy_setup:
            self.policy.setup()

        """
        must be called after all nodes are made
        """
        if self.task_id is not None:
            self.task_id = torch.tensor([[self.task_id]], device=globals.CONFIG.device) # 2d #type:ignore

        # save a handle
        self.batch_loader = globals.DEFAULT_BATCH_LOADER

        # make the scheduler, hard-coded except prediction_type which callers
        # may override via globals.CONFIG.prediction_type before calling setup()
        self.noise_scheduler = DDIMScheduler(
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            beta_start=0.0001,
            clip_sample=True,
            num_train_timesteps=globals.CONFIG.common_noise_scheduler.num_train_timesteps, # type:ignore
            prediction_type=getattr(globals.CONFIG.common_noise_scheduler, "prediction_type", "epsilon"),
        )
        self.noise_scheduler.set_timesteps(globals.CONFIG.num_inference_steps) # type:ignore

        self.original_policy_noise_scheduler = None


    def set_policy(self, policy):
        self.policy = policy
        self.original_policy_noise_scheduler = self.policy.noise_scheduler

    def save_obs(self, obs):
        self.current_obs = obs


    def norm_gpu_obs(self, obs_dict_np):
        # wrap in a data dict, which is what the batch loader expects
        dd = {"obs": obs_dict_np}

        dd_torch = pytorch_util.dict_to_torch(dd)

        # must normalize ourselves, here, because of the way co-training works with separate normalizers per dataset
        ndd_torch = self.batch_loader.transfer_and_norm(dd_torch)

        nobs_torch = ndd_torch['obs']

        return nobs_torch

    def unnorm_cpu_action(self, naction_gpu):
        naction_gpu_dd = {"action": naction_gpu}

        action_dd = self.batch_loader.unnorm_and_transfer(naction_gpu_dd)

        action_cpu = action_dd['action']

        return action_cpu

    def norm_gpu_action(self, action_cpu):
        action_cpu_dd = {"action": action_cpu}

        naction_dd = self.batch_loader.transfer_and_norm(action_cpu_dd)

        naction_gpu = naction_dd['action']

        return naction_gpu

    def GetObs(self):
        obs = self.current_obs

        if obs is None:
            return None

        # get obs keys to use
        obs_keys_to_use: list = globals.CONFIG.obs_keys_to_use #type:ignore

        # patch-token groups (e.g. DINOv3 patches) bypass obs_encoder entirely and
        # are read straight out of nobs by DiffusionModel._encode_patch_groups, so
        # they're deliberately excluded from obs_keys_to_use (obs_encoder would
        # flatten and destroy the grid) but still must survive this filtering step.
        patch_keys_to_use: list = list(getattr(self.policy, "patch_keys_to_use", []) or [])

        keys_to_keep = obs_keys_to_use + patch_keys_to_use

        # confirm all keys are in obs
        for key in keys_to_keep:
            assert(key in obs)

        # only keep the ones in obs_keys_to_use / patch_keys_to_use
        obs = {key: obs[key] for key in keys_to_keep}

        for key in keys_to_keep:
            val = obs[key]
            if isinstance(val, np.ndarray):
                # only the action dim
                if val.ndim == 1:
                    # add batch and traj dim
                    val2 = np.expand_dims(val, axis=(0,1))

                # batch and action dim
                elif val.ndim == 2:
                    # add a traj dim
                    val2 = np.expand_dims(val, axis=1)
                else:
                    val2 = val

                # save it
                obs[key] = val2

        # done
        return obs


    def RunInference(self, obs_dict_np, progress=None, progress_valid=None):

        # run inference
        with torch.no_grad():
            nobs_torch = self.norm_gpu_obs(obs_dict_np)

            device = next(self.policy.model.parameters()).device
            progress_t = None
            progress_valid_t = None
            if progress is not None:
                # (1,1) -- same convention as self.task_id, batch size 1 for ROS inference
                progress_t = torch.tensor([[progress]], dtype=torch.float32, device=device)
            if progress_valid is not None:
                progress_valid_t = torch.tensor([[progress_valid]], dtype=torch.float32, device=device)

            # inside predict_action -> conditional_sample is where the iteration occurs. `for t in scheduler.timesteps`
            naction_gpu, all_nactions_gpu = self.policy.infer(
                nobs_torch, task_id=self.task_id,
                progress=progress_t, progress_valid=progress_valid_t)

            future_actions = self.unnorm_cpu_action(naction_gpu)
            all_actions = self.unnorm_cpu_action(all_nactions_gpu)

            return future_actions, all_actions

    def infer(self, progress=None, progress_valid=None):
        """
        progress: fraction through the episode (0..1), or None if this
            checkpoint wasn't trained with a progress token.
        progress_valid: whether `progress` is meaningful for this rollout
            (e.g. False for sub-task-only checkpoints) -- see
            Honda1Sampler/DexNexTransformerForDiffusion's use_progress_token.
        """
        # get observation
        obs_dict_np = self.GetObs()

        if obs_dict_np is None:
            print("No obs yet.")
            return None, None

        # replace the policy's scheduler for inference
        device = next(self.policy.model.parameters()).device
        self.noise_scheduler.timesteps = self.noise_scheduler.timesteps.to(device)
        self.policy.noise_scheduler = self.noise_scheduler

        # run inference
        tic = utils.tic()
        action, all_actions = self.RunInference(obs_dict_np, progress=progress, progress_valid=progress_valid)
        toc = utils.toc(tic)
        globals.log_one_if_exists("profiling/SimpleInference.RunInference", toc)

        # put the original back in for training (only if it was saved — not used in ROS2 inference flow)
        if self.original_policy_noise_scheduler is not None:
            self.policy.noise_scheduler = self.original_policy_noise_scheduler

        # back to numpy
        action = action.numpy()
        all_actions = all_actions.numpy()

        # set nans to zero, print a warning
        if np.isnan(all_actions).any():
            print("Warning: NaN values found in actions. Setting NaNs to zero.")
            all_actions = np.where(np.isnan(all_actions), 0.0, all_actions)
            action = np.where(np.isnan(action), 0.0, action)

        return action, all_actions

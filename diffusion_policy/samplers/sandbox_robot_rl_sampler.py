from typing import Optional
import numpy as np
import numba
import scipy.stats
import hydra
from diffusion_policy.common.replay_buffer import ReplayBuffer
import diffusion_policy.globals as globals
from diffusion_policy import utils
from collections import defaultdict
from tqdm import tqdm

import copy
from omegaconf import OmegaConf, open_dict
from operator import itemgetter

import diffusion_policy.globals as globals

from diffusion_policy.common.sarsa_sampler import DatasetSampler, EpisodeSampler

import diffusion_policy.globals as globals

class SandboxRobotRLEpisodeSampler(EpisodeSampler):
    """
    Get a sample from an episode for the object manipulation policy.
    
    mapping: (state, object id, object target pose) --> Traj[actions].
    """
    
    def get_action_sample(self, ep_idx):
        """
        For an action, we want a sequence from ep_idx - n_obs_steps to ep_idx + horizon.
        """
        action_key = globals.CONFIG.action_key # type: ignore
        
        # make indices which are episode-relative
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx # type:ignore
        
        # get the sample
        sample = self.indices.get_sequence_by_train_indices_and_key(indices, action_key)
        
        return sample
    
    def get_obs_sample(self, ep_idx):
        """
        """
        obs_keys_to_load: list = globals.CONFIG.obs_keys_to_load # type: ignore
        
        
        obs_sample = {}
        
        # assemble the obs sample
        for obs_key in obs_keys_to_load: # type:ignore
            obs_sample[obs_key] = self.get_key_sample(obs_key, ep_idx)
        
        return obs_sample
    

    
    def get_sample(self, ep_idx):
        """
        for RL, we need obs, obs_next, action, action_next, reward, and not_done. (s, a, r) will be in the R.B., not_done is computed based on the episode length.
        """
        assert(ep_idx >= 0)
        # assert(ep_idx <= len(self)-2) # must be -2 since we get the next obs & action
        assert(ep_idx <= len(self)-1) # allow ep_idx to be == len(self)-1. In that case, obs_next will be a repeat of obs
        
        H = len(globals.CONFIG.action_rel_indices) # type: ignore
        
        sample = {}
        sample["obs"] = self.get_obs_sample(ep_idx)
        sample["obs_next"] = self.get_obs_sample(ep_idx + H)
        
        sample["action"] = self.get_action_sample(ep_idx)
        sample["action_next"] = self.get_action_sample(ep_idx + H)
        
        
        sample["reward"] = self.get_reward(ep_idx)
        sample["not_done"] = self.get_not_done(ep_idx)


        return sample
    
class SandboxRobotActionOnlyEpisodeSampler(SandboxRobotRLEpisodeSampler):
    
    def get_sample(self, ep_idx):
        """
        for RL, we need obs, obs_next, action, action_next, reward, and not_done. (s, a, r) will be in the R.B., not_done is computed based on the episode length.
        """
        assert(ep_idx >= 0)
        # assert(ep_idx <= len(self)-2) # must be -2 since we get the next obs & action
        assert(ep_idx <= len(self)-1) # allow ep_idx to be == len(self)-1. In that case, obs_next will be a repeat of obs
        
        sample = {}
        
        sample["action"] = self.get_action_sample(ep_idx)
        

        return sample
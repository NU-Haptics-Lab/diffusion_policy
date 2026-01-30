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


class ManipAnythingEpisodeSampler(EpisodeSampler):
    """
    Get a sample from an episode for the object manipulation policy.
    
    mapping: (state, object id, object target pose) --> Traj[actions].
    """
    
    def get_action_sample(self, ep_idx):
        """
        For an action, we want a sequence from ep_idx - n_obs_steps to ep_idx + horizon.
        """
        # make indices which are episode-relative
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx # type:ignore
        
        # ensure trajectory doesn't go past the object target pose timestamp
        is_past = indices > self.target_idx
        is_past_indices = np.where(is_past)[0]
        
        if len(is_past_indices) > 0:
            # get the integer element
            last_valid_idx = is_past_indices[0]
        
            # past-fill any indices after the target idx
            indices[last_valid_idx:] = self.target_idx
        
        # get the sample
        sample = self.indices.get_sequence_by_train_indices_and_key(indices, "action")
        
        # reset to ensure we aren't using the previous target idx
        self.target_idx = None
        
        return sample
    
    def get_obs_sample(self, ep_idx):
        """
        randomly sample an object pose between ep_idx and the end of the episode, and set that as the target pose.
        
        data-pt keys:
        datapt = {
            'state': state, FK, and haptics,
            'object_id': ,
            'object_target_pose': ,
            'action': ,
        }
        """
        # randomly sample the target idx -- episode-relative
        target_idx = np.random.randint(ep_idx, len(self))
        
        # save it for use in get_action_sample
        self.target_idx = target_idx
        
        # get target pose
        target_pose = self.get_key_sample("object_target_pose", target_idx)
        
        object_id = self.get_key_sample("object_id", ep_idx),
        
        # get the state components
        state = self.get_key_sample("state", ep_idx)
        
        # default output dict
        default_obs_sample = {
            'state': state,
            'object_id': object_id,
            'object_target_pose': target_pose,
        }
        
        # actual
        obs_sample = {}
        
        # assemble the obs sample
        for obs_keys in globals.CONFIG.obs_keys_to_load: # type:ignore
            obs_sample[obs_keys] = default_obs_sample[obs_keys]
        
        return obs_sample
    

    
    def get_sample(self, ep_idx):
        """
        similar to its parent, but no _next obs nor action
        """
        assert(ep_idx >= 0)
        # assert(ep_idx <= len(self)-2) # must be -2 since we get the next obs & action
        assert(ep_idx <= len(self)-1) # allow ep_idx to be == len(self)-1. In that case, obs_next will be a repeat of obs
        
        sample = {}
        sample["obs"] = self.get_obs_sample(ep_idx)
        
        sample["action"] = self.get_action_sample(ep_idx)

        sample["reward"] = self.get_reward(ep_idx)

        sample["not_done"] = self.get_not_done(ep_idx)

        ## Meta data
        sample["task_id"] = self.get_task_id(ep_idx)
        
        # explicit q-val
        sample["qval"] = self.get_qval(ep_idx)
        
        ## My Add-ons
        # ep len
        sample["rb_index"] = self.get_rb_index(ep_idx)
        sample["ep_len"] = np.array([len(self)])

        return sample
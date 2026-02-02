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


class MageHandEpisodeSampler(EpisodeSampler):
    """
    Get a sample from an episode for the object manipulation policy.
    
    mapping: (state, object id, object target pose) --> Traj[actions].
    """
    def get_mage_hand_action_trajectory(self, ep_idx, key):
        """
        For an action, we want a sequence from ep_idx - n_obs_steps to ep_idx + horizon.
        """
        # safety assertion
        assert(self.target_idx is not None)
        
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
        
        # get the sample. actions are just future states
        key_action = self.indices.get_sequence_by_train_indices_and_key(indices, key)
        
        return key_action
        
    def get_joint_action_sample(self, ep_idx):
        joint_action = self.get_mage_hand_action_trajectory(ep_idx, "joint_state")
        
        return joint_action
    
    def get_rel_pose_action_sample(self, ep_idx):
        # action is the same as the state
        pose_action = self.get_mage_hand_action_trajectory(ep_idx, "pose_state")
        
        # make relative to pose at ep_idx
        ref_pose = self.get_key_sample("pose_state", ep_idx)
        
        rel_pose_action = utils.drake_compute_rel_pose(
            src_pose=ref_pose,
            dst_pose=pose_action
        )
        
        return rel_pose_action
    
    def get_rel_object_target_pose(self, ep_idx, target_idx):
        # get target pose
        object_target_pose = self.get_key_sample("object_target_pose", target_idx)
        
        # make the relative object target pose
        pose_state = self.get_key_sample("pose_state", ep_idx)
        
        # compute rel pose
        rel_object_target_pose = utils.drake_compute_rel_pose(
            src_pose=pose_state,
            dst_pose=object_target_pose
        )
        
        return rel_object_target_pose
    
    def get_rel_object_target_pos(self, ep_idx, target_idx):
        rel_pose = self.get_rel_object_target_pose(ep_idx, target_idx)
        
        rel_pos = rel_pose[4:7]
        return rel_pos
        
    def get_all_rel_object_target_pose(self):
        all_rel_object_poses = []
        
        # iterate over each ep idx
        # for ep_idx in range(len(self)):
        # using tqdm
        for ep_idx in tqdm(range(len(self)), desc="Computing rel object target poses"):
            
            # iterate over ep_idx + 1 to end of episode
            # for target_idx in range(ep_idx + 1, len(self)):
            # using tqdm
            for target_idx in tqdm(range(ep_idx + 1, len(self)), desc="Computing rel object target poses (inner)"):
                # set the target idx
                # get rel object target pose
                rel_object_target_pose = self.get_rel_object_target_pose(ep_idx, target_idx)
                
                # append
                all_rel_object_poses.append(rel_object_target_pose)
                
        # stack
        out = np.stack(all_rel_object_poses, axis=0)
        
        # reshape to 2d
        out2 = out.reshape(-1, 7)
        
        return out2
    
    def get_all_rel_object_target_pos(self):
        all_rel_object_target_pos = self.get_all_rel_object_target_pose()
        
        # only take the position part
        all_rel_object_target_pos = all_rel_object_target_pos[:, 4:7]
        
        return all_rel_object_target_pos
    
    def get_all_rel_pose_actions(self):
        all_rel_pose_actions = []
        
        # iterate over each ep idx
        # for ep_idx in range(len(self)):
        # using tqdm for progress bar
        for ep_idx in tqdm(range(len(self)), desc="Computing rel pose actions (outer)"):
            
            # iterate over ep_idx + 1 to end of episode
            # for target_idx in range(ep_idx + 1, len(self)):
            # using tqdm for progress bar
            for target_idx in tqdm(range(ep_idx + 1, len(self)), desc="Computing rel pose actions (inner)"):
                # set the target idx
                self.target_idx = target_idx
                
                # get rel pose action
                rel_pose_action = self.get_rel_pose_action_sample(ep_idx)
                
                # append
                all_rel_pose_actions.append(rel_pose_action)
        
        # since rel_pose_action is already 2d, we have to vstack instead of convert to array
        out = np.vstack(all_rel_pose_actions)
    
        return out
    
    def get_obs_sample(self, ep_idx):
        """
        randomly sample an object pose between ep_idx and the end of the episode, and set that as the target pose.
        
        data-pt keys:
        
        datapt = {
            'pose_state': pose_state,
            'joint_state': joint_state,
            'aux_state': aux_state,
            'object_id': 0, # from objects.yaml
            'object_target_pose': object_target_pose,
            'not_done': True,
            'reward': 0.0,
            'task_id': 0, # no task id
            'qval': 0.0,
        }
        """
        # randomly sample the target idx -- episode-relative
        target_idx = np.random.randint(ep_idx, len(self))
        
        # save it for use in get_action_sample
        self.target_idx = target_idx
        
        # get the state components
        joint_state = self.get_key_sample("joint_state", ep_idx)
        
        # haptics
        haptics = self.get_key_sample("haptics", ep_idx)
        
        # rel-fk
        rel_fk = self.get_key_sample("rel_fk", ep_idx)
        
        rel_object_target_pose = self.get_rel_object_target_pose(ep_idx, target_idx)
        
        # default output dict
        default_obs_sample = {
            'joint_state': joint_state,
            'haptics': haptics,
            'rel_fk': rel_fk,
            'rel_object_target_pose': rel_object_target_pose,
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
        
        ## observations
        sample["obs"] = self.get_obs_sample(ep_idx)
        
        ## actions
        sample["joint_action"] = self.get_joint_action_sample(ep_idx)
        
        sample["rel_pose_action"] = self.get_rel_pose_action_sample(ep_idx)

        ## reward & not done
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

        
        # reset to ensure we don't use the previous target idx next time
        self.target_idx = None
        return sample
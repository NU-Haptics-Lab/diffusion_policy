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

import math


class MageHandEpisodeSampler(EpisodeSampler):
    """
    Get a sample from an episode for the object manipulation policy.
    
    mapping: (state, object id, object target pose) --> Traj[actions].
    """
    def total_unique_datapoints(self):
        """ formula is ep_len**2/2 """
        ep_len = len(self)
        total = ep_len**2/2
        return total
    
    def get_post_target_indices(self, ep_idx):
        # make indices which are episode-relative
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx # type:ignore
        
        is_past = indices > self.target_idx
        is_past_indices = np.where(is_past)[0]
        
        return is_past_indices
    
    def get_mage_hand_action_trajectory(self, ep_idx, key):
        """
        For an action, we want a sequence
        """
        # safety assertion
        assert(self.target_idx is not None)
        
        # make indices which are episode-relative
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx # type:ignore
        
        # ensure trajectory doesn't go past the object target pose timestamp
        is_past_indices = self.get_post_target_indices(ep_idx)
        
        if len(is_past_indices) > 0:
            # get the integer element
            last_valid_idx = is_past_indices[0]
        
            # past-fill any indices after the target idx
            indices[last_valid_idx:] = self.target_idx
        
        # get the sample. actions are just future states
        key_action = self.indices.get_sequence_by_train_indices_and_key(indices, key)
        
        return key_action
        
    def get_joint_action_sample(self, ep_idx):
        """ joint actions are absolute, so no modification needed to the forward-fill part of the trajectory """
        joint_action = self.get_mage_hand_action_trajectory(ep_idx, "joint_state")
        
        return joint_action
    
    def set_stationary(self, rel_pose_action, first_past_idx):
        # set to stationary
        rel_pose_action[first_past_idx:, 0:4] = np.array([1.0, 0.0, 0.0, 0.0]) # quat
        rel_pose_action[first_past_idx:, 4:7] = np.array([0.0, 0.0, 0.0])       # pos
        
        return rel_pose_action
    
    def get_rel_pose_action_sample(self, ep_idx):
        """ rel pose actions are relative, so we need to reset the forward-fill part of the trajectory to `staitonary`, aka zeros for pos and 1, 0, 0, 0 for quat """
        # action is the same as the state
        pose_action = self.get_mage_hand_action_trajectory(ep_idx, "pose_state")
        
        # make relative to pose at ep_idx
        ref_pose = self.get_key_sample("pose_state", ep_idx)
        
        rel_pose_action = self.get_rel_pose(
            src=ref_pose,
            dst=pose_action
        )
        
        return rel_pose_action
    
    def get_rel_pos_action_sample(self, ep_idx):
        rel_pose_action = self.get_rel_pose_action_sample(ep_idx)
        
        rel_pos_action = rel_pose_action[:, 4:7]
        return rel_pos_action
    
    def get_rel_quat_action_sample(self, ep_idx):
        rel_pose_action = self.get_rel_pose_action_sample(ep_idx)
        
        rel_quat_action = rel_pose_action[:, 0:4]
        return rel_quat_action
    
    def get_rel_pose(self, src, dst):
        """
        experiment with different options?
        """   
        # option A: complete relative pose, hard to debug if things are going wrong. Also pos direction is dependent on future quat, they're coupled, so perhaps that's a harder learning problem
        if False:
            rel_pose = utils.drake_compute_rel_pose(
                src_pose=src,
                dst_pose=dst
            )
            
        # option B: represent orient as yaw, pitch, roll (in that order), and recognize the fact that once gravity is turned on, the problem will be pos & yaw invariant, but will vary w.r.t. pitch & roll, so we can decouple pos & yaw from pitch & roll. If this option is used, then we must provide the current pose's pitch and roll in the obs.
        if True:
            rel_pose = utils.get_rel_yaw_pose(
                src,
                dst
            )
        
        return rel_pose
    
    def get_rel_pos_only(self, src_v, dst_v):
        """
        compute rel pos only, dst w.r.t. src
        """
        assert(src_v.shape[-1] == 3)
        assert(dst_v.shape[-1] == 3)
        rel_pos = dst_v - src_v
        return rel_pos
        
    def get_rel_pos_only_from_pose(self, src_pose, dst_pose):
        """
        compute rel pos only, dst w.r.t. src
        both poses are [7] wxyz, xyz
        """
        # extract pos
        src_pos = src_pose[..., 4:7]
        dst_pos = dst_pose[..., 4:7]
        
        rel_pos = self.get_rel_pos_only(src_pos, dst_pos)
        
        return rel_pos
    
    def get_rel_object_target_pose(self, ep_idx, target_idx):
        # get target pose
        object_target_pose = self.get_key_sample("object_target_pose", target_idx)
        
        # make the relative object target pose
        pose_state = self.get_key_sample("pose_state", ep_idx)
        
        # compute rel pose
        rel_object_target_pose = self.get_rel_pose(
            src = pose_state,
            dst = object_target_pose
        )
        
        return rel_object_target_pose
    
    def get_rel_object_target_pos_only(self, ep_idx, target_idx):
        """
        only considers position part of the pose when computing relative position
        """
        # get target pose
        object_target_pose = self.get_key_sample("object_target_pose", target_idx)
        
        # make the relative object target pose
        pose_state = self.get_key_sample("pose_state", ep_idx)
        
        # compute rel pose
        rel_object_target_pos = self.get_rel_pos_only_from_pose(
            src_pose = pose_state,
            dst_pose = object_target_pose
        )
        
        return rel_object_target_pos
    
    def get_rel_object_target_pos(self, ep_idx, target_idx):
        rel_pose = self.get_rel_object_target_pose(ep_idx, target_idx)
        
        rel_pos = rel_pose[..., 4:7]
        return rel_pos
    
    def get_rel_object_pos(self, ep_idx):
        # same interface, but now the target is just this ep_idx
        return self.get_rel_object_target_pos(ep_idx, ep_idx)
    
    def get_rel_object_pos_only(self, ep_idx):
        """
        only considers position part of the pose when computing relative position
        """
        # same interface, but now the target is just this ep_idx
        return self.get_rel_object_target_pos_only(ep_idx, ep_idx)
        
    def get_all_rel_object_target_pose(self):
        all_rel_object_poses = []
        
        # iterate over each ep idx
        for ep_idx in range(len(self)):
        # using tqdm
        # for ep_idx in tqdm(range(len(self)), desc="getting rel object target poses (outer)"):
            
            # iterate over ep_idx + 1 to end of episode
            for target_idx in range(ep_idx + 1, len(self)):
            # using tqdm
            # for target_idx in tqdm(range(ep_idx + 1, len(self)), desc="getting rel object target poses (inner)"):
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
    
    def get_all_rel_object_pos(self):
        # iterate over each ep idx
        all_rel_object_pos = []
        for ep_idx in range(len(self)):
        # using tqdm for progress bar
        # for ep_idx in tqdm(range(len(self)), desc="getting rel object pos (outer)"):
            # get rel object pos
            rel_object_pos = self.get_rel_object_pos(ep_idx)
            
            # append
            all_rel_object_pos.append(rel_object_pos)
            
        # since rel_object_pos is already 2d, we have to vstack instead of convert to array
        out = np.vstack(all_rel_object_pos)
        
        return out
    
    def get_all_rel_pos_actions(self):
        all_rel_pos_actions = []
        
        # iterate over each ep idx
        for ep_idx in range(len(self)):
            # iterating over each target idx is very slow, so to save time just set target_idx to len(self) - 1
            target_idx = len(self) - 1
            
            # set the target idx
            self.target_idx = target_idx
            
            # get rel pos action
            rel_pos_action = self.get_rel_pos_action_sample(ep_idx)
            
            # append
            all_rel_pos_actions.append(rel_pos_action)
        
        # since rel_pos_action is already 2d, we have to vstack instead of convert to array
        out = np.vstack(all_rel_pos_actions)
    
        return out
    
    def get_pose_pitch_roll(self, ep_idx):
        """
        extract pitch and roll from the pose at ep_idx.
        
        rel_pose_state from self.get_rel_pose
        """
        pose_state = self.get_key_sample("pose_state", ep_idx)
        
        rel_pose_state = self.get_rel_pose(
            src = pose_state,
            dst = pose_state
        )
        
        # pitch, roll w.r.t. rel pose --- this might be the same value as pose_state, and if so we can save some compute by exploiting that fact. w/e though
        rel_pitch_roll = utils.drake_extract_pitch_roll_from_pose(rel_pose_state)
        
        # must ensure it's 2d
        rel_pitch_roll = rel_pitch_roll.reshape(1, -1)
        
        return rel_pitch_roll
    
    def get_all_pose_pitch_roll(self):
        all_pose_pitch_roll = []
        
        # iterate over each ep idx
        for ep_idx in range(len(self)):
            # get pose pitch roll
            pose_pitch_roll = self.get_pose_pitch_roll(ep_idx)
            
            # append
            all_pose_pitch_roll.append(pose_pitch_roll)
        
        # stack
        out = np.stack(all_pose_pitch_roll, axis=0)
        
        return out
    
    def get_rel_fk(self, ep_idx):
        """
        get rel fk at ep_idx
        """
        # get fk at ep_idx
        fk = self.get_key_sample("fk", ep_idx)
        
        # get pose at ep_idx
        pose_state = self.get_key_sample("pose_state", ep_idx)
        
        rel_fk = utils.drake_compute_rel_fk(pose_state, fk, self.get_rel_pose)
        
        return rel_fk
    
    def get_all_rel_fk(self):
        all_rel_fk = []
        
        # iterate over each ep idx
        for ep_idx in range(len(self)):
            # get rel fk
            rel_fk = self.get_rel_fk(ep_idx)
            
            # append
            all_rel_fk.append(rel_fk)
        
        # stack
        out = np.stack(all_rel_fk, axis=0)
        
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
        rel_fk = self.get_rel_fk(ep_idx)
        
        # rel_object_target_pose = self.get_rel_object_target_pose(ep_idx, target_idx)
        
        rel_object_target_pos = self.get_rel_object_target_pos(ep_idx, target_idx)
        
        rel_object_pos = self.get_rel_object_pos(ep_idx)
        
        pose_pitch_roll = self.get_pose_pitch_roll(ep_idx)
        
        # default output dict
        default_obs_sample = {
            'pose_pitch_roll': pose_pitch_roll,
            'joint_state': joint_state,
            'haptics': haptics,
            'rel_fk': rel_fk,
            'rel_object_pos': rel_object_pos,
            'rel_object_target_pos': rel_object_target_pos,
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
        
        sample["rel_pos_action"] = self.get_rel_pos_action_sample(ep_idx)
        
        sample["rel_quat_action"] = self.get_rel_quat_action_sample(ep_idx)

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
    
class MageHandDatasetSampler(DatasetSampler):
    """ exactly the same, but has a total_unique_datapoints function """
    def Init(self, ep_mask):
        out = super().Init(ep_mask)
        
        total = self.total_unique_datapoints()
        print(f"Total unique datapoints in MageHandDatasetSampler: {total}")
        
        return out
        
    
    def total_unique_datapoints(self):
        total = 0
        ep_sampler: MageHandEpisodeSampler
        for ep_sampler in self.get_ep_list(): #type:ignore
            total += ep_sampler.total_unique_datapoints()
        
        return total
        
    
            
    def make_episode(self, rb_episode_end, rb_offset, tr_ep_offset): #type:ignore
        """ 
        custom: skip the first couple indices of magehand because right now my state saver spends a couple steps snapping to the haptx glove pose
        """
        beginning_of_ep_step_modifier = 5
        rb_offset += beginning_of_ep_step_modifier
        
        assert(self.inlier_mask is not None)
        
        # already made
        if not rb_episode_end in self.ep_samplers:
            # check if rb_offset is valid
            if rb_offset >= rb_episode_end:
                print("Skipping episode creation because rb_offset >= rb_episode_end")
                return None
            
            # make the ep sampler
            ep_sampler = self.ep_sampler_class(
                self.rb_id,
                rb_offset,
                rb_episode_end,
                self.inlier_mask[rb_offset:rb_episode_end]
            )

            self.ep_samplers[rb_episode_end] = ep_sampler
            
            self.tr_ep_offsets.append(tr_ep_offset)
            
        # return the ep
        return self.ep_samplers[rb_episode_end]
    
    def get_all_rel_fk(self):
        all_rel_fk = []
        
        ep_sampler: MageHandEpisodeSampler
        for ep_sampler in tqdm(self.get_ep_list(), desc="Getting all rel fk"):
            ep_rel_fk = ep_sampler.get_all_rel_fk()
            all_rel_fk.append(ep_rel_fk)
        
        out = np.vstack(all_rel_fk)
        
        return out
    
    def get_all_rel_object_target_pos(self):
        all_rel_object_target_pos = []
        
        ep_sampler: MageHandEpisodeSampler
        for ep_sampler in tqdm(self.get_ep_list(), desc="Getting all rel object target pos"):
            ep_rel_object_target_pos = ep_sampler.get_all_rel_object_target_pos()
            all_rel_object_target_pos.append(ep_rel_object_target_pos)
        
        out = np.vstack(all_rel_object_target_pos)
        
        return out
    
    def get_all_rel_pos_actions(self):
        all_rel_pos_actions = []
        
        ep_sampler: MageHandEpisodeSampler
        for ep_sampler in tqdm(self.get_ep_list(), desc="Getting all rel pos actions"):
            ep_rel_pos_actions = ep_sampler.get_all_rel_pos_actions()
            all_rel_pos_actions.append(ep_rel_pos_actions)
        
        out = np.vstack(all_rel_pos_actions)
        
        return out
    
    def get_all_rel_object_pos(self):
        all_rel_object_pos = []
        
        ep_sampler: MageHandEpisodeSampler
        for ep_sampler in tqdm(self.get_ep_list(), desc="Getting all rel object pos"):
            ep_rel_object_pos = ep_sampler.get_all_rel_object_pos()
            all_rel_object_pos.append(ep_rel_object_pos)
        
        out = np.vstack(all_rel_object_pos)
        
        return out
    
    def get_all_pose_pitch_roll(self):
        all_pose_pitch_roll = []
        
        ep_sampler: MageHandEpisodeSampler
        for ep_sampler in tqdm(self.get_ep_list(), desc="Getting all pose pitch roll"):
            ep_pose_pitch_roll = ep_sampler.get_all_pose_pitch_roll()
            all_pose_pitch_roll.append(ep_pose_pitch_roll)
        
        out = np.vstack(all_pose_pitch_roll)
        
        return out
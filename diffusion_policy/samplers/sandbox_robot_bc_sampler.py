from typing import Optional
import numpy as np
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

import diffusion_policy.common.sarsa_sampler as sarsa_sampler


# import DatasetSampler, EpisodeSampler

import diffusion_policy.globals as globals


class SandboxRobotBCEpisodeSampler(sarsa_sampler.EpisodeSampler):
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
        similar to its parent, but no _next obs nor action
        """
        assert(ep_idx >= 0)
        # assert(ep_idx <= len(self)-2) # must be -2 since we get the next obs & action
        assert(ep_idx <= len(self)-1) # allow ep_idx to be == len(self)-1. In that case, obs_next will be a repeat of obs
        
        sample = {}
        sample["obs"] = self.get_obs_sample(ep_idx)
        
        sample["action"] = self.get_action_sample(ep_idx)

        return sample
    
    
class SandboxRobotBCEpisodeSamplerBackwardsCompat(SandboxRobotBCEpisodeSampler):
    """
    backwards compat with old zarr datasets which only have ["img", "img2", "state", "action", "reward"]
    """
    def make_default_obs(self):
        """
        get the shape meta from config, use obs_keys_to_load to determine the necessary inputs, make all zeros arrays
        """
        shape_meta = globals.CONFIG.shape_meta # type: ignore
        obs_keys_to_load: list = globals.CONFIG.obs_keys_to_load # type: ignore
        
        obs_sample = {}
        
        horizon = 1
        
        for obs_key in obs_keys_to_load:
            shape = shape_meta[obs_key].shape
            shape = np.array(shape)
            
            # add traj dim if necessary
            # if shape.ndim == 1:
            #     shape = (horizon, shape[0])
            shape = (horizon, *shape)
            
            # make the default
            obs_sample[obs_key] = np.zeros(shape, dtype=np.float32)
            
        return obs_sample
    
    def make_default_action(self):
        """
        get the shape meta from config, use action_rel_indices to determine the necessary inputs, make all zeros arrays
        """
        shape_meta = globals.CONFIG.shape_meta # type: ignore
        action_rel_indices: list = globals.CONFIG.action_rel_indices # type: ignore
        
        horizon = len(action_rel_indices)
        
        # for backwards compat, we assume that the action key is "action"
        action_shape = shape_meta["action"].shape
        
        # add traj dimension
        shape = (horizon, *action_shape)
        
        return np.zeros(shape, dtype=np.float32)
    
    def get_obs_sample(self, ep_idx):
        """
        must define a custom mapping. Reference: dexnex_projects ... gen_dataset_hitl.py. 'state' is the robot joint pos, 'action' is the joint_command (aka next position)
        
        from actor_critic_19.yaml:
        state: shape: 38 ## gofa (6), wr (2), th (5), ff (4), mf (4), biotacs (5), th-pos (3), ff-pos (3), mf-pos (3), block xyz (3)
        
        action: shape: 21 gofa (6), wr (2), th (5), ff (4), mf (4)
   
        from actor_critic_15.yaml:
        shape_meta:
            action:
                shape:
                - 21 # gofa (6), wr (2), th (5), ff (4), mf (4)

            img: # overhead cam
                shape:
                - 3
                - 192
                - 192
            img2: # wrist cam
                shape:
                - 3
                - 192
                - 192
            
            state: # the name "state" came from the zarr generation script
                shape:
                - 35 # gofa (6), wr (2), th (5), ff (4), mf (4), biotacs (5), th-pos (3), ff-pos (3), mf-pos (3)
   
        """
        obs_keys_to_load: list = globals.CONFIG.obs_keys_to_load # type: ignore
        
        obs_sample = self.make_default_obs()
        
        # for now, only allow horizon of 1
        assert(obs_sample["robot_joint_pos"].shape[0] == 1)
        
        # get the state
        state = self.get_key_sample("state", ep_idx)[0]
        
        # check the length to infer the states
        if len(state) == 38:
            # actor_critic_19.yaml
            robot_joint_pos = state[:21]
            biotacs = state[21:26]
            fk = state[26:35]
            block_xyz = state[35:38]
            
        elif len(state) == 35:
            # actor_critic_15.yaml
            robot_joint_pos = state[:21]
            biotacs = state[21:26]
            fk = state[26:35]
            
            # default
            block_xyz = np.zeros(3)
            
        else:
            raise
            
        # fill in the obs sample
        ln = min(len(robot_joint_pos), len(obs_sample["robot_joint_pos"]))
        obs_sample["robot_joint_pos"][0, :ln] = robot_joint_pos[:ln]

        """
        OBJECT_IDS = {
            "reserved": 0,
            "block": 1,
            "bin_with_divider": 2,
            "cube_with_one_green_face": 3,
            "rotating_puck": 4,
            "ring1": 5,
            "ring2": 6,
            "ring3": 7,
            "ring_toy_base": 8,
            "pushT": 9,
        }
        """
        
        # object pos
        id1 = 1 * 3
        id2 = id1 + 3
        obs_sample["object_pos"][0, id1:id2] = block_xyz[:3]
        
        # from avatar_drake_sim ... tasks.py
        """
        # Task Catalog
        | task_id | Description   | env
        |---|----------------------|---|
        | 0 | reserved - all tasks | all |
        | 1 | single_box_and_bins  | SIM |
        | 2 | orient_cube          | SIM |
        | 3 | rotate_puck          | SIM |
        | 4 | ring_stack           | SIM |
        | 5 | block transfer   | IRL
        | 6 | bnb 1cam             | IRL |
        | 7 | bnb 2cam             | IRL|
        | 8 | static 1 block | SIM|
        | 9 | randomized 1 block | SIM |
        | 10 | stack rings | IRL |
        | 11 | unstack rings | IRL |
        | 12 | stack block bridge | IRL |
        | 13 | unstack block bridge | IRL |
        | 14 | jar on lid | IRL |
        | 15  | jar off lid | IRL |
        | 16  | binning objects | IRL |
        | 17  | folding | IRL |
        | 18  | unfolding | IRL |
        | 19  | push penny off table | SIM |
        | 20  | peg in hole (object) | SIM |
        """
        TASKS_IDS = {
            "single_box_and_bins": 1,
            "orient_cube": 2,
            "rotate_puck": 3,
            "ring_stack": 4,
        }
        
        RB_ID_TO_TASK_ID = {
            "block_transfer": 5,
            "bnb_1cam": 6,
            "bnb_2cams": 7,
            "sim_hitl_toby": 8,
        }
        task_id = RB_ID_TO_TASK_ID[self.rb_id]
        
        obs_sample["task_obs"][0, task_id] = 1.0
        
        return obs_sample

        
    
    def get_action_sample(self, ep_idx):
        """
        For an action, we want a sequence from ep_idx - n_obs_steps to ep_idx + horizon.
        """
        # make indices which are episode-relative
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx # type:ignore
            
        # get action
        action = self.indices.get_sequence_by_train_indices_and_key(indices, "action")
        
        # confirm the length of the action dim
        assert(len(action[-1]) == 21)
        
        action_sample = self.make_default_action()
        
        ln = min(len(action[-1]), len(action_sample[-1]))
        action_sample[:, :ln] = action[:, :ln]
        
        return action_sample
    

    def get_sample(self, ep_idx):
        """
        just for easy debugging
        """
        sample = super().get_sample(ep_idx)
        
        return sample
    

# class BackwardsCompatSamplerModifier:
#     def modify_obs_sampler(self, obs_sample):
        
#         # for now, only allow horizon of 1
#         assert(obs_sample["robot_joint_pos"].shape[0] == 1)
        
#         # get the state
#         state = self.get_key_sample("state", ep_idx)[0]
        
#         # check the length to infer the states
#         if len(state) == 38:
#             # actor_critic_19.yaml
#             robot_joint_pos = state[:21]
#             biotacs = state[21:26]
#             fk = state[26:35]
#             block_xyz = state[35:38]
            
#         elif len(state) == 35:
#             # actor_critic_15.yaml
#             robot_joint_pos = state[:21]
#             biotacs = state[21:26]
#             fk = state[26:35]
            
#             # default
#             block_xyz = np.zeros(3)
            
#         else:
#             raise
            
#         # fill in the obs sample
#         ln = min(len(robot_joint_pos), len(obs_sample["robot_joint_pos"]))
#         obs_sample["robot_joint_pos"][0, :ln] = robot_joint_pos[:ln]

#         """
#         OBJECT_IDS = {
#             "reserved": 0,
#             "block": 1,
#             "bin_with_divider": 2,
#             "cube_with_one_green_face": 3,
#             "rotating_puck": 4,
#             "ring1": 5,
#             "ring2": 6,
#             "ring3": 7,
#             "ring_toy_base": 8,
#             "pushT": 9,
#         }
#         """
        
#         # object pos
#         id1 = 1 * 3
#         id2 = id1 + 3
#         obs_sample["object_pos"][0, id1:id2] = block_xyz[:3]
        
#         # from avatar_drake_sim ... tasks.py
#         """
#         # Task Catalog
#         | task_id | Description   | env
#         |---|----------------------|---|
#         | 0 | reserved - all tasks | all |
#         | 1 | single_box_and_bins  | SIM |
#         | 2 | orient_cube          | SIM |
#         | 3 | rotate_puck          | SIM |
#         | 4 | ring_stack           | SIM |
#         | 5 | block transfer   | IRL
#         | 6 | bnb 1cam             | IRL |
#         | 7 | bnb 2cam             | IRL|
#         | 8 | static 1 block | SIM|
#         | 9 | randomized 1 block | SIM |
#         | 10 | stack rings | IRL |
#         | 11 | unstack rings | IRL |
#         | 12 | stack block bridge | IRL |
#         | 13 | unstack block bridge | IRL |
#         | 14 | jar on lid | IRL |
#         | 15  | jar off lid | IRL |
#         | 16  | binning objects | IRL |
#         | 17  | folding | IRL |
#         | 18  | unfolding | IRL |
#         | 19  | push penny off table | SIM |
#         | 20  | peg in hole (object) | SIM |
#         """
#         TASKS_IDS = {
#             "single_box_and_bins": 1,
#             "orient_cube": 2,
#             "rotate_puck": 3,
#             "ring_stack": 4,
#         }
        
#         RB_ID_TO_TASK_ID = {
#             "block_transfer": 5,
#             "bnb_1cam": 6,
#             "bnb_2cams": 7,
#             "sim_hitl_toby": 8,
#         }
#         task_id = RB_ID_TO_TASK_ID[self.rb_id]
        
#         obs_sample["task_obs"][0, task_id] = 1.0
        
#         return obs_sample
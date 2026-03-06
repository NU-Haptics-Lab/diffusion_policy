

"""
Which datasets will I be using?

How to decide the control panels? We want the minimum numberof inputs? Or just a diff one-hot vector for each dataset? That's fine if the reward was static for the dataset, but what if the reward is changing throughout the dataset? Like in RL using scheduling / reverse curriculum generation? In that case 

/media/dexnex_ssd/data/sandbox/online_rl_replay_buffer_rl_6.zarr -- RL reward only
/media/dexnex_ssd/data/sandbox/online_rl_replay_buffer_rl_5.zarr -- RL reward only

/media/dexnex_ssd/data/bnb/bt.reward.zarr - block transfer task, task rewards

/media/dexnex_ssd/data/bnb/1cam.reward.zarr - 1 cam bnb, task rewards

/media/dexnex_ssd/data/bnb/2cams.reward.zarr - 2 cam bnb, task rewards

/media/dexnex_ssd/data/bnb/sim_hitl_toby.reward.zarr - sim bnb, task rewards

/media/dexnex_ssd/data/bnb/sim_online_rl1.reward.zarr - 0% s.r.

/media/dexnex_ssd/data/bnb/actor_critic_15.reward.zarr - the big one, 92% s.r. I believe

/media/dexnex_ssd/data/sim_1block_randomized/all.reward.zarr - not entirely sure



paths = [
"/media/dexnex_ssd/data/bnb/bt.reward.zarr",
"/media/dexnex_ssd/data/bnb/1cam.reward.zarr",
"/media/dexnex_ssd/data/bnb/2cams.reward.zarr",
"/media/dexnex_ssd/data/bnb/sim_hitl_toby.reward.zarr",
"/media/dexnex_ssd/data/bnb/sim_hitl_not_toby.reward.zarr",
"bnb/actor_critic_15.reward.zarr",
"sim_1block_randomized/on_task.reward.zarr",

[x] sandbox/online_rl_replay_buffer_rl_6.zarr
[x] sandbox/online_rl_replay_buffer_rl_5.zarr

these are very small and kinda suck. Ignore for now.
"sandbox/sandbox2.zarr",
"sandbox/sandbox3.zarr",
]

Total number of episodes: 10693
Total number of steps: 784920

ugh, I'm not sure how to use bt/1cam/2cams, etc. They have (state, position_as_action, reward) tuples. 
options: 
* compute a torque traj from the position traj 
    will work in this case but is it generally true that a conversion will always exist from the dataset action to my action? not necessarily (ex: wheel action to robot joint action). for the same embodiment though, there will probably be a mapping.
* add robot_pos_action as a control mode, and then add a control and can train like normal. 
* treat actions as actually just another observation. I don't hate the idea. at run time we then have the choice of pos or torque control.
ugh, just realized I can't treat actions like obs because of the way we do conditioning. actions must be treated differently due to the UNet + FiLM architecture.
Unless... I move away from the unet + FiLM and just use a transformer.

right now the fastest will be just doubling my action space from 30 joint torques to 30 joint torques + 30 joint positions.

make torque the first 30, position the next 30
"""









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

from diffusion_policy.common.multi_replay_buffer_sampler import (
    MultiReplayBufferDatasetSampler,
)

import diffusion_policy.globals as globals

from diffusion_policy.common.replay_buffer_loader import ReplayBufferLoader

from diffusion_policy.samplers.sandbox_robot_rl_sampler import (
    SandboxRobotRLEpisodeSampler,
)



# class SamplerOptions:
#     def __init__(self,
#                  bc_only: bool = False,
#                  rl_only: bool = False,
#                  contains_task_id: bool = False,
#                  task_id: int = 0, # lets us condition on samples doing different tasks
#                  reward_id: int = 0, # lets us condition on samples with differing reward functions
#                  ):
#         self.bc_only = bc_only
#         self.rl_only = rl_only
#         self.contains_task_id = contains_task_id
#         self.task_id = task_id
#         self.reward_id = reward_id
        
#     def get_control_panel_inputs(self):
#         """
#         output will be an array which converts the options into a some-hot float array which allows us to condition our policy on the different options.
        
#         order:
#         one-hot task id array
#         one-hot reward id array
#         bc only flag
#         rl only flag
        
#         """
#         # max values from sandbox_common, when this file was created.
#         max_nb_joint_pos = 30
#         max_nb_joint_vel = 30
#         max_nb_touch_pos = 10
#         max_nb_touch_force = 10
#         max_nb_objects = 20
#         max_nb_tasks = 30
#         max_traj_horizon = 16

class BC(EpisodeSampler):
    """
    just for BC
    """
    def get_action_sample(self, ep_idx):
        raise
    
    def get_obs_sample(self, ep_idx):
        raise
    
    def get_sample(self, ep_idx):
        """
        only need obs and action
        """
        assert(ep_idx >= 0)
        # assert(ep_idx <= len(self)-2) # must be -2 since we get the next obs & action
        assert(ep_idx <= len(self)-1) # allow ep_idx to be == len(self)-1. In that case, obs_next will be a repeat of obs
        
        sample = {}
        sample["obs"] = self.get_obs_sample(ep_idx)
        
        sample["action"] = self.get_action_sample(ep_idx)

        return sample
    
# class GoalConditionedBC(EpisodeSampler):
#     """
#     Define a loss based off whether s' matches s'_desired, or not. This is a way to do BC when the target dataset doesn't contain the same action as the origin dataset.
    
#     If our policy takes action a in state s and ends up in state s', then the MSE loss between s' and s'_d tells us how fitting a was. RL-7 outputs a single value, so in this case it's learning to predict the MSE_loss
#     """
    
#     def get_sample(self, ep_idx):
#         """
#         need my obs, my action, my next obs, and the desired obs
#         """
#         assert(ep_idx >= 0)
#         # assert(ep_idx <= len(self)-2) # must be -2 since we get the next obs & action
#         assert(ep_idx <= len(self)-1) # allow ep_idx to be == len(self)-1. In that case, obs_next will be a repeat of obs
        
#         sample = {}
#         sample["obs"] = self.get_obs_sample(ep_idx)
#         sample["obs_next"] = self.get_obs_sample(ep_idx+1)
        
#         sample["action"] = self.get_action_sample(ep_idx)
        
#         sample["obs_next_desired"] = self.get_obs_desired_sample(ep_idx+1)

#         return sample
        
        
# alias

def get_zeros_obs_sample():
    obs_sample = {}
    obs_sample["robot_joint_pos"] = np.zeros(30)
    obs_sample["robot_joint_vel"] = np.zeros(30)
    obs_sample["object_quat"] = np.zeros(40) # max from sandbox_common.MaxParams
    obs_sample["object_pos"] = np.zeros(60) # max from sandbox_common.MaxParams
    
    return obs_sample

class RL(SandboxRobotRLEpisodeSampler):
    """
    just for RL (bellman eq)
    """
    pass
    
class RL1(RL):
    """
    position actions, different tasks, different rewards
    
    "bnb/bt.reward.zarr",
    "bnb/1cam.reward.zarr",
    "bnb/2cams.reward.zarr",
    "bnb/sim_hitl_toby.reward.zarr",
    "bnb/sim_hitl_not_toby.reward.zarr",
    "bnb/actor_critic_15.reward.zarr",
    "sim_1block_randomized/on_task.reward.zarr",
    """
    def get_obs_sample(self, ep_idx):
        """
        obs_keys_to_use:
        - robot_joint_pos
        - robot_joint_vel
        - object_quat
        - object_pos
        """
        obs_sample = get_zeros_obs_sample()
        
        # assemble the obs sample
        obs_sample["robot_joint_pos"] = self.get_key_sample("state", ep_idx)[0:21]
        obs_sample["task_obs"][0] = 1.0
        
        return obs_sample
    
    def get_action_sample(self, ep_idx):
        action_key = "action"
        
        # make indices which are episode-relative
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx # type:ignore
        
        # get the sample
        my_sample = self.indices.get_sequence_by_train_indices_and_key(indices, action_key)
        
        h = len(globals.CONFIG.action_rel_indices) # type: ignore
        a = globals.CONFIG.shape_meta.action.shape[0] # type: ignore
        sample = np.zeros((1, h, a))
        
        # make torque the first 30, position the next 30
        sample[:, :, 30:30+21] = my_sample
        
        return sample
        
    
class RL2(RL):
    """
    autonomous, same action, same task, same rewards, minimal change needed
    
    "sandbox/online_rl_replay_buffer_rl_6.zarr",
    "sandbox/online_rl_replay_buffer_rl_5.zarr",    
    """
    def get_obs_sample(self, ep_idx):
        obs_sample = super().get_obs_sample(ep_idx)
        
        # add missing obs
        
        return obs_sample
    
    def get_action_sample(self, ep_idx):
        action_key = "robot_joint_action"
        
        # make indices which are episode-relative
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx # type:ignore
        
        # get the sample
        my_sample = self.indices.get_sequence_by_train_indices_and_key(indices, action_key)
        
        h = len(globals.CONFIG.action_rel_indices) # type: ignore
        a = globals.CONFIG.shape_meta.action.shape[0] # type: ignore
        sample = np.zeros((1, h, a))
        
        # make torque the first 30, position the next 30
        sample[:, :, 0:] = my_sample
        
        return sample
    
    
    
    
if __name__ == "__main__":
    # setup the rb loaders
    rbl = ReplayBufferLoader(
        rb_paths={
            "rl_6": "/media/dexnex_ssd/data/sandbox/online_rl_replay_buffer_rl_6.zarr",
            "rl_5": "/media/dexnex_ssd/data/sandbox/online_rl_replay_buffer_rl_5.zarr",
        },
        modes={
            "rl_6": "r",
            "rl_5": "r",
        },
        do_loading=True,
    )
    rbl.setup()
    # save to globals
    globals.REPLAY_BUFFER_LOADER = rbl
    
    # make the config
    str = """
common_indices:
    _target_: diffusion_policy.common.sarsa_sampler.Indices
    rb_id: default
    rb_episode_start_idx: 0
    rb_episode_end: 0
    training_episode_start: 0
    training_episode_end: 0
    pad_before: 0
    pad_after: 0
    debug: false

obs_keys_to_load:
  - robot_joint_pos
  - robot_joint_vel
  
action_rel_indices: [-1, 0, 1, 2, 3, 4, 5, 6]
action_key: robot_joint_action

"""
    config = OmegaConf.create(str)
    globals.CONFIG = config
    
    
    # setup the samplers
    rb2 = MultiReplayBufferDatasetSampler(
        rb_ids = ["rl_6", "rl_5"],
        description="rl2",
        ep_sampler_class_str = "diffusion_policy.samplers.sandbox_robot_rl_7_sampler.RL2",
    )
    rb2.setup()
    rb2.InitAll()
    
    print("total number of samples in rb2: ", len(rb2))
    
    # sample from the samplers
    for i in range(10):
        ds_tr_idx = np.random.randint(0, len(rb2)-1)
        sample = rb2.get_sample(ds_tr_idx)
        print(sample)
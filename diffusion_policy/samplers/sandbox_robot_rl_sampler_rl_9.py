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

from diffusion_policy.common.multi_replay_buffer_sampler import (
    MultiReplayBufferDatasetSampler,
)

from diffusion_policy.samplers.sandbox_robot_rl_sampler import (
    SandboxRobotRLEpisodeSampler,
)

# class RL9(SandboxRobotRLEpisodeSampler):
#     """
#     Get a sample from an episode for the object manipulation policy.
    
#     mapping: (state, object id, object target pose) --> Traj[actions].
#     """
#     def get_reward(self, ep_idx):
#         original = super().get_reward(ep_idx)
        
#         if self.rb_id == "rb_8"
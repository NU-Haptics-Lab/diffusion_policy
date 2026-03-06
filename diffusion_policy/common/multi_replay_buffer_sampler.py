
"""


Be able to sample from multiple replay buffers and treat them as one dataset




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

import torch as th

from diffusion_policy.common.sarsa_sampler import (
    DatasetSampler,
    EpisodeSampler,
    get_lower_bound_idx,
)

class MultiReplayBufferDatasetSampler(DatasetSampler):
    # override
    def __init__(self,
            rb_ids: list,
            description: str = "",
            ep_sampler_class_str: str = "diffusion_policy.common.sarsa_sampler.EpisodeSampler", # I don't love this design
            use_cap_rewards = False,
            reward_cap = 50.0,
            ):
        self.rb_ids = rb_ids
        self.description = description
        self.ep_sampler_class_str = ep_sampler_class_str
        self.use_cap_rewards = use_cap_rewards
        self.reward_cap = reward_cap
        
        # convert text to class object using hydra
        self.ep_sampler_class = hydra.utils.get_class(ep_sampler_class_str)
        
    # override
    def setup(self):
        """
        must be called after all nodes are made
        """
        # refs
        self.replay_buffers: dict[str, ReplayBuffer] = {rb_id: globals.REPLAY_BUFFER_LOADER[rb_id] for rb_id in self.rb_ids} #type:ignore
        
        self.initd = False
        self.ep_mask = None
        self.ep_samplers: dict[tuple[str, int], EpisodeSampler] = {} #type:ignore
        self.qvals = None
        # self.inlier_mask = None
        

    #override
    def Init(self,
              ep_mask
              ):
        raise
        
    # override
    def InitAll(self):
        self.ep_samplers = {} #type:ignore

        self.make_multiple_replay_buffer_episodes()
            
        self.initd = True
            
        self.print_nb_successes()
        
    # override
    def reinit_all(self):
        """
        just iterate over the new eps
        """
        raise NotImplementedError()
            
    # new
    def make_episode_per_rb(self, rb_id, rb_episode_end, rb_offset, tr_ep_offset):
        
        training_episode_start = rb_offset
        training_episode_end = rb_episode_end
        
        mask = np.ones(training_episode_end - training_episode_start, dtype=bool)
        
        
        # already made
        if not (rb_id, rb_episode_end) in self.ep_samplers:
            # make the ep sampler
            ep_sampler = self.ep_sampler_class(
                rb_id = rb_id,
                rb_episode_start_idx = rb_offset,
                rb_episode_end = rb_episode_end,
                training_episode_start = training_episode_start,
                training_episode_end = training_episode_end,
                mask = mask,
                use_cap_rewards = self.use_cap_rewards,
                reward_cap = self.reward_cap,
            )
            ep_sampler.setup()

            self.ep_samplers[(rb_id, rb_episode_end)] = ep_sampler
            
            self.tr_ep_offsets.append(tr_ep_offset)
            
            
        # return the ep
        return self.ep_samplers[(rb_id, rb_episode_end)]
            
            
    # new
    def make_episodes_per_rb(self, rb_id, episode_ends, ep_mask, tr_ep_offset):
        """ Using the replay buffer's episode_ends, make episode sampler classes  """


        # first ep offset
        rb_offset = 0        

        # one episode sampler per episode
        for idx, rb_episode_end in tqdm(enumerate(episode_ends)): #type:ignore
            # if skip a.k.a. episode mask
            if ep_mask is None or ep_mask[idx]:
                # make the episode (or retrieve it)
                ep_sampler = self.make_episode_per_rb(rb_id, rb_episode_end, rb_offset, tr_ep_offset)

                if ep_sampler is not None:
                    # add the length of the training episode
                    tr_ep_offset += len(ep_sampler)
                    
            # set rb offset to the old rb_episode_end
            rb_offset = rb_episode_end
            
        return tr_ep_offset
            

    # new
    def make_multiple_replay_buffer_episodes(self):
        """
        make multiple replay buffer episodes, and concatenate them into one big episode sampler. This is used for the multi-replay-buffer dataset sampler, where we want to treat multiple replay buffers as one big dataset.
        """
        # 
        
        # training episode ends. Copy from the ep sampler classes so we can use the efficient binary-search np.searchsorted method when converting from training index to episode
        self.tr_ep_offsets = []
        tr_ep_offset = 0
        
        # loop over replay buffers
        for rb_id, rb in self.replay_buffers.items():
            n_eps = len(rb.episode_ends) #type:ignore
            
            ep_mask = np.ones(n_eps, dtype=bool)
            
            # make the episode (or retrieve it)
            tr_ep_offset = self.make_episodes_per_rb(rb_id, rb.episode_ends, ep_mask, tr_ep_offset)
            

    # override
    def copy(self):
        """
        make a copy
        """
        copy = MultiReplayBufferDatasetSampler(
            rb_ids = self.rb_ids,
            ep_sampler_class_str = self.ep_sampler_class_str,
            use_cap_rewards = self.use_cap_rewards,
            reward_cap = self.reward_cap,
        )
        copy.setup()
        
        return copy
    
    # override, use self.description instead of self.rb_id
    def print_nb_successes(self):
        successes, count = self.get_nb_successes()
                
        print("{}: nb successes: {}/{}".format(self.description, successes, count))
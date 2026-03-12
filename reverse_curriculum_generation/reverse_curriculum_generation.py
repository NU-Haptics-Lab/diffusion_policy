

"""


"""


import numpy as np

import diffusion_policy.globals as globals

# sarsa sampler
from diffusion_policy.common.sarsa_sampler import (
    DatasetSampler, 
    EpisodeSampler,
)


import avatar_drake_sim.sims.sandbox.sandbox_common as commons

# my globals, kind of like a singleton
DIFFICULTY_SCALE = 0.0 # 0.0 to 1.0
SUCCESS_RATE = 0.0
NB_ATTEMPTS = 0

# def update_difficulty_scale(outcome):
#     global DIFFICULTY_SCALE

#     delta = 0.005 # 0.5%
    
#     # simple update rule: if outcome is success, increase difficulty, if failure, decrease difficulty
#     if outcome:
#         DIFFICULTY_SCALE = min(1.0, DIFFICULTY_SCALE + delta)
#     else:
#         DIFFICULTY_SCALE = max(0.0, DIFFICULTY_SCALE - delta)

#     # save to commons as well
#     commons.DIFFICULTY_SCALE = DIFFICULTY_SCALE
    
def update_difficulty_scale2(outcome):
    """
    update the success rate and increase difficulty if s.r. >50%, decrease if s.r. <50%
    """
    global SUCCESS_RATE, NB_ATTEMPTS, DIFFICULTY_SCALE
    
    delta = 0.005 # 0.5%

    NB_ATTEMPTS += 1
    SUCCESS_RATE += (outcome - SUCCESS_RATE) / NB_ATTEMPTS # could use the last x attempts instead of all attempts

    if SUCCESS_RATE > 0.5:
        DIFFICULTY_SCALE = min(1.0, DIFFICULTY_SCALE + delta)
    else:
        DIFFICULTY_SCALE = max(0.0, DIFFICULTY_SCALE - delta)
        
    # save to commons as well
    commons.DIFFICULTY_SCALE = DIFFICULTY_SCALE
    
def update_difficulty_scale3(outcome):
    """
    just use a low pass filter
    """
    global DIFFICULTY_SCALE
    alpha = 0.01 # smoothing factor
    
    DIFFICULTY_SCALE = alpha * outcome + (1 - alpha) * DIFFICULTY_SCALE

class Node:
    """
    one node per episode. When updating, decide whether to move up or down the markov chain depending on the current success rate
    """
    def __init__(self,
                 ep: EpisodeSampler,
                 ) -> None:
        self.ep = ep
        
        # my members
        # self.avg_success_rate = 0.0
        # self.nb_attempts = 0
        self.ep_len = len(self.ep)
        
        # self.reset()
        
    # def clip_ep_idx(self):
    #     self.current_ep_idx = np.clip(self.current_ep_idx, 0, self.ep_len-1)
        
    # def reset(self):
    #     self.current_ep_idx = self.ep_len-1
        
    # def update_ep_idx(self):
    #     """
    #     if s.r. is >50%, make the node harder. If s.r. <50%, make the node easier
    #     """
    #     if self.avg_success_rate > 0.5:
    #         self.current_ep_idx -= 1
    #     else:
    #         self.current_ep_idx += 1
            
    #     self.clip_ep_idx()
        
    # def update(self, outcome):
    #     """
    #     update my avg success rate using the moving average fcn
    #     """
    #     # update nb attempts
    #     self.nb_attempts += 1
        
    #     # avg function
    #     self.avg_success_rate += outcome / self.nb_attempts
        
    #     self.update_ep_idx()
        
    # def get_current_timeout(self):
    #     steps_to_end = len(self.ep) - self.current_ep_idx
        
    #     # give the policy 3x this many steps.
    #     timeout = steps_to_end * 3 * commons.LEARNING_RATE_DT
        
    #     return timeout
        
    # def get_current_rb_idx(self):
    #     return self.ep.get_id(self.current_ep_idx)
    
    # def get_current_state_dict(self):
    #     return self.ep.get_obs_sample(self.current_ep_idx)
    
    def get_difficulty_scaled_state_dict(self):
        """
        from the continuous global difficulty scale
        """
        # convert difficulty scale to an ep idx
        ep_idx_floor = np.floor((1.0 - DIFFICULTY_SCALE) * (self.ep_len - 1))
        ep_idx_ceil = np.ceil((1.0 - DIFFICULTY_SCALE) * (self.ep_len - 1))

        sample_floor = self.ep.get_obs_sample(int(ep_idx_floor))
        sample_ceil = self.ep.get_obs_sample(int(ep_idx_ceil))

        # interpolate between the two samples
        if ep_idx_floor == ep_idx_ceil:
            return sample_floor
        else:
            alpha = (1.0 - DIFFICULTY_SCALE) * (self.ep_len - 1) - ep_idx_floor
            sample = {}
            for key in sample_floor:
                sample[key] = (1 - alpha) * sample_floor[key] + alpha * sample_ceil[key]
            return sample
        
    
class ReverseCurriculumGeneration:
    def __init__(self,
                 sampler: DatasetSampler,
                 ) -> None:
        self.sampler = sampler
    
        # my members
        self.nodes = []
        self.current_node: Node | None = None
        self.total_attempts = 0
        self.total_successes = 0
        
    def setup(self):
        # # rb handle
        # assert(globals.REPLAY_BUFFER_LOADER is not None)
        # self.rb = globals.REPLAY_BUFFER_LOADER[self.rb_key]
        
        eps = self.sampler.get_ep_list()
        self.nodes = [Node(ep) for ep in eps]
        
    def update(self, outcome):
        global DIFFICULTY_SCALE
        # update it
        # # WON"T WORK WITH VECTOR ENVS
        # if self.current_node is not None:
        #     self.current_node.update(outcome)
        
        # won't work with vector env, will ahve to make a discrete state
        self.total_attempts += 1
        self.total_successes += outcome
        
        # just do it here
        sr = self.total_successes / self.total_attempts
        if sr > 0.5:
            update_difficulty_scale3(1.0)
        else:
            update_difficulty_scale3(0.0)

        # update global
        # update_difficulty_scale3(outcome)

        # log the difficulty scale
        if globals.LOGGER is not None:
            globals.LOGGER.log_one("difficulty_scale", DIFFICULTY_SCALE)
    
    # def get_current_timeout(self):
    #     if self.current_node is not None:
    #         return self.current_node.get_current_timeout()
    #     else:
    #         # default timeout
    #         return commons.MAX_EPISODE_TIME
    
    # def get_a_sample_idx(self):
    #     """
    #     draw a node from our open set
    #     """
    #     # pick a random node
    #     node_idx = np.random.choice(len(self.nodes))
    #     node = self.nodes[node_idx]
        
    #     # get the rb idx from the node
    #     rb_idx = node.get_current_rb_idx()
        
    #     # we're done
    #     return rb_idx
    
    def get_a_sample_state_dict(self):
        """
        draw a node from our open set, and get the state dict for the current rb idx of that node
        """
        # pick a random node
        node_idx = np.random.choice(len(self.nodes))
        node = self.nodes[node_idx]
        self.current_node = node
        
        # get the state dict for that rb idx
        state_dict = node.get_difficulty_scaled_state_dict()
        
        # we're done
        return state_dict
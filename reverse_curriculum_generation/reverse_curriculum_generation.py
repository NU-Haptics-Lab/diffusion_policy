

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



class Node:
    """
    one node per episode. When updating, decide whether to move up or down the markov chain depending on the current success rate
    """
    def __init__(self,
                 ep: EpisodeSampler,
                 ) -> None:
        self.ep = ep
        
        # my members
        self.avg_success_rate = 0.0
        self.nb_attempts = 0
        self.ep_len = len(self.ep)
        
        self.reset()
        
    def clip_ep_idx(self):
        self.current_ep_idx = np.clip(self.current_ep_idx, 0, self.ep_len-1)
        
    def reset(self):
        self.current_ep_idx = self.ep_len-1
        
    def update_ep_idx(self):
        """
        if s.r. is >50%, make the node harder. If s.r. <50%, make the node easier
        """
        if self.avg_success_rate > 0.5:
            self.current_ep_idx -= 1
        else:
            self.current_ep_idx += 1
            
        self.clip_ep_idx()
        
    def update(self, outcome):
        """
        update my avg success rate using the moving average fcn
        """
        # update nb attempts
        self.nb_attempts += 1
        
        # avg function
        self.avg_success_rate += outcome / self.nb_attempts
        
        self.update_ep_idx()
        
    def get_current_timeout(self):
        steps_to_end = len(self.ep) - self.current_ep_idx
        
        # give the policy 3x this many steps.
        timeout = steps_to_end * 3 * commons.LEARNING_RATE_DT
        
        return timeout
        
    def get_current_rb_idx(self):
        return self.ep.get_id(self.current_ep_idx)
    
    def get_current_state_dict(self):
        return self.ep.get_obs_sample(self.current_ep_idx)
        
    
class ReverseCurriculumGeneration:
    def __init__(self,
                 sampler: DatasetSampler,
                 ) -> None:
        self.sampler = sampler
    
        # my members
        self.nodes = []
        self.current_node: Node | None = None
        
    def setup(self):
        # # rb handle
        # assert(globals.REPLAY_BUFFER_LOADER is not None)
        # self.rb = globals.REPLAY_BUFFER_LOADER[self.rb_key]
        
        eps = self.sampler.get_ep_list()
        self.nodes = [Node(ep) for ep in eps]
        
    def update(self, outcome):
        # update it
        # WON"T WORK WITH VECTOR ENVS
        if self.current_node is not None:
            self.current_node.update(outcome)
    
    def get_current_timeout(self):
        if self.current_node is not None:
            return self.current_node.get_current_timeout()
        else:
            # default timeout
            return commons.MAX_EPISODE_TIME
    
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
        state_dict = node.get_current_state_dict()
        
        # we're done
        return state_dict
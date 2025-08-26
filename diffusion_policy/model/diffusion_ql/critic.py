import copy
import torch
import torch.nn as nn
import robomimic.models.base_nets as rmbn
from robomimic.models.obs_nets import ObservationEncoder
from robomimic.models.base_nets import MLP


from diffusion_policy.model.components.trunk import Trunk
from diffusion_policy.model.components.leaf import Leaf
from diffusion_policy.model.components.tree import Tree
from diffusion_policy.model.obs_encoder import ObsEncoderMaker

from diffusion_policy.model.components.dexnex_layers import CascadingCNNSpatialSoftmax

import diffusion_policy.globals as globals

# example, from https://github.com/NU-Haptics-Lab/Diffusion-Policies-for-Offline-RL#


"""
TODO - understand how to use the same weights for multiple image inputs while ensuring proper forward/backward passes are done, and the weights are updated correctly. I know this repo does it, just need to study it a bit
"""

class QLDenser(nn.Module):
    """
    Standard MLP ... could just use robomimic's
    """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, 1))
        
    def forward(self, image_features, non_image_features):
        x = torch.cat([image_features, non_image_features])
        x = self.model(input)
        return x

class QLModel(nn.Module):
    def __init__(self,
                 obs_encoder_maker: ObsEncoderMaker
                 ):
        super().__init__()
        
        # get the robomimic obs-encoder
        obs_encoder: ObservationEncoder = obs_encoder_maker.get()
        
        # calc the obs output
        
        # rootcaps
        rootcaps = {}
        
        # roots
        roots = {}

        # trunk
        trunk = nn.Sequential(
            obs_encoder,
            QLDenser(obs_encoder.output_shape())
        )
        
        branches = {}

        # leafs
        leafs = {}
        for key, val in enumerate(globals.REPLAY_BUFFER_LOADER.rbs.items()):
            leafs[key] = QLDenser(<>)

        # tree
        self.tree = Tree(
                         rootcaps,
                         roots,
                         trunk,
                         branches,
                         leafs)

    def forward(self, state_dict, action, options=None):
        """
        how is the state dict passed in?
        """
        pass
        <>
            
        

class DoubleCritic(nn.Module):
    def __init__(self,
                 qlmodel: QLModel
                 ):
        super().__init__()
        
        self.q1_model = qlmodel
        self.q2_model = copy.deepcopy(self.q1_model)
    
    def forward(self, state_dict, action, options=None):
        return self.q1_model(state_dict, action, options), self.q2_model(state_dict, action, options)
               
    def q1(self, state_dict, action):
        return self.q1_model(state_dict, action)
    
    def q_min(self, state_dict, action):
        q1, q2 = self.forward(state_dict, action)
        return torch.min(q1, q2)
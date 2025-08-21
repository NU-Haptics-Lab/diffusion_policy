import torch
import torch.nn as nn

from diffusion_policy.model.components.trunk import Trunk
from diffusion_policy.model.components.leaf import Leaf
from diffusion_policy.model.components.tree import Tree

from diffusion_policy.model.components.dexnex_layers import CascadingCNNSpatialSoftmax

import diffusion_policy.globals as globals

# example, from https://github.com/NU-Haptics-Lab/Diffusion-Policies-for-Offline-RL#


"""
TODO - understand how to use the same weights for multiple image inputs while ensuring proper forward/backward passes are done, and the weights are updated correctly. I know this repo does it, just need to study it a bit
"""

class QLImageEncoder(nn.Module):
    """
    Convert images into features with expected x, y values
    """
    def __init__(self):
        super().__init__()
        
        # for now just use resnet18
        self.model = CascadingCNNSpatialSoftmax()
        
    def forward(self, input):
        x = self.model(input)
        return x
    

class QLDenser(nn.Module):
    """
    Standard MLP
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
    def __init__(self):
        super().__init__()
        
        # rootcaps
        rootcaps = {}
        
        # roots
        roots = {}

        # trunk
        trunk = nn.Sequential(
            QLImageEncoder(),
            QLDenser(<>)
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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.q1_model = QLModel()
        self.q2_model = QLModel()
    
    def forward(self, state_dict, action, options=None):
        return self.q1_model(state_dict, action, options), self.q2_model(state_dict, action, options)
               
    def q1(self, state_dict, action):
        return self.q1_model(state_dict, action)
    
    def q_min(self, state_dict, action):
        q1, q2 = self.forward(state_dict, action)
        return torch.min(q1, q2)
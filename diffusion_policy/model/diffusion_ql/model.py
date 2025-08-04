import torch
import torch.nn as nn

from diffusion_policy.model.trunk import Trunk
from diffusion_policy.model.branch import Branch
from diffusion_policy.model.tree import Tree

# example, from https://github.com/NU-Haptics-Lab/Diffusion-Policies-for-Offline-RL#


"""
TODO - understand how to use the same weights for multiple image inputs while ensuring proper forward/backward passes are done, and the weights are updated correctly. I know this repo does it, just need to study it a bit
"""

class QLImageEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = <>
        
    def forward(self, input):
        x = self.model(input)
        return x
    

class QLDenser(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, 1))
        
    def forward(self, input):
        x = self.model(input)
        return x
    
class QLBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, 1))
        
    def forward(self, input):
        x = self.model(input)
        return x

class QLModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # construct the image encoder
        image_encoder = QLImageEncoder()
        
        # construct the denser part of the network
        denser = QLDenser(<>)

        # trunk
        trunk = Trunk(
            image_encoder,
            denser
        )

        # branches
        branches = {}
        for key, val in enumerate(CONFIG.datasets):
            branches[key] = Branch(QLBranch())

        # tree
        self.tree = Tree(trunk, branches)

    def forward(self, input, key):
        x = self.tree.forward(input, key)
        return x
        
    

    def forward(self, state_dict, action):
        # begin assembling features
        features = action
        
        # iterate over state history
        for <> in <>:
            # pass images into image encoder
            image_features = <>()
            
            # concat with rest of the features
            features = torch.cat([features, image_features])
            
            # concat the non-image features
            features = torch.cat([features, joint_state])
            features = torch.cat([features, tactile])
            
        # pass all the features into the rest of the net
        x = self.denser(features)
        
        # we're done
        return x
            
        

class QLCritic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.q1_model = QLModel()
        self.q2_model = QLModel()
    
    def forward(self, state_dict, action):
        return self.q1_model(state_dict, action), self.q2_model(state_dict, action)
               
    def q1(self, state_dict, action):
        return self.q1_model(state_dict, action)
    
    def q_min(self, state_dict, action):
        q1, q2 = self.forward(state_dict, action)
        return torch.min(q1, q2)
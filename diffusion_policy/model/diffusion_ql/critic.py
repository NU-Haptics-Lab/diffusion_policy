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
from diffusion_policy.common.pytorch_util import dict_apply

import diffusion_policy.globals as globals

# example, from https://github.com/NU-Haptics-Lab/Diffusion-Policies-for-Offline-RL#

class BaseCritic:
    """
    placeholder
    """
    def __init__(self):
        pass

"""
TODO - understand how to use the same weights for multiple image inputs while ensuring proper forward/backward passes are done, and the weights are updated correctly. I know this repo does it, just need to study it a bit
"""

class QLDenser(nn.Module):
    """
    Standard MLP ... could just use robomimic's
    """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Mish(),
                    )
        
    def forward(self, image_features, non_image_features):
        x = torch.cat([image_features, non_image_features])
        x = self.model(input)
        return x
    
    def output_shape(self, input_shape=None):
        return self.hidden_dim

class QLModel(nn.Module):
    def __init__(self,
                 obs_encoder_maker: ObsEncoderMaker,
                 if_stack_history: bool = True,
                 ):
        super().__init__()
        self.obs_encoder_maker = obs_encoder_maker
        self.if_stack_history = if_stack_history
        
        # if we're stacking the history
        if self.if_stack_history:
            self.MakeStackObsEncoder()

        # get the robomimic obs-encoder
        self.obs_encoder: ObservationEncoder = self.obs_encoder_maker.get()
                    
        # rootcaps
        rootcaps = {}
        
        # roots
        roots = {}

        # trunk
        # MLP based off obs encoder output shape. Output is a list, should be 1 long, so extract the first and only element
        self.trunk_denser = QLDenser(self.obs_encoder.output_shape()[0])
        trunk = nn.Sequential(
            self.obs_encoder,
            self.trunk_denser
        )
        
        branches = {}

        # leafs
        leafs = {}
        for key, val in enumerate(globals.REPLAY_BUFFER_LOADER.rbs.items()):
            leafs[key] = QLDenser(self.trunk_denser.output_shape())

        # tree
        self.tree = Tree(
                         rootcaps,
                         roots,
                         trunk,
                         branches,
                         leafs)
        
    def MakeStackObsEncoder(self):
        # if we're stacking the history, we must modify the shape meta 
        # local copy
        rgbs = copy.deepcopy(self.obs_encoder_maker.rgbs)
        lowdims = copy.deepcopy(self.obs_encoder_maker.rgbs)
        ch = self.obs_encoder_maker.ch
        cw = self.obs_encoder_maker.cw

        len_history = len(globals.CONFIG.obs_rel_indices)

        # iterate over keys
        for key, val in list(rgbs.items()) + list(lowdims.items()):
            # get the shape
            shape = val.shape

            # only need to stack the first axis
            shape[0] *= len_history

        # make the obs encoder maker
        oem = ObsEncoderMaker(rgbs, lowdims, ch, cw)

        # save it 
        self.obs_encoder_maker = oem
        
    def StackHistory(self, dd):
        def fcn(x):
            # assumes the data is [batch, history, ...]
            x = torch.flatten(x, start_dim=1, end_dim=2)
            return x
        
        out = dict_apply(dd, fcn)
        return out

    def forward(self, state_dict, action, options: dict = None):
        # stack the state history
        if self.if_stack_history:
            state_dict = self.StackHistory(state_dict)
            
        # combine the state and actions
        inputs_dict = state_dict
        inputs_dict["action"] = action

        self.tree.forward_options(inputs_dict, options)
            
        

class DoubleCritic(nn.Module, BaseCritic):
    def __init__(self,
                 qlmodel: QLModel
                 ):
        super().__init__()
        
        self.q1_model = qlmodel
        self.q2_model = copy.deepcopy(self.q1_model)
    
    def forward(self, state_dict, action, options: dict = None):
        q1 = self.q1_model(state_dict, action, options)
        q2 = self.q2_model(state_dict, action, options)
        return q1, q2
               
    def q1(self, state_dict, action):
        return self.q1_model(state_dict, action)
    
    def q_min(self, state_dict, action):
        q1, q2 = self.forward(state_dict, action)
        return torch.min(q1, q2)
    

def test():
    import hydra
    from omegaconf import OmegaConf
    txt = """
shape_meta:
  action:
    shape:
    - 21 # gofa (6), wr (2), th (5), ff (4), mf (4)

  img:
    shape:
    - 3
    - 192
    - 192
  img2:
    shape:
    - 3
    - 192
    - 192
  
  state: # the name "state" came from the zarr generation script
    shape:
    - 35 # gofa (6), wr (2), th (5), ff (4), mf (4), biotacs (5), th-pos (3), ff-pos (3), mf-pos (3)
    
qlmodel:
    _target_: diffusion_policy.model.diffusion_ql.critic.QLModel

    obs_encoder_maker:
        _target_: diffusion_policy.model.obs_encoder.ObsEncoderMaker

        rgbs:
            img:
                shape: ${shape_meta.img.shape}
            img2:
                shape: ${shape_meta.img2.shape}
            
        ch: 184
        cw: 184

        lowdims:
            state:
                shape: ${shape_meta.state.shape}
                
            action:
                shape: ${shape_meta.action.shape}
    """
    config = OmegaConf.create(txt)
    
    class Obj(object):
        pass
    
    globals.REPLAY_BUFFER_LOADER = Obj()
    globals.REPLAY_BUFFER_LOADER.rbs = {"test1": 1, "test2": 2}

    x: DoubleCritic = hydra.utils.instantiate(config)
    
    
    pass
    
    
if __name__ == "__main__":
    test()
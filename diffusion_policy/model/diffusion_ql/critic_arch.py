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

from diffusion_policy.utils import print_nb_params


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
        
    def forward(self, inputs):
        # x = torch.cat([image_features, non_image_features])
        x = self.model(inputs)
        if x._version == 1:
            print("hi")
            pass
        return x
    
    def output_shape(self, input_shape=None):
        return self.hidden_dim

class QLModel(nn.Module):
    def __init__(self,
                 obs_encoder_maker: ObsEncoderMaker,
                 if_stack_history: bool = True,
                 trunk_hidden_dim = 256,
                 leaf_hidden_dim = 128,
                 use_tree = False
                 ):
        super().__init__()
        self.obs_encoder_maker = obs_encoder_maker
        self.if_stack_history = if_stack_history
        self.use_tree = use_tree
        
        # if we're stacking the history
        if self.if_stack_history:
            self.MakeStackObsEncoder()

        # get the robomimic obs-encoder
        self.obs_encoder: ObservationEncoder = self.obs_encoder_maker.get()
                    
        # rootcaps
        rootcaps = nn.ModuleDict()
        
        # roots
        roots = nn.ModuleDict()

        # trunk
        # MLP based off obs encoder output shape. Output is a list, should be 1 long, so extract the first and only element
        dense_input = self.obs_encoder.output_shape()[0] + len(globals.CONFIG.action_rel_indices) * globals.CONFIG.shape_meta.action.shape[0]
        self.trunk_denser = QLDenser(dense_input, hidden_dim=trunk_hidden_dim)
        trunk = self.trunk_denser
        
        branches = nn.ModuleDict()

        # leafs
        if self.use_tree:
            leafs = nn.ModuleDict()
            for key, val in globals.REPLAY_BUFFER_LOADER.rbs.items():
                leafs[key] = nn.Sequential(
                    QLDenser(self.trunk_denser.output_shape(), hidden_dim=leaf_hidden_dim),
                    nn.Linear(leaf_hidden_dim, 1) # critic must output a single q-value
                )

            # tree
            self.tree = Tree(
                            rootcaps,
                            roots,
                            trunk,
                            branches,
                            leafs)
        else:
            t = nn.Sequential(
                trunk,
                nn.Linear(trunk.output_shape(), 1)
            )
            self.tree = t
            self.policy = t
        
    def MakeStackObsEncoder(self):
        # if we're stacking the history, we must modify the shape meta 
        # local copy
        rgbs = copy.deepcopy(self.obs_encoder_maker.rgbs)
        lowdims = copy.deepcopy(self.obs_encoder_maker.lowdims)
        ch = self.obs_encoder_maker.ch
        cw = self.obs_encoder_maker.cw

        len_history = len(globals.CONFIG.obs_rel_indices)

        # iterate over keys
        for key, val in list(rgbs.items()) + list(lowdims.items()):
            # get the shape
            shape = val.shape

            # only need to stack the first axis (the channels)
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
    
    def forward_obs(self, state, action):
        s = self.obs_encoder(state)
        
        # stack the actions
        a = torch.flatten(action, start_dim=1, end_dim=2)
        
        # concat along last dimension
        x = torch.concat([s, a], dim=-1)
        
        return x

    def forward(self, state_dict, action, options: dict = None):
        # stack the state history
        if self.if_stack_history:
            state_dict = self.StackHistory(state_dict)
            
        # encode the inputs
        x = self.forward_obs(state_dict, action)

        if self.use_tree:
            x = self.tree.forward_options(x, options)
        else:
            x = self.policy(x)
        
        return x
    
class QLModelSimple(QLModel):
    def __init__(self,
                 obs_encoder_maker: ObsEncoderMaker,
                 hidden_dim = 256
                 ):
        nn.Module.__init__(self)
        
        self.obs_encoder_maker = obs_encoder_maker

        # get the robomimic obs-encoder
        self.obs_encoder: ObservationEncoder = self.obs_encoder_maker.get()
        
        # just the observation, no actions
        dense_input = self.obs_encoder.output_shape()[0]
        
        self.dense = nn.Sequential(
                QLDenser(dense_input, hidden_dim = hidden_dim),
                nn.Linear(hidden_dim, 1) # critic must output a single q-value
            )
        
    def forward(self, state_dict, action, options: dict = None):
            
        # encode the inputs
        x = self.obs_encoder(state_dict)

        x = self.dense(x)
        
        return x
    
            
        

class DoubleCritic(nn.Module, BaseCritic):
    def __init__(self,
                 qlmodel: QLModel,
                 use_double_q = True
                 ):
        nn.Module.__init__(self)
        self.use_double_q = use_double_q
        
        self.q1_model = qlmodel
        
        if self.use_double_q:
            self.q2_model = copy.deepcopy(self.q1_model)

        print_nb_params(self.q1_model, "Critic params")
    
    def forward(self, state_dict, action, options: dict = None):
        q1 = self.q1_model(state_dict, action, options)
        
        if self.use_double_q:
            q2 = self.q2_model(state_dict, action, options)
        else:
            q2 = None
            
        return q1, q2
               
    def q1(self, state_dict, action):
        return self.q1_model(state_dict, action)
    
    def q_min(self, state_dict, action):
        q1, q2 = self.forward(state_dict, action)
        
        
        if self.use_double_q:
            out = torch.min(q1, q2)
        else:
            out = q1
            
        return out
    

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
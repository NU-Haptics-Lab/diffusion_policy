
import torch
from torch import nn

import numpy as np
from diffusion_policy.dataset.batch_loader import BatchLoader
from diffusion_policy.common import pytorch_util

class Attractor(nn.Module):
    """
    place a point-attractor at a specific xyz location, use that attractor to calculate an explicit gradient with which to inform the actor
    
    
    Because normalized actions are soft-bounded to [-1, 1], we would expect a max joint error of 2
    so a max summed joint error of 2 * action_size = 42
    so our weighting should be ~ 0.01 so that our loss is ~[0, 1]
    """
    def __init__(self,
                 weight,
                 *args, 
                 batch_loader: BatchLoader = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.weight = weight
        self.batch_loader = batch_loader
        
        self.setup()
        
    def setup(self):
            
        # do FK to get the target
        # self.target_state: np.array = self.fk(state, action)
        
        # taken from the demo dataset when the arm was near the block
        # shape must be [1, 1, action size]
        js = np.array([ 1.6091695 ,  1.1892474 , -0.7316676 , -0.41493666,  0.50543344,
        2.3390458 , -0.22566482, -0.48511946,  0.3490656 ,  0.07753903,
        1.570794  ,  0.02636739,  0.3490656 , -0.17283447,  1.5707941 ,
        0.26340103,  0.3931825 ,  1.2217318 ,  0.20944086, -0.56853324,
        0.15276071])
        
        # resize in place
        js.resize((1, 1, js.shape[0]))
        
        ## normalize
        # wrap in a data dict, which is what the batch loader expects
        dd = {"action": js}
        
        # put into 
        dd_torch = pytorch_util.dict_to_torch(dd)
        
        # norm and on GPU
        ndd_torch = self.batch_loader.transfer_and_norm(dd_torch)
        
        # save
        self.n_js = ndd_torch['action']
        pass
        
    def RelToAbsActions(self, state, action):
        if False:
            # rel_a = action
            # s = state[:, ] + rel_a
            pass
            
        else:
            s = action
            
        return s
        
    def forward(self, state, action):
        """
        inputs: state, action
        """
        
        # if relative
        s = self.RelToAbsActions(state, action)
        
        # s shape should be [batch size, history size, action size]
        
        # get the delta state
        ds = s - self.n_js
        
        # divide by radius of region of interest
        dr = ds / 1.0
        
        # gofa hack to account for greater link length
        dr[:, :, 0:6] = dr[:, :, 0:6] * 2.0
        
        # power 3 each element (or 2.0, or 1.0)
        ds3 = torch.pow(dr, 1.0)
        
        # abs
        abs1 = torch.abs(ds3)
        
        # sum along the action dim
        sum1 = torch.sum(abs1, dim = 2)
        
        # weight
        l = sum1 * self.weight
        
        # done, shape should be [batch size, history size, 1]
        return l
    
class EnergyPenalty(Attractor):
    """
    Penalize high energy usage
    """
    def setup(self):
        # no setup needed
        pass
    
    def forward(self, state, action):
        """
        compute the energy usage for
        """
        # get next state
        
        # if relative
        s = self.RelToAbsActions(state, action)
        
        # get the delta state
        r_shifted_s = s[:, :, 1:]
        l_shifted_s = s[:, :, 0:-1]
        
        ds = torch.abs(r_shifted_s - l_shifted_s)
        
        # power 2 each element
        ds2 = torch.pow(ds, 2.0)
        
        # gofa hack to account for greater link mass
        ds2[:, :, 0:6] = ds2[:, :, 0:6] * 4.0
        
        # sum along the action dim
        sum1 = torch.sum(ds2, dim = 2)
        
        # weight
        l = sum1 * self.weight
        
        # done, shape should be [batch size, history size, 1]
        return l
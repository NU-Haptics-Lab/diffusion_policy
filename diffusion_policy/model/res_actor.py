
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1DBasic
from torch import nn
import einops
from einops.layers.torch import Rearrange
import torch

import diffusion_policy.globals as globals


class ResActor(nn.Module):
    """
    Residual Actor model
    input: s, a
    output: delta-action
    """
    def __init__(self, 
                 model: ConditionalUnet1DBasic,
                 scale,
                 ) -> None:
        super().__init__()
        
        self.model = model
        self.scale = scale
        
    def forward(self, inputs, global_cond):
        """
        get the model outputs, soft-limit them so we only do local search. In other words enforce only local perturbations.
        
        inputs: inputs - should be the action
                global_cond - should be the state
        outputs the final NON-RES action
        """
        sample = einops.rearrange(inputs, 'b h t -> b t h')
        global_cond2 = einops.rearrange(global_cond, 'b h t -> b (h t)')

        x = self.model(sample, global_cond=global_cond2)
    
        # map actions from [-inf, inf] to [-scale, scale]
        x = self.scale * nn.Tanh()(x)
        
        # add on the residual
        assert(inputs.shape == x.shape)
        x = inputs + x
        
        # return the new actions
        return x
    
    @torch.no_grad()
    def infer(self, inputs, global_cond):
        return self.forward(inputs, global_cond)
    
    def reset(self):
        pass
    
class ResInference:
    def __init__(self) -> None:
        self.actor: DiffusionModel = globals.MODELS["actor"].get_model() #type:ignore
        self.res_actor: ResActor = globals.MODELS["res_actor"].get_model() #type:ignore
        
        # need for compat
        self.noise_scheduler = None
        
    @torch.no_grad()
    def infer(self, nobs_torch, task_id):
        
        # denoise using diffusion
        _, _, all_a = self.actor.infer(nobs_torch, task_id, self.noise_scheduler)
        
        # extract the state
        s = nobs_torch['state']
        
        # add on residual actions
        all_a2 = self.res_actor.infer(all_a, s)
        
        future_a = self.actor.get_future_actions(all_a2)
        
        # middle return value for compat, not used anymore
        return future_a, future_a, all_a2
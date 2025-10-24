
import torch
import torch.nn as nn
import torch.nn.utils
import diffusion_policy.globals as globals

from diffusion_policy.losses.batch_loss import WeightedBatchLoss
from diffusion_policy.losses.losses import Losses

from .common import CalcSumLoss

def MaxGrad(model):
    grad_clip = 0.0
    for param in model.get_model().parameters():
        g = param.grad
        if g is not None:
            grad_clip = max(grad_clip, g.max().item())
    return grad_clip

class StepTrainer:
    """
    Responsible for training for one step.

    Gets the summed loss from any number of WeightedBatchLoss's and then performs back propagation.

    It's a good idea to sum the weighted losses from all co-training datasets before doing back-prop because it should reduce variance and improve training stability, I believe.
    """

    def __init__(self,
                 w_batch_losses: dict,
                 grad_norm = 1.0
        ):
        # handle to node
        self.w_batch_losses = w_batch_losses
        self.grad_norm = grad_norm
        
    def backprop_model_loss(self, model, loss: torch.Tensor, grad_norm):
        # can skip backprop if the loss is zero (aka the loss was skipped due to freqs)
        if loss == 0.0:
            return
        
        # back propagation
        loss.backward(retain_graph=True)
        
        # if clipping the gradients
        if grad_norm is not None and grad_norm > 0: 
            # torch.nn.utils.clip_grad_value_(model.get_model().parameters(), clip_value=grad_norm)
            
            # this returns the total_norm, PRE normalization
            norms = torch.nn.utils.clip_grad_norm_(model.get_model().parameters(), max_norm=grad_norm) #type:ignore
            max_grad_norm = norms.max().item() # norms is a single value
        else:
            max_grad_norm = None 
                    
        return max_grad_norm
        
        
    def train_actor(self, actor_losses):
        if "actor" not in globals.CONFIG.models_to_train: #type:ignore
            return
        
        model = globals.MODELS["actor"]
        
        def fcn(name, grad_norm=self.grad_norm):
            l = actor_losses[name]
            grad_clip = self.backprop_model_loss(model, l, grad_norm)
            return grad_clip
        
        # first get the bc gradient norm
        globals.MODELS.reset() 
        bc_grad_norm = fcn("bc")
        
        # now get the dql gradient norm
        # globals.MODELS.reset() 
        # dql_max_grad = fcn("dql")
        
        # reset the gradients so we can limit the DQL gradient first
        globals.MODELS.reset() 
        
        self.dql_grad_ratio = 0.1
        if True and bc_grad_norm is not None:
            dql_grad_clip = bc_grad_norm * self.dql_grad_ratio
            
            # now do scaled dql loss. dql grad's are now 10% of bc's
            dql_max_grad = fcn("dql", dql_grad_clip)
            
            # if dql_max_grad is not None and dql_max_grad > 0.0:
            #     globals.LOGGER.log_one("actor/grad_max/dql", dql_max_grad)
            
            
        # now that the most constraining gradient has been clipped, add back on the bc gradient, NO GRAD NORMING now
        bc_grad_norm_2 = fcn("bc", 99.0)
            
        # add on the attractor loss, NO GRAD NORMING now
        attractor_max_grad = fcn("attractor", 99.0)
        
        # finally, step the model if grad > 0.0
        if attractor_max_grad is not None and attractor_max_grad > 0.0:
            model.step()
            
        # logging
        if bc_grad_norm is not None and bc_grad_norm > 0.0:
            globals.LOGGER.log_one("actor/grad_max/bc", bc_grad_norm)
        
        pass
        
            
            
        

    def train(self):
        # initialize a zero loss variable
        total_losses = CalcSumLoss(self.w_batch_losses)
        
        dd = {}
        
        # actor must be first, else you'll get an in-place operation error
        # if "actor" in total_losses.keys():
        #     # can fix by re-ordering in the yaml
        #     assert("actor" == list(total_losses.keys())[0])
            
        # actor
        if "actor" in total_losses.keys():
            self.train_actor(total_losses['actor'])
        
        # do one at a time
        # hack
        for key, loss in [('critic', total_losses['critic'])]:
            # reset optimizer gradients
            globals.MODELS.reset() 
            
            # can skip backprop if the loss is zero (aka the loss was skipped due to freqs)
            if loss == 0.0:
                continue
            
            if key not in globals.CONFIG.models_to_train: #type:ignore
                continue
            
            model = globals.MODELS[key]
            
            # back propagation
            loss.backward()
            
            # logging
            dd[key + ": weighted sum loss"] = loss
            
            model = globals.MODELS[key]
            
            # if clipping the gradients
            if self.grad_norm > 0: 
                # torch.nn.utils.clip_grad_value_(model.get_model().parameters(), clip_value=self.grad_norm) #type:ignore
                
                # maxgrad = MaxGrad(model)
                # dd[key + ": Grad Max"] = maxgrad
                norms = torch.nn.utils.clip_grad_norm_(model.get_model().parameters(), max_norm=self.grad_norm) #type:ignore
                dd[key + ": grad_norm"] = norms.max().item()
            
            # step
            model.step()
        
        globals.LOGGER.log(dd)
        
        # reset optimizer gradients
        globals.MODELS.reset() 
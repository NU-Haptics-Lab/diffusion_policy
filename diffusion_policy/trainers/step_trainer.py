
import torch
import torch.nn as nn
import torch.nn.utils
import diffusion_policy.globals as globals
from diffusion_policy import utils

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
                 w_batch_losses: dict[str, WeightedBatchLoss],
                 grad_norm = 1.0
        ):
        # handle to node
        self.w_batch_losses = w_batch_losses
        self.grad_norm = grad_norm
        
    def setup(self):
        # global params
        self.use_bc_loss = globals.CONFIG.use_bc_loss # type:ignore
        self.models_to_train = globals.CONFIG.models_to_train # type:ignore
        
        for k, v in self.w_batch_losses.items():
            v.setup()
        
    def backprop_model_loss(self, model, loss: torch.Tensor, grad_norm):
        # can skip backprop if the loss is zero (aka the loss was skipped due to freqs)
        if loss == 0.0:
            return
        
        if grad_norm == 0.0:
            # don't include this loss if the grad-norm is zero
            return
        
        # back propagation
        loss.backward(retain_graph=True)
        
        # if clipping the gradients
        if grad_norm is not None and grad_norm > 0: 
            # torch.nn.utils.clip_grad_value_(model.get_model().parameters(), clip_value=grad_norm)
            
            # this returns the total_norm, PRE normalization
            norms = torch.nn.utils.clip_grad_norm_(model.get_model().parameters(), max_norm=grad_norm) #type:ignore
            
            # norms is a single value
            max_grad_norm = norms.max().item() 
        else:
            max_grad_norm = None 
                    
        return max_grad_norm
        
        
    def train_actor(self, actor_losses):
        if "actor" not in self.models_to_train:
            return
        
        model = globals.MODELS["actor"]
        
        def fcn(name, grad_norm=self.grad_norm):
            l = actor_losses[name]
            grad_clip = self.backprop_model_loss(model, l, grad_norm)
            return grad_clip
        
        # first get the bc gradient norm
        globals.MODELS.reset() 
        bc_grad_norm = None
        if self.use_bc_loss:
            bc_grad_norm = fcn("bc")
            
            if bc_grad_norm is not None and bc_grad_norm > 0.0:
                globals.LOGGER.log_one("actor/grad_max/bc", bc_grad_norm)
        
        # just realized it only makes sense to tie dql grad to BC grad for datasets that we're cloning behavior from (aka, non online-rl datasets), because it'll be a very large arbitrary value for online-rl datasets
        
        # But I think we still want the online experience to inform the actor via the critic
        
        # right now I sum up losses across tasks before this method, perhaps it makes more sense to return all losses for all tasks separately and then choose how to deal with them at this level
        
        # that'll require a small rewrite, so for now just reduce the sim_online_rlx task weights
                
        # reset the gradients so we can limit the DQL gradient first
        globals.MODELS.reset() 
        
        self.dql_grad_ratio = 0.01
        if True and bc_grad_norm is not None:
            dql_max_grad = bc_grad_norm * self.dql_grad_ratio
        else:
            dql_max_grad = 1.0 # 0.05 # keep at 1.0 to be just like the original DQL paper's code
            
            # now do scaled dql loss. dql grad's are now 10% of bc's
        # if my understanding of the math is correct, then reducing grad_norm by 10x is the same as reducing l.r. by 10x, AS LONG as this is the only grad term.
        # I find that when the BC grad is ~0.1 that the behavior is decent, so set the DQL grad_norm = to 0.05??
        dql_max_grad = fcn("dql", grad_norm=dql_max_grad)
            
        if dql_max_grad is not None and dql_max_grad > 0.0:
            globals.LOGGER.log_one("actor/grad_max/dql", dql_max_grad)
            
            # if dql_max_grad is not None and dql_max_grad > 0.0:
            #     globals.LOGGER.log_one("actor/grad_max/dql", dql_max_grad)
            
            
        # now that the most constraining gradient has been clipped, add back on the bc gradient, NO GRAD NORMING now
        if self.use_bc_loss:
            bc_grad_norm_2 = fcn("bc", 99.0)
            
        # add on the attractor loss
        attractor_max_grad = fcn("attractor", 99.0)
        if attractor_max_grad is not None and attractor_max_grad > 0.0:
            globals.LOGGER.log_one("actor/grad_max/attractor", attractor_max_grad)
        
        # finally, step the model if grad > 0.0
        if utils.GlobalStepFreqTrigger('actor'):
            model.step()
            
        # logging
        
        pass
        
            
            
        

    def train(self):
        globals.MODELS.reset() 
        # initialize a zero loss variable
        total_losses = CalcSumLoss(self.w_batch_losses)
        
        dd = {}
        
        # actor must be first, else you'll get an in-place operation error
        if "actor" in total_losses.keys():
            # can fix by re-ordering in the yaml
            assert("actor" == list(total_losses.keys())[0])
            
            # hackish
            actor_losses = total_losses['actor']
            loss_sum = actor_losses['dql'] + actor_losses['bc'] + actor_losses['attractor']
            
            # hack sum up losses
            total_losses['actor'] = loss_sum
            
        # # actor
        # if "actor" in total_losses.keys():
        #     self.train_actor(total_losses['actor'])
        
        # do one at a time
        # hack
        # for key, loss in [('critic', total_losses['critic'])]:
        for key, loss in total_losses.items():
            # reset optimizer gradients
            globals.MODELS.reset() 
            
            # can skip backprop if the loss is zero (aka the loss was skipped due to freqs)
            if loss == 0.0:
                continue
            
            if key not in self.models_to_train:
                continue
            
            # nan protection
            if torch.isnan(loss):
                raise ValueError(f"Loss for {key} is NaN. Check the BatchLoss method.")
            
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
            
    def reset(self):
        # reset batch losses
        for k, v in self.w_batch_losses.items():
            v.reset()
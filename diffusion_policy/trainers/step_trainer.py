
import torch
import torch.nn as nn
import torch.nn.utils
import diffusion_policy.globals as globals

from diffusion_policy.losses.batch_loss import WeightedBatchLoss
from diffusion_policy.losses.losses import Losses

from .common import CalcSumLoss


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

    def train(self):
        # initialize a zero loss variable
        total_losses = CalcSumLoss(self.w_batch_losses)
        
        dd = {}
        
        # actor must be first, else you'll get an in-place operation error
        if "actor" in total_losses.keys():
            # can fix by re-ordering in the yaml
            assert("actor" == list(total_losses.keys())[0])
        
        # do one at a time
        for key, loss in total_losses.items():
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
            
        # # now step
        # for key in total_losses:
        #     if key not in globals.CONFIG.models_to_train:
        #         continue
            
            model = globals.MODELS[key]
            
            # if clipping the gradients
            if self.grad_norm > 0: 
                norms = torch.nn.utils.clip_grad_norm_(model.get_model().parameters(), max_norm=self.grad_norm, norm_type=2) #type:ignore
                dd[key + ": Grad Norm"] = norms.max().item()
            
            # step
            model.step()
        
        globals.LOGGER.log(dd)
        
        # reset optimizer gradients
        globals.MODELS.reset() 

        # logging
        # self.raw_loss_cpu = total_loss.item()
        
        # not very helpful for tracking training progress
        # log = {
        #     'total_train_loss': self.raw_loss_cpu,
        # }
        # globals.LOGGER.log(log)
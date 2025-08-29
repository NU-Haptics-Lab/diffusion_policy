
import torch
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
                 w_batch_losses: dict
        ):
        # handle to node
        self.w_batch_losses = w_batch_losses

    def train(self):
        # initialize a zero loss variable
        total_losses = CalcSumLoss(self.w_batch_losses)
        # fjskdl = Losses(total_losses)
        
        # do one at a time
        for key, val in total_losses.items():
            # reset gradients
            globals.MODELS.reset() 
            
            # back propagation
            val.backward()
            
            # step
            globals.MODELS[key].step()
        
        # reset gradients
        globals.MODELS.reset() 

        # # calculate gradients for all losses and for all datasets simultaneously
        # fjskdl.backward()

        # # now we can step all of the models
        # globals.MODELS.Step()

        # logging
        # self.raw_loss_cpu = total_loss.item()
        
        # not very helpful for tracking training progress
        # log = {
        #     'total_train_loss': self.raw_loss_cpu,
        # }
        # globals.LOGGER.log(log)
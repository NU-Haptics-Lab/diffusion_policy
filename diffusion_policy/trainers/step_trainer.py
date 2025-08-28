
import torch
import diffusion_policy.globals as globals

from diffusion_policy.losses.batch_loss import WeightedBatchLoss

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
        total_loss = CalcSumLoss(self.w_batch_losses)

        # calculate gradients for all datasets simultaneously
        total_loss.backward()

        # now we can step all of the models
        globals.MODELS.Step()

        # logging
        raw_loss_cpu = total_loss.item()
        # tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
        # train_losses.append(raw_loss_cpu)
        
        step_log = {
            'total_train_loss': raw_loss_cpu,
        }
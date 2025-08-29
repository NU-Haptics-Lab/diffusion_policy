import torch
import numpy as np

import diffusion_policy.globals as globals
from diffusion_policy.losses.batch_loss import WeightedBatchLoss

from diffusion_policy.common.pytorch_util import dict_tensor_to

def CalcSumLoss(w_batch_losses):
    # initialize a zero loss variable
    # TODO: make the shape not hard coded
    total_losses = {
        'actor': torch.tensor([0.0], requires_grad=True),
        'critic': torch.tensor([0.0], requires_grad=True),
    }
    
    # move to device
    device = torch.device(globals.CONFIG.device)
    total_losses = dict_tensor_to(total_losses, device)

    batch_loss: WeightedBatchLoss
    # iterate over the batch losses and get the total loss
    for key, batch_loss in w_batch_losses.items():
        losses = batch_loss.compute_weighted_loss()
        
        for key, val in losses.items():
            total_losses[key] = total_losses[key] + val
        
    return total_losses


def CalcSumEval(w_batch_losses):
    # initialize a zero loss variable
    total_evals = np.zeros(2)

    batch_loss: WeightedBatchLoss
    # iterate over the batch losses and get the total loss
    for key, batch_loss in w_batch_losses.items():
        wevals = batch_loss.compute_weighted_eval()

        total_evals += wevals
        
    return total_evals
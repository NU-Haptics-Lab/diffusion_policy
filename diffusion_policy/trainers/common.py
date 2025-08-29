import torch
import numpy as np

import diffusion_policy.globals as globals
from diffusion_policy.losses.batch_loss import WeightedBatchLoss

def CalcSumLoss(w_batch_losses):
    # initialize a zero loss variable
    total_loss = torch.tensor([0.0], requires_grad=True)
    
    # move to device
    device = torch.device(globals.CONFIG.device)
    total_loss = total_loss.to(device)

    batch_loss: WeightedBatchLoss
    # iterate over the batch losses and get the total loss
    for key, batch_loss in w_batch_losses.items():
        loss = batch_loss.compute_weighted_loss()

        total_loss = total_loss + loss
        
    return total_loss


def CalcSumEval(w_batch_losses):
    # initialize a zero loss variable
    total_evals = np.zeros(2)

    batch_loss: WeightedBatchLoss
    # iterate over the batch losses and get the total loss
    for key, batch_loss in w_batch_losses.items():
        wevals = batch_loss.compute_weighted_eval()

        total_evals += wevals
        
    return total_evals
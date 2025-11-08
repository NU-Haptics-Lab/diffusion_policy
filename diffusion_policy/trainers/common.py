import torch
import numpy as np

import diffusion_policy.globals as globals
from diffusion_policy.losses.batch_loss import WeightedBatchLoss

from diffusion_policy.common.pytorch_util import dict_tensor_to, dict_apply

def CalcSumLoss(w_batch_losses):
    # initialize a zero loss torch tensor
    # TODO: make the shape not hard coded
    total_losses = {
        'actor': {
            'bc': torch.tensor([0.0], requires_grad=True),
            'dql': torch.tensor([0.0], requires_grad=True),
            'attractor': torch.tensor([0.0], requires_grad=True),
        },
        'critic': torch.tensor([0.0], requires_grad=True),
    }
    
    # for key in globals.CONFIG.models_to_train:
    #     total_losses[key] = torch.tensor([0.0], requires_grad=True)
    
    # move total_losses to device
    device = torch.device(globals.CONFIG.device)
    total_losses = dict_tensor_to(total_losses, device)

    batch_loss: WeightedBatchLoss
    
    # iterate over the batch losses and get the total loss
    for key, batch_loss in w_batch_losses.items():
        # task toggle check
        if key not in globals.CONFIG.tasks_to_use:
            continue
        
        # dict
        losses = batch_loss.compute_weighted_loss()
        
        # TODO: write a proper recursion .... and a class
        for key, val in losses.items():
            if not isinstance(val, dict):
                total_losses[key] = total_losses[key] + val
            else:
                for key2, val2 in val.items():
                    total_losses[key][key2] = total_losses[key][key2] + val2
        
    return total_losses


def CalcSumEval(w_batch_losses):
    # initialize a zero loss variable
    total_evals = np.zeros(2)

    batch_loss: WeightedBatchLoss
    # iterate over the batch losses and get the total loss
    for key, batch_loss in w_batch_losses.items():
        # task toggle check
        if key not in globals.CONFIG.tasks_to_use:
            continue
        
        wevals = batch_loss.compute_weighted_eval()

        total_evals += wevals
        
    return total_evals
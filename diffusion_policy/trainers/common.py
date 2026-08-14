import torch
import numpy as np

import diffusion_policy.globals as globals
from diffusion_policy.losses.batch_loss import WeightedBatchLoss
from diffusion_policy import utils

from diffusion_policy.common.pytorch_util import dict_tensor_to, dict_apply

def CalcSumLoss(w_batch_losses):
    # initialize a zero loss torch tensor
    # TODO: make the shape not hard coded
    total_losses = {
        'actor': {
            'bc': torch.tensor([0.0], requires_grad=True),
            'dql': torch.tensor([0.0], requires_grad=True),
            'attractor': torch.tensor([0.0], requires_grad=True),
            'keypoint': torch.tensor([0.0], requires_grad=True),
            'pellet_xyz': torch.tensor([0.0], requires_grad=True),
        },
        'res_actor': utils.InitZeroTensorOnDevice(),
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
        if key not in globals.CONFIG.tasks_to_use: #type:ignore
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
    """
    Returns (total_evals, per_dataset_evals): total_evals is the [loss, mse]
    pair summed (weight-combined) across every dataset/rb_id in
    w_batch_losses, as before. per_dataset_evals is a dict rb_id -> raw
    (un-weighted) [loss, mse] for that dataset alone, from the SAME eval()
    call used to build total_evals (not a second call -- eval() advances the
    val dataloader, so calling it twice per rb_id would consume two different
    batches and double the validation forward-pass cost).
    """
    # initialize a zero loss variable? CPU
    total_evals = np.zeros(2)
    per_dataset_evals = {}

    batch_loss: WeightedBatchLoss
    # iterate over the batch losses and get the total loss
    for key, batch_loss in w_batch_losses.items():
        # len check
        if len(batch_loss) == 0:
            continue

        # task toggle check
        if key not in globals.CONFIG.tasks_to_use:
            continue

        raw_evals = np.array(batch_loss.batch_loss.eval())
        per_dataset_evals[key] = raw_evals

        total_evals += batch_loss.weight * raw_evals

    return total_evals, per_dataset_evals
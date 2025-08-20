import torch

def CalcSumLoss(w_batch_losses):
    # initialize a zero loss variable
    total_loss = torch.tensor([0.0], requires_grad=True)

    # iterate over the batch losses and get the total loss
    for key, batch_loss in w_batch_losses.items():
        loss = batch_loss.compute_weighted_loss()

        total_loss += loss
        
    return total_loss
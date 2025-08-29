import numpy as np

import diffusion_policy.globals as globals

from diffusion_policy.dataset.batch_loader import BatchLoader
from diffusion_policy.losses.actor_loss import ActorLoss
from diffusion_policy.losses.critic_loss import CriticLoss


class BatchLoss:
    """
    Responsible for computing loss for one batch.
    Bundles the batch loader for a dataset and the policy heads associated with that dataset. Useful in co-training
    """

    def __init__(self,
        batch_loader: BatchLoader,
        eta: float # weight of the critic loss
        ):
        self.batch_loader = batch_loader
        self.eta = eta
        
        # save handles to nodes
        self.actor = globals.MODELS["actor"]
        self.actor_model = globals.MODELS["actor"].get_model()
        self.critic = globals.MODELS["critic"]
        
        # get the rb_id
        self.rb_id = self.batch_loader.rb_id

    def compute_loss(self):
        """
        Train for one batch.
        """
        # get the batch from the batch loader
        nbatch = next(self.batch_loader)

        # get the actor loss
        actor_loss = self.actor.loss(nbatch, self.rb_id)

        # get the critic loss
        critic_loss = self.critic.loss(nbatch, self.rb_id)

        # weighted sum them
        loss = actor_loss + self.eta * critic_loss

        # we're done
        return loss
    
    def eval(self):
        # get batch
        nbatch = next(self.batch_loader)

        # get the actor loss
        actor_loss = self.actor.loss(nbatch, self.rb_id)

        # get the action mse error
        action_mse_error = self.actor_model.get_val_action_mse_error(nbatch)

        return actor_loss, action_mse_error

    
class WeightedBatchLoss:
    """
    Bundles a BatchLoss and a weight parameter. Useful in co-training
    """
    def __init__(self,
        batch_loss: BatchLoss,
        weight: float
        ):
        self.batch_loss = batch_loss
        self.weight = weight

    def compute_weighted_loss(self):
        wloss = self.weight * self.batch_loss.compute_loss()
        return wloss
    
    def compute_weighted_eval(self):
        wevals = self.weight * np.array(self.batch_loss.eval())
        return wevals

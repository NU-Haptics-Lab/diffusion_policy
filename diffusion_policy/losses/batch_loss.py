import numpy as np

import diffusion_policy.globals as globals

from diffusion_policy.dataset.batch_loader import BatchLoader
from diffusion_policy.losses.actor_loss import ActorLoss
from diffusion_policy.model.diffusion_ql.diffusion_ql_loss import CriticLoss

from diffusion_policy.common.pytorch_util import dict_apply


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
        # # get the batch from the batch loader
        # nbatch = next(self.batch_loader)

        # # get the BC loss
        # actor_loss = self.actor.loss(nbatch, self.rb_id)

        # # get the DQL loss
        # critic_loss = self.critic.loss(nbatch, self.rb_id)

        # # weighted sum them
        # loss = actor_loss + self.eta * critic_loss

        # we're done
        return loss
    
    def eval(self):
        # get batch
        nbatch = next(self.batch_loader)

        # get the actor loss
        actor_loss = self.actor.loss(nbatch, self.rb_id)

        # get the action mse error
        action_mse_error = self.actor_model.get_val_action_mse_error(nbatch)

        return actor_loss.cpu(), action_mse_error
    
class DQLBatchLoss(BatchLoss):
    """
    Compute actor and critic loss and return them in a dictionary
    """
    
    def compute_loss(self):
        """
        compute loss for one batch.
        """
        losses = {}
        
        # get the batch from the batch loader
        nbatch = next(self.batch_loader)
        
        if "actor" in globals.CONFIG.models_to_train:
            # get the BC loss
            bc_loss = self.actor.loss(nbatch, self.rb_id)
            losses['actor'] = bc_loss
        
        if "critic" in globals.CONFIG.models_to_train:
            # get the DQL losses
            dql_actor_loss, dql_critic_loss = self.critic.loss(nbatch, self.rb_id)
            losses['critic'] = dql_critic_loss
            
        # # TODO: if train actor and use dql
        # if True:
        #     losses['actor'] = losses['actor'] + self.eta * dql_actor_loss
        
        # we're done
        return losses
    

class CriticBatchLoss(BatchLoss):
    """
    Compute actor and critic loss and return them in a dictionary
    """
    
    def compute_loss(self):
        """
        compute loss for one batch.
        """
        # get the batch from the batch loader
        nbatch = next(self.batch_loader)

        # get the DQL losses
        dql_actor_loss, dql_critic_loss = self.critic.loss(nbatch, self.rb_id)
        
        # summed critic loss
        critic_loss = dql_critic_loss
        
        losses = {
            'critic': critic_loss
        }
        
        # we're done
        return losses
    
    # TODO: rename all eval to validate
    def eval(self):
        # get batch, batch_loader handles train vs val mode
        nbatch = next(self.batch_loader)

        # get the critic loss
        dql_actor_loss, dql_critic_loss = self.critic.loss(nbatch, self.rb_id)
    
        return dql_critic_loss.cpu()
        

    
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
        losses = self.batch_loss.compute_loss()
        
        # apply the weighting to each loss
        wloss = dict_apply(losses, lambda x: self.weight * x)
        return wloss
    
    # TODO: rename eval to validate
    def compute_weighted_eval(self):
        # this batch_loss should already be in val mode
        wevals = self.weight * np.array(self.batch_loss.eval())
        return wevals

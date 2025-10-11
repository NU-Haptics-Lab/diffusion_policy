import numpy as np

import torch

import diffusion_policy.globals as globals

from diffusion_policy.dataset.batch_loader import BatchLoader
from diffusion_policy.losses.actor_loss import ActorLoss
from diffusion_policy.model.diffusion_ql.diffusion_ql_loss import CriticLoss

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sarsa_sampler import DatasetSampler, Indices
from diffusion_policy.dataset.train_and_val import TrainAndVal

from diffusion_policy.model.model import ModelEmaOptim
from diffusion_policy.model.diffusion_ql.diffusion_ql_loss import CriticLoss

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
        self.actor: ModelEmaOptim = globals.MODELS["actor"] #type:ignore
        self.actor_model = self.actor.get_model()
        self.critic: CriticLoss = globals.MODELS["critic"] #type:ignore
        
        # get the rb_id
        self.rb_id = self.batch_loader.rb_id

        # my members
        self.current_batch: dict

    def compute_loss(self):
        """
        Train for one batch.
        """
        # # get the batch from the batch loader
        nbatch = next(self.batch_loader)
        self.current_batch = nbatch # save

        # # get the BC loss
        actor_loss = self.actor.loss(nbatch, self.rb_id)

        # # get the DQL loss
        # critic_loss = self.critic.loss(nbatch, self.rb_id)

        # # weighted sum them
        # loss = actor_loss + self.eta * critic_loss

        # we're done
        return actor_loss
    
    def eval(self):
        # get batch
        nbatch = next(self.batch_loader)

        # get the actor loss
        actor_loss = self.actor.loss(nbatch, self.rb_id)

        # get the action mse error
        action_mse_error = self.actor_model.get_val_action_mse_error(nbatch)

        return actor_loss.cpu(), action_mse_error
    
class TTREfficiencyWeightedBatchLoss(BatchLoss):
    """
    TTR - time-to-reward.
    Weight each sample in the batch based on the time-to-reward effiency
    """
    def __init__(self, batch_loader, eta):
        super().__init__(batch_loader, eta)

        self.setup()

    def setup(self):
        """
        compute the efficiencies.

        Might have to correct if we pad the dataset episodes, not sure
        """
        ttrs_reversed = []
        dataloader: TrainAndVal = globals.DATALOADERS[self.rb_id]
        dss: DatasetSampler = dataloader.sampler #type:ignore ide error from sars vs sarsa. can ignore

        # traverse the dataset one episode at a time, from end of the dataset to the beginning
        for ep in reversed(dss.episodes):
            # loop vars
            reward_timestep = None

            # traverse backwards
            for i in reversed(range(len(ep))):
                r = ep.get_reward(i)

                # update reward
                if r > 0.0:
                    # check for false positive
                    # assume any time-to-reward less than 2.0 seconds (20 steps) was a false positive from the end of the previous episode so don't update the reward
                    if i > 20:
                        reward_timestep = i

                # check if we have a reward timestamp
                if reward_timestep is None:
                    ttr = None
                else:
                    # timesteps-to-reward
                    ttr = reward_timestep - i
                
                id = ep.get_id(i)
                ttrs_reversed.append(ttr)

        # reverse back to forward
        ttrs = ttrs_reversed.reverse()

        ## post process to remove outliers
        ttrs = np.array(ttrs)

        # replace nan's with the max so they'll map to 0.0 extra weight
        ttrs[np.isnan(ttrs)] = np.max(ttrs)
        
        # efficiency between [0, 1] where 0 == max ttr, and 1 == min ttr
        self.efficiencies = (np.max(ttrs) - ttrs) / (np.max(ttrs) - np.min(ttrs))

    def get_weights(self, indices):
        """
        Each sample starts with weight 1.0, and is given a higher weight if its trajectory more efficiently gets to a reward state
        """
        b = indices.shape[0]
        weights = torch.ones([b, 1])

        # time-to-reward efficiency
        ttr_eff = self.efficiencies[indices]

        # want to give a sample more weight if it's more efficient
        weights += ttr_eff

        return weights


    def compute_loss(self):
        loss = super().compute_loss()

        indices = self.current_batch["rb_index"]

        batch_weights = self.get_weights(indices)

        # element-wise multiplication
        weighted_loss = torch.mul(loss, batch_weights)

        return weighted_loss
    
class DQLBatchLoss(BatchLoss):
    """
    Compute actor and critic loss and return them in a dictionary
    """
    
    def compute_loss(self):
        """
        compute loss for one batch.
        """
        losses = {}
        
        # flags
        models_to_train: list = globals.CONFIG.models_to_train # type: ignore
        use_dql: bool = globals.CONFIG.use_dql # type: ignore

        train_actor = "actor" in models_to_train
        train_critic = "critic" in models_to_train
        need_dql_actor_loss = train_actor and use_dql
        
        # get the batch from the batch loader
        nbatch = next(self.batch_loader)
        
        if train_actor:
            # get the BC loss
            bc_loss = self.actor.loss(nbatch, self.rb_id)
            losses['actor'] = bc_loss
        
        # need critic loss if we're training critic, need actor loss if we're using dql
        if train_critic or need_dql_actor_loss:
            # get the DQL losses
            dql_actor_loss, dql_critic_loss = self.critic.loss(nbatch, self.rb_id)
            
            if train_critic:
                losses['critic'] = dql_critic_loss
            
            # dql_actor_loss shouldn't be None as long as need_dql_actor_loss is True, but I had to add it for pylance
            if need_dql_actor_loss and dql_actor_loss is not None:
                losses['actor'] = losses['actor'] + self.eta * dql_actor_loss
        
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

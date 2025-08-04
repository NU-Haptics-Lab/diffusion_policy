import diffusion_policy.globals

from diffusion_policy.dataset.batch_loader import BatchLoader
from diffusion_policy.trainers.actor_trainer import ActorTrainer
from diffusion_policy.losses.critic_loss import CriticTrainer


class BatchLoss:
    """
    Responsible for computing loss for one batch.
    Bundles the batch loader for a dataset and the policy heads associated with that dataset. Useful in co-training
    """

    def __init__(self,
        batch_loader: BatchLoader,
        actor
        ):
        pass

    def compute_loss(self):
        """
        Train for one batch.
        """
        # get the batch from the batch loader
        nbatch = next(self.batch_loader)

        # get the actor loss
        actor_loss = self.actor_loss(nbatch)

        # get the critic loss
        critic_loss = self.critic_loss(nbatch)

        # weighted sum them
        loss = actor_loss + self.eta * critic_loss

        # we're done
        return loss
    
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
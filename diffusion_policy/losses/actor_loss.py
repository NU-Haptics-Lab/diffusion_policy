

class ActorLoss:
    """
    OBSOLETED -- use Model instead
    """

    def __call__(self, nbatch):
        # compute loss
        raw_loss = self.model.compute_loss(nbatch)
        loss = raw_loss / CONFIG.training.gradient_accumulate_every

        return loss
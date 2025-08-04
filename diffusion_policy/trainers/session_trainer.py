import diffusion_policy.globals
from diffusion_policy.trainers.epoch_trainer import EpochTrainer

class SessionTrainer:
    """
    Replacing workspaces, since that name is not descriptive.

    SessionTrainer is responsible for a training session, including loading data, policies, training, evaluation, and logging
    """
    def __init__(self,
        epoch_trainer: EpochTrainer,
        nb_epochs: int
        ):
        # 
        self.epoch_trainer = epoch_trainer
        self.nb_epochs = nb_epochs

    def train(self):
        """
        Train for one session
        """
        for nb_epoch in range(self.nb_epochs):
            # train for one epoch
            self.epoch_trainer.train()

            # end of epoch logging
            pass

        # end of session stuff
        pass
import diffusion_policy.globals as globals
from diffusion_policy.trainers.epoch_trainer import EpochTrainer, EpochEvaluator

class SessionTrainer:
    """
    Replacing workspaces, since that name is not descriptive.

    SessionTrainer is responsible for a training session, including loading data, policies, training, evaluation, and logging
    """
    def __init__(self,
        epoch_trainer: EpochTrainer,
        epoch_evaluator: EpochEvaluator,
        nb_epochs: int
        ):
        # 
        self.epoch_trainer = epoch_trainer
        self.epoch_evaluator = epoch_evaluator
        self.nb_epochs = nb_epochs

    def train(self):
        """
        Train for one session
        """
        for nb_epoch in range(self.nb_epochs):
            # train for one epoch
            self.epoch_trainer.train()
            
            # eval for one epoch
            self.epoch_evaluator.eval()

            # end of epoch logging
            globals.EPOCH += 1
            globals.LOGGER.log_one("epoch", globals.EPOCH)
            
            # checkpoints
            globals.CHECKPOINTER.save()
            
            

        # end of session stuff
        pass
import diffusion_policy.globals as globals
from diffusion_policy.trainers.epoch_trainer import EpochTrainer, EpochValidator
from diffusion_policy.rollout import Rollout

class SessionTrainer:
    """
    Replacing workspaces, since that name is not descriptive.

    SessionTrainer is responsible for a training session, including loading data, policies, training, evaluation, and logging
    """
    def __init__(self,
        epoch_trainer: EpochTrainer,
        epoch_evaluator: EpochValidator,
        rollouts: Rollout = None,
        nb_epochs: int = 0,
        ):
        # 
        self.epoch_trainer = epoch_trainer
        self.epoch_evaluator = epoch_evaluator
        self.rollouts = rollouts
        self.nb_epochs = nb_epochs

    def run(self):
        """
        Run for one session
        """
        globals.LOGGER.log_one("epoch", globals.EPOCH) # edge-case compat for when we save <1 epoch after beginning the session
            
        while globals.EPOCH < globals.CONFIG.total_num_epochs: #type:ignore
            # train for one epoch
            self.epoch_trainer.train()
            
            # eval for one epoch
            self.epoch_evaluator.validate()
            
            # rollouts
            if self.rollouts is not None:
                self.rollouts.run()
            
            # checkpoints
            globals.CHECKPOINTER.save()

            # end of epoch logging
            globals.EPOCH += 1
            globals.LOGGER.log_one("epoch", globals.EPOCH)
            
            

        # end of session stuff
        pass
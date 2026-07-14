import diffusion_policy.globals as globals
from diffusion_policy.trainers.epoch_trainer import EpochTrainer, EpochValidator

class SessionTrainer:
    """
    Replacing workspaces, since that name is not descriptive.

    SessionTrainer is responsible for a training session, including loading data, policies, training, evaluation, and logging
    """
    def __init__(self,
        epoch_trainer: EpochTrainer,
        epoch_evaluator: EpochValidator,
        nb_epochs: int = 0,
        ):
        # 
        self.epoch_trainer = epoch_trainer
        self.epoch_evaluator = epoch_evaluator
        self.nb_epochs = nb_epochs
        
    def setup(self):
        self.epoch_trainer.setup()
        self.epoch_evaluator.setup()

    def run(self):
        """
        Run for one session
        """
        globals.LOGGER.log_one("epoch", globals.EPOCH) # edge-case compat for when we save <1 epoch after beginning the session

        # save an initial checkpoint before any training occurs, so step 0 is always recoverable
        # (bypass force_save's top-k logic here, since no metrics have been logged yet)
        if not globals.CHECKPOINTER.resume:
            globals.CHECKPOINTER.save_checkpoint(tag='step_0')
            if globals.CHECKPOINTER.save_last_ckpt:
                globals.CHECKPOINTER.save_checkpoint()

        while globals.EPOCH < globals.CONFIG.total_num_epochs: #type:ignore
            # train for one epoch
            self.epoch_trainer.train()
            
            # eval for one epoch
            self.epoch_evaluator.validate()
            
            # checkpoints
            globals.CHECKPOINTER.save()

            # end of epoch logging
            globals.EPOCH += 1
            globals.LOGGER.log_one("epoch", globals.EPOCH)
            
            

        # end of session stuff
        pass
            
    def reset(self):
        self.epoch_trainer.reset()
        
        self.epoch_evaluator.reset()
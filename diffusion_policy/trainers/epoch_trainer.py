import diffusion_policy.globals
from diffusion_policy.trainers.step_trainer import StepTrainer


class EpochTrainer:
    """
    Responsible for a training for one epoch.
    """

    def __init__(self,
            step_trainer: StepTrainer,
            nb_batches: int
            ):
        self.step_trainer = step_trainer
        self.nb_batches = nb_batches
        


    def train(self):
        """
        Train for one epoch. 
        Done once we've gone through the entire dataset...? Doesn't quite work with co-training off multiple datasets
        """
        for nb in range(self.nb_batches):
            # train for one batch
            self.step_trainer.train()

            # end of batch logging
            pass

        # end of epoch stuff
        pass
import diffusion_policy.globals
from diffusion_policy.trainers.step_trainer import StepTrainer


class EpochTrainer:
    """
    Responsible for a training for one epoch.
    """

    def __init__(self,
            step_trainer: StepTrainer
            ):
        self.step_trainer = step_trainer


    def train(self):
        """
        Train for one epoch. 
        Done once we've gone through the entire dataset...? Doesn't quite work with co-training off multiple datasets
        """
        done = False
        while not done:
            # train for one batch
            self.step_trainer.train()

            # end of batch logging
            pass

            # check if done?
            done = <>

        # end of epoch stuff
        pass
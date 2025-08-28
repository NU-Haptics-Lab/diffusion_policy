import torch
import diffusion_policy.globals as globals
from diffusion_policy.trainers.step_trainer import StepTrainer
from .common import CalcSumLoss
from diffusion_policy.utils import EveryEpoch
import tqdm

class Epoch:
    """
    Responsible for doing something for one epoch.
    """

    def __init__(self,
            nb_batches: int
            ):
        self.nb_batches = nb_batches

class EpochEvaluator(Epoch):
    """
    Responsible for evaluating for one epoch.
    """

    def __init__(self,
            nb_batches: int,
            val_every: int,
            w_batch_losses: dict
            ):
        self.nb_batches = nb_batches
        self.val_every = val_every
        self.w_batch_losses = w_batch_losses
        
        # handle to node
        self.models = globals.MODELS

    @torch.no_grad()
    def eval(self):
        """
        eval for one epoch.
        """
        if EveryEpoch(self.val_every):
            # switch to eval mode
            self.models.eval()
            
            for nb in tqdm.tqdm(range(self.nb_batches), desc=f"Evaluation epoch {globals.EPOCH}", leave=False):
                # eval for one batch
                total_loss = CalcSumLoss(self.w_batch_losses)

                # end of batch logging
                pass

            # end of epoch stuff
            pass
        
            # switch to training mode
            self.models.train()
    
    
class EpochTrainer(Epoch):
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
        for nb in tqdm.tqdm(range(self.nb_batches), desc=f"Training epoch {globals.EPOCH}", leave=False):
            # train for one batch
            self.step_trainer.train()

            # end of batch logging
            pass

        # end of epoch stuff
        pass
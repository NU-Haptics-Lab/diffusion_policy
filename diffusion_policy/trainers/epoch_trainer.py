import torch
import diffusion_policy.globals as globals
from diffusion_policy.trainers.step_trainer import StepTrainer
from .common import CalcSumLoss, CalcSumEval
from diffusion_policy.utils import EveryEpoch
import tqdm
import numpy as np
from diffusion_policy.rollout import Rollout

import avatar_drake_sim.sims.sandbox.sandbox_common as commons

class Epoch:
    """
    Responsible for doing something for one epoch.
    """

    def __init__(self,
            nb_batches: int
            ):
        self.nb_batches = nb_batches

class EpochValidator(Epoch):
    """
    Responsible for validating for one epoch.
    """

    def __init__(self,
            nb_batches: int,
            val_every: int,
            w_batch_losses: dict
            ):
        self.nb_batches = nb_batches
        self.val_every = val_every
        self.w_batch_losses = w_batch_losses
        
    def setup(self):
        # handle to node
        self.models = globals.MODELS
        
        for k, v in self.w_batch_losses.items():
            v.setup()

    @torch.no_grad()
    def validate(self):
        """
        validate for one epoch.
        """
        # check for no length
        if self.nb_batches == 0 or len(self.w_batch_losses) == 0:
            return
        
        if EveryEpoch(self.val_every):
            # switch to eval mode
            self.models.eval()
            
            # setup logging
            step_log = {}
            val_losses = []
            val_action_mse_errors = []
            # per-dataset (rb_id) raw, un-weighted breakdown -- the combined
            # val/loss and val/action_mse_error above mix datasets together
            # (weighted by each dataset's cotrain loss weight), which can
            # hide one dataset's val performance behind another's
            per_dataset_losses = {}
            per_dataset_mses = {}

            # evaluate for a nb of batches
            for nb in tqdm.tqdm(range(self.nb_batches), desc=f"Evaluation epoch {globals.EPOCH}", leave=False):
                # eval for one batch
                total_evals, per_dataset_evals = CalcSumEval(self.w_batch_losses)

                # TODO: this is specific to actor eval, doesn't consider the critic eval. Must fix.

                # end of batch logging
                val_losses.append(total_evals[0])
                val_action_mse_errors.append(total_evals[1])

                for rb_id, (raw_loss, raw_mse) in per_dataset_evals.items():
                    per_dataset_losses.setdefault(rb_id, []).append(raw_loss)
                    per_dataset_mses.setdefault(rb_id, []).append(raw_mse)

            # finish logging
            if len(val_losses) > 0:
                val_loss = torch.mean(torch.tensor(val_losses)).item()
                # log epoch average validation loss
                step_log['val/loss'] = val_loss

            if len(val_action_mse_errors) > 0:
                val_action_mse_error = torch.mean(torch.tensor(val_action_mse_errors)).item()
                # log epoch average validation loss
                step_log['val/action_mse_error'] = val_action_mse_error

            for rb_id, vals in per_dataset_losses.items():
                step_log[f'val/loss_{rb_id}'] = torch.mean(torch.tensor(vals)).item()
            for rb_id, vals in per_dataset_mses.items():
                step_log[f'val/action_mse_error_{rb_id}'] = torch.mean(torch.tensor(vals)).item()

            #
            globals.LOGGER.log(step_log)
        
            # switch to training mode
            self.models.train()
            
    def reset(self):
        # reset batch losses
        for k, v in self.w_batch_losses.items():
            v.reset()
    
class EpochTrainer(Epoch):
    """
    Responsible for a training for one epoch.
    """

    def __init__(self,
            step_trainer: StepTrainer,
            nb_batches: int,
            rollouts: Rollout | None = None,
            ):
        self.step_trainer = step_trainer
        self.nb_batches = nb_batches
        self.rollouts = rollouts
        
    def setup(self):
        self.step_trainer.setup()
        
        if self.rollouts is not None:
            self.rollouts.setup()

    def train(self):
        """
        Train for one epoch. 
        Done once we've gone through the entire dataset...? Doesn't quite work with co-training off multiple datasets
        """
        losses = []
        
        for nb in tqdm.tqdm(range(self.nb_batches), desc=f"Training epoch {globals.EPOCH}", leave=False):
            # train for one step
            self.step_trainer.train()
            
            # rollouts
            if self.rollouts is not None:
                self.rollouts.run()

            # update the global step count
            globals.STEP += 1
            commons.CURRENT_PARAMS.rl_step = globals.STEP
            
            
            ## end of step logging
            log = {
                "global_step": globals.STEP,
            }
            globals.LOGGER.log(log)

        # end of epoch stuff
        # log = {
        #     "epoch_avg_loss": np.mean(losses),
        # }
        # globals.LOGGER.log(log)
            
    def reset(self):
        self.step_trainer.reset()
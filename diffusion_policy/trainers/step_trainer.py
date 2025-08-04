
import torch
import diffusion_policy.globals
from diffusion_policy.globals import CONFIG

from diffusion_policy.losses.batch_loss import WeightedBatchLoss


class StepTrainer:
    """
    Responsible for training for one step.

    Gets the summed loss from any number of WeightedBatchLoss's and then performs back propagation.

    It's a good idea to sum the weighted losses from all co-training datasets before doing back-prop because it should reduce variance and improve training stability, I believe.
    """

    def __init__(self,
            w_batch_losses = dict
        ):
        self.w_batch_losses = w_batch_losses

    def train(self):
        # initialize a zero loss variable
        total_loss = torch.tensor([0.0], requires_grad=True)

        # iterate over the batch losses and get the total loss
        for key, batch_loss in enumerate(self.w_batch_losses):
            loss = batch_loss.compute_weighted_loss()

            total_loss += loss

        # calculate gradients for all datasets simultaneously
        total_loss.backward()

        # step optimizer
        if self.global_step % CONFIG.training.gradient_accumulate_every == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            lr_scheduler.step()
        
        # update ema
        if CONFIG.training.use_ema:
            ema.step(self.model)

        # logging
        raw_loss_cpu = raw_loss.item()
        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
        train_losses.append(raw_loss_cpu)
        step_log = {
            'train_loss': raw_loss_cpu,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'lr': lr_scheduler.get_last_lr()[0]
        }

        # do some stuff if this is the last step
        is_last_batch = (batch_idx == (len(meta_dataset)-1))
        if not is_last_batch:
            # log of last step is combined with validation and rollout
            wandb_run.log(step_log, step=self.global_step)
            json_logger.log(step_log)
            self.global_step += 1

        # if we're done training
        if (CONFIG.training.max_train_steps is not None) \
            and batch_idx >= (CONFIG.training.max_train_steps-1):
            return False
        
        # we're done
        return True
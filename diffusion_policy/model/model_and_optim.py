import hydra
import torch
import diffusion_policy.globals as globals
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler

class ModelandOptim:
    """
    Convenience class which bundles a model, an optimizer, and a learning rate scheduler
    """
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer_target: str,
                 optimizer_cfg: dict,
                 lr_scheduler = "cosine",
                 lr_warmup_steps = 500,
                 gradient_accumulate_every = 1,
                 ):
        self.model = model
        self.gradient_accumulate_every = gradient_accumulate_every
        
        # make the optimizer class
        cls = hydra.utils.get_class(optimizer_target)
        self.optimizer = cls(
                **optimizer_cfg, 
                params=self.model.parameters()
                )
        
        # transfer to GPU
        optimizer_to(self.optimizer, globals.CONFIG.device)
        
        # device transfer of the model, since I own it
        device = torch.device(globals.CONFIG.device)
        self.model.to(device)
        
        # make the LR scheduler
        self.lr_scheduler = get_scheduler(
            lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=(
                globals.CONFIG.session_trainer.epoch_trainer.nb_batches * globals.CONFIG.total_num_epochs) // gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=globals.STEP-1
        )
        
    def loss(self, nbatch):
        # compute loss for the model
        raw_loss = self.model.loss(nbatch)
        loss = raw_loss / self.gradient_accumulate_every
        
        # for logging
        self.raw_loss_cpu = raw_loss.item()

        return loss
    
    def step(self):
        if globals.STEP % self.gradient_accumulate_every == 0:
            self.optimizer.step()
            self.lr_scheduler.step()
            
    def step_log(self):
        step_log = {
            'train_loss': self.raw_loss_cpu,
            'lr': self.lr_scheduler.get_last_lr()[0]
        }
        return step_log
    
    def get_model(self):
        return self.model
    
    def reset(self):
        self.optimizer.zero_grad()
    
    def denoise(self, 
            nobs_dict,
            noise_scheduler,
            ):
        return self.model.denoise(nobs_dict, noise_scheduler)
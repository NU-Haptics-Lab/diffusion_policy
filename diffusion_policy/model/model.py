import copy
import torch
import hydra

import diffusion_policy.globals as globals
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler

from .ema import Ema
from .optim import Optim

class Ema:
    """
    Ema model mixin
    """
    def __init__(self,
            model,
            use_ema: bool,
            ema_cfg,
            ):
        # configure model
        self.model = model
        self.use_ema = use_ema
        self.ema_cfg = ema_cfg

        self.ema_model = None
        if self.use_ema:
            self.ema_model = copy.deepcopy(self.model.get_model())

        # configure ema
        self.ema: EMAModel = None
        if self.use_ema:
            assert(self.ema_model is not None)
                
            # make the ema model using the passed in ema_cfg
            self.ema = EMAModel(
                **self.ema_cfg,
                model=self.ema_model)
        
        # device transfer of the ema, since I own it
        device = torch.device(globals.CONFIG.device)
        if self.ema_model is not None:
            self.ema_model.to(device)
    
    def update_ema(self):
        if self.use_ema:
            self.ema.step(self.model.get_model())

class Optim:
    """
    Torch optimizer and learning rate scheduler mixin
    """
    def __init__(self,
                 model: torch.nn.Module = None,
                 optimizer_target: str = None,
                 optimizer_cfg: dict = None,
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
    
    def step(self):
        if globals.STEP % self.gradient_accumulate_every == 0:
            self.optimizer.step()
            self.lr_scheduler.step()
            
    def step_log(self):
        step_log = {
            'lr': self.lr_scheduler.get_last_lr()[0]
        }
        return step_log
    
    def reset(self):
        self.optimizer.zero_grad()

class Model():
        
    def __init__(self,
                 model):
        # configure model
        self.model = model

        # device transfer of the model, since I own it
        device = torch.device(globals.CONFIG.device)
        self.model.to(device)
            
    def loss(self, nbatch, task_id):
        """ alias for compute_loss """
        return self.model.loss(nbatch)
        
    def step(self):
        self.model.step()
        
    def denoise(self, 
            nobs_dict,
            noise_scheduler,
            ):
        return self.model.denoise(nobs_dict, noise_scheduler)
    
    def get_model(self):
        return self.model.get_model()
    
    def reset(self):
        self.model.reset()

class ModelEmaOptim(Model, Ema, Optim):
    """
    Mixin class with a model, an ema model, and an optimizer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

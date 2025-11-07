import copy
import torch
import torch.nn as nn
import hydra

import diffusion_policy.globals as globals
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.common.checkpointer import TopKCheckpointManager
from diffusion_policy.model.diffusion_model import DiffusionModel

class Base(nn.Module):
    def __init__(self,
                #  checkpointer: TopKCheckpointManager
                 ) -> None:
        # must init nn.Module
        super().__init__()
        
        # self.checkpointer = checkpointer
        
    def reset(self):
        pass
    
    def step(self):
        pass
    
    # def save(self):
    #     self.checkpointer.save()
        
    # def load(self):
    #     self.checkpointer.load()
        
    
    # def eval(self):
    #     pass
    
    # def train(self):
    #     pass

class Ema(Base):
    """
    Ema model mixin
    """
    def __init__(self,
            use_ema: bool,
            ema_cfg,
            ):
        super().__init__()
        
        # configure model
        self.use_ema = use_ema
        self.ema_cfg = ema_cfg
            
    def setup(self, model: DiffusionModel):
        self.model = model

        self.ema_model = None
        if self.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure ema
        self.ema: EMAModel
        if self.use_ema:
            assert(self.ema_model is not None)
                
            # make the ema model using the passed in ema_cfg
            self.ema = EMAModel(
                **self.ema_cfg,
                model=self.ema_model)
        
        # device transfer of the ema, since I own it
        device = torch.device(globals.CONFIG.device) #type:ignore
        if self.ema_model is not None:
            self.ema_model.to(device)
        
    
    def update_ema(self):
        if self.use_ema:
            self.ema.step(self.model)
            
    def step(self):
        self.update_ema()
        
    def get_model(self):
        return self.ema_model

class Optim(Base):
    """
    Torch optimizer and learning rate scheduler mixin
    """
    def __init__(self,
                 optimizer_target: str = "",
                 optimizer_cfg: dict = {},
                 lr_scheduler = "cosine",
                 lr_warmup_steps = 500,
                 gradient_accumulate_every = 1,
                 grad_norm = 1.0
                 ):
        super().__init__()
        
        self.gradient_accumulate_every = gradient_accumulate_every
        self.optimizer_target = optimizer_target
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_type = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.grad_norm = grad_norm
        
    def setup(self, model):
        self.model = model
        
        # make the optimizer class
        cls = hydra.utils.get_class(self.optimizer_target)
        self.optimizer = cls(
                **self.optimizer_cfg, 
                params = self.model.parameters()
                )
        
        # transfer to GPU
        optimizer_to(self.optimizer, globals.CONFIG.device) #type:ignore
        
        # extract config vals
        nb_batches = globals.CONFIG.session_trainer.epoch_trainer.nb_batches # type:ignore
        total_num_epochs = globals.CONFIG.total_num_epochs #type:ignore
        
        # make the LR scheduler
        self.lr_scheduler = get_scheduler(
            self.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps = self.lr_warmup_steps,
            num_training_steps=(
                nb_batches * total_num_epochs) // self.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=globals.STEP-1 -1 # -2 to work with the initial step value of 1
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

class Model(Base):
        
    def __init__(self,
                 model: DiffusionModel,
                 ):
        super().__init__()
        
        # configure model
        self.model = model

        # device transfer of the model, since I own it
        device = torch.device(globals.CONFIG.device) #type:ignore because pylance can't do dynamic type checking
        self.model.to(device)
            
    def loss(self, nbatch, task_id):
        """ alias for compute_loss """
        return self.model.loss(nbatch, task_id)
        
    def denoise(self, 
            nobs_dict,
            noise_scheduler,
            ):
        
        assert(isinstance(self.model, DiffusionModel))
        return self.model.denoise(nobs_dict, noise_scheduler)
    
    def get_model(self):
        # model should be a diffusion model
        return self.model
    
    def reset(self):
        self.model.reset()

class ModelEmaOptim(Base):
    """
    Mixin class with a model, an ema model, and an optimizer
    """
    def __init__(self,
            model: Model,
            ema: Ema,
            optim: Optim
    ):
        super().__init__()
        
        self.model = model
        self.ema = ema
        self.optim = optim
        
        # setup
        self.ema.setup(self.model.get_model())
        self.optim.setup(self.model.get_model())
            
    def loss(self, nbatch, task_id):
        return self.model.loss(nbatch, task_id)
    
    def get_model(self):
        return self.model.get_model()
    
    def get_ema_model(self):
        return self.ema.get_model()
    
    def step(self):
        self.model.step()
        self.ema.step()
        self.optim.step()
        
    def reset(self):
        self.model.reset()
        self.ema.reset()
        self.optim.reset()
        
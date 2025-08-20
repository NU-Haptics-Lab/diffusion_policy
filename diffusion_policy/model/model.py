import torch
from diffusion_policy.globals import CONFIG
from diffusion_policy.model.diffusion.ema_model import EMAModel
import copy
import hydra

class Model:
    """
    Convenience model class which bundles a model and an ema model (if used)
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
            self.ema_model = copy.deepcopy(self.model)

        # configure ema
        self.ema: EMAModel = None
        if self.use_ema:
            self.ema = hydra.utils.instantiate(
                self.ema,
                model=self.ema_model)
        
        # device transfer of the ema, since I own it
        device = torch.device(CONFIG.training.device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        optimizer_to(self.optimizer, device)
        
        # init in train mode
        self.train()

    def compute_loss(self, nbatch):
        # compute loss
        raw_loss = self.run_model.compute_loss(nbatch)
        loss = raw_loss / CONFIG.training.gradient_accumulate_every

        return loss
    
    def update_ema(self):
        if self.use_ema:
            self.ema_model.step(self.model)
            
    def eval(self):
        """
        switch to eval mode
        """
        if self.use_ema:
            self.run_model = self.ema_model
            
    def train(self):
        """
        switch to train mode
        """
        self.run_model = self.model
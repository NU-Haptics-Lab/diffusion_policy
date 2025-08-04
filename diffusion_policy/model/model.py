import torch
from diffusion_policy.globals import CONFIG
from diffusion_policy.model.diffusion.ema_model import EMAModel
import copy
import hydra

class Model:
    """
    Convenience model class which bundles a policy and an ema policy (if used)
    """
    def __init__(self,
            model     
            ):
        # configure model
        self.model = model

        self.ema_model = None
        if CONFIG.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure ema
        ema: EMAModel = None
        if CONFIG.training.use_ema:
            ema = hydra.utils.instantiate(
                CONFIG.ema,
                model=self.ema_model)
        
        # device transfer of the ema, since I own it
        device = torch.device(CONFIG.training.device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        optimizer_to(self.optimizer, device)

    def compute_loss(self, nbatch):
        # compute loss
        raw_loss = self.model.compute_loss(nbatch)
        loss = raw_loss / CONFIG.training.gradient_accumulate_every

        return loss
    
    def update_ema(self):
        if CONFIG.training.use_ema:
            ema.step(self.model)
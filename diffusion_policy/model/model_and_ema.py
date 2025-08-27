import torch
import diffusion_policy.globals as globals
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.diffusion_model import DiffusionModel
import copy
import hydra

class ModelandEMA:
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
        
        # init in train mode
        self.train()
        
    def compute_loss(self, nbatch):
        return self.model.compute_loss(nbatch)
    
    def loss(self, nbatch, task_id):
        """ alias for compute_loss """
        return self.compute_loss(nbatch)
    
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
        
    def step(self):
        self.model.step()
        self.update_ema()
        
    def denoise(self, 
            nobs_dict,
            noise_scheduler,
            ):
        return self.model.denoise(nobs_dict, noise_scheduler)
        

def test():
    import hydra
    from omegaconf import OmegaConf
    txt = """
_target_: diffusion_policy.model.model_and_ema.ModelandEMA
use_ema: true

model:
    _target_: torch.nn.Module

ema_cfg:
    _target_: diffusion_policy.model.diffusion.ema_model.EMAModel
    inv_gamma: 1.0
    max_value: 0.9999
    min_value: 0.0
    power: 0.75
    update_after_step: 0
    """
    config = OmegaConf.create(txt)

    # x = hydra.utils.instantiate(config)
    x = ModelandEMA(model=config.model, use_ema=config.use_ema, ema_cfg=config.ema_cfg)
    
    pass
    
    
if __name__ == "__main__":
    test()
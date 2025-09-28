import wandb
import collections
from omegaconf import OmegaConf

import diffusion_policy.globals as globals
from diffusion_policy.utils import get_output_dir

class Logging:
    """
    class to set up wandb and log to it
    """
    
    def __init__(self,
                 use_wandb = False,
                 output_dir = None,
                 wandb_cfg: dict = None
                 ):
        self.use_wandb = use_wandb
        self.output_dir = get_output_dir(output_dir)
        self.wandb_cfg = wandb_cfg
        
        self.reset()
        
        if self.use_wandb:
            self.wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(globals.CONFIG, resolve=True),
                **wandb_cfg
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )
            
    def reset(self):
        self.data = {}
            
    def save(self, data):
        """
        Only saves the most recent data for each key, NOT all
        """
        # for key, val in data.items():
        #     self.data[key] = val
        self.data.update(data)
        
    def log(self, data):
        self.save(data)
        
        if self.use_wandb:
            self.wandb_run.log(
                data,
                step=globals.STEP
            )
            
    def log_one(self, label, datapoint):
        dd = {label: datapoint}
        self.log(dd)
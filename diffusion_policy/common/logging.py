import wandb

class Logging:
    """
    class to set up wandb and log to it
    """
    
    def __init__(self,
                 use_wandb = False
                 ):
        self.use_wandb = use_wandb
        
        if self.use_wandb:
            self.wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(CONFIG, resolve=True),
                **CONFIG.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )
        
    def log(self, data):
        if self.use_wandb:
            self.wandb_run.log(
                data,
                step=globals.STEP
            )
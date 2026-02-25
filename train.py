"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

"""
Design principle - I hate all the discourse around how to pass data between siblings, cousins, parents, children, etc., etc., etc. IMO it creates so much innecessary boilerplate code, passthrough functions, and needless confusion.
So, here's the idea: write code like normal, breaking down functionality into appropriate classes and using composition to create modular code. 

If each child instance has EXACTLY ONE parent (composition to be clear, not inheritance), then everyone is happy.

BUT, as soon as a child instance must be accessed by anything other than its single parent, split it off and refactor it into an independent global singleton, a "Node" (to borrow ROS nomenclature), so that any arbitrary number of classes can access it without needing to pass the class instance handles every which way to Sunday.

For example:
1 zarr dataset -> 1 replay buffer -> 1 sampler -> 1 dataset => everyone's happy, 1 hierarchical class structure. BUT, since both the actor and critic want to access that dataset's batch data ... sampler now becomes the top-level of a Node, which actor_trainer and critic_trainer can access via global singletons.

"""


import sys
import atexit
import signal
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf, DictConfig
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import click

# global config
import diffusion_policy.globals as globals

from diffusion_policy.trainers.session_trainer import SessionTrainer

from diffusion_policy.common.checkpointer import TopKCheckpointManager

# to combat dataloader deadlock
import torch
import torch.multiprocessing

from diffusion_policy.globals import (
    load_config_to_global,
    load_global_config_and_save_to_globals_dict,
)


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Register the del resolver
OmegaConf.register_new_resolver("del", lambda: None)

# final save if the program was ctrl+c'd
def Shutdown():
    if input("Save a checkpoint? y/n") == 'y':
        # checkpoints
        globals.CHECKPOINTER.save()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config')),
)
def main(cfg: DictConfig):
    
    # whether we're debugging
    if cfg.debug:  # type: ignore
        # no resuming
        cfg.resume = False # type:ignore
        
        cfg.common_dataset.options.common.batch_size = 4 # type: ignore
        cfg.total_num_epochs = 4 # type: ignore
        cfg.batches_per_epoch = 4 # type: ignore
        cfg.common_dataset.options.train.num_workers = 0 # type: ignore
        cfg.common_dataset.options.train.persistent_workers = False # type: ignore
        # cfg.common_noise_scheduler.num_train_timesteps = 10 # type: ignore
        # cfg.models.models.critic.num_inference_steps = 4 # type: ignore
        
        # test small network
        # cfg.models.models.actor.model.model.down_dims = (16, 32, 64)
        
        # no online logging
        cfg.logging.use_wandb = False # type: ignore
        
        # testing checkpointing
        cfg.checkpoint.checkpoint_every = 1 # type: ignore
        
        # testing validation
        cfg.val_every = 1 # type: ignore
        
        # testing freq
        # for key in cfg.step_freqs:
        #     cfg.step_freqs[key] = 1
        
        # testing rollouts
        cfg.session_trainer.epoch_trainer.rollouts.freq = 1 # type:ignore
        cfg.session_trainer.epoch_trainer.rollouts.num_rollouts_per_trigger = 1
        cfg.session_trainer.epoch_trainer.rollouts.warmup_nb_steps = 0
        
        # # no rollouts
        # cfg.session_trainer.epoch_trainer.rollouts.use_online_rollout = False # type:ignore
        
        torch.autograd.set_detect_anomaly(True) # type: ignore
        
    # load using a common local loader, put in a globals dict
    load_global_config_and_save_to_globals_dict(cfg, "train")
    
    # now load the config to the global singletons
    load_config_to_global("train")
    
    # run it
    print("Begin running.")
    try:
        globals.SESSION_TRAINER.run()
    except KeyboardInterrupt:
        if input("Save a checkpoint? y/n ") == 'y':
            # checkpoints
            assert(isinstance(globals.CHECKPOINTER, TopKCheckpointManager))
            assert(globals.CHECKPOINTER is not None)
            globals.CHECKPOINTER.force_save()
    print("Done")

if __name__ == "__main__":
    # only need the following if using my meta dataset
    # torch.multiprocessing.set_start_method('spawn') # or 'forkserver'
    main() #type:ignore

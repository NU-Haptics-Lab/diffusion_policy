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
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import click

# global config
import diffusion_policy.globals as globals

from diffusion_policy.trainers.session_trainer import SessionTrainer

# to combat dataloader deadlock
import torch
import torch.multiprocessing

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
def main(cfg: OmegaConf):
    # save config into the global config
    globals.CONFIG = cfg
    
    # Temporarily disable strict mode to add new keys
    OmegaConf.set_struct(globals.CONFIG, False) # type: ignore
    
    # whether we want to run single threaded for debugging purposes
    if globals.CONFIG.single_thread:
        globals.CONFIG.common_dataset.options.train.num_workers = 0 # type: ignore
        globals.CONFIG.common_dataset.options.train.persistent_workers = False # type: ignore
        
        globals.CONFIG.logging.use_wandb = False # type: ignore
        
    
    # whether we're debugging
    if globals.CONFIG.debug:  # type: ignore
        # no resuming
        globals.CONFIG.resume = False # type:ignore
        
        globals.CONFIG.common_dataset.options.common.batch_size = 4 # type: ignore
        globals.CONFIG.total_num_epochs = 4 # type: ignore
        globals.CONFIG.batches_per_epoch = 4 # type: ignore
        globals.CONFIG.common_dataset.options.train.num_workers = 0 # type: ignore
        globals.CONFIG.common_dataset.options.train.persistent_workers = False # type: ignore
        # globals.CONFIG.common_noise_scheduler.num_train_timesteps = 10 # type: ignore
        # globals.CONFIG.models.models.critic.num_inference_steps = 4 # type: ignore
        
        # test small network
        # globals.CONFIG.models.models.actor.model.model.down_dims = (16, 32, 64)
        
        # no online logging
        globals.CONFIG.logging.use_wandb = False # type: ignore
        
        # testing checkpointing
        globals.CONFIG.checkpoint.checkpoint_every = 1 # type: ignore
        
        # testing validation
        globals.CONFIG.val_every = 1 # type: ignore
        
        # testing freq
        # for key in globals.CONFIG.step_freqs:
        #     globals.CONFIG.step_freqs[key] = 1
        
        # testing rollouts
        globals.CONFIG.session_trainer.epoch_trainer.rollouts.freq = 1 # type:ignore
        globals.CONFIG.session_trainer.epoch_trainer.rollouts.num_rollouts_per_trigger = 1 # type:ignore
        
        # # no rollouts
        # globals.CONFIG.session_trainer.epoch_trainer.rollouts.use_online_rollout = False # type:ignore
        
        torch.autograd.set_detect_anomaly(True) # type: ignore
    
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(globals.CONFIG) # type: ignore
    
    # apply overrides, much faster than merge
    OmegaConf.unsafe_merge(globals.CONFIG, globals.CONFIG.override) # type: ignore
    print("Config merged.")
        
    # spin up the logger
    globals.LOGGER = hydra.utils.instantiate(globals.CONFIG.logging) # type: ignore
    print("Logger spun.")
    
    # spin up the replay buffer loader
    globals.REPLAY_BUFFER_LOADER = hydra.utils.instantiate(globals.CONFIG.replay_buffer_loader) # type: ignore
    print("Replay Buffer Loader spun.")
    
    # spin up the dataloaders
    globals.DATALOADERS = hydra.utils.instantiate(globals.CONFIG.dataloaders) # type: ignore
    print("Dataloaders spun.")
    
    # spin up the dataloaders
    globals.DEFAULT_BATCH_LOADER = hydra.utils.instantiate(globals.CONFIG.default_batch_loader) # type: ignore
    print("DEFAULT_BATCH_LOADER spun.")
        
    # spin up the models
    globals.MODELS = hydra.utils.instantiate(globals.CONFIG.models) # type: ignore
    print("Models spun.")
    
    # spin up the checkpointer
    globals.CHECKPOINTER = hydra.utils.instantiate(globals.CONFIG.checkpoint) # type: ignore
    print("Checkpointer spun.")

    # spin up the session trainer
    # cls = hydra.utils.get_class(cfg._target_)
    globals.SESSION_TRAINER: SessionTrainer = hydra.utils.instantiate(globals.CONFIG.session_trainer) # type: ignore
    print("Session Trainer spun.")
    
    # if resuming, load
    globals.CHECKPOINTER.load()

    # run it
    print("Begin running.")
    try:
        globals.SESSION_TRAINER.run()
    except KeyboardInterrupt:
        if input("Save a checkpoint? y/n ") == 'y':
            # checkpoints
            globals.CHECKPOINTER.force_save()
    print("Done")

if __name__ == "__main__":
    # only need the following if using my meta dataset
    # torch.multiprocessing.set_start_method('spawn') # or 'forkserver'
    main() #type:ignore

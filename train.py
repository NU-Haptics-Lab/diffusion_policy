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
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import click

# global config
from diffusion_policy.config.config import CONFIG
from diffusion_policy.trainers.session_trainer import SessionTrainer

# to combat dataloader deadlock
import torch.multiprocessing

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config')),
)
def main(cfg: OmegaConf):
    # save config into the global config
    global CONFIG
    CONFIG = cfg
    
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    # spin up the batch loader
    global BATCH_LOADER
    BATCH_LOADER = hydra.utils.instantiate(CONFIG.batch_loader)

    # spin up the session trainer
    cls = hydra.utils.get_class(cfg._target_)
    session_trainer = hydra.utils.instantiate(CONFIG.session_trainer)

    # run it
    session_trainer.train()

if __name__ == "__main__":
    # only need the following if using my meta dataset
    torch.multiprocessing.set_start_method('spawn') # or 'forkserver'
    main()

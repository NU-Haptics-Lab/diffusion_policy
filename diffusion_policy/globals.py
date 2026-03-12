import os
import hydra
from omegaconf import OmegaConf, DictConfig
import pathlib
import copy
import random
import wandb
import tqdm
import numpy as np

import torch
import torch.nn as nn
from diffusion_policy.common.replay_buffer_loader import ReplayBufferLoader
from diffusion_policy.model.models import Models
from diffusion_policy.common.checkpointer import TopKCheckpointManager
from diffusion_policy.common.logging import Logging
import diffusion_policy.dataset.train_and_val as train_and_val 
# import DataLoaders


# import diffusion_policy.dataset.batch_loader as batch_loader
# import BatchLoader
# from diffusion_policy.trainers.session_trainer import SessionTrainer

from contextlib import contextmanager



"""
the current config key
"""
CURRENT_CONFIG_KEY = ""

"""
Global config access
"""
CONFIG: DictConfig | None = None # backwards compat

"""
Contains replay buffers for each dataset specified
"""
REPLAY_BUFFER_LOADER: ReplayBufferLoader | None

"""
Contains both train and val data loaders
"""
DATALOADERS: train_and_val.DataLoaders | None

"""
Contains all models used in training
"""
MODELS: Models | None = None

""" Epoch count """
EPOCH = 1 # start on 1 so that the fcn EveryEpoch doesn't fire on the first epoch

""" Training step count """
STEP = 1 # start on 1 so that rollouts won't trigger immediately

""" Logger """
LOGGER: Logging | None = None

""" Checkpointer """
CHECKPOINTER: TopKCheckpointManager | None

""" Default batch loader for normalizing """
# DEFAULT_BATCH_LOADER: batch_loader.BatchLoader
DEFAULT_BATCH_LOADER = None


# SESSION_TRAINER: SessionTrainer
SESSION_TRAINER = None

#######################################################################
#######################################################################
#######################################################################
#######################################################################

class GlobalConfig:
    """
    Global config access
    """
    def __init__(self,      
                
        CONFIG: DictConfig | None,
        
        REPLAY_BUFFER_LOADER: ReplayBufferLoader | None,
        
        DATALOADERS: train_and_val.DataLoaders | None,
        
        LOGGER: Logging | None,
        
        CHECKPOINTER: TopKCheckpointManager | None,
        
        DEFAULT_BATCH_LOADER,
        
        SESSION_TRAINER,
        
        MODELS: Models | None = None,
        
        EPOCH = 1, # start on 1 so that the fcn EveryEpoch doesn't fire on the first epoch
        
        STEP = 1, # start on 1 so that rollouts won't trigger immediately
        
        ):
        self.CONFIG = CONFIG
        self.REPLAY_BUFFER_LOADER = REPLAY_BUFFER_LOADER
        self.DATALOADERS = DATALOADERS
        self.MODELS = MODELS
        self.LOGGER = LOGGER
        self.CHECKPOINTER = CHECKPOINTER
        self.DEFAULT_BATCH_LOADER = DEFAULT_BATCH_LOADER
        self.SESSION_TRAINER = SESSION_TRAINER
        self.EPOCH = EPOCH
        self.STEP = STEP
        
GLOBALS_DICT: dict[str, GlobalConfig] = {}


########################################
########################################
########################################
########################################
########################################

def load_global_config(cfg: DictConfig, cfg_key):
    """
    loads the global config locally, does not save to the current global vars
    """
    # Temporarily disable strict mode to add new keys
    OmegaConf.set_struct(cfg, False) # type: ignore
    
    # whether we want to run single threaded for debugging purposes
    if 'single_thread' in cfg and cfg.single_thread:
        cfg.common_dataset.options.train.num_workers = 0 # type: ignore
        cfg.common_dataset.options.train.persistent_workers = False # type: ignore
        
        cfg.logging.use_wandb = False # type: ignore
    
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg) # type: ignore
    
    # apply overrides, much faster than merge
    if 'override' in cfg:
        OmegaConf.unsafe_merge(cfg, cfg.override) # type: ignore
        
    # spin up the logger
    if 'logging' in cfg:
        LOGGER = hydra.utils.instantiate(cfg.logging) # type: ignore
    else:
        LOGGER = None
    
    # spin up the replay buffer loader
    if 'replay_buffer_loader' in cfg:
        REPLAY_BUFFER_LOADER = hydra.utils.instantiate(cfg.replay_buffer_loader) # type: ignore
    else:
        REPLAY_BUFFER_LOADER = None
    
    # spin up the dataloaders
    if 'dataloaders' in cfg:
        DATALOADERS = hydra.utils.instantiate(cfg.dataloaders) # type: ignore
    else:
        DATALOADERS = None
    
    # spin up the dataloaders
    if 'default_batch_loader' in cfg:
        DEFAULT_BATCH_LOADER = hydra.utils.instantiate(cfg.default_batch_loader) # type: ignore
    else:
        DEFAULT_BATCH_LOADER = None
        
    # spin up the models
    if 'models' in cfg:
        MODELS = hydra.utils.instantiate(cfg.models) # type: ignore
    else:
        MODELS = None
    
    # spin up the checkpointer
    if 'checkpoint' in cfg:
        CHECKPOINTER = hydra.utils.instantiate(cfg.checkpoint) # type: ignore
    else:
        CHECKPOINTER = None

    # spin up the session trainer
    if 'session_trainer' in cfg:
        SESSION_TRAINER: SessionTrainer = hydra.utils.instantiate(cfg.session_trainer) # type: ignore
    else:
        SESSION_TRAINER = None

    # make the struct
    globals_struct = GlobalConfig(
        CONFIG=cfg,
        REPLAY_BUFFER_LOADER    =REPLAY_BUFFER_LOADER,
        DATALOADERS             =DATALOADERS,
        LOGGER                  =LOGGER,
        DEFAULT_BATCH_LOADER    =DEFAULT_BATCH_LOADER,
        MODELS                  =MODELS,
        CHECKPOINTER            =CHECKPOINTER,
        SESSION_TRAINER         =SESSION_TRAINER
    )
        
    # must now save
    GLOBALS_DICT[cfg_key] = globals_struct
    
    # must do the following within a context to maintain the correct global state
    with use_config(cfg_key):
        ## now must call setup methods for all nodes
        if LOGGER is not None:
            LOGGER.setup()
            
        if REPLAY_BUFFER_LOADER is not None:
            REPLAY_BUFFER_LOADER.setup()
            
        if DATALOADERS is not None:
            DATALOADERS.setup()
        
        if DEFAULT_BATCH_LOADER is not None:
            DEFAULT_BATCH_LOADER.setup()
            
        if MODELS is not None:
            MODELS.setup()
            
        if CHECKPOINTER is not None:
            CHECKPOINTER.load()
        
        if SESSION_TRAINER is not None:        
            SESSION_TRAINER.setup()
        

    
def load_global_config_from_path(cfg_path, cfg_key):
    cfg = OmegaConf.load(cfg_path)
    load_global_config(cfg, cfg_key) #type:ignore
    
# alias
def load_global_config_from_path_and_save_to_globals_dict(cfg_path, cfg_key):    
    load_global_config_from_path(cfg_path, cfg_key)
    
# def load_global_config_from_path_and_save_to_globals(cfg_path, cfg_key):    
#     load_global_config_from_path(cfg_path, cfg_key)

def load_global_config_and_save_to_globals_dict(cfg, cfg_key):
    load_global_config(cfg, cfg_key)

def load_config_to_global(config_key):
    global CURRENT_CONFIG_KEY, CONFIG, REPLAY_BUFFER_LOADER, DATALOADERS, LOGGER, CHECKPOINTER, DEFAULT_BATCH_LOADER, MODELS, SESSION_TRAINER
    
    if config_key not in GLOBALS_DICT:
        print("Warning, config key: {} not in GLOBALS_DICT".format(config_key))
        return
    
    global_config = GLOBALS_DICT[config_key]

    CURRENT_CONFIG_KEY       = config_key
    CONFIG                   = global_config.CONFIG
    REPLAY_BUFFER_LOADER     = global_config.REPLAY_BUFFER_LOADER
    DATALOADERS              = global_config.DATALOADERS
    LOGGER                   = global_config.LOGGER
    DEFAULT_BATCH_LOADER     = global_config.DEFAULT_BATCH_LOADER
    MODELS                   = global_config.MODELS
    CHECKPOINTER             = global_config.CHECKPOINTER
    SESSION_TRAINER          = global_config.SESSION_TRAINER
    
    pass
    
def load_config_direct_to_global(config):
    # this is for when we don't want to save to the GLOBALS_DICT, but just want to load a config directly to the global vars
    load_global_config(config, "direct_load")
    load_config_to_global("direct_load")
    
@contextmanager
def use_config(cfg_key):
    backup = CURRENT_CONFIG_KEY
    
    # load my config
    try:
        yield load_config_to_global(cfg_key)
        
    # revert to backup config
    finally:
        load_config_to_global(backup)
        
def save_current_config_to_globals(cfg_key):
    global CURRENT_CONFIG_KEY, CONFIG, REPLAY_BUFFER_LOADER, DATALOADERS, LOGGER, CHECKPOINTER, DEFAULT_BATCH_LOADER, MODELS, SESSION_TRAINER
    
    globals_struct = GlobalConfig(
        CONFIG=CONFIG,
        REPLAY_BUFFER_LOADER    =REPLAY_BUFFER_LOADER,
        DATALOADERS             =DATALOADERS,
        LOGGER                  =LOGGER,
        DEFAULT_BATCH_LOADER    =DEFAULT_BATCH_LOADER,
        MODELS                  =MODELS,
        CHECKPOINTER            =CHECKPOINTER,
        SESSION_TRAINER         =SESSION_TRAINER
    )
    
    GLOBALS_DICT[cfg_key] = globals_struct
    CURRENT_CONFIG_KEY = cfg_key
    
def log_one_if_exists(label, datapoint):
    if LOGGER is not None:
        LOGGER.log_one(label, datapoint)
        
def get_current_globals():
    return {
        "CONFIG": CONFIG,
        "REPLAY_BUFFER_LOADER": REPLAY_BUFFER_LOADER,
        "DATALOADERS": DATALOADERS,
        "LOGGER": LOGGER,
        "CHECKPOINTER": CHECKPOINTER,
        "DEFAULT_BATCH_LOADER": DEFAULT_BATCH_LOADER,
        "MODELS": MODELS,
        "SESSION_TRAINER": SESSION_TRAINER,
    }
    
def set_current_globals(globals_dict: dict):
    global CONFIG, REPLAY_BUFFER_LOADER, DATALOADERS, LOGGER, CHECKPOINTER, DEFAULT_BATCH_LOADER, MODELS, SESSION_TRAINER
    
    CONFIG = globals_dict["CONFIG"]
    REPLAY_BUFFER_LOADER = globals_dict["REPLAY_BUFFER_LOADER"]
    DATALOADERS = globals_dict["DATALOADERS"]
    LOGGER = globals_dict["LOGGER"]
    CHECKPOINTER = globals_dict["CHECKPOINTER"]
    DEFAULT_BATCH_LOADER = globals_dict["DEFAULT_BATCH_LOADER"]
    MODELS = globals_dict["MODELS"]
    SESSION_TRAINER = globals_dict["SESSION_TRAINER"]
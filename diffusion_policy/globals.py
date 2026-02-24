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
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.replay_buffer_loader import ReplayBufferLoader
from diffusion_policy.model.models import Models
from diffusion_policy.common.checkpointer import TopKCheckpointManager
from diffusion_policy.common.logging import Logging
from diffusion_policy.dataset.train_and_val import DataLoaders
from diffusion_policy.dataset.batch_loader import BatchLoader
from diffusion_policy.trainers.session_trainer import SessionTrainer

"""
Global config access
"""
CONFIG: OmegaConf # backwards compat

"""
Contains replay buffers for each dataset specified
"""
REPLAY_BUFFER_LOADER: ReplayBufferLoader

"""
Contains both train and val data loaders
"""
DATALOADERS: DataLoaders

"""
Contains all models used in training
"""
MODELS: Models = None # type:ignore

""" Epoch count """
EPOCH = 1 # start on 1 so that the fcn EveryEpoch doesn't fire on the first epoch

""" Training step count """
STEP = 1 # start on 1 so that rollouts won't trigger immediately

""" Logger """
LOGGER: Logging

""" Checkpointer """
CHECKPOINTER: TopKCheckpointManager

""" Default batch loader for normalizing """
DEFAULT_BATCH_LOADER: BatchLoader


SESSION_TRAINER: SessionTrainer

class Globals:
    """
    Global config access
    """
    def __init__(self,      
                 
        CONFIG: DictConfig,
        
        REPLAY_BUFFER_LOADER: ReplayBufferLoader,
        
        DATALOADERS: DataLoaders,
        
        LOGGER: Logging,
        
        CHECKPOINTER: TopKCheckpointManager,
        
        DEFAULT_BATCH_LOADER: BatchLoader,
        
        SESSION_TRAINER: SessionTrainer,
        
        MODELS: Models = None, # type:ignore
        
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
        
GLOBALS_DICT = {}
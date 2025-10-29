import os
import hydra
from omegaconf import OmegaConf
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
from diffusion_policy.model.diffusion_ql.diffusion_ql_loss import CriticLoss
from diffusion_policy.model.models import Models
from diffusion_policy.common.checkpointer import TopKCheckpointManager
from diffusion_policy.common.logging import Logging

"""
Global config access
"""
CONFIG: OmegaConf

"""
Contains replay buffers for each dataset specified
"""
REPLAY_BUFFER_LOADER: ReplayBufferLoader

"""
Contains both train and val data loaders
"""
DATALOADERS: dict

"""
Contains all models used in training
"""
MODELS: Models

""" Epoch count """
EPOCH = 1 # start on 1 so that the fcn EveryEpoch doesn't fire on the first epoch

""" Training step count """
STEP = 0

""" Logger """
LOGGER: Logging

""" Checkpointer """
CHECKPOINTER: TopKCheckpointManager

""" Sim Env? """
# SIM_ENV
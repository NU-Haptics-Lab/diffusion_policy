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
from diffusion_policy.losses.critic_loss import CriticLoss
from diffusion_policy.model.models import Models

"""
Global config access
"""
CONFIG: OmegaConf = None

"""
Contains replay buffers for each dataset specified
"""
REPLAY_BUFFER_LOADER: ReplayBufferLoader = None

"""
Contains both train and val data loaders
"""
DATALOADERS = None

"""
Contains all models used in training
"""
MODELS: Models = None



""" Epoch count """
EPOCH = 0
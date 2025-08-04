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

"""
Global config access
"""

CONFIG: OmegaConf = None

DATASET: BaseImageDataset = None

REPLAY_BUFFER_LOADER: ReplayBufferLoader = None
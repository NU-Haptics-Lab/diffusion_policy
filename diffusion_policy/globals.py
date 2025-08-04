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

"""
Global config access
"""

CONFIG: OmegaConf

DATASET: BaseImageDataset
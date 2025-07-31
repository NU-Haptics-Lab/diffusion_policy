import hydra
from omegaconf import OmegaConf
from diffusion_policy.dataset.base_dataset import BaseImageDataset

"""
Global config access
"""

CONFIG: OmegaConf

DATASET: BaseImageDataset
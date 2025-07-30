import hydra
from omegaconf import OmegaConf

"""
Global config access
"""

CONFIG: OmegaConf

ZARR_TO_DATA_KEYS = {
    "img", "image",
    "img2", "image2",
    "state", "lowdim_obs"
}

BATCH_STRUCTURE = [
    "obs": {
        "image",
        "image2",
        "lowdim_obs"
    },
    "action",
    "reward",
    "not_done"
]
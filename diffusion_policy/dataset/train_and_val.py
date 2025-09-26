import copy
from omegaconf import OmegaConf

from typing import Dict
import torch
from torch.utils.data import DataLoader as torchDataLoader
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sars_sampler import (
    DatasetSampler, get_val_mask, downsample_mask, get_not_done)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
import diffusion_policy.globals as globals

# TODO: move this class to its own file
class DexNexDataset(BaseImageDataset):
    def __init__(self,
                 sampler: DatasetSampler
        ):
        self.sampler = sampler
    
    def _fix_obs(self, obs):
        """
        Fix the messups in the zarr dataset. 
        """
        # Moveaxis moved to the dataset generation script to save training time
        # now I must do this to be backwards compatable with my messed up dataset order. whoops!
        for key in globals.CONFIG.obs_keys_to_use:
            if "img" in key:
                obs[key] = np.moveaxis(obs[key], 2, 1)

    def _sample_to_data(self, sample):
        """
        custom fix for our zarr dataset, as well as casting the data down to float32 to save space
        """
        # fix this state
        self._fix_obs(sample["obs"])
        self._fix_obs(sample["obs_next"])
        
        # convert all data to float32 to save space
        def fcn(x):
            out = x.astype(np.float32) # returns a copy
            return out
        data = dict_apply(sample, fcn)

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a single sample from the dataset
        """
        # get sample from the sampler, axes are [T, ...], T - trajectory. The sampler already puts the data in a dictionary.
        sample = self.sampler.get_sample(idx)
        
        # convert the sample to a neural net compatible data dict 
        data = self._sample_to_data(sample)

        # put in a torch array
        torch_data = dict_apply(data, torch.from_numpy)

        return torch_data

    def __len__(self) -> int:
        return len(self.sampler)
    

class TrainAndVal:
    """
    Dataset to provide (s, a, r, s') samples to a torch dataloader. Formerly named dexnex_2cams_image_ql_dataset.py:DexNexDataset but that name isn't descriptive.

    note: s and s' are actually observations.

    zarr dataset keys:
    - img
    - img2
    - state
    """
    def __init__(self,
            sampler: DatasetSampler, # default sampler, not init'd
            options: dict,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            ):
        
        super().__init__()
        rb_id = sampler.rb_id
        self.options = options

        # get nb episodes
        nb_episodes = globals.REPLAY_BUFFER_LOADER[rb_id].n_episodes
        
        val_mask = get_val_mask(
            n_episodes=nb_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        # downsamples if max_train_episodes is not None
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # make train sampler with train mask
        self.train_sampler = copy.deepcopy(sampler)
        self.train_sampler.Init(train_mask)

        # make an exact copy
        self.val_sampler = copy.deepcopy(sampler)
        self.val_sampler.Init(val_mask)
        
        # make the datasets
        self.train_dataset = DexNexDataset(self.train_sampler)
        self.val_dataset = DexNexDataset(self.val_sampler)
        
        # make the train & val config
        train_cfg = copy.deepcopy(options.common)
        val_cfg = copy.deepcopy(options.common)
        OmegaConf.unsafe_merge(train_cfg, options.train)
        OmegaConf.unsafe_merge(val_cfg, options.val)
        
        # make the train & val dataloader
        self.train_dataloader = torchDataLoader(
            self.train_dataset,
            **train_cfg
        )
        self.val_dataloader = torchDataLoader(
            self.val_dataset,
            **val_cfg
        )
        
        # dict access
        self.dd = {}
        self.dd["train"] = self.train_dataloader
        self.dd["val"] = self.val_dataloader
        
    def __getitem__(self, key):
        return self.dd[key]
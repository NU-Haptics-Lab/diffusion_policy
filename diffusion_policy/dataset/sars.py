from typing import Dict
import torch
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


class SARSDataset(BaseImageDataset):
    """
    Dataset to provide (s, a, r, s') samples to a torch dataloader. Formerly named dexnex_2cams_image_ql_dataset.py:DexNexDataset but that name isn't descriptive.

    note: s and s' are actually observations.

    zarr dataset keys:
    - img
    - img2
    - state
    """
    def __init__(self,
            sampler: DatasetSampler,
            shape_meta: dict,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            state_length=8,
            history_indices=[]
            ):
        
        super().__init__()
        self.sampler = sampler
        
        ## block taken from real_pusht_image_dataset.py for huge training speedup
        rgb_keys = ['img', 'img2']
        lowdim_keys = ['state']

        self.dataset_keys = rgb_keys + lowdim_keys
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.state_length = state_length
        self.history_indices = history_indices
        self.device = torch.device(globals.CONFIG.training.device)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    

    def __len__(self) -> int:
        return len(self.sampler)
    
    def _collate_state(self, sample, suffix=""):
        agent_pos = sample['state' + suffix][:][:, :self.state_length].astype(np.float32)
        
        # Moveaxis moved to the dataset generation script to save training time
        # now I must do this to be backwards compatable with my messed up dataset order. whoops!
        image = np.moveaxis(sample['img' + suffix], 2, 1)
        image2 = np.moveaxis(sample['img2' + suffix], 2, 1)

        out = {
            'image': image,  # T, 3, 96, 96
            'image2': image2,
            'agent_pos': agent_pos,  # T, self.state_length
        }

        return out
        

    def _sample_to_data(self, sample):
        """
        output -- data: {s, a, r, s', not_done}
        """
        # collate this state
        state = self._collate_state(sample)

        # current state
        data = {
            'obs': state,
            'action': sample['action'].astype(np.float32), # T, self.state_length
            'reward': sample['reward'].astype(np.float32),
            'not_done': sample['not_done'].astype(np.float32)
        }
        
        # next state
        data['obs_next'] = self._collate_state(sample, suffix="_next")
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a single sample from the dataset
        """
        # get sample from the sampler, axes are [T, ...], T - trajectory
        sample = self.sampler.sample_sequence(idx)
        
        # convert the sample to a neural net compatible data dict 
        data = self._sample_to_data(sample)

        # put in a torch array
        torch_data = dict_apply(data, torch.from_numpy)

        return torch_data
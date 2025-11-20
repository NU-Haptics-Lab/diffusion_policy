from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler_ql import (
    SequenceSampler, get_val_mask, downsample_mask, get_not_done)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
from diffusion_policy.globals import CONFIG


class DexNexDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            state_length=8,
            history_indices=[]
            ):
        
        super().__init__()
        # self.replay_buffer = ReplayBuffer.copy_from_path(
        #     zarr_path, keys=['img', 'state', 'action'])
        
        # this will load directly from disk, and not into RAM. There's no noticeable slowdown. You really don't want to load into RAM, so that we don't save the entire dataset into each checkpoint during pickling
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path)
        
        print("Replay buffer nb datapoints: ", self.replay_buffer.n_steps) 
        print("Replay buffer nb episodes: ", self.replay_buffer.n_episodes) 
        
        ## block taken from real_pusht_image_dataset.py for huge training speedup
        rgb_keys = ['img', 'img2']
        lowdim_keys = ['state']
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps
        ## end block
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
            history_indices=history_indices
            )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.state_length = state_length
        self.n_obs_steps = n_obs_steps
        self.history_indices = history_indices
        self.device = torch.device(CONFIG.training.device)

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
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps) # trajectory slice
        
        # hard-coded state keys, obtained from the rosbag-to-zarr dataset conversion script
        state_keys = [
            "img",
            "img2",
            "state",
        ]
        
        agent_pos = sample['state' + suffix][T_slice][:, :self.state_length].astype(np.float32)
        
        # Moveaxis moved to the dataset generation script to save training time
        # now I must do this to be backwards compatable with my messed up dataset order. whoops!
        image = np.moveaxis(sample['img' + suffix][T_slice], 2, 1)
        image2 = np.moveaxis(sample['img2' + suffix][T_slice], 2, 1)

        out = {
            'image': image,  # T, 3, 96, 96
            'image2': image2,
            'agent_pos': agent_pos,  # T, self.state_length
        }

        return out
        

    def _sample_to_data(self, sample):
        # image = sample['img']
        # image2 = sample['img2']
        
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
        # data['obs_next'] = self._collate_state(sample, suffix="_next")
        
        # history of states
        for val in self.history_indices:
            suffix = "_" + str(val)
            data['obs' + suffix] = self._collate_state(sample, suffix=suffix)
        
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


def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = DexNexDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

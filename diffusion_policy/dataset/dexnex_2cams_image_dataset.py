from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat

""" 
joint limits for the normalizer. Only thing that matters is the scale and offset. stat's aren't used in normalization

in normalizer.py, if forward: x = x * scale + offset
can use normalize_util.get_range_normalizer_from_stat to do the math for us, just need to set stat['max'] and stat['min']

"""
# some values from my joint_limits.yaml, others from the urdf or 
# https://www.shadowrobot.com/wp-content/uploads/2022/03/shadow_dexterous_hand_e_technical_specification.pdf
JOINT_LIMITS = np.array([ 
    [0.52, 2.5], # gofa
    [-1.0, 2.0], # gofa
    [-1.0, 1.1], # gofa
    [-1.0, 1.1], # gofa
    [-0.6, 2.0], # gofa
    [-2.5, 2.5], # gofa
    [-0.489, 0.140], # lh_WRJ2
    [-0.698, 0.489], # lh_WRJ1
    [-0.35, 0.35], # "lh_FFJ4",
    [-0.262, 1.571], # "lh_FFJ3",
    [0, 1.57], # "lh_FFJ2",
    [0, 1.57], # "lh_FFJ1",
    [-0.35, 0.35], # "lh_MFJ4",
    [-0.262, 1.571], # "lh_MFJ3",
    [0, 1.57], # "lh_MFJ2",
    [0, 1.57], # "lh_MFJ1",
    [-1.047, 1.04], # "lh_THJ5",
    [0, 1.22], # "lh_THJ4",
    [-0.209, 0.20], # "lh_THJ3",
    [-0.7, 0.7], # "lh_THJ2", # urdf is slightly wider than tech spec. Go with the wider one to be safe
    [-0.262, 1.571], # "lh_THJ1",
], dtype='float32')

HAPTICS = np.array(5 * [[0.0, 1.0]], dtype='float') # haptics were already normalized from 0 to 1

# values taken from the normalizer fit function, then rounded to the nearest meter
mins = [0.2193,  1.1434,  0.1246,  0.2273,  1.1635,  0.1175, 0.2029,  1.1771,  0.1151]
maxs = [0.5204, 1.3973, 0.4805, 0.5474, 1.4750, 0.5378, 0.5411, 1.4861, 0.5283]
FINGERTIP_POS = np.stack((np.floor(mins), np.ceil(maxs)), axis=1)

LIMITS = np.concatenate((JOINT_LIMITS, HAPTICS, FINGERTIP_POS), axis=0, dtype='float32')

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
            state_length=8
            ):
        
        super().__init__()
        # self.replay_buffer = ReplayBuffer.copy_from_path(
        #     zarr_path, keys=['img', 'state', 'action'])
        
        # this will load directly from disk, and not into RAM. There's no noticeable slowdown. You really don't want to load into RAM, so that we don't save the entire dataset into each checkpoint during pickling
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path)
        
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
            key_first_k=key_first_k
            )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.state_length = state_length
        self.n_obs_steps = n_obs_steps

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

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][..., :self.state_length]
        }
        # normalizer = LinearNormalizer()
        # normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        normalizer = LinearNormalizer()
        normalizer['agent_pos'] = get_range_normalizer_from_stat({'min': LIMITS[:, 0], 'max': LIMITS[:, 1]})
        normalizer['action'] = get_range_normalizer_from_stat({'min': JOINT_LIMITS[:, 0], 'max': JOINT_LIMITS[:, 1]})

        normalizer['image'] = get_image_range_normalizer()
        normalizer['image2'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps) # trajectory slice
        
        agent_pos = sample['state'][T_slice][:, :self.state_length].astype(np.float32)
        
        # Moveaxis moved to the dataset generation script to save training time
        # now I must do this to be backwards compatable with my messed up dataset order. whoops!
        image = np.moveaxis(sample['img'][T_slice], 2, 1)
        image2 = np.moveaxis(sample['img2'][T_slice], 2, 1)
        # image = sample['img']
        # image2 = sample['img2']

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'image2': image2,
                'agent_pos': agent_pos, # T, self.state_length
            },
            'action': sample['action'].astype(np.float32) # T, self.state_length
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        
        data = self._sample_to_data(sample)
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

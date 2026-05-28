"""Dataset for PolyUMI-exported ReplayBuffer zarrs.

Schema (produced by polyumi_ingest export-dp):
    img:    (T, 256, 256, 3) float32 in [0, 1]
    state:  (T, 8)            float32  [x, y, z, qx, qy, qz, qw, gripper_m]
    action: (T, 8)            float32  same absolute pose; converted to relative
                                       per-sample at training time below.

`__getitem__` returns the standard diffusion-policy dict::

    {'obs': {'image': (T, 3, 256, 256), 'agent_pos': (T, 8)},
     'action': (T, 8)}  # row 0 = identity, rows >0 = T_init^-1 ∘ T_k
"""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


def to_relative_action(abs_action: np.ndarray) -> np.ndarray:
    """Convert an absolute pose+gripper trajectory to relative-to-first-step.

    Input  (T, 8) = [x, y, z, qx, qy, qz, qw, gripper] in scipy quaternion order.
    Output (T, 8) where row 0 is identity ([0,0,0, 0,0,0,1, gripper_0]) and row k is
    ``T_init^-1 ∘ T_k`` expressed in the same parameterization, with ``qw >= 0`` to
    remove quaternion double-cover ambiguity.
    """
    t = abs_action[:, :3]
    q = abs_action[:, 3:7]
    grip = abs_action[:, 7:8]

    rot = Rotation.from_quat(q)
    rot_init_inv = rot[0].inv()
    t_rel = rot_init_inv.apply(t - t[0])
    q_rel = (rot_init_inv * rot).as_quat()
    flip = q_rel[:, 3] < 0
    q_rel[flip] *= -1.0

    rel = np.concatenate([t_rel, q_rel, grip], axis=1).astype(np.float32)
    rel[0, :3] = 0.0
    rel[0, 3:7] = [0.0, 0.0, 0.0, 1.0]
    return rel


class PolyUMIImageDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
    ):
        super().__init__()
        # Open on-disk; lazy reads keep memory low and avoid pickling the whole dataset
        # into checkpoints. Same pattern as DexNexDataset.
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(mask=~val_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self) -> 'PolyUMIImageDataset':
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def _sample_relative_actions(self, n: int = 1024, seed: int = 0) -> np.ndarray:
        """Sample ``n`` random horizons, return their relative actions stacked as (n*T, 8).

        Used to fit the action normalizer on the relative parameterization actually seen
        at training time (absolute optitrack values would yield a wildly wrong scale).
        """
        n = min(n, len(self.sampler))
        if n == 0:
            return np.zeros((0, 8), dtype=np.float32)
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self.sampler), size=n, replace=False)
        chunks = []
        for i in idxs:
            sample = self.sampler.sample_sequence(int(i))
            chunks.append(to_relative_action(sample['action']))
        return np.concatenate(chunks, axis=0)

    def get_normalizer(self, mode: str = 'limits', **kwargs) -> LinearNormalizer:
        data = {
            'action': self._sample_relative_actions(),
            'agent_pos': np.asarray(self.replay_buffer['state']),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._sample_relative_actions())

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image = np.moveaxis(sample['img'], -1, 1).astype(np.float32)  # (T,3,256,256), [0,1]
        agent_pos = sample['state'].astype(np.float32)                # (T, 8)
        action = to_relative_action(sample['action'])                  # (T, 8) relative
        return {
            'obs': {
                'image': image,
                'agent_pos': agent_pos,
            },
            'action': action,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)

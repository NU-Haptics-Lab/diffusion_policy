"""Dataset for PolyUMI-exported ReplayBuffer zarrs.

Schema (produced by polyumi_ingest export-dp):
    img:    (T, 256, 256, 3) float32 in [0, 1]
    state:  (T, 8)            float32  [x, y, z, qx, qy, qz, qw, gripper_m]
    action: (T, 8)            float32  same absolute pose; converted to relative
                                       per-sample at training time below.

`__getitem__` returns the standard diffusion-policy dict::

    {'obs': {'image': (T, 3, 256, 256), 'agent_pos': (T, 1)},
     'action': (T, 8)}  # row 0 = identity, rows >0 = T_init^-1 ∘ T_k

The lowdim observation is gripper width only — the absolute optitrack pose is intentionally
excluded because (1) the policy predicts pose deltas from a fixed origin (the start of each
horizon), so absolute pose carries no learnable signal, and (2) it would couple the policy
to the specific optitrack world frame.

The action normalizer is fit on *relative* actions to match what the network sees at train
time. Absolute optitrack values would yield a wildly wrong scale.
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


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton quaternion multiplication in scipy [x, y, z, w] order.

    For unit quaternions a, b representing rotations R_a, R_b: ``_quat_mul(a, b)``
    represents the composition that, applied to a vector v, gives ``R_a(R_b(v))``.
    Matches ``(Rotation.from_quat(a) * Rotation.from_quat(b)).as_quat()``.
    """
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` (..., 3) by unit quaternion ``q`` (..., 4) in [x, y, z, w] order."""
    qv = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def to_relative_action_torch(abs_action: torch.Tensor) -> torch.Tensor:
    """Torch version of :func:`to_relative_action`, GPU-friendly (no host sync)."""
    t = abs_action[:, :3]
    q = abs_action[:, 3:7]
    grip = abs_action[:, 7:8]

    # Inverse of a unit quaternion [x, y, z, w] is [-x, -y, -z, w].
    q0 = q[0]
    q0_inv = torch.cat((-q0[:3], q0[3:4]))                  # (4,)

    t_rel = _quat_apply(q0_inv.expand_as(q), t - t[0])      # (T, 3)
    q_rel = _quat_mul(q0_inv.expand_as(q), q)               # (T, 4)
    q_rel = torch.where(q_rel[:, 3:4] < 0, -q_rel, q_rel)   # hemisphere flip

    out = torch.cat((t_rel, q_rel, grip), dim=1)            # (T, 8)
    # Force exact identity at row 0 — protects against floating-point drift in the
    # very first frame, which the policy uses as the trajectory anchor.
    identity = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        dtype=out.dtype, device=out.device,
    )
    out = torch.cat((
        torch.cat((identity, grip[0])).unsqueeze(0),        # (1, 8)
        out[1:],                                            # (T-1, 8)
    ), dim=0)
    return out


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
        cache_to_gpu: bool | str | None = None,
    ):
        """
        cache_to_gpu: materialize the full replay buffer as torch tensors on GPU.
            Eliminates per-step disk I/O and host→device transfers. Accepts:
              - ``None`` / ``False``: disabled (default; on-disk lazy reads).
              - ``True``: use ``cuda`` (current default CUDA device).
              - explicit device string, e.g. ``"cuda:0"``.
            When enabled, set ``dataloader.num_workers=0`` and ``pin_memory=false`` in
            both train and val loaders — GPU tensors can't cross process boundaries or
            be pinned.
        """
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

        # GPU cache (optional). Pre-permute img to CHW so consumers don't pay that cost.
        self._gpu_data: dict[str, torch.Tensor] | None = None
        if cache_to_gpu:
            device = torch.device('cuda' if cache_to_gpu is True else cache_to_gpu)
            img = np.asarray(self.replay_buffer['img'])              # (N, H, W, 3) float32
            img = np.moveaxis(img, -1, 1).copy()                     # (N, 3, H, W)
            state = np.asarray(self.replay_buffer['state'])          # (N, 8)
            action = np.asarray(self.replay_buffer['action'])        # (N, 8)
            self._gpu_data = {
                'img': torch.from_numpy(img).to(device),
                'state': torch.from_numpy(state).to(device),
                'action': torch.from_numpy(action).to(device),
            }

    def get_validation_dataset(self) -> 'PolyUMIImageDataset':
        val_set = copy.copy(self)
        # Shallow copy shares self._gpu_data (intentional).
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
        at training time.
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
            # Only the gripper-width column (state[:, 7]) feeds the policy at inference;
            # fit the normalizer on that same column so train and inference scales match.
            'agent_pos': np.asarray(self.replay_buffer['state'])[:, 7:8],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._sample_relative_actions())

    def __len__(self) -> int:
        return len(self.sampler)

    # ---- per-sample sampling ------------------------------------------------

    def _gpu_slice(self, key: str, idx: int) -> torch.Tensor:
        """Mirror SequenceSampler.sample_sequence for a single key, on GPU."""
        buf_start, buf_end, sam_start, sam_end = (int(x) for x in self.sampler.indices[idx])
        tensor = self._gpu_data[key]  # type: ignore[index]
        sub = tensor[buf_start:buf_end]
        if sam_start > 0 or sam_end < self.horizon:
            full = torch.zeros(
                (self.horizon,) + tensor.shape[1:],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            if sam_start > 0:
                full[:sam_start] = sub[0]
            if sam_end < self.horizon:
                full[sam_end:] = sub[-1]
            full[sam_start:sam_end] = sub
            sub = full
        return sub

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._gpu_data is not None:
            image = self._gpu_slice('img', idx)                  # (T, 3, H, W) float32 [0,1]
            state = self._gpu_slice('state', idx)                # (T, 8)
            action = self._gpu_slice('action', idx)              # (T, 8) absolute
            action_rel = to_relative_action_torch(action)        # stays on GPU; no host sync

            return {
                'obs': {
                    'image': image,
                    'agent_pos': state[:, 7:8].contiguous(),  # gripper-width only
                },
                'action': action_rel,
            }

        # CPU / on-disk path
        sample = self.sampler.sample_sequence(idx)
        image = np.moveaxis(sample['img'], -1, 1).astype(np.float32)  # (T,3,H,W) [0,1]
        agent_pos = sample['state'][:, 7:8].astype(np.float32)        # gripper-width only
        action = to_relative_action(sample['action'])                 # (T, 8) relative
        data = {
            'obs': {
                'image': image,
                'agent_pos': agent_pos,
            },
            'action': action,
        }
        return dict_apply(data, torch.from_numpy)

import copy
from collections import OrderedDict
from omegaconf import OmegaConf

from typing import Dict
import torch
from torch import nn
from torch.utils.data import DataLoader as torchDataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np
import scipy.stats
import copy
from tqdm import tqdm
from diffusion_policy import utils
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer


import diffusion_policy.common.sarsa_sampler as sarsa_sampler 
# # import (
#     DatasetSampler, get_val_mask, downsample_mask, get_not_done)


from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
import diffusion_policy.globals as globals
from diffusion_policy.model.diffusion_ql.diffusion_ql_loss import CriticLoss

# TODO: move this class to its own file
class DexNexDataset(BaseImageDataset):
    def __init__(self,
                 sampler
        ):
        assert(isinstance(sampler, sarsa_sampler.DatasetSampler))
        self.sampler = sampler
        
    def reinit_all(self):
        # nothing to do
        pass
    
    def _fix_obs(self, obs):
        """
        Fix the messups in the zarr dataset. 
        """
        # Moveaxis moved to the dataset generation script to save training time
        # now I must do this to be backwards compatable with my messed up dataset order. whoops!
        obs_keys_to_use = globals.CONFIG.obs_keys_to_use # type: ignore
        for key in obs_keys_to_use:
            if "img" in key:
                obs[key] = np.moveaxis(obs[key], 2, 1)
                
        # now each image should be [batch, history, channels, pixelsx, pixelsy]

    def _sample_to_data(self, sample):
        """
        custom fix for our zarr dataset, as well as casting the data down to float32 to save space
        """
        # fix this state
        self._fix_obs(sample["obs"])
        # self._fix_obs(sample["obs_next"])
        
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
    
    def get_key(self, key):
        return self.sampler.get_key(key)
    
    def get_qvals(self):
        return self.sampler.get_qvals()
    
    def get_ep_lengths(self):
        return self.sampler.get_ep_lengths_arr()
        
        
    

def _nested_stack_to_device(samples, device):
    """Recursively stack a list of nested dicts/tensors and move to device."""
    if isinstance(samples[0], dict):
        return {k: _nested_stack_to_device([s[k] for s in samples], device) for k in samples[0]}
    return torch.stack(samples).to(device)


def _nested_index(data, idx):
    if isinstance(data, dict):
        return {k: _nested_index(v, idx) for k, v in data.items()}
    return data[idx]


def _detect_zero_fill_obs_keys(dataset):
    """
    Returns the subset of obs_keys_to_load that this dataset's sampler
    zero-fills rather than reading from the replay buffer (EpisodeSampler.
    resolve_rb_key returns None for these -- see AIETErlenmeyerFlask3SimSampler/
    RealSampler). Returns None if that can't be determined (sampler doesn't
    expose get_ep_list/episode samplers).

    GPUCachedDataset uses this to avoid ever materializing these as full
    (N, 1, *shape) tensors -- some zero-filled keys use the OTHER data
    source's shape (e.g. sim zero-fills the 14x14x384 vision patch keys it
    has no use for), so naively stacking/concatenating them across the whole
    dataset can dwarf the size of the actual data by orders of magnitude.
    They're synthesized instead, cheaply, per __getitem__ call.
    """
    sampler = dataset.sampler
    if not hasattr(sampler, 'get_ep_list'):
        return None
    eps = list(sampler.get_ep_list())
    if not eps:
        return None
    obs_keys_to_load = list(globals.CONFIG.obs_keys_to_load)  # type: ignore
    return {key for key in obs_keys_to_load if eps[0].resolve_rb_key(key) is None}


def _vectorized_load_dataset(dataset: 'DexNexDataset', device: torch.device):
    """
    Fast path for GPUCachedDataset. dataset[i] independently re-extracts and
    re-converts each sample's action window, but adjacent samples' windows
    (per action_rel_indices) overlap almost entirely -- redundant work. This
    instead loads each raw replay-buffer array once per episode and gathers
    ALL of that episode's valid obs/action windows with a handful of
    vectorized numpy operations, instead of one Python-level call per sample.

    Returns (data, zero_fill_obs_keys) on success, or None (caller should
    fall back to the per-sample path) if the sampler isn't the expected
    DatasetSampler-of-EpisodeSamplers shape, OR if a key still can't be found
    in the replay buffer even after resolving it through the episode
    sampler's resolve_rb_key/resolve_action_rb_key hooks (identity by
    default; samplers doing custom key aliasing or zero-filling -- e.g.
    AIETErlenmeyerFlask3SimSampler aliasing joint_positions to joint_states,
    or zero-filling keys the other data source doesn't have -- override those
    hooks so this fast path stays correct instead of reading the wrong data).

    zero_fill_obs_keys are deliberately excluded from `data['obs']` -- see
    GPUCachedDataset's docstring for why (a zero-filled key using the other
    data source's shape can dwarf the real data if materialized densely for
    every sample). The caller synthesizes them cheaply per __getitem__ instead.
    """
    sampler = dataset.sampler
    if not hasattr(sampler, 'get_ep_list'):
        return None

    obs_keys_to_load = list(globals.CONFIG.obs_keys_to_load)  # type: ignore
    action_key = globals.CONFIG.action_key  # type: ignore
    action_rel_indices = np.array(globals.CONFIG.action_rel_indices)  # type: ignore

    eps = list(sampler.get_ep_list())
    assert len(eps) > 0
    # all episodes share the same underlying ReplayBuffer -- load each raw
    # array exactly ONCE, not once per episode (that was the original bug:
    # np.asarray(rb[key]) inside the episode loop re-decompressed the whole
    # zarr array on every single episode iteration).
    rb = eps[0].indices.replay_buffer

    # resolve each obs/action key to its actual rb key (None => zero-fill),
    # via the episode sampler's hooks -- identity by default, overridden by
    # samplers that alias or zero-fill keys
    obs_key_map = {key: eps[0].resolve_rb_key(key) for key in obs_keys_to_load}

    # some samplers' action isn't a literal rb array at all -- e.g.
    # AIETAlignmentSim3Sampler's is [rel_palm_pose_xyz_rpy, gripper_value],
    # computed by subtracting/concatenating two DIFFERENT rb arrays, not an
    # alias of a single one. Those samplers expose build_action_chunk(train_idx)
    # (batched, per-episode equivalent of get_action_trajectory) instead of
    # relying on resolve_action_rb_key's identity/single-key-alias model.
    build_action_chunk = getattr(eps[0], "build_action_chunk", None)
    raw_action_array = None
    if build_action_chunk is None:
        resolved_action_key = eps[0].resolve_action_rb_key(action_key)

    raw_obs_arrays = {}
    zero_fill_obs_keys = set()
    try:
        for key, rb_key in obs_key_map.items():
            if rb_key is None:
                zero_fill_obs_keys.add(key)
            else:
                raw_obs_arrays[key] = np.asarray(rb[rb_key])
        if build_action_chunk is None:
            raw_action_array = np.asarray(rb[resolved_action_key])
    except KeyError as e:
        print(f"_vectorized_load_dataset declining: key {e} not found in the replay buffer "
              f"even after key-alias resolution")
        return None
    # task_id/subtask_id/data_source are all OPTIONAL top-level sample
    # fields -- not every sampler emits every one of them (e.g. data_source
    # is only set by the aiet_erlenmeyer_flask_3 cotraining Real/Sim
    # subclasses' get_sample(); AIETAlignmentSimSampler never sets it at
    # all). Probe a real sample once to find out which of them THIS sampler
    # actually produces, rather than assuming all three -- building/
    # including a field the sampler doesn't emit would both waste work and
    # make the safety check below KeyError on expected[key].
    probe_sample = dataset[0]
    scalar_fields = [f for f in ("task_id", "subtask_id", "data_source") if f in probe_sample]

    # ep_id (AIETErlenmeyerFlaskSampler.get_sample) is a THIRD kind of scalar
    # field, distinct from both of the above: it's neither a raw per-timestep
    # rb array (there's no rb['ep_id']) nor a single dataset-wide constant --
    # it's constant PER EPISODE (== that episode's own rb_episode_end), so it
    # needs its own per-episode-broadcast handling below rather than
    # resolve_scalar_field's "constant or rb array" logic, and must be
    # excluded from extra_fields (below) so that bucket doesn't try
    # rb['ep_id'] and decline vectorization entirely for every sampler that
    # emits ep_id.
    has_ep_id = 'ep_id' in probe_sample

    # some samplers hardcode these in get_sample() regardless of what's in
    # the rb (or even when the rb has no such array at all to read a
    # per-sample value from) -- constant_fields takes precedence over
    # reading the rb, matching that behavior exactly instead of guessing a
    # generic fallback that may not match (see get_constant_sample_fields's
    # docstring; this is what caught the sim dataset's data_source being
    # read back as 0 instead of the hardcoded 1).
    constant_fields = eps[0].get_constant_sample_fields()
    generic_fallback = {"task_id": 24.0, "subtask_id": 0.0, "data_source": 0.0}

    def resolve_scalar_field(field_name):
        if field_name in constant_fields:
            return None, constant_fields[field_name]
        try:
            return np.asarray(rb[field_name]), None
        except KeyError:
            return None, generic_fallback[field_name]

    raw_scalar_arrays = {}
    scalar_fallbacks = {}
    for field_name in scalar_fields:
        raw_scalar_arrays[field_name], scalar_fallbacks[field_name] = resolve_scalar_field(field_name)

    # any OTHER top-level fields beyond obs/action/task_id/subtask_id/
    # data_source (e.g. AIETAlignmentSimSampler's wrist_pellet_keypoints_px/
    # _valid/target_pellet_valid) -- assumed to be direct, unaliased rb
    # arrays of the same name, read once and fancy-indexed per episode just
    # like obs keys. If that assumption is wrong for some future sampler
    # (e.g. it needs the aliasing/constant-field handling task_id/
    # subtask_id/data_source get above), this declines to vectorize rather
    # than risk silently building incorrect data.
    extra_fields = [k for k in probe_sample if k not in ("obs", "action", "ep_id") and k not in scalar_fields]
    raw_extra_arrays = {}
    try:
        for field_name in extra_fields:
            raw_extra_arrays[field_name] = np.asarray(rb[field_name])
    except KeyError as e:
        print(f"_vectorized_load_dataset declining: extra top-level field {e} not found "
              f"directly in the replay buffer")
        return None

    # zero-fill keys are never read/allocated per-sample here -- for a data
    # source that doesn't have them at all (e.g. sim zero-filling the real
    # side's 14x14x384 vision patch keys), a dense (N, 1, *shape) tensor of
    # zeros can dwarf the size of the actual data by orders of magnitude.
    # GPUCachedDataset synthesizes these on the fly in __getitem__ instead.
    per_key_obs = {k: [] for k in obs_keys_to_load if k not in zero_fill_obs_keys}
    action_chunks = []
    scalar_chunks = {field_name: [] for field_name in scalar_fields}
    extra_chunks = {field_name: [] for field_name in extra_fields}
    ep_id_chunks = [] if has_ep_id else None

    for ep in tqdm(eps, desc="gpu-preload (vectorized, per-episode)"):
        indices = ep.indices
        train_idx = np.array(list(indices.get_all_train_indices()))
        if len(train_idx) == 0:
            continue
        n = len(train_idx)

        # obs: single "current" index per sample (obs_rel_indices == [0]).
        # get_key_sample wraps this in a length-1 list, giving each sample a
        # (1, D) shape (a history-length-1 axis) -- add that axis back here.
        obs_rb_idx = indices.get_rb_indices(train_idx)  # (N,)
        for key in per_key_obs:
            per_key_obs[key].append(raw_obs_arrays[key][obs_rb_idx][:, None, ...])

        # action: one column per action_rel_indices offset -> (N, L, *feat)
        if build_action_chunk is not None:
            action_chunks.append(ep.build_action_chunk(train_idx))
        else:
            action_rb_idx = np.stack(
                [indices.get_rb_indices(train_idx + off) for off in action_rel_indices],
                axis=1,
            )
            action_chunks.append(raw_action_array[action_rb_idx])

        # task_id / subtask_id / data_source (whichever this sampler actually
        # emits, per scalar_fields above): (1,)-shaped value per sample,
        # either read from the rb or filled with whichever fallback applies
        # (the sampler's hardcoded constant, if it has one -- see above --
        # else the generic default matching get_sample's rb-absent behavior)
        for field_name in scalar_fields:
            raw_array = raw_scalar_arrays[field_name]
            if raw_array is not None:
                scalar_chunks[field_name].append(raw_array[obs_rb_idx][:, None])
            else:
                scalar_chunks[field_name].append(np.full((n, 1), scalar_fallbacks[field_name], dtype=np.float32))

        # extra top-level fields (e.g. wrist_pellet_keypoints_px/_valid,
        # target_pellet_valid): same "current" indexing as obs, add back the
        # history-length-1 axis get_key_sample would have produced
        for field_name in extra_fields:
            extra_chunks[field_name].append(raw_extra_arrays[field_name][obs_rb_idx][:, None, ...])

        # ep_id: constant for every sample in this episode (its own
        # rb_episode_end), not read from anywhere -- see has_ep_id above
        if has_ep_id:
            ep_id_chunks.append(np.full((n, 1), float(ep.rb_episode_end), dtype=np.float32))

    obs = {}
    for key in per_key_obs:
        full = np.concatenate(per_key_obs[key], axis=0).astype(np.float32)
        if "img" in key:
            # matches DexNexDataset._fix_obs's moveaxis(obs[key], 2, 1), shifted
            # by one axis since a batch dim is now at position 0
            full = np.moveaxis(full, 3, 2)
        obs[key] = torch.from_numpy(full)

    data = {
        'obs': obs,
        'action': torch.from_numpy(np.concatenate(action_chunks, axis=0).astype(np.float32)),
    }
    for field_name in scalar_fields:
        data[field_name] = torch.from_numpy(np.concatenate(scalar_chunks[field_name], axis=0).astype(np.float32))
    for field_name in extra_fields:
        data[field_name] = torch.from_numpy(np.concatenate(extra_chunks[field_name], axis=0).astype(np.float32))
    if has_ep_id:
        data['ep_id'] = torch.from_numpy(np.concatenate(ep_id_chunks, axis=0).astype(np.float32))

    # safety check: this bypasses the normal per-sample path entirely, so
    # verify it against the known-correct dataset[i] for a handful of random
    # indices before trusting it for training.
    rng = np.random.default_rng(0)
    check_idx = rng.choice(len(dataset), size=min(20, len(dataset)), replace=False)
    for i in tqdm(check_idx, desc="gpu-preload (verifying against per-sample path)"):
        expected = dataset[int(i)]
        for key, val in data.items():
            if key == 'obs':
                for obs_key, obs_val in val.items():
                    if not torch.allclose(obs_val[i], expected['obs'][obs_key]):
                        raise RuntimeError(
                            f"_vectorized_load_dataset mismatch at sample {i}, obs key '{obs_key}': "
                            f"vectorized={obs_val[i]} expected={expected['obs'][obs_key]}")
            elif not torch.allclose(val[i], expected[key]):
                raise RuntimeError(
                    f"_vectorized_load_dataset mismatch at sample {i}, key '{key}': "
                    f"vectorized={val[i]} expected={expected[key]}")

    return dict_apply(data, lambda t: t.to(device)), zero_fill_obs_keys


class GPUCachedDataset(torch.utils.data.Dataset):
    """
    Loads an entire DexNexDataset into GPU memory up front.
    __getitem__ then returns directly from the pre-loaded tensors with no
    zarr / numpy overhead per step.

    Obs keys the sampler zero-fills (EpisodeSampler.resolve_rb_key returns
    None -- e.g. sim zero-filling the real side's vision keys) are never
    stacked/concatenated into a full (N, 1, *shape) tensor: for a zero-filled
    key that uses the OTHER data source's shape, that dense tensor can be
    orders of magnitude bigger than the dataset's actual data (observed: a
    250k-sample sim dataset that's ~30MB of real data ballooned past 50GB of
    host RAM and got OOM-killed trying to materialize two zero-filled
    14x14x384 vision keys it has no use for). These keys are synthesized as
    small zero tensors per __getitem__ call instead.
    """
    def __init__(self, dataset: 'DexNexDataset', device: torch.device, num_workers: int = 0, vectorized: bool = False):
        self._source = dataset  # kept for attribute delegation
        self.device = device
        self._len = len(dataset)
        self.zero_fill_obs_keys = set()
        if self._len == 0:
            self._data = None
            return
        print(f"Preloading {self._len} samples to {device} ...")

        self._data = None
        if vectorized:
            result = _vectorized_load_dataset(dataset, device)
            if result is None:
                print("_vectorized_load_dataset declined (unsupported sampler shape); falling back to per-sample loading")
            else:
                self._data, self.zero_fill_obs_keys = result

        if self._data is None:
            self.zero_fill_obs_keys = _detect_zero_fill_obs_keys(dataset) or set()

            def strip_zero_fill(sample):
                if self.zero_fill_obs_keys:
                    sample = dict(sample)
                    sample['obs'] = {k: v for k, v in sample['obs'].items() if k not in self.zero_fill_obs_keys}
                return sample

            if num_workers > 0:
                # parallelize the CPU-bound per-sample construction (zarr reads,
                # astype, torch.from_numpy) across worker processes, same
                # mechanism as normal training dataloading -- only the final
                # GPU stack/transfer below stays in the main process.
                loader = torchDataLoader(
                    dataset,
                    batch_size=1,
                    num_workers=num_workers,
                    collate_fn=lambda batch: strip_zero_fill(batch[0]),
                    shuffle=False,
                )
                samples = [s for s in tqdm(loader, desc="gpu-preload", total=self._len)]
            else:
                samples = [strip_zero_fill(dataset[i]) for i in tqdm(range(self._len), desc="gpu-preload")]
            self._data = _nested_stack_to_device(samples, device)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        if self._data is None:
            raise IndexError("GPUCachedDataset is empty")
        sample = _nested_index(self._data, idx)
        for key in self.zero_fill_obs_keys:
            shape = globals.CONFIG.shape_meta[key].shape  # type: ignore
            sample['obs'][key] = torch.zeros(1, *shape, dtype=torch.float32, device=self.device)
        return sample

    def __getattr__(self, name):
        # Delegate any attribute not defined here to the source DexNexDataset
        # (e.g. get_ep_lengths, get_qvals, get_key)
        return getattr(self._source, name)


class GPUDatasetView(torch.utils.data.Dataset):
    """
    A zero-copy view into a GPUCachedDataset restricted to a subset of indices.
    Multiple views can share the same underlying GPU tensors.
    """
    def __init__(self, cache: GPUCachedDataset, indices: np.ndarray):
        self._cache = cache
        self._indices = indices  # int array into cache

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._cache[int(self._indices[idx])]

    def get_ep_lengths(self):
        return self._cache.get_ep_lengths()[self._indices]

    def get_qvals(self):
        return self._cache.get_qvals()[self._indices]

    def __getattr__(self, name):
        return getattr(self._cache, name)


def _nested_nbytes(data):
    if isinstance(data, dict):
        return sum(_nested_nbytes(v) for v in data.values())
    return data.element_size() * data.nelement()


def _nested_to_device(data, device):
    if isinstance(data, dict):
        return {k: _nested_to_device(v, device) for k, v in data.items()}
    return data.to(device)


class GPUSampleCache(torch.utils.data.Dataset):
    """
    Lazily caches each sample on GPU the first time `dataset[idx]` is
    accessed -- decode + host-to-device transfer happens once per idx, every
    later access to that same idx returns the already-resident GPU tensors.

    Unlike GPUCachedDataset (which preloads everything up front and requires
    the whole dataset to fit in GPU memory), this fills in lazily and can be
    bounded to a byte budget smaller than the full dataset.

    evict=True bounds memory via LRU eviction against max_bytes.

    evict=False never evicts an already-cached sample. Rationale: training
    reshuffles every epoch over a fixed episode pool, so an evicted sample
    just gets reloaded and re-cached again later at the same cost -- LRU
    eviction under a tight budget can end up paying the CPU/decompression
    cost repeatedly for the same samples instead of once. With evict=False,
    once max_bytes worth of samples are cached, NEW (never-before-seen)
    samples are simply served uncached rather than evicting anything --
    already-cached samples keep being served from GPU for free, growth just
    stops. If max_bytes is None and evict=False, the cache is unbounded and
    will grow to the size of every distinct sample ever accessed.

    Must be used with num_workers=0 -- CUDA tensors can't cross a forked
    worker process, and the whole point is a single cache shared across
    every access from the main process.
    """
    def __init__(self, dataset, device, max_bytes=None, evict=True):
        self._source = dataset
        self.device = device
        self.max_bytes = max_bytes
        self.evict = evict
        self._cache = OrderedDict()  # idx -> nested dict of GPU tensors, ordered by last access
        self._cache_bytes = 0

        if not evict and max_bytes is None:
            print("GPUSampleCache: evict=False with no max_bytes set -- cache will grow "
                  "unbounded (up to the full dataset's GPU footprint) as new samples are seen.")

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx: int):
        cached = self._cache.get(idx)
        if cached is not None:
            if self.evict:
                self._cache.move_to_end(idx)
            return cached

        sample_gpu = _nested_to_device(self._source[idx], self.device)
        nbytes = _nested_nbytes(sample_gpu)

        if self.max_bytes is not None and self._cache_bytes + nbytes > self.max_bytes:
            if self.evict:
                while self._cache and self._cache_bytes + nbytes > self.max_bytes:
                    _, oldest = self._cache.popitem(last=False)
                    self._cache_bytes -= _nested_nbytes(oldest)
            else:
                # budget's full and we're not allowed to evict -- serve this
                # sample without memoizing it, leave the existing cache intact
                return sample_gpu

        self._cache[idx] = sample_gpu
        self._cache_bytes += nbytes
        return sample_gpu

    def __getattr__(self, name):
        # Delegate any attribute not defined here to the source dataset
        # (e.g. get_ep_lengths, get_qvals, get_key)
        return getattr(self._source, name)


class TrainAndVal:
    """
    Wrapper for a dataset sampler. Computes val and train masks and uses those to create a torch dataloader (along with a handle to a replay buffer) 

    Dataset to provide (s, a, r, s') samples to a torch dataloader. Formerly named dexnex_2cams_image_ql_dataset.py:DexNexDataset but that name isn't descriptive.

    note: s and s' are actually observations.

    zarr dataset keys:
    - img
    - img2
    - state
    """
    def __init__(self,
            sampler, # default sampler, not init'd
            options: dict,
            seed=42,
            val_ratio=0.0,
            split_val_by_episode=False, # the split is ALWAYS whole-episode now (DatasetSampler.get_sample_mask returns a per-episode mask; there's no per-timestep sub-selection support below the episode level). If True, val episodes are chosen so their combined timestep count approximates val_ratio of the total (weighted by episode length); if False (default), val_ratio instead selects that fraction of the EPISODE COUNT directly (simpler, but the resulting timestep fraction can drift from val_ratio if episode lengths vary a lot). See DatasetSampler.get_sample_mask.
            max_train_episodes=None,
            whether_to_use = True,
            use_weighted_dataloader = False,
            use_val_set = True, # if false, train_sampler == sampler, will save time during RL
            preload_to_gpu = False, # load entire dataset into GPU memory before training
            preload_vectorized = False, # use _vectorized_load_dataset instead of per-sample loading; falls back automatically if unsupported
            use_gpu_sample_cache = False, # lazily cache each sample on GPU on first access (GPUSampleCache); mutually exclusive with preload_to_gpu
            gpu_sample_cache_max_bytes = None, # byte budget for the above; None = unbounded
            gpu_sample_cache_evict = True, # LRU-evict to stay under the budget; False = never evict, just stop caching new samples once full
            ):
        assert(isinstance(sampler, sarsa_sampler.DatasetSampler))
        assert not (preload_to_gpu and use_gpu_sample_cache), \
            "preload_to_gpu and use_gpu_sample_cache are mutually exclusive -- both make the dataset serve samples off GPU-resident tensors"

        self.sampler = sampler # should be the entire dataset
        self.rb_id = sampler.rb_id
        self.options = options
        self.val_ratio = val_ratio
        self.split_val_by_episode = split_val_by_episode
        self.seed = seed
        self.max_train_episodes = max_train_episodes
        self.whether_to_use = whether_to_use
        self.use_weighted_dataloader = use_weighted_dataloader
        self.use_val_set = use_val_set
        self.preload_to_gpu = preload_to_gpu
        self.preload_vectorized = preload_vectorized
        self.use_gpu_sample_cache = use_gpu_sample_cache
        self.gpu_sample_cache_max_bytes = gpu_sample_cache_max_bytes
        self.gpu_sample_cache_evict = gpu_sample_cache_evict

        self.is_setup = False
        
    def setup(self):
        self.sampler.setup()

        # only init if we're being trained off of
        # tasks_to_use = globals.CONFIG.tasks_to_use #type:ignore
        # if self.rb_id in tasks_to_use:
        
        self.init()
            
        # init the sampler regardless?
        # self.sampler.InitAll()
        
        self.is_setup = True
            
    # def compute_weights(self, dataset: DexNexDataset):
    #     """
    #     Use critic to compute weights 
    #     """
    #     critic: CriticLoss = globals.MODELS["critic"] #type:ignore
    #     qvals = []
        
    #     a = dataset.get_key('action')
    #     s = dataset.get_key('state')
        
    #     # add history dim
    #     s2 = np.expand_dims(s, axis=1)
        
    #     b = {'obs':
    #             {'state': s2
    #             }
    #         }
        
    #     # note: this won't work if padding is not zero...
    #     # for batch in samples:
    #         #
    #         # a = batch['action']
            
    #         # # add batch dim
    #         # batch2 = utils.add_batch_dim(batch)
    #         # a2 = torch.unsqueeze(a, dim=0)
            
    #     nbatch, na = utils.norm_gpu_critic(b, a)
                    
    #     # get the qval
    #     qvals = critic.infer(nbatch, na, self.rb_id)
            
    #         #
    #         # qvals.append(qval)
            
    #     # range: [-inf, 1], but more likely [-100, 1]
    #     # qvals = torch.tensor(qvals)
        
    #     # map to range [0, 1]
    #     sqvals = nn.Tanh()(qvals) / 2.0 + 0.5
    #     sqvals2 = torch.squeeze(sqvals)
            
    #     weights = sqvals2.cpu().numpy()
        
    #     return weights
    
    
    def compute_weights(self, dataset: DexNexDataset):
        """
        Use qvals to compute weights
        """
        qvals = dataset.get_qvals()
        
        # map to range [0, 1]
        sqvals = np.tanh(qvals) / 2.0 + 0.5 #type:ignore
        # sqvals2 = torch.squeeze(sqvals)
            
        # weights = sqvals2.cpu().numpy()
        weights = sqvals
        
        return weights
    
    
    def compute_weights2(self, dataset: DexNexDataset):
        """
        Use ep lens to compute weights. Use exponential decay based off episode length
        """
        # get episode lengths
        ep_lens = dataset.get_ep_lengths()
        
        # max reward at the min ep length (fastest time-to-completion)
        def max_reward():
            return ep_lens.min()
        
        # mean at the max reward
        mu = max_reward()
        
        # std dev from the data
        std = np.std(ep_lens)
        
        # scale the std dev by some value
        std2 = std * 2.0
        
        # calc how much each value is LESS than mu. x should now be [-inf, 0]
        x = mu - ep_lens
        
        # find the probability, range [0, 0.5] (x=0 is 50% prob)
        probs = scipy.stats.norm.cdf(x, 0, std2)
        
        # scale range to [0, 1.0]
        weights = probs * 2.0
        
        return weights
        
        
    def _resolve_device(self):
        return torch.device(globals.CONFIG.device if globals.CONFIG is not None and hasattr(globals.CONFIG, 'device') else 'cuda')  # type: ignore

    def _maybe_gpu_cache(self, base_dataset):
        """
        Wraps base_dataset in whichever GPU caching strategy is configured
        (at most one of the two -- enforced in __init__), or returns it
        unwrapped if neither is enabled.
        """
        if self.preload_to_gpu:
            device = self._resolve_device()
            return GPUCachedDataset(base_dataset, device, num_workers=self.options.train.num_workers, vectorized=self.preload_vectorized) #type:ignore
        elif self.use_gpu_sample_cache:
            device = self._resolve_device()
            return GPUSampleCache(base_dataset, device, max_bytes=self.gpu_sample_cache_max_bytes, evict=self.gpu_sample_cache_evict)
        else:
            return base_dataset

    def make_dataloader(self, dataset, cfg):
        if self.preload_to_gpu or self.use_gpu_sample_cache:
            # CUDA tensors can't be shared with forked worker processes, and
            # the cache only helps if every access goes through the same
            # cache instance in the main process. Workers are also pointless
            # once data is served straight off the GPU.
            cfg = dict(cfg)
            cfg['num_workers'] = 0
            cfg.pop('pin_memory', None)
            cfg.pop('persistent_workers', None)

        use = self.use_weighted_dataloader
        
        # if MODELS hasn't been initialized yet, just use the regular dataloader until it is
        # use = use and globals.MODELS is not None
        
        # only do if we're training from it
        
        
        if use:
            # get the weights
            weights = self.compute_weights2(dataset)
            
            assert(len(weights) == len(dataset))
            
            sampler = WeightedRandomSampler(list(weights), len(dataset))
            
            # make the sampler
            dataloader = torchDataLoader(
                dataset,
                sampler=sampler,
                **cfg
                )
            
        else:
            dataloader = torchDataLoader(
                dataset,
                **cfg
                )
        
        return dataloader
            
    def init_no_val(self):
        """
        same as init, but condensed for readability
        """
        # init the sampler
        self.sampler.InitAll()

        # save a ref, make the dexnex dataset
        self.train_sampler = self.sampler
        self.train_dataset = self._maybe_gpu_cache(DexNexDataset(self.train_sampler))

        # train config
        train_cfg = copy.deepcopy(self.options.common) # type: ignore
        OmegaConf.unsafe_merge(train_cfg, self.options.train) # type: ignore

        # torch dataloader
        self.train_dataloader = self.make_dataloader(self.train_dataset, train_cfg)
        
        # dict access
        self.dd = {}
        self.dd["all"] = self.train_dataloader
        self.dd["train"] = self.train_dataloader
        self.dd["val"] = None
        
        print(f"{self.rb_id}: train={len(self.train_dataloader)} batches ({len(self.train_dataset)} samples)  val=None")

    def reinit_no_val(self):
        """
        able to just add new episodes since train == all. Saves time.
        """
        with utils.profile_section("reinit_no_val.reinit_all"):
            self.sampler.reinit_all()

            if isinstance(self.train_dataset, (GPUCachedDataset, GPUSampleCache)):
                # both caches key/index off the underlying dataset, which is
                # being rebuilt here (new episodes) -- must be rebuilt from
                # scratch rather than reused, or cached entries would answer
                # with stale data for indices that now point at different samples
                self.train_dataset = self._maybe_gpu_cache(DexNexDataset(self.train_sampler))
            else:
                self.train_dataset.reinit_all()
        
        # train config
        train_cfg = copy.deepcopy(self.options.common) # type: ignore
        OmegaConf.unsafe_merge(train_cfg, self.options.train) # type: ignore
        
        # torch dataloader
        with utils.profile_section("reinit_no_val.make_dataloader"):
            self.train_dataloader = self.make_dataloader(self.train_dataset, train_cfg)
        
        # dict access
        self.dd = {}
        self.dd["all"] = self.train_dataloader
        self.dd["train"] = self.train_dataloader
        self.dd["val"] = None
        
        print("New length of train_dataloader: {} batches".format(len(self.train_dataloader)))
        
    def init(self):
        if not self.whether_to_use:
            return
        
        # get nb episodes
        nb_episodes = globals.REPLAY_BUFFER_LOADER[self.rb_id].n_episodes # type:ignore
        
        doing_bc = True
        if globals.CONFIG is not None:
            if "doing_bc" in globals.CONFIG:
                doing_bc = globals.CONFIG.doing_bc # type: ignore
        
        if nb_episodes == 0 and doing_bc:
            print("No episodes in replay buffer, did you forget to seed the online RL replay buffer? Aka copy/paste a good starting RB and rename it to: {}".format(self.rb_id))
            raise
        
        elif nb_episodes == 0 and not doing_bc:
            print("No episodes in replay buffer, but we're not doing BC. Be sure to set the training warmup so that rollouts are collected before training")
            return
        
        # no val?
        if not self.use_val_set:
            self.init_no_val()
            return
        
        # first, init the all sampler
        self.sampler.InitAll()

        # get the val mask -- sized to the CURRENTLY-INCLUDED episode count
        # (i.e. len(self.sampler.get_ep_list())), which only equals the raw
        # replay-buffer episode count when InitAll() didn't filter anything
        # out (e.g. via episode_filter_key/value). Init()/make_episodes()
        # index by RAW episode position (enumerate(replay_buffer.episode_ends)),
        # so this must be scattered back into raw-episode-index space before
        # being passed to Init() -- otherwise a filtered sampler (fewer
        # included episodes than raw episodes) hits an out-of-bounds index
        # once make_episodes() walks past the included count.
        val_mask = self.sampler.get_sample_mask(
            ratio = self.val_ratio,
            seed = self.seed,
            split_by_episode = self.split_val_by_episode,
        )

        # self.sampler.ep_mask is the raw-length inclusion mask InitAll() just
        # built (all-True if unfiltered); its True positions are exactly the
        # raw episode indices get_ep_list()/get_sample_mask() enumerate over,
        # in the same (ascending) order.
        raw_ep_mask = self.sampler.ep_mask
        included_raw_indices = np.where(raw_ep_mask)[0]
        assert len(included_raw_indices) == len(val_mask)

        def _expand_to_raw(mask_over_included):
            raw = np.zeros_like(raw_ep_mask, dtype=bool)
            raw[included_raw_indices] = mask_over_included
            return raw

        val_mask = _expand_to_raw(val_mask)

        # get the train mask -- excluded (filtered-out) episodes must stay
        # excluded for train too, not fall back into it just because they're
        # not in val_mask
        train_mask = raw_ep_mask & ~val_mask

        # make train sampler with train mask
        self.train_sampler = self.sampler.copy()
        self.train_sampler.Init(train_mask)

        # make an exact copy
        self.val_sampler = copy.deepcopy(self.sampler)
        self.val_sampler.Init(val_mask)
        
        # make the datasets
        if self.preload_to_gpu:
            # Load the full dataset once, then create zero-copy views for train/val.
            device = self._resolve_device()
            self.all_dataset = GPUCachedDataset(DexNexDataset(self.sampler), device, num_workers=self.options.train.num_workers, vectorized=self.preload_vectorized) #type:ignore

            # GPUDatasetView indexes into all_dataset's FLAT sample space
            # (len(self.sampler), e.g. every valid timestep across every
            # included episode) -- NOT episode-index space. train_mask/
            # val_mask are raw-episode-index-space (length == total raw
            # episode count, one bool per episode, see _expand_to_raw
            # above), so `np.where(train_mask)[0]` previously handed
            # GPUDatasetView a handful of small EPISODE positions (e.g.
            # 0-700) instead of the thousands of sample indices those
            # episodes actually span -- confirmed directly: train/val
            # "sample counts" exactly matched episode counts, and val
            # samples' ep_id never matched val_sampler's own episode list
            # (the tiny integer "indices" landed on essentially arbitrary
            # early flat-index samples unrelated to the intended held-out
            # episodes). Convert to per-episode boundaries in self.sampler's
            # own iteration order (ascending raw index among included
            # episodes -- the same order DexNexDataset(self.sampler) uses to
            # map a flat index to an episode) and expand each selected
            # episode's boolean flag into its full run of flat sample
            # indices instead.
            ep_lengths = np.array(self.sampler.get_ep_lengths())
            ep_offsets = np.concatenate([[0], np.cumsum(ep_lengths)])[:-1]
            train_ep_sel = train_mask[included_raw_indices]
            val_ep_sel = val_mask[included_raw_indices]

            def _flat_sample_indices(ep_sel):
                chosen = np.where(ep_sel)[0]
                if len(chosen) == 0:
                    return np.array([], dtype=int)
                return np.concatenate([
                    np.arange(ep_offsets[i], ep_offsets[i] + ep_lengths[i]) for i in chosen
                ])

            train_indices = _flat_sample_indices(train_ep_sel)
            val_indices = _flat_sample_indices(val_ep_sel)
            self.train_dataset = GPUDatasetView(self.all_dataset, train_indices)
            self.val_dataset = GPUDatasetView(self.all_dataset, val_indices)
        else:
            # train/val samplers index disjoint sample sets, so each gets its
            # own cache instance rather than sharing one like the preload_to_gpu
            # zero-copy-view trick above (no overlap to de-duplicate)
            self.train_dataset = self._maybe_gpu_cache(DexNexDataset(self.train_sampler))
            self.val_dataset = self._maybe_gpu_cache(DexNexDataset(self.val_sampler))
            self.all_dataset = self._maybe_gpu_cache(DexNexDataset(self.sampler))
        
        # make the train & val config
        train_cfg = copy.deepcopy(self.options.common) # type: ignore
        val_cfg = copy.deepcopy(self.options.common) # type: ignore
        OmegaConf.unsafe_merge(train_cfg, self.options.train) # type: ignore
        OmegaConf.unsafe_merge(val_cfg, self.options.val) # type: ignore
        
        # make the train & val dataloader
        self.all_dataloader = self.make_dataloader(self.all_dataset, train_cfg)
        self.train_dataloader = self.make_dataloader(self.train_dataset, train_cfg)
        
        if len(self.val_dataset) > 0:
            self.val_dataloader = self.make_dataloader(self.val_dataset, val_cfg)
        
        # dict access
        self.dd = {}
        self.dd["all"] = self.all_dataloader
        self.dd["train"] = self.train_dataloader
        
        if len(self.val_dataset) > 0:
            self.dd["val"] = self.val_dataloader
        else:
            print("Warning: len(self.val_dataset) == 0 for rb_id: {}".format(self.rb_id))
            self.dd["val"] = None
        
        val_len = len(self.val_dataloader) if self.dd["val"] is not None else 0
        print(f"{self.rb_id}: all={len(self.all_dataloader)} batches ({len(self.all_dataset)} samples)  "
              f"train={len(self.train_dataloader)} batches ({len(self.train_dataset)} samples)  "
              f"val={val_len} batches ({len(self.val_dataset)} samples)")
        
    # def reinit(self):
    #     # only init if we're being trained off of
    #     tasks_to_use = globals.CONFIG.tasks_to_use #type:ignore
    #     if self.rb_id in tasks_to_use:
    #         self.init()
    
    def reinit(self):
        """
        if we're not using a val set, then train == all, so we can just add new episodes to save a lot of re-indexing time. If not, we have to do a full init because the training/val masks will not be valid
        """
        if self.use_val_set:
            self.init()
        else:
            self.reinit_no_val()
        
        
    def __getitem__(self, key):
        return self.dd[key]
    

class DataLoaders:
    def __init__(self,
                 d: dict[int, TrainAndVal]
                 ) -> None:
        self.d = d
        
    def setup(self):
        for key, d in self.d.items():
            d.setup()
        
    def __getitem__(self, key):
        return self.d[key]
        
    def reinit(self):
        for key, d in self.d.items():
            d.reinit()
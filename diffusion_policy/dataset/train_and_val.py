import copy
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


class GPUCachedDataset(torch.utils.data.Dataset):
    """
    Loads an entire DexNexDataset into GPU memory up front.
    __getitem__ then returns directly from the pre-loaded tensors with no
    zarr / numpy overhead per step.
    """
    def __init__(self, dataset: 'DexNexDataset', device: torch.device):
        self._source = dataset  # kept for attribute delegation
        self._len = len(dataset)
        if self._len == 0:
            self._data = None
            return
        print(f"Preloading {self._len} samples to {device} ...")
        samples = [dataset[i] for i in tqdm(range(self._len), desc="gpu-preload")]
        self._data = _nested_stack_to_device(samples, device)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        if self._data is None:
            raise IndexError("GPUCachedDataset is empty")
        return _nested_index(self._data, idx)

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
            max_train_episodes=None,
            whether_to_use = True,
            use_weighted_dataloader = False,
            use_val_set = True, # if false, train_sampler == sampler, will save time during RL
            preload_to_gpu = False, # load entire dataset into GPU memory before training
            ):
        assert(isinstance(sampler, sarsa_sampler.DatasetSampler))

        self.sampler = sampler # should be the entire dataset
        self.rb_id = sampler.rb_id
        self.options = options
        self.val_ratio = val_ratio
        self.seed = seed
        self.max_train_episodes = max_train_episodes
        self.whether_to_use = whether_to_use
        self.use_weighted_dataloader = use_weighted_dataloader
        self.use_val_set = use_val_set
        self.preload_to_gpu = preload_to_gpu

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
        
        
    def make_dataloader(self, dataset, cfg):
        if self.preload_to_gpu:
            # CUDA tensors can't be shared with forked worker processes.
            # Workers are also pointless — data is already on GPU.
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
        base = DexNexDataset(self.train_sampler)
        if self.preload_to_gpu:
            device = torch.device(globals.CONFIG.device if globals.CONFIG is not None and hasattr(globals.CONFIG, 'device') else 'cuda')  # type: ignore
            self.train_dataset = GPUCachedDataset(base, device)
        else:
            self.train_dataset = base

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

            if isinstance(self.train_dataset, GPUCachedDataset):
                # GPU cache must be rebuilt from scratch when new episodes arrive
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

        # get the val mask
        val_mask = self.sampler.get_sample_mask(
            ratio = self.val_ratio,
            seed = self.seed
        )
        
        # get the train mask
        train_mask = ~val_mask

        # make train sampler with train mask
        self.train_sampler = self.sampler.copy()
        self.train_sampler.Init(train_mask)

        # make an exact copy
        self.val_sampler = copy.deepcopy(self.sampler)
        self.val_sampler.Init(val_mask)
        
        # make the datasets
        if self.preload_to_gpu:
            # Load the full dataset once, then create zero-copy views for train/val.
            device = torch.device(globals.CONFIG.device if globals.CONFIG is not None and hasattr(globals.CONFIG, 'device') else 'cuda')  # type: ignore
            self.all_dataset = GPUCachedDataset(DexNexDataset(self.sampler), device)
            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            self.train_dataset = GPUDatasetView(self.all_dataset, train_indices)
            self.val_dataset = GPUDatasetView(self.all_dataset, val_indices)
        else:
            self.train_dataset = DexNexDataset(self.train_sampler)
            self.val_dataset = DexNexDataset(self.val_sampler)
            self.all_dataset = DexNexDataset(self.sampler)
        
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
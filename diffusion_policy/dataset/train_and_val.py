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
from diffusion_policy import utils
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sarsa_sampler import (
    DatasetSampler, get_val_mask, downsample_mask, get_not_done)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
import diffusion_policy.globals as globals
from diffusion_policy.model.diffusion_ql.diffusion_ql_loss import CriticLoss

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
            sampler: DatasetSampler, # default sampler, not init'd
            options: dict,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            whether_to_use = True,
            use_weighted_dataloader = False,
            ):
        self.sampler = sampler
        self.rb_id = sampler.rb_id
        self.options = options
        self.val_ratio = val_ratio
        self.seed = seed
        self.max_train_episodes = max_train_episodes
        self.whether_to_use = whether_to_use
        self.use_weighted_dataloader = use_weighted_dataloader

        # only init if we're being trained off of
        tasks_to_use = globals.CONFIG.tasks_to_use #type:ignore
        if self.rb_id in tasks_to_use:
            self.init()
            
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
            
        
    def init(self):
        if not self.whether_to_use:
            return
        
        # get nb episodes
        nb_episodes = globals.REPLAY_BUFFER_LOADER[self.rb_id].n_episodes # type:ignore
        
        if nb_episodes == 0:
            raise

        # TODO: rewrite to use datapoints instead of episodes...
        val_mask = get_val_mask(
            n_episodes=nb_episodes, 
            val_ratio=self.val_ratio,
            seed=self.seed)
        train_mask = ~val_mask

        # downsamples if max_train_episodes is not None
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=self.max_train_episodes, 
            seed=self.seed)

        # make train sampler with train mask
        self.train_sampler = copy.deepcopy(self.sampler)
        self.train_sampler.Init(train_mask)

        # make an exact copy
        self.val_sampler = copy.deepcopy(self.sampler)
        self.val_sampler.Init(val_mask)

        # init the original sampler
        all = np.logical_or(val_mask, train_mask)
        self.sampler.Init(all)
        
        # make the datasets
        self.train_dataset = DexNexDataset(self.train_sampler)
        self.val_dataset = DexNexDataset(self.val_sampler)
        
        # make the train & val config
        train_cfg = copy.deepcopy(self.options.common) # type: ignore
        val_cfg = copy.deepcopy(self.options.common) # type: ignore
        OmegaConf.unsafe_merge(train_cfg, self.options.train) # type: ignore
        OmegaConf.unsafe_merge(val_cfg, self.options.val) # type: ignore
        
        # make the train & val dataloader
        self.train_dataloader = self.make_dataloader(self.train_dataset, train_cfg)
        self.val_dataloader = self.make_dataloader(self.val_dataset, val_cfg)
        
        # dict access
        self.dd = {}
        self.dd["train"] = self.train_dataloader
        self.dd["val"] = self.val_dataloader
        
        print(self.rb_id + ": len train dataset (nb batches): {}".format(len(self.train_dataloader)))
        
    def reinit(self):
        # only init if we're being trained off of
        tasks_to_use = globals.CONFIG.tasks_to_use #type:ignore
        if self.rb_id in tasks_to_use:
            self.init()
        
        
    def __getitem__(self, key):
        return self.dd[key]
    

class DataLoaders:
    def __init__(self,
                 d: dict[int, TrainAndVal]
                 ) -> None:
        self.d = d
        
    def __getitem__(self, key):
        return self.d[key]
        
    def reinit(self):
        for key, d in self.d.items():
            d.reinit()
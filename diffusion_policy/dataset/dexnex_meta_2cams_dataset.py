from typing import Dict
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from omegaconf import OmegaConf, open_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

import hydra
from collections import defaultdict

from diffusion_policy.config.config import CONFIG

"""
A dataset composed of several other datasets. To be used in co-training.
"""

# slice_dict_by_subkey_hash = {}
def slice_dict_by_subkey(dict, prefix):
    # if this method is too slow, use the hash table feature
    # if prefix in slice_dict_by_subkey_hash:
    #     return slice_dict_by_subkey_hash[prefix]
    
    out = {}
    for key in dict:
        if prefix in key:
            out[key] = dict[key]
            
    return out
    
def get_history_dict(dict, key):
    """ Just an semantic alias for slice_dict_by_subkey """
    return slice_dict_by_subkey(dict, key)

def get_dict_value_by_suffix(dict, key, suffix):
    suffix = "_" + str(suffix)
    return dict[key + suffix]

def get_history_value(dict, key, history_index):
    return get_dict_value_by_suffix(dict, key, history_index)

def get_obs_history_value(dict, history_index):
    return get_history_value(dict, 'obs', history_index)
    

class MetaDataset():
    def __init__(self, num_batches, cfg):
        cfg_dataset = cfg.task.dataset
        ratios = cfg_dataset.ratios
        zarr_paths = cfg_dataset.zarr_paths
        batch_size = cfg_dataset.batch_size
        num_workers = cfg_dataset.num_workers
        
        # params
        self.cfg = cfg
        self.num_batches = num_batches
        self.ratios = ratios
        self.zarr_paths = zarr_paths
        self.device = torch.device(cfg.training.device)
        
        self.timeout = 0 # seconds, I assume. Has to be zero if num_workers = 0
        
        # containers
        self.datasets = []
        self.dataloaders = []
        self.dataset_iterators = []
        self.dataset_batch_sizes = []
        self.normalizers = []
        
        # iter count
        self.count = 0
        
        # for each dataset
        for idx in range(len(zarr_paths)):
            zarr_path = zarr_paths[idx]
            ratio = ratios[idx]
            
            # calc batch size
            dataset_batch_size = round( ratio * batch_size )
            
            # add the zarr path
            with open_dict(cfg):
                cfg_dataset.dataset.zarr_path = zarr_path
                
                if num_workers[idx] > 0:
                    cfg.dataloader.persistent_workers = True
                else:
                    cfg.dataloader.persistent_workers = False
            
            # configure dataset
            dataset: BaseImageDataset
            dataset = hydra.utils.instantiate(cfg_dataset.dataset)
            assert isinstance(dataset, BaseImageDataset)
            dataloader = DataLoader(dataset,
                                    batch_size=dataset_batch_size,
                                    timeout=self.timeout,
                                    num_workers=num_workers[idx],
                                    **cfg.dataloader)
            
            # move normalizer to device
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(dataset.get_normalizer().state_dict())
            normalizer = normalizer.to(self.device)
            self.normalizers.append(normalizer)

            # append the dataloaders
            self.datasets.append(dataset)
            self.dataloaders.append(dataloader)
            self.dataset_batch_sizes.append(dataset_batch_size)
            
            # sanity
            print("dataset length: ", len(dataset))
            
        # TEST
        for idx, _ in enumerate(self):
            break
        pass
        

    def get_validation_dataset(self, num_val_batches):
        val_set = copy.copy(self)
        
        # overwrite
        val_set.num_batches = num_val_batches
        
        # reset lists, otherwise it'll overwrite them in self. Stupid python.
        val_set.datasets = []
        val_set.dataloaders = []
        
        for idx in range(len(self.datasets)):
            # dataset
            val_set.datasets.append(self.datasets[idx].get_validation_dataset())
            
            # dataloader
            val_set.dataloaders.append(
                DataLoader(
                    val_set.datasets[idx], 
                    batch_size=self.dataset_batch_sizes[idx],
                    timeout=self.timeout,
                    **self.cfg.val_dataloader))
            
        return val_set

    # def get_normalizer(self, mode='limits', **kwargs):
    #     <>
        # need to think about this for a moment
        # need to use the same normalizer for ALL included datasets
        # option 1: combine all datasets and calc the overall normalizer
        # option 2: use the pre-existing calculations to determine the correct normalizer for all datasets
        # option 3: use each normalizer for each mini-batch separately. Pros: no extra work needed, can keep datasets totally separate. Generalizes to datasets with different embodiments, as in, I could train on open-x-emobodiment data and n.b.d. I don't 
        # option 4: can incorporate a task mask, so the input state is the sum of all task's input states. Essentially treating each task as a different embodiment
        
    def get_normalizers(self):
        return self.normalizers
        

    def __len__(self) -> int:
        return self.num_batches
    
    # implicitly called at the start of loops
    def __iter__(self):
        # reset count
        self.count = 0
        self.dataset_iterators = []
        
        # print("dataset count: {}".format(self.count))
        # print("dataset len: {}".format(len(self)))
        
        # iterate 
        for dataloader in self.dataloaders:
            # force a reshuffle
            self.dataset_iterators.append(iter(dataloader))
            
        return self
    
    def normalize_key_and_history(self, normalizer, batch, key):
        # setup return var
        out = {}
        
        # get the appropriate normalizer
        n = normalizer[key]
        
        # get the dict of all data this normalizer pertains to
        d = get_history_dict(batch, key)
        
        # iterate over appropriate data points
        for key2, val2 in enumerate(d):
            out[key2] = n(val2)
            
        # we're done
        return out
    
    def get_single_field_normalizer_by_key(self, normalizer, key):
        # kinda hacky, so it goes. At least the hack is contained within this method instead of spread out
        if key in normalizer:
            return normalizer[key]
        
        elif key == "obs":
            obs_n = normalizer
            
    def normalize_obs_impl(self, normalizer, obs_dict):
        """ This method uses the obs dictionary itself """
        return normalizer.normalize(obs_dict)
        
    def normalize_obs(self, normalizer, batch):
        """ This method assumes the input batch has an 'obs' key """
        nobs = self.normalize_obs_impl(normalizer, batch['obs'])
        return nobs
        
    def normalize_action(self, normalizer, batch):
        if 'action' in batch:
            naction = normalizer['action'].normalize(batch['action'])
        else:
            naction = None
        return naction
    
    def normalize_batch(self, batch, idx):
        """
        batch - a dictionary on the GPU
        idx - which dataset to use
        """
        # get normalizer for this dataset
        normalizer = self.normalizers[idx]
        
        # observations
        nobs = self.normalize_obs(normalizer, batch)
        
        # actions
        naction = self.normalize_action(normalizer, batch)
        
        # put into data dict
        ndata = {
            "nobs": nobs,
            "naction": naction
        }
        
        # next state
        obs = get_dict_value_by_suffix(batch, "obs", "next")
        nnobs = self.normalize_obs_impl(normalizer, obs)
        ndata["nobs_next"] = nnobs
        
        # history of states
        for val in CONFIG.history_indices:
            obs = get_obs_history_value(batch, val)
            hnobs = self.normalize_obs_impl(normalizer, obs)
            ndata["nobs_" + str(val)] = hnobs
            
        # we're done
        return nobs, naction
    
    def batch_to_gpu(self, batch):
        batch_gpu = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
        return batch_gpu
        
    # def normalize_gpu(self, batch_gpu, idx):
    #     # convert from numpy to torch, add a batch dimension, and transfer to the GPU
    #     # batch_gpu = dict_apply(batch, 
    #     #         lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
        
    #     # normalize the batch
    #     ndata = self.normalize_batch(batch_gpu, idx)
        
    #     return ndata
        
        
    def unnormalize(self, nresult):
        # extract normalized actions
        naction = nresult['naction'] # [B, T, ...]
        naction_pred = nresult['naction_pred'] #[B, T, ...]
        
        action = torch.zeros(naction.shape, device=naction.device)
        action_pred = torch.zeros(naction_pred.shape, device=naction_pred.device)
        
        i0 = 0
        for idx in range(len(self.dataset_iterators)):
            normalizer = self.normalizers[idx]
            batch_size = self.dataset_batch_sizes[idx]
                
            # extract the correct samples in the batch for this dataset
            this_naction = naction[i0:i0+batch_size]
            this_naction_pred = naction_pred[i0:i0+batch_size]
            
            # unnormalize
            this_action = normalizer['action'].unnormalize(this_naction)
            this_action_pred = normalizer['action'].unnormalize(this_naction_pred)
            
            # save into the result
            action[i0:i0+batch_size] = this_action
            action_pred[i0:i0+batch_size] = this_action_pred
            
            i0 += batch_size
        
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result
    
    # def unnormalize(self, nresult, normalizer):
    #     # TODO: when you want to unnormalize using the passed in normalizer
    #     pass

    def next_batch(self, idx):

        # dataset specific vars
        iterator = self.dataset_iterators[idx]
        normalizer = self.normalizers[idx]
        dataloader = self.dataloaders[idx]
        dataset = self.datasets[idx]
        
        try:
            batch = next(iterator) # output: dict
        # except can occur from a timeout or a StopIteration
        # I'm still not sure why the timeout's are occuring, but this will get around it
        except Exception as e:
            print("next(iterator) except: ")
            print(e)
            print("len(dataset): ", len(dataset))
            
            # reshuffle this iterator
            self.dataset_iterators[idx] = iter(dataloader)
            iterator = self.dataset_iterators[idx]
            
            # get a new batch
            batch = next(iterator) # output: dict
        
        # put on gpu
        batch_gpu = self.batch_to_gpu(batch)
        
        # normalizer = dataset.get_normalizer()
        ndata = self.normalize_batch(batch_gpu, idx)
        
        # we're done
        return ndata

    def concat_datas(self, ndatas):
        """
        Concatenate all data with the same key. Right now doesn't work with nested dicts...

        ndatas - list of dicts of data on GPU
        """
        tensors = defaultdict(list)
        out_d = {}

        # flatten the data
        for ndata in ndatas:
            for key, val in enumerate(ndata):
                # if it's a nested dict
                if isinstance(val, dict):
                    pass
                    
                else:
                    tensors[key].append(val)
                    
        # concat the data
        for key, val in enumerate(tensors):
            out_d[key] = torch.cat(val)

        # we're done
        return out_d
        
    
    # called each for-loop
    def __next__(self):
        """
        This method will assemble a single batch consisting of data from each dataset, with the ratio defined in the config yaml.
        """
        
        if self.count < self.num_batches:
            # increment our count
            self.count += 1
            
            # containers
            tensors = defaultdict(list)
            ndatas = []
            
            # process each batch
            for idx in range(len(self.dataset_iterators)):
                ndata = self.next_batch(idx)

                # add to list
                ndatas.append(ndata)

            # concatenate all data with the same key
            out_d = self.concat_datas(ndatas)
                
            # return the data
            return out_d
        else:    
            # we've done num_batches
            # print("Iteration done. Count: {}".format(self.count))
            raise StopIteration









import tqdm
import torch
import numpy as np
import zarr
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common import pytorch_util
import diffusion_policy.globals as globals
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

from diffusion_policy.common.replay_buffer import ReplayBuffer

from diffusion_policy.model.common import normalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
from diffusion_policy.common.normalize_util import get_identity_normalizer_from_stat
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.train_and_val import TrainAndVal

from diffusion_policy.samplers.mage_hand_sampler import MageHandEpisodeSampler, MageHandDatasetSampler

""" 
joint limits for the normalizer. Only thing that matters is the scale and offset. stat's aren't used in normalization

in normalizer.py, if forward: x = x * scale + offset
can use normalize_util.get_range_normalizer_from_stat to do the math for us, just need to set stat['max'] and stat['min']

s: 0.00: mean 1.46 std 0.13 min 1.16 max 1.93
s: 1.00: mean 1.29 std 0.32 min 0.39 max 2.27
s: 2.00: mean -0.91 std 0.58 min -3.01 max 0.52
s: 3.00: mean -0.01 std 0.57 min -2.00 max 2.00
s: 4.00: mean 0.25 std 0.44 min -1.38 max 1.72
s: 5.00: mean 2.07 std 0.55 min -0.11 max 2.50
s: 6.00: mean -0.16 std 0.21 min -0.49 max 0.15
s: 7.00: mean -0.12 std 0.26 min -0.70 max 0.49
s: 8.00: mean -0.02 std 0.20 min -0.35 max 0.35
s: 9.00: mean 0.24 std 0.49 min -0.26 max 1.55
s: 10.00: mean 1.40 std 0.37 min -0.00 max 1.57
s: 11.00: mean 0.19 std 0.36 min -0.00 max 1.57
s: 12.00: mean 0.15 std 0.17 min -0.35 max 0.35
s: 13.00: mean 0.24 std 0.48 min -0.26 max 1.54
s: 14.00: mean 1.31 std 0.42 min -0.00 max 1.57
s: 15.00: mean 0.18 std 0.38 min -0.00 max 1.57
s: 16.00: mean -0.24 std 0.25 min -1.04 max 0.91
s: 17.00: mean 1.13 std 0.10 min 0.65 max 1.22
s: 18.00: mean 0.02 std 0.19 min -0.21 max 0.21
s: 19.00: mean 0.68 std 0.04 min -0.17 max 0.70
s: 20.00: mean 0.60 std 0.34 min -0.26 max 1.56
s: 21.00: mean 0.39 std 0.49 min 0.00 max 1.00
s: 22.00: mean 0.22 std 0.41 min 0.00 max 1.00
for s 23
for s 24
s: 25.00: mean 0.38 std 0.49 min 0.00 max 1.00
s: 26.00: mean 0.35 std 0.12 min -0.03 max 0.68
s: 27.00: mean 1.42 std 0.06 min 1.18 max 1.56
s: 28.00: mean 0.20 std 0.13 min 0.06 max 0.71
s: 29.00: mean 0.37 std 0.13 min -0.10 max 0.73
s: 30.00: mean 1.47 std 0.06 min 1.23 max 1.67
s: 31.00: mean 0.21 std 0.15 min 0.06 max 0.75
s: 32.00: mean 0.35 std 0.13 min -0.13 max 0.71
s: 33.00: mean 1.49 std 0.07 min 1.22 max 1.70
s: 34.00: mean 0.21 std 0.15 min 0.06 max 0.73
s: 35.00: mean 0.36 std 0.13 min 0.09 max 0.79
s: 36.00: mean 1.45 std 0.07 min 1.18 max 1.66
s: 37.00: mean 0.13 std 0.15 min -12.44 max 0.50

"""

"""
action: 0.00: min 1.16 max 1.93
action: 1.00: min 0.38 max 2.56
action: 2.00: min -3.00 max 0.46
action: 3.00: min -2.00 max 2.00
action: 4.00: min -1.38 max 1.72
action: 5.00: min -0.12 max 2.50
action: 6.00: min -0.49 max 0.14
action: 7.00: min -0.70 max 0.49
action: 8.00: min -0.35 max 0.35
action: 9.00: min -0.26 max 1.57
action: 10.00: min 0.00 max 1.57
action: 11.00: min 0.00 max 1.57
action: 12.00: min -0.35 max 0.35
action: 13.00: min -0.26 max 1.57
action: 14.00: min 0.00 max 1.57
action: 15.00: min 0.00 max 1.57
action: 16.00: min -1.05 max 0.93
action: 17.00: min 0.65 max 1.22
action: 18.00: min -0.21 max 0.21
action: 19.00: min -0.08 max 0.70
action: 20.00: min -0.26 max 1.56
"""

# some values from my joint_limits.yaml, others from the urdf or 
# https://www.shadowrobot.com/wp-content/uploads/2022/03/shadow_dexterous_hand_e_technical_specification.pdf
JOINT_LIMITS = np.array([ 
    [0.52, 2.5], # gofa
    [-1.0, 2.6], # gofa
    [-3.0, 1.1], # gofa
    [-2.0, 2.0], # gofa
    [-2.0, 2.0], # gofa
    [-2.5, 2.5], # gofa
    [-0.49, 0.150], # lh_WRJ2
    [-0.7, 0.5], # lh_WRJ1
    [-0.35, 0.35], # "lh_FFJ4",
    [-0.262, 1.571], # "lh_FFJ3",
    [0, 1.57], # "lh_FFJ2",
    [0, 1.57], # "lh_FFJ1",
    [-0.35, 0.35], # "lh_MFJ4",
    [-0.262, 1.571], # "lh_MFJ3",
    [0, 1.57], # "lh_MFJ2",
    [0, 1.57], # "lh_MFJ1",
    [-1.05, 1.04], # "lh_THJ5",
    [0, 1.22], # "lh_THJ4",
    [-0.21, 0.21], # "lh_THJ3",
    [-0.7, 0.7], # "lh_THJ2", # urdf is slightly wider than tech spec. Go with the wider one to be safe
    [-0.262, 1.571], # "lh_THJ1",
], dtype='float32')

HAPTICS = np.array(5 * [[0.0, 1.0]], dtype='float') # haptics were already normalized from 0 to 1

# values taken from the normalizer fit function, then rounded to the nearest meter
# mins = [0.2193,  1.1434,  0.1246,  0.2273,  1.1635,  0.1175, 0.2029,  1.1771,  0.1151]
# maxs = [0.5204, 1.3973, 0.4805, 0.5474, 1.4750, 0.5378, 0.5411, 1.4861, 0.5283]
# FINGERTIP_POS = np.stack((np.floor(mins), np.ceil(maxs)), axis=1)

"""
From validate_rewards for fingertip pos's
range: 0.10214976966381073, 0.6771937012672424
range: 1.1958625316619873, 1.5864757299423218
range: 0.060191262513399124, 0.5472645163536072
range: 0.0736096128821373, 0.7232524156570435
range: 1.251955509185791, 1.6685783863067627
range: 0.05885033309459686, 0.598027229309082
range: 0.04745176061987877, 0.7142849564552307
range: 1.2671566009521484, 1.6714200973510742
range: 0.05856914445757866, 0.5928069353103638

take 2:
s: 26.00: mean 0.35 std 0.12 min -0.03 max 0.68
s: 27.00: mean 1.42 std 0.06 min 1.18 max 1.56
s: 28.00: mean 0.20 std 0.13 min 0.06 max 0.71
s: 29.00: mean 0.37 std 0.13 min -0.10 max 0.73
s: 30.00: mean 1.47 std 0.06 min 1.23 max 1.67
s: 31.00: mean 0.21 std 0.15 min 0.06 max 0.75
s: 32.00: mean 0.35 std 0.13 min -0.13 max 0.71
s: 33.00: mean 1.49 std 0.07 min 1.22 max 1.70
s: 34.00: mean 0.21 std 0.15 min 0.06 max 0.73


"""
FINGERTIP_POS = np.array([
    [-1.0, 1.0], # th-x
    [0.0, 2.0], # th-y
    [0, 1.0], # th-z
    [-1.0, 1.0], # ff-x
    [0.0, 2.0], # ff-y
    [0.0, 1.0], # ff-z
    [-1.0, 1.0], # mf-x
    [0.0, 2.0], # mf-y
    [0.0, 1.0], # mf-z
], dtype='float32')

"""
block xyz limits
range: 0.123125821352005, 0.7750506401062012
range: 1.24249267578125, 1.6347488164901733
range: 0.05169874057173729, 0.4972562789916992
"""
BLOCK = np.array([
    [-1.0, 1.0], # x
    [0.0, 2.0], # y
    [-0.05, 1.0], # z
], dtype='float32')

LIMITS = np.concatenate((JOINT_LIMITS, HAPTICS, FINGERTIP_POS, BLOCK), axis=0, dtype='float32')


class DataArray:
    """
    A class for handling a single data array and performing desired operations on it, e.g. normalize, unnormalize
    """
    def __init__(self, 
                 normalizer: SingleFieldLinearNormalizer,
                 descriptor = "",
                 strict: bool = True,
                 clamp = True, # UNUSED whether to clamp the normalized values to [-1, 1]
                 ):
        self.normalizer = normalizer
        self.descriptor = descriptor
        self.strict = strict
        self.clamp = globals.CONFIG.clamp # type:ignore
        
        # transfer to device, since I own normalizer
        # if hasattr(globals.CONFIG, "device"):
        # assert(hasattr(globals.CONFIG, "device"))

        device_type: str = globals.CONFIG.device # type: ignore
        device = torch.device(device_type) 
        self.normalizer.to(device)

        self.reset()

    def reset(self):
        self.datapoint = None

    def set(self, dp: torch.Tensor):
        self.datapoint = dp

    def get(self):
        if self.strict and self.datapoint is None:
            print("Forgot to set the datapoint")
            raise

        return self.datapoint

    def normalize(self):
        if self.strict and self.datapoint is None:
            print("Forgot to set the datapoint")
            raise
        
        if self.datapoint is None:
            return

        self.datapoint = self.normalizer.normalize(self.datapoint)
        
        if self.clamp:
            self.datapoint = torch.clamp(self.datapoint, -1.0, 1.0)
            
            pass
            
            
    
    def unnormalize(self):
        if self.strict and self.datapoint is None:
            print("Forgot to set the datapoint")
            raise
        
        if self.datapoint is None:
            return
        
        # must clamp the normalized datapoint first
        if self.clamp:
            self.datapoint = torch.clamp(self.datapoint, -1.0, 1.0)
            
            pass

        # and then unnormalize
        self.datapoint = self.normalizer.unnormalize(self.datapoint)
    
class NestedDataArray:
    """
    A class for holding a nested data structure of DataArray's. Each branch = NestedDataArray, each leaf = DataArray
    """
    def __init__(self,
                 descriptor = "",
                 strict = True
                 ):
        self.nest = {}
        self.descriptor = descriptor
        self.strict = strict

    def reset(self):
        for key, val in self.nest.items():
            val.reset()

    def set(self, data):
        """
        Data - a nested dict of torch.Tensors
        """
        # NOTE: should be called after set_normalizers
        for key, val in data.items():
        #     if isinstance(val, dict):
        #         self.nest[key] = NestedDataArray()
                
        #     else:
        #         self.nest[key] = DataArray()
            
            # works whether val is a dict or a torch.Tensor since the syntax is the same
            self.nest[key].set(val)

    def get(self):
        """
        extract a nested dict of torch.Tensor's
        """
        out = {}

        # works whether val is a NestedDataArray or a DataArray since the syntax is the same
        for key, val in self.nest.items():
            dp = val.get()
            
            # skip if none or if it's an empty dict
            if dp is not None and not (isinstance(dp, dict) and len(dp) == 0):
                out[key] = val.get()

        return out

    def set_normalizers(self, nested_normalizers):
        # reset nest
        self.nest = {}

        for key, val in nested_normalizers.items():
            # leaf
            if isinstance(val, SingleFieldLinearNormalizer):
                da = DataArray(val, descriptor=key, strict=self.strict, clamp=val.clamp)
                self.nest[key] = da

            # another branch
            else:
                nda = NestedDataArray(descriptor=key, strict=self.strict)
                nda.set_normalizers(val)
                self.nest[key] = nda

    # no grads wanted
    @torch.no_grad()
    def normalize(self):
        # works whether val is a NestedDataArray or a DataArray since the syntax is the same
        val: NestedDataArray | DataArray
        for key, val in self.nest.items():
            val.normalize()

    # no grads wanted
    @torch.no_grad()
    def unnormalize(self):
        # works whether val is a NestedDataArray or a DataArray since the syntax is the same
        for key, val in self.nest.items():
            val.unnormalize()

class Batch:
    """
    Maintain a batch & associated state info
    """
    def __init__(self, idx=None, batch=None):
        self.idx = idx
        self.batch = batch

    def Set(self, batch, idx):
        self.batch = batch
        self.idx = idx

class BatchLoader:
    """
    Responsible for handling requests for batchs of trainable data.
    Uses the torch dataloader to handle data shuffling and querying the underlying sampler.
    Uses NestedDataArray to handle normalizing / unnormalizing.

    I've put the normalizer here because we only normalize after getting a full batch (on GPU btw), which happens one level above the data-loader, and the dataset sits below the data-loader
    """
    def __init__(self,
                 rb_id: str,
                 train_or_val: str,
                 use_dataloader: bool = True,
                 strict: bool = True
            ):
        self.rb_id = rb_id
        self.train_or_val = train_or_val
        self.use_dataloader = use_dataloader
        self.strict = strict
        
        self.dataloaders = None #type:ignore
        
        # only continue if we're being trained off of or special rb_id of default
        tasks_to_use = globals.CONFIG.tasks_to_use #type:ignore
        if (self.rb_id not in tasks_to_use) and not self.rb_id == "default":
            return
        
        # if we actually want to use a data-loader. might not when we're doing inference but still need the task-ids
        if self.use_dataloader and not self.rb_id == "default":
            # get a handle to the dataloader "Node".
            self.dataloaders: TrainAndVal = globals.DATALOADERS[self.rb_id]
        
        self.nested_data_array = NestedDataArray("top-level", strict=self.strict)
        

        device_type: str = globals.CONFIG.device # type: ignore
        self.device = torch.device(device_type) 
        
        # will make the nested data array structure
        self.init_normalizers()
        
        # reset
        self.reset()
        
    def get_static_nns(self):
        obs = {}
        obs_keys_to_load = globals.CONFIG.obs_keys_to_load # type: ignore
        if "state" in obs_keys_to_load:
            obs['state'] = get_range_normalizer_from_stat(
                    {'min': LIMITS[:, 0], 'max': LIMITS[:, 1]})
            
        if "img" in obs_keys_to_load:
            obs['img'] = get_image_range_normalizer()
            
        if "img2" in obs_keys_to_load:
            obs['img2'] = get_image_range_normalizer()

        
        act = get_range_normalizer_from_stat(
                {'min': JOINT_LIMITS[:, 0], 'max': JOINT_LIMITS[:, 1]}
                )
        
        ## rb index
        rb_index = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                )
        rb_index.clamp = False
        
        ## task id
        task_id = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                )
        task_id.clamp = False
        
        ## qvals, clamp to [-1, 1]
        qval = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                )
        
        ## ep length
        ep_len = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                )
        ep_len.clamp = False
        
        nns = {
            'obs': obs,
            'obs_next': obs,
            'action': act,
            'action_next': act,
            'not_done': get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                ),
            'reward': get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                ),
            'rb_index': rb_index,
            'task_id': task_id,
            'qval': qval,
            'ep_len': ep_len,
        }
        
        return nns
    
    def get_fitted_nns(self):
        nns = self.get_static_nns()
        
        # use the 'all' zarr array data
        rb: ReplayBuffer = globals.REPLAY_BUFFER_LOADER['all'] #type:ignore
        s: zarr.Array = rb['state'] #type:ignore
        a: zarr.Array = rb['action'] #type:ignore
        
        # only fit the state, and action. No image
        nn: normalizer.SingleFieldLinearNormalizer = nns['obs']['state']
        # nn.fit(s, mode='gaussian')
        nn.fit(s, mode='limits')
        
        nn = nns['action']
        # nn.fit(a, mode='gaussian')
        nn.fit(a, mode='limits')
        
        # _next's
        if 'obs_next' in nns:
            nns['obs_next']['state'] = nns['obs']['state']
            
        if 'action_next' in nns:
            nns['action_next'] = nns['action']
        
        return nns
        
        

    def init_normalizers(self):
        """
        Structure, must be same as the shape_meta structure.

        """
        if False:
            nns = self.get_static_nns()
        else:
            nns = self.get_fitted_nns()
            
        
        self.nested_data_array.set_normalizers(nns)
        
    def reset(self):
        # reset my count
        self.count = 0

        # forces a reshuffle
        if self.use_dataloader and self.dataloaders is not None:
            dataloader = self.dataloaders[self.train_or_val]
            self.iterator = iter(dataloader)
            
            # print("New len iterator: {}".format(len(dataloader)))
        else:
            self.iterator = None

    # implicitly called at the start of loops
    def __iter__(self):
        """
        Required for tqdm (progress bar) compatibility
        """

        self.reset()

        return self
    
    def transfer_to_gpu(self, batch):
        batch_gpu = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

        return batch_gpu
    
    def operate_on_batch(self, batch, normalize=True):
        """
        batch - a torch-gpu nested dict 
        
        """
        # reset to clear any old data
        self.nested_data_array.reset()

        # update the nested data array data
        self.nested_data_array.set(batch)

        if normalize:
            # run the normalizer
            self.nested_data_array.normalize()
        else:
            self.nested_data_array.unnormalize()

        # extract the normalized data
        nbatch = self.nested_data_array.get()
        
        # we're done
        return nbatch
    
    def normalize_batch(self, batch):
        return self.operate_on_batch(batch, normalize=True)
    
    def unnormalize_batch(self, batch):
        return self.operate_on_batch(batch, normalize=False)
    
    def transfer_and_norm(self, batch):
        batch_gpu = self.transfer_to_gpu(batch)
        
        ndata = self.normalize_batch(batch_gpu)
        
        return ndata
    
    def unnorm_and_transfer(self, nbatch_gpu):
        data_gpu = self.unnormalize_batch(nbatch_gpu)
        
        data = pytorch_util.dict_tensor_to(data_gpu, 'cpu')
        
        return data
    
    def get_batch(self):
        """
        To be called when we're using a dataloader. i.e. during training
        """
        # None protection
        assert(self.iterator is not None)

        try:
            batch = next(self.iterator) # output: dict
        # except can occur from a timeout or a StopIteration
        # I'm still not sure why the timeout's are occuring, but this will get around it
        except Exception as e:
            # print("next(iterator) except: ")
            # print(e)
            
            # # reset
            # print("shuffle")
            self.reset()
            
            # get a new batch
            batch = next(self.iterator) # output: dict

        ndata = self.transfer_and_norm(batch)
        
        # we're done
        return ndata
    
    # called each for-loop
    def __next__(self):
        nbatch = self.get_batch()

        return nbatch
        
        # if self.count < globals.CONFIG.num_train_batches:
        #     # increment our count
        #     self.count += 1

        
        # else:
        #     # we've done num_batches
        #     # print("Iteration done. Count: {}".format(self.count))
        #     raise StopIteration
        
class ManipAnythingBatchLoader(BatchLoader):
    def get_static_nns(self):
        
        obs = {}
        obs_keys_to_load = globals.CONFIG.obs_keys_to_load # type: ignore
        if "state" in obs_keys_to_load:
            obs['state'] = get_range_normalizer_from_stat(
                    {'min': LIMITS[:, 0], 'max': LIMITS[:, 1]})
            
        if "object_target_pose" in obs_keys_to_load:
            obs['object_target_pose'] = get_identity_normalizer_from_stat(
                {'min': np.zeros(7, dtype=np.float32)}
                )
            
        if "object_id" in obs_keys_to_load:
            obs['object_id'] = get_identity_normalizer_from_stat(
                {'min': np.zeros(1, dtype=np.float32)}
                )
            
        
        act = get_range_normalizer_from_stat(
                {'min': JOINT_LIMITS[:, 0], 'max': JOINT_LIMITS[:, 1]}
                )
        
        ## rb index
        rb_index = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                )
        rb_index.clamp = False
        
        ## task id
        task_id = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                )
        task_id.clamp = False
        
        ## qvals, clamp to [-1, 1]
        qval = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                )
        
        ## ep length
        ep_len = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                )
        ep_len.clamp = False
        
        nns = {
            'obs': obs,
            'action': act,
            'not_done': get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                ),
            'reward': get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                ),
            'rb_index': rb_index,
            'task_id': task_id,
            'qval': qval,
            'ep_len': ep_len,
        }
        
        return nns
    
    
    def get_fitted_nns(self):
        """
        must be done after normalization, otherwise the stats won't be uniform
        """
        raise
        # nns = self.get_static_nns()
        
        # # use the 'all' zarr array data
        # rb: ReplayBuffer = globals.REPLAY_BUFFER_LOADER['all'] #type:ignore
        # s: zarr.Array = rb['state'] #type:ignore
        # a: zarr.Array = rb['action'] #type:ignore
        
        # # only fit the state, and action. No image
        # nn: normalizer.SingleFieldLinearNormalizer = nns['obs']['state']
        # # nn.fit(s, mode='gaussian')
        # nn.fit(s, mode='limits')
        
        # nn = nns['action']
        # # nn.fit(a, mode='gaussian')
        # nn.fit(a, mode='limits')
        
        # # _next's
        # if 'obs_next' in nns:
        #     nns['obs_next']['state'] = nns['obs']['state']
            
        # if 'action_next' in nns:
        #     nns['action_next'] = nns['action']
        
        return nns
    

class MageHandBatchLoader(BatchLoader):
    def get_static_nns(self):
        nns = {}
        obs = {}
        obs_keys_to_load = globals.CONFIG.obs_keys_to_load # type: ignore
        
        # default identity normalizer
        for obs_key in obs_keys_to_load:
            # get shape meta from the config
            nb = globals.CONFIG.shape_meta[obs_key].shape # type: ignore
            
            
            obs[obs_key] = get_identity_normalizer_from_stat(
                {'min': np.zeros(nb, dtype=np.float32)}
                )
            
        act_keys = ['joint_action', 'rel_pos_action', 'rel_quat_action']
        for act_key in act_keys:
            # get shape meta from the config
            nb = globals.CONFIG.shape_meta[act_key].shape # type: ignore
            
            nns[act_key] = get_identity_normalizer_from_stat(
                {'min': np.zeros(nb, dtype=np.float32)}
                )
        
        ## rb index
        rb_index = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                )
        rb_index.clamp = False
        
        ## task id
        task_id = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                )
        task_id.clamp = False
        
        ## qvals, clamp to [-1, 1]
        qval = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                )
        
        ## ep length
        ep_len = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                )
        ep_len.clamp = False
        
        nns = nns | {
            'obs': obs,
            'not_done': get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                ),
            'reward': get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
                ),
            'rb_index': rb_index,
            'task_id': task_id,
            'qval': qval,
            'ep_len': ep_len,
        }
        
        return nns
    
    def update_nns_from_saved_stats(self, nns):
        # yaw rel:
        # Fitted nn: rel_object_target_pos: min Parameter containing:
        # tensor([-0.9948, -0.9292, -0.3309]) max Parameter containing:
        # tensor([0.8686, 0.7937, 1.2904])
        min = [-0.99, -0.93, -0.33]
        max = [0.86, 0.79, 1.3]
        
        # make a single field linear normalizer
        normalizer = get_range_normalizer_from_stat(
            {'min': np.array(min, dtype=np.float32),
             'max': np.array(max, dtype=np.float32)}
        )
        
        # set it
        nns['obs']['rel_object_target_pos'] = normalizer
        
        # yaw rel:
        # Fitted nn: rel_pos_action: min Parameter containing:
        # tensor([-0.6253, -0.7827, -1.1875]) max Parameter containing:
        # tensor([0.7943, 0.8647, 1.0851])
        min = [-0.63, -0.78, -1.19]
        max = [0.79, 0.86, 1.09]
        normalizer = get_range_normalizer_from_stat(
            {'min': np.array(min, dtype=np.float32),
             'max': np.array(max, dtype=np.float32)}
        )
        
        # set it
        nns['rel_pos_action'] = normalizer
        
        # yaw rel:
        # Fitted nn: rel_object_pos: min Parameter containing:
        # tensor([-0.8819, -0.8892, -0.2709]) max Parameter containing:
        # tensor([0.8685, 0.7369, 1.2904])
        min = [-0.88, -0.89, -0.27]
        max = [0.87, 0.74, 1.29]
        normalizer = get_range_normalizer_from_stat(
            {'min': np.array(min, dtype=np.float32),
             'max': np.array(max, dtype=np.float32)}
        )
        nns['obs']['rel_object_pos'] = normalizer
        
        # yaw rel:
        # Fitted nn: pose_pitch_roll: min Parameter containing:
        # tensor([-0.5999, -3.1281]) max Parameter containing:
        # tensor([1.5325, 3.1075])
        min = [-0.6, -3.13]
        max = [1.53, 3.11]
        normalizer = get_range_normalizer_from_stat(
            {'min': np.array(min, dtype=np.float32),
             'max': np.array(max, dtype=np.float32)}
        )
        nns['obs']['pose_pitch_roll'] = normalizer
        
        # Fitted nn: rel_fk: min Parameter containing:
        # tensor([-0.3321, -0.3412, -0.3299, -0.3946, -0.4109, -0.4037, -0.3786, -0.4136,
        #         -0.4078, -0.3651, -0.4055, -0.4020, -0.3508, -0.3984, -0.3924]) max Parameter containing:
        # tensor([0.3118, 0.3328, 0.3303, 0.3916, 0.4106, 0.4120, 0.4001, 0.4133, 0.4160,
        #         0.4051, 0.4094, 0.4120, 0.4000, 0.3981, 0.4036])
        min = [-0.33, -0.34, -0.33, -0.39, -0.41, -0.40, -0.38, -0.41,
               -0.41, -0.37, -0.41, -0.40, -0.35, -0.40, -0.39]
        max = [0.31, 0.33, 0.33, 0.39, 0.41, 0.41, 0.40, 0.41,
               0.42, 0.41, 0.41, 0.41, 0.40, 0.40, 0.40]
        normalizer = get_range_normalizer_from_stat(
            {'min': np.array(min, dtype=np.float32),
                'max': np.array(max, dtype=np.float32)}
        )
        nns['obs']['rel_fk'] = normalizer
        
        return nns
    
    def shrink_nn(self, nn, shrink_rate):
        input_stats_dict = nn.get_input_stats()
        min = input_stats_dict['min']
        max = input_stats_dict['max']
        
        # shrink it
        center = (min + max) / 2.0
        half_range = (max - min) / 2.0 * shrink_rate
        new_min = center - half_range
        new_max = center + half_range
        
        normalizer = get_range_normalizer_from_stat(
            {'min': new_min,
             'max': new_max}
        )
        return normalizer
    
    def shrink_nns_inplace(self, nns, key):
        shrink_rate = 0.8
        
        nn = nns[key]
        nn = self.shrink_nn(nn, shrink_rate)
        
        nns[key] = nn
    
    def shrink_nns(self, nns):
        """
        in inference, often the inputs are OOD, and are clamped to 1.0. Because we set the normalizer to the dataset min/max, very few training points will be clamped, maybe making inference harder.
        
        Can experiment with shrinking the min/max a bit to force more clamping during training.
        
        only the inputs matter, so only modify obs
        
        note** from the histogram analysis, vast majority of input points follow a gaussian, meaning that shrinking the min/max will not affect many points.
        """
        # get rel_object_target_pos min/max
        self.shrink_nns_inplace(nns['obs'], 'rel_object_target_pos')
        
        # rel object pos
        self.shrink_nns_inplace(nns['obs'], 'rel_object_pos')
        
        
    
    def fit_nn(self, data, nn: SingleFieldLinearNormalizer, descriptor = ""):
        nn.fit(data, mode='limits')
        
        # print the results for min max
        input_stats_dict = nn.get_input_stats()
        print("Fitted nn: {}: min {} max {}".format(descriptor, input_stats_dict['min'], input_stats_dict['max']))
        
    def fit_rel_object_target_pos(self, nns):
        """
        only pos, NO QUAT
        """
        # get ref to dataloaders
        dls: TrainAndVal = globals.DATALOADERS[self.rb_id]
        
        # must get a reference to the mage hand sampler. Use sampler because it contains the entire dataset
        sampler: MageHandDatasetSampler = dls.sampler #type:ignore
        
        d_np = sampler.get_all_rel_object_target_pos()
        
        self.fit_nn(d_np, nns['obs']['rel_object_target_pos'], "rel_object_target_pos")
        
        
        if True:
            self.plot_histogram(d_np, descriptor="rel_object_target_pos")
        
    def fit_rel_pos_action(self, nns):
        """
        only pos, NO QUAT
        """
        # get ref to dataloaders
        dls: TrainAndVal = globals.DATALOADERS[self.rb_id]
        
        # must get a reference to the mage hand sampler. Use sampler because it contains the entire dataset
        sampler: MageHandDatasetSampler = dls.sampler #type:ignore
        
        d_np = sampler.get_all_rel_pos_actions()
        ###
        
        self.fit_nn(d_np, nns['rel_pos_action'], "rel_pos_action")
        
    def plot_histogram(self, data, descriptor=""):
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.hist(data, bins=100)
        plt.title("Histogram of {}".format(descriptor))
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.grid()
        plt.show()
        
    def fit_rel_object_pos(self, nns):
        """
        only pos, NO QUAT
        """
        # get ref to dataloaders
        dls: TrainAndVal = globals.DATALOADERS[self.rb_id]
        
        # must get a reference to the mage hand sampler. Use sampler because it contains the entire dataset
        sampler: MageHandDatasetSampler = dls.sampler #type:ignore
        
        d_np = sampler.get_all_rel_object_pos()
        ###
        
        self.fit_nn(d_np, nns['obs']['rel_object_pos'], "rel_object_pos")
        
        if True:
            self.plot_histogram(d_np, descriptor="rel_object_pos")
            
    def fit_pose_pitch_roll(self, nns):
        """
        must be calculated from dataset
        """
        # get ref to dataloaders
        dls: TrainAndVal = globals.DATALOADERS[self.rb_id]
        
        # must get a reference to the mage hand sampler. Use sampler because it contains the entire dataset
        sampler: MageHandDatasetSampler = dls.sampler #type:ignore
        
        d_np = sampler.get_all_pose_pitch_roll()
        
        self.fit_nn(d_np, nns['obs']['pose_pitch_roll'], "pose_pitch_roll")
        
    def fit_rel_fk(self, nns):
        """
        only pos, NO QUAT
        """
        # get ref to dataloaders
        dls: TrainAndVal = globals.DATALOADERS[self.rb_id]
        
        # must get a reference to the mage hand sampler. Use sampler because it contains the entire dataset
        sampler: MageHandDatasetSampler = dls.sampler #type:ignore
        
        d_np = sampler.get_all_rel_fk()
        ###
        
        self.fit_nn(d_np, nns['obs']['rel_fk'], "rel_fk")
    
    def get_fitted_nns(self):
        """
        joint states can be fitted normally
        
        we don't want to fit rel-orientations because range(quat) is already [-1, 1]
        
        ********we DON"T want to fit any quaternions, because they're already normalized to [-1, 1]
        
        we do want to fit rel-positions because they aren't normalized, including FK
        
        
        biotacs are already normalized from 0 to 1, so no fitting needed
        
        pose_pitch_roll must be fit, and must be calculated
        
        for any key which is passed directly into the sample, we can just call fit_nn. 
            - joint_state
            - rel_fk
            - joint_action
        
        For every other key in the sample, we must first calculate all values from the dataset, then fit
        
        RECALL: all normalizers are based off either the replay buffer (not dependent on train/val) or the sampler member variables (also not dependent on train/val) so no matter where this is being used it should be consistent, albeit inefficient if done multiple times.
        """
        nns = self.get_static_nns()
        
        # uses entire dataset
        rb: ReplayBuffer = globals.REPLAY_BUFFER_LOADER['all'] #type:ignore
        
        # joint state isn't modified and has no quats
        self.fit_nn(rb['joint_state'], nns['obs']['joint_state'], "joint_state")
        
        # same nn for joint action as for joint state
        nns['joint_action'] = nns['obs']['joint_state']
        
        # all calculated inputs, not directly from the RB
        if False:
            # rel object target pos is calculated per sample, so we must extract all values first
            self.fit_rel_object_target_pos(nns)
            
            # rel_pos_action is also calculated per sample
            self.fit_rel_pos_action(nns)
            
            # rel object pos
            self.fit_rel_object_pos(nns)
            
            # pose pitch roll
            self.fit_pose_pitch_roll(nns)
            
            self.fit_rel_fk(nns)
            
        else:
            nns = self.update_nns_from_saved_stats(nns)
            
        # experiment with shrinking the input ranges a bit. idk if it helps
        if True:
            self.shrink_nns(nns)
            
        return nns
    
    def __next__(self):
        """
        only difference is that we need to concatenate the joint_action and the rel_pose_action and save it as 'action'.
        """
        nbatch = super().__next__()
        
        # get joint_action and rel_pose_action
        joint_action = nbatch['joint_action']
        rel_quat_action = nbatch['rel_quat_action']
        rel_pos_action = nbatch['rel_pos_action']
        
        # concat
        action = torch.cat([
            joint_action, 
            rel_quat_action,
            rel_pos_action,
            ], dim=-1)
        
        # save
        nbatch['action'] = action

        return nbatch

class NestedBatchLoader(dict):
    """
    Extend BatchLoader functionality to multiple BatchLoaders. Useful when doing co-training on different datasets
    """
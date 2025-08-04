import torch
import numpy as np
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.globals import CONFIG
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
from diffusion_policy.common.normalize_util import get_identity_normalizer_from_stat
from diffusion_policy.dataset.base_dataset import BaseImageDataset

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


class DataArray:
    """
    A class for handling a single data array and performing desired operations on it, e.g. normalize, unnormalize
    """
    def __init__(self, 
                 normalizer: SingleFieldLinearNormalizer
                 ):
        self.datapoint = None
        self.normalizer = normalizer

    def set(self, dp: torch.Tensor):
        self.datapoint = dp

    def get(self):
        if self.datapoint is None:
            print("Forgot to set the datapoint")
            raise

        return self.datapoint

    def normalize(self):
        if self.datapoint is None:
            print("Forgot to set the datapoint")
            raise

        self.datapoint = self.normalizer.normalize(self.datapoint)
    
    def unnormalize(self):
        if self.datapoint is None:
            print("Forgot to set the datapoint")
            raise

        self.datapoint = self.normalizer.unnormalize(self.datapoint)
    
class NestedDataArray:
    """
    A class for holding a nested data structure of DataArray's. Each branch = NestedDataArray, each leaf = DataArray
    """
    def __init__(self):
        self.nest = {}

    def set(self, data):
        """
        Data - a nested dict of torch.Tensors
        """
        # works whether val is a dict or a torch.Tensor since the syntax is the same
        for key, val in enumerate(data):
            self.nest[key].set(data)

    def get(self):
        """
        extract a nested dict of torch.Tensor's
        """
        out = {}

        # works whether val is a NestedDataArray or a DataArray since the syntax is the same
        for key, val in enumerate(self.nest):
            out[key] = val.get()

        return out

    def set_normalizers(self, nested_normalizers):
        # reset nest
        self.nest = {}

        for key, val in enumerate(nested_normalizers):
            # leaf
            if isinstance(val, SingleFieldLinearNormalizer):
                da = DataArray(val)
                self.nest[key] = da

            # another branch
            else:
                nda = NestedDataArray()
                self.nest[key] = nda.set_normalizers(val)


    def normalize(self):
        # works whether val is a NestedDataArray or a DataArray since the syntax is the same
        for key, val in enumerate(self.nest):
            val.normalize()

    def unnormalize(self):
        # works whether val is a NestedDataArray or a DataArray since the syntax is the same
        for key, val in enumerate(self.nest):
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
            dataset: BaseImageDataset
            ):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset,
            batch_size=dataset_batch_size,
            timeout=self.timeout,
            num_workers=num_workers[idx],
            **cfg.dataloader)
        
        self.nested_data_array = NestedDataArray()
        
        self.device = torch.device(CONFIG.training.device)

    def init_normalizer(self):
        """
        Structure, must be same as the shape_meta structure.

        """

        obs = {
                'lowdim_obs': get_range_normalizer_from_stat(
                    {'min': LIMITS[:, 0], 'max': LIMITS[:, 1]}
                    ),
                'image': get_image_range_normalizer(),
                'image2': get_image_range_normalizer(),
            }
        
        nn = {
            'obs': obs,
            'next_obs': obs,
            'action': get_range_normalizer_from_stat(
                {'min': JOINT_LIMITS[:, 0], 'max': JOINT_LIMITS[:, 1]}
                ),
            'not_done': get_identity_normalizer_from_stat(
                {'min', np.array([1])}
            ),
            'reward': get_identity_normalizer_from_stat(
                {'min', np.array([1])}
                )
        }
        
        self.nested_data_array.set_normalizers(nn)
        
    def reset(self):
        # reset my count
        self.count = 0

        # forces a reshuffle
        self.iterator = iter(self.dataloader)

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
    
    def normalize_batch(self, batch):
        """
        batch - a torch-gpu nested dict 
        
        """
        # update the nested data array data
        self.nested_data_array.set(batch)

        # run the normalizer
        self.nested_data_array.normalize()

        # extract the normalized data
        nbatch = self.nested_data_array.get()

        
        # we're done
        return nbatch

    
    def get_batch(self):

        try:
            batch = next(self.iterator) # output: dict
        # except can occur from a timeout or a StopIteration
        # I'm still not sure why the timeout's are occuring, but this will get around it
        except Exception as e:
            print("next(iterator) except: ")
            print(e)
            print("len(dataset): ", len(self.dataset))
            
            # reshuffle this iterator
            self.iterator = iter(self.dataloader)
            
            # get a new batch
            batch = next(self.iterator) # output: dict

        batch_gpu = self.transfer_to_gpu(batch)
        
        ndata = self.normalize_batch(batch_gpu)
        
        # we're done
        return ndata
    
    # called each for-loop
    def __next__(self):
        
        if self.count < self.num_batches:
            # increment our count
            self.count += 1

            nbatch = self.get_batch()

            return nbatch
        
        else:
            # we've done num_batches
            # print("Iteration done. Count: {}".format(self.count))
            raise StopIteration

class NestedBatchLoader(dict):
    """
    Extend BatchLoader functionality to multiple BatchLoaders. Useful when doing co-training on different datasets
    """
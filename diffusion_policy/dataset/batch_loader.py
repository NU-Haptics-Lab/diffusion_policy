








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

import diffusion_policy.samplers.mage_hand_sampler as mage_hand_sampler
# import MageHandEpisodeSampler, MageHandDatasetSampler

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
                #  clamp = True, # UNUSED whether to clamp the normalized values to [-1, 1]
                 ):
        self.normalizer = normalizer
        self.descriptor = descriptor
        self.strict = strict
        self.clamp = globals.CONFIG.clamp # type:ignore
        
        if self.clamp:
            self.clamp_value = globals.CONFIG.clamp_value #type:ignore
        else:
            self.clamp_value = None
        
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
            self.datapoint = torch.clamp(self.datapoint, -self.clamp_value, self.clamp_value)
            
            pass
            
            
    
    def unnormalize(self):
        if self.strict and self.datapoint is None:
            print("Forgot to set the datapoint")
            raise
        
        if self.datapoint is None:
            return
        
        # must clamp the normalized datapoint first
        if self.clamp:
            self.datapoint = torch.clamp(self.datapoint, -self.clamp_value, self.clamp_value)
            
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
                da = DataArray(val, descriptor=key, strict=self.strict)
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
                 strict: bool = True,
                 normalizer_fit_rb_ids: 'list[str] | None' = None, # which rb_id(s) to pool together when FITTING this BatchLoader's normalizer stats (see init_normalizers/get_fitted_nns). None (default) = just this loader's own rb_id, i.e. an independent per-source fit -- the correct default for genuinely distinct-schema sources. Pass the SAME list (e.g. every cotrain rb_id) on every BatchLoader instance that should share one normalizer, so they agree on one explicit, deterministic cache key instead of accidentally sharing (or not) based on class name + construction order -- see init_normalizers's docstring for the bug this replaces.

            ):
        self.rb_id = rb_id
        self.train_or_val = train_or_val
        self.use_dataloader = use_dataloader
        self.strict = strict
        self.normalizer_fit_rb_ids = list(normalizer_fit_rb_ids) if normalizer_fit_rb_ids else [rb_id]

        self.dataloaders = None #type:ignore
        self.iterator = None

        self.is_setup = False
        
    def __len__(self):
        if self.use_dataloader and self.dataloaders is not None:
            dataloader = self.dataloaders[self.train_or_val]
            if dataloader is None:
                return 0
            else:
                return len(dataloader)
        else:
            return 0
        
    def setup(self):
        
        # # only continue if we're being trained off of or special rb_id of default
        # tasks_to_use = globals.CONFIG.tasks_to_use #type:ignore
        # if (self.rb_id not in tasks_to_use) and not self.rb_id == "default":
        #     return
        
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
        
        self.is_setup = True
    
    def fit_nn(self, data, nn: SingleFieldLinearNormalizer, descriptor = "", mode = 'limits'):
        """
        mode='limits' (default): scale+offset map the data's observed
        min/max exactly onto [-1, 1] -- appropriate for obs keys, where you
        want the full observed range representable.

        mode='gaussian': scale+offset map to (roughly) unit variance/zero
        mean instead (see normalizer.py's _fit -- actually 1.5 std, so ~87%
        of a normal distribution lands inside [-1, 1], not a hard min/max
        clamp). Better matches the actual noising process diffusion trains
        against (DDPM's forward process adds Gaussian noise to the ACTION
        target, so a target distribution that's already closer to Gaussian
        is a more consistent match than one artificially stretched to fill
        [-1, 1] by its rarest outlier). Only used for AIETErlenmeyerFlask4BatchLoader's
        action normalizer currently -- min/max outlier values under this mode
        can legitimately fall outside [-1, 1] (unlike 'limits'), which
        interacts with common_noise_scheduler's clip_sample: true (clips the
        model's predicted x0 to [-1, 1] during reverse sampling) -- a
        long-tailed action's true target may get clipped there; worth
        revisiting if that turns out to matter.
        """
        nn.fit(data, mode=mode)
        
        # print the results for min max
        input_stats_dict = nn.get_input_stats()
        # print("Fitted nn: {}: min {} max {}".format(descriptor, input_stats_dict['min'], input_stats_dict['max']))
        
        # outlier protection
        min_val = np.array(input_stats_dict['min']).min()
        max_val = np.array(input_stats_dict['max']).max()
        
        min_allowed = -1000.0
        max_allowed = 1000.0
        
        if min_val < min_allowed or max_val > max_allowed:
            raise ValueError("Fitted nn {} has min {} or max {} outside of allowed range [{}, {}]. Check your data for outliers.".format(descriptor, min_val, max_val, min_allowed, max_allowed))
        
        
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
        
        

    def _flatten_normalizers(self, nns, prefix=()):
        """
        Nested dict of {str: SingleFieldLinearNormalizer | dict} -> flat
        {path_tuple: normalizer.state_dict()}. Mirrors the recursion
        NestedDataArray.set_normalizers already does to tell leaves from
        branches.
        """
        flat = {}
        for key, val in nns.items():
            path = prefix + (key,)
            if isinstance(val, SingleFieldLinearNormalizer):
                flat[path] = val.state_dict()
            elif isinstance(val, dict):
                flat.update(self._flatten_normalizers(val, path))
        return flat

    def _apply_normalizer_state_dict(self, nns, flat_state_dict, prefix=()):
        """Inverse of _flatten_normalizers -- loads saved stats onto the
        (unfitted, shape-only) normalizers get_static_nns() produced."""
        for key, val in nns.items():
            path = prefix + (key,)
            if isinstance(val, SingleFieldLinearNormalizer):
                if path in flat_state_dict:
                    val.load_state_dict(flat_state_dict[path])
            elif isinstance(val, dict):
                self._apply_normalizer_state_dict(val, flat_state_dict, path)

    def init_normalizers(self):
        """
        Structure, must be same as the shape_meta structure.

        Fitting (get_fitted_nns) requires the real dataset/sampler machinery
        and can be slow (e.g. AIETAlignmentSim3BatchLoader's action
        normalizer walks every episode) -- so the first time a given
        (class, normalizer_fit_rb_ids) combination actually fits, its stats
        are cached in globals.NORMALIZER_STATE_DICTS (checkpointed by
        TopKCheckpointManager, see checkpointer.py). Later instantiations
        sharing that same combination -- e.g. default_batch_loader during
        standalone testing/inference, after a checkpoint has been loaded, or
        several cotrain streams deliberately sharing one normalizer -- reuse
        those cached stats instead of re-fitting from scratch.

        cache_key is (class name, sorted normalizer_fit_rb_ids) -- explicit
        and deterministic, NOT dependent on which BatchLoader instance
        happens to call setup() first. This replaces a previous bug: caching
        by class name ALONE meant every BatchLoader subclass instance
        (regardless of its own rb_id) silently shared whichever instance's
        fit ran first -- for cotrains reusing one BatchLoader class across
        multiple rb_ids (e.g. AIETErlenmeyerFlask4BatchLoader across
        erlenmeyer/erlenmeyer_align/sim_alignment), this happened to already
        match the desired "share one normalizer" outcome, purely because
        globals.py's setup order always constructs default_batch_loader (a
        single fixed rb_id) before the per-stream BatchLoaders -- but it was
        an accident of construction order + shared class name, not a
        deliberate, robust guarantee: a future subclass split, reordering, or
        a stray rb_id-specific override would silently break it with no
        error. normalizer_fit_rb_ids (constructor param) now makes the
        intended sharing (or non-sharing) explicit in the yaml instead.
        """
        cache_key = f"{type(self).__name__}:{','.join(sorted(self.normalizer_fit_rb_ids))}"

        if cache_key in globals.NORMALIZER_STATE_DICTS:
            nns = self.get_static_nns()
            self._apply_normalizer_state_dict(nns, globals.NORMALIZER_STATE_DICTS[cache_key])
            print(f"[BatchLoader:{self.rb_id}] reusing cached normalizer stats for group '{cache_key}'")
        else:
            nns = self.get_fitted_nns()
            globals.NORMALIZER_STATE_DICTS[cache_key] = self._flatten_normalizers(nns)
            print(f"[BatchLoader:{self.rb_id}] fit NEW normalizer stats for group '{cache_key}' "
                  f"(pooled from rb_ids={self.normalizer_fit_rb_ids})")

        self.nested_data_array.set_normalizers(nns)
        
    def reset(self):
        # reset my count
        self.count = 0

        # forces a reshuffle
        if self.use_dataloader and self.dataloaders is not None:
            dataloader = self.dataloaders[self.train_or_val]
            
            if dataloader is None:
                self.iterator = None
                print("Note: setting batch_loader.iterator to None.")
            else:
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
        
        # assert all finite values
        def fn(x):
            if not torch.isfinite(x).all():
                raise ValueError("Batch contains non-finite values.")
        pytorch_util.dict_apply_inplace(ndata, fn)
        
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
        
        
    def fit_rel_object_target_pos(self, nns):
        """
        only pos, NO QUAT
        """
        # get ref to dataloaders
        dls: TrainAndVal = globals.DATALOADERS[self.rb_id]
        
        # must get a reference to the mage hand sampler. Use sampler because it contains the entire dataset
        sampler: mage_hand_sampler.MageHandDatasetSampler = dls.sampler #type:ignore
        
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
        sampler: mage_hand_sampler.MageHandDatasetSampler = dls.sampler #type:ignore
        
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
        sampler: mage_hand_sampler.MageHandDatasetSampler = dls.sampler #type:ignore
        
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
        sampler: mage_hand_sampler.MageHandDatasetSampler = dls.sampler #type:ignore
        
        d_np = sampler.get_all_pose_pitch_roll()
        
        self.fit_nn(d_np, nns['obs']['pose_pitch_roll'], "pose_pitch_roll")
        
    def fit_rel_fk(self, nns):
        """
        only pos, NO QUAT
        """
        # get ref to dataloaders
        dls: TrainAndVal = globals.DATALOADERS[self.rb_id]
        
        # must get a reference to the mage hand sampler. Use sampler because it contains the entire dataset
        sampler: mage_hand_sampler.MageHandDatasetSampler = dls.sampler #type:ignore
        
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
    
class SandboxRobotBCBatchLoader(BatchLoader):
    def get_static_nns(self):
        """
        pretty much all identity normalizers here.
        """
        nns = {}
        obs = {}
        obs_keys_to_load = globals.CONFIG.obs_keys_to_load # type: ignore
        act_key = globals.CONFIG.action_key # type: ignore
        
        # default identity normalizer
        for obs_key in obs_keys_to_load:
            # get shape meta from the config
            nb = globals.CONFIG.shape_meta[obs_key].shape # type: ignore
            
            obs[obs_key] = get_identity_normalizer_from_stat(
                {'min': np.zeros(nb, dtype=np.float32)}
                )
            
        # actions
        # get shape meta from the config
        nb = globals.CONFIG.shape_meta[act_key].shape # type: ignore
        
        # required keyword
        # nns["action"] = get_identity_normalizer_from_stat(
        #     {'min': np.zeros(nb, dtype=np.float32)}
        #     )
        # custom for torque
        max_torques = 1.0 * np.ones(nb, dtype=np.float32)
        max_torques[0:6] = np.array([
            500.0,
            500.0,
            200.0,
            150.0,
            40.0,
            40.0,
        ])
        nns["action"] = get_range_normalizer_from_stat(
            {'min': -max_torques, 'max': max_torques}
        )
        
        # final assembly
        nns = nns | {
            'obs': obs,
        }
        
        # we're done
        return nns
    
    def get_fitted_nns(self):
        # default nns
        nns = self.get_static_nns()
        
        # uses the 'all' dataset ... special keyword for now...
        rb: ReplayBuffer = globals.REPLAY_BUFFER_LOADER['all'] #type:ignore
        obs_keys_to_load: list = globals.CONFIG.obs_keys_to_load # type: ignore
        
        # each observation
        for obs_key in obs_keys_to_load:
            # only fit if obs_key is in the rb, otherwise keep default identity normalizer
            if obs_key in rb:
                self.fit_nn(rb[obs_key], nns['obs'][obs_key], obs_key)
                
        # for the action
        # act_key = globals.CONFIG.action_key # type: ignore
        # self.fit_nn(rb[act_key], nns['action'], act_key)
        
        return nns
    
class SandboxRobotRLBatchLoader(SandboxRobotBCBatchLoader):
    """
    same exact as SandboxRobotBCBatchLoader, but duplicate obs into obs_next and action into action_next
    """
    
    def get_fitted_nns(self):
        nns = super().get_fitted_nns()
        
        # duplicate obs into obs_next
        nns['obs_next'] = nns['obs']
        
        # duplicate action into action_next
        nns['action_next'] = nns['action']
        
        # must overwrite the previous action to use our action normalizer
        nns['obs']['robot_joint_prev_action'] = nns['action']
        nns['obs_next']['robot_joint_prev_action'] = nns['action_next']
        
        
        nns['not_done'] = get_identity_normalizer_from_stat({'min': np.array([0], dtype=np.float32)})
        
        nns['reward'] = get_identity_normalizer_from_stat({'min': np.array([0], dtype=np.float32)})
        
        return nns

class NestedBatchLoader(dict):
    """
    Extend BatchLoader functionality to multiple BatchLoaders. Useful when doing co-training on different datasets
    """
    
class AIETErlenmeyerFlaskBatchLoader(BatchLoader):
    """
    BatchLoader for task 24 (erlenmeyer flask insertion).

    Zarr keys produced by gen_dataset.py:
      joint_states          [T, 30]  — gofa(6)+wrist(2)+th(5)+ff(4)+mf(4)+rf(4)+lf(5)
      joint_commands        [T, 30]  — same layout, used as action
      wrist_cam_features    [T, 768] — DINOv2 ViT-B/14 CLS token
      overhead_cam_features [T, 768] — DINOv2 ViT-B/14 CLS token
      biotac_lh             [T, D]   — already normalised to [0, 1]

    Joint limits in JOINT_LIMITS only cover 21 joints (no rf/lf), so both
    joint_states and joint_commands are fitted from data in get_fitted_nns.
    biotac_lh keeps its identity normalizer (already [0, 1]).
    
    For now, don't normalize the dinov2 features
    """

    def get_static_nns(self):
        """
        default identity normalizers
        
        nns['action'] is hard-coded in diffusion_model and must be used
        """
        nns = {}
        obs = {}
        obs_keys_to_load = globals.CONFIG.obs_keys_to_load  # type: ignore

        for obs_key in obs_keys_to_load:
            nb = globals.CONFIG.shape_meta[obs_key].shape  # type: ignore
            obs[obs_key] = get_identity_normalizer_from_stat(
                {'min': np.zeros(nb, dtype=np.float32)}
            )

        act_key = globals.CONFIG.action_key  # type: ignore
        nb_act = globals.CONFIG.shape_meta[act_key].shape  # type: ignore
        nns['action'] = get_identity_normalizer_from_stat(
            {'min': np.zeros(nb_act, dtype=np.float32)}
        )

        nns['obs'] = obs
        
        
        # task id, identity normalizer
        nns['task_id'] = get_identity_normalizer_from_stat(
            {'min': np.array([0], dtype=np.float32)}
        )

        # subtask id, identity normalizer -- must not be touched by normalization
        nns['subtask_id'] = get_identity_normalizer_from_stat(
            {'min': np.array([0], dtype=np.float32)}
        )

        return nns

    def get_fitted_nns(self):
        nns = self.get_static_nns()

        rb: ReplayBuffer = globals.REPLAY_BUFFER_LOADER['all']  # type: ignore
        obs_keys_to_load: list = globals.CONFIG.obs_keys_to_load  # type: ignore
        act_key: str = globals.CONFIG.action_key  # type: ignore
        
        obs_keys_to_normalize = [
            "joint_states",
        ]

        for obs_key in obs_keys_to_normalize:
            assert obs_key in obs_keys_to_load, f"Observation key '{obs_key}' is not in the list of keys to load."
            self.fit_nn(rb[obs_key], nns['obs'][obs_key], obs_key)

        if act_key in rb:
            self.fit_nn(rb[act_key], nns['action'], act_key)

        return nns


class AIETErlenmeyerFlask3BatchLoader(BatchLoader):
    """
    Shared BatchLoader for aiet_erlenmeyer_flask_3's cotraining of the real
    erlenmeyer-flask dataset with the sim alignment dataset -- used for BOTH
    rb_ids (real and sim), just pointed at different rb_id/sampler pairs in
    the yaml. Registers identity normalizers for the UNION of every key
    either data source's sampler emits (AIETErlenmeyerFlask3RealSampler /
    AIETErlenmeyerFlask3SimSampler), including the shared joint_states key
    and the new data_source tag, so nbatch has a uniform schema regardless of
    which data source produced it -- the model masks out data-source-inapplicable
    tokens itself (sim_only_token_keys/real_only_token_keys), this loader
    just needs every key to exist so normalization doesn't KeyError.

    joint_states is fit from whichever rb_id this specific loader instance is
    attached to (self.rb_id) -- real and sim each get their own fit, since
    they're two different underlying datasets sharing one conceptual key.
    Everything else uses identity normalizers, same "for now, don't
    normalize" precedent as AIETErlenmeyerFlaskBatchLoader's vision features.
    """

    def get_static_nns(self):
        nns = {}
        obs = {}
        obs_keys_to_load = globals.CONFIG.obs_keys_to_load  # type: ignore

        for obs_key in obs_keys_to_load:
            nb = globals.CONFIG.shape_meta[obs_key].shape  # type: ignore
            obs[obs_key] = get_identity_normalizer_from_stat(
                {'min': np.zeros(nb, dtype=np.float32)}
            )

        act_key = globals.CONFIG.action_key  # type: ignore
        nb_act = globals.CONFIG.shape_meta[act_key].shape  # type: ignore
        nns['action'] = get_identity_normalizer_from_stat(
            {'min': np.zeros(nb_act, dtype=np.float32)}
        )

        nns['obs'] = obs

        for key in ['task_id', 'subtask_id', 'data_source']:
            nns[key] = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
            )

        return nns

    def get_fitted_nns(self):
        nns = self.get_static_nns()

        rb: ReplayBuffer = globals.REPLAY_BUFFER_LOADER[self.rb_id]  # type: ignore
        if "joint_states" in rb:
            self.fit_nn(rb["joint_states"], nns['obs']["joint_states"], "joint_states")
        elif "joint_positions" in rb:
            # sim rb: joint_states is aliased from joint_positions by the sampler
            self.fit_nn(rb["joint_positions"], nns['obs']["joint_states"], "joint_states")

        act_key = globals.CONFIG.action_key  # type: ignore
        if act_key in rb:
            self.fit_nn(rb[act_key], nns['action'], act_key)

        return nns


class AIETAlignmentSimBatchLoader(BatchLoader):
    """
    BatchLoader for the sim-only pellet alignment task.

    Zarr keys (relevant obs subset -- see aiet_alignment_sim.yaml for the
    full obs_keys_to_load, and AIETAlignmentSimSampler for the pellet
    keypoint/validity fields, which are top-level sample fields, not obs):
      joint_positions             [T, 30] -- gofa(6)+wr(2)+th(5)+ff(4)+mf(4)+rf(4)+lf(5);
                                    used as both an obs key and the action key
      wrist_camera_pose_xyz_rpy   [T, 6]  -- forearm-camera pose, xyz + roll/pitch/yaw
      target_pellet_location      [T, 3]  -- xyz; PelletLocalizer's xyz-regression
                                    aux-loss target, not a model input
      wrist_camera_patch_features [T, 14, 14, 384] -- DINOv3 patch tokens, fed to
                                    PelletLocalizer only, bypasses obs_encoder entirely
                                    (identity-normalized -- "don't normalize the DINO
                                    features" precedent, same as aiet_erlenmeyer_flask_3)

    joint_positions, wrist_camera_pose_xyz_rpy, and target_pellet_location
    have no known a-priori limits for this sim setup (unlike JOINT_LIMITS
    for the real robot), so they're fit directly from the dataset
    (mode='limits') in get_fitted_nns, same as the action.
    """

    IMAGE_OBS_KEYS = []  # nothing loaded as raw pixels anymore (wrist_camera_image was replaced by wrist_camera_patch_features)
    OBS_KEYS_TO_FIT = ["joint_positions", "wrist_camera_pose_xyz_rpy", "target_pellet_location"]

    def get_static_nns(self):
        nns = {}
        obs = {}
        obs_keys_to_load = globals.CONFIG.obs_keys_to_load  # type: ignore

        for obs_key in obs_keys_to_load:
            if obs_key in self.IMAGE_OBS_KEYS:
                obs[obs_key] = get_range_normalizer_from_stat(
                    {'min': np.array([0], dtype=np.float32), 'max': np.array([255], dtype=np.float32)}
                )
            else:
                nb = globals.CONFIG.shape_meta[obs_key].shape  # type: ignore
                obs[obs_key] = get_identity_normalizer_from_stat(
                    {'min': np.zeros(nb, dtype=np.float32)}
                )

        act_key = globals.CONFIG.action_key  # type: ignore
        nb_act = globals.CONFIG.shape_meta[act_key].shape  # type: ignore
        nns['action'] = get_identity_normalizer_from_stat(
            {'min': np.zeros(nb_act, dtype=np.float32)}
        )

        nns['obs'] = obs

        # the shared AIETErlenmeyerFlaskSampler always emits task_id/subtask_id
        # (with fallback defaults when a dataset doesn't have them, as here) --
        # register identity normalizers so NestedDataArray has somewhere to put
        # them. Harmless: embed_task_id is false for this sim-only task.
        nns['task_id'] = get_identity_normalizer_from_stat(
            {'min': np.array([0], dtype=np.float32)}
        )
        nns['subtask_id'] = get_identity_normalizer_from_stat(
            {'min': np.array([0], dtype=np.float32)}
        )

        # pellet-keypoint aux-loss targets (AIETAlignmentSimSampler only) --
        # identity normalizers: these are supervision targets for
        # BatchLoss's keypoint loss, not policy obs, and that loss does its
        # own px -> [-1,1] conversion to match the model's spatial-softmax
        # grid convention exactly, so nothing should rescale them here
        nns['wrist_pellet_keypoints_px'] = get_identity_normalizer_from_stat(
            {'min': np.zeros((4, 2), dtype=np.float32)}
        )
        nns['wrist_pellet_keypoints_valid'] = get_identity_normalizer_from_stat(
            {'min': np.zeros((4,), dtype=np.float32)}
        )
        nns['target_pellet_valid'] = get_identity_normalizer_from_stat(
            {'min': np.zeros((1,), dtype=np.float32)}
        )

        return nns

    def get_fitted_nns(self):
        nns = self.get_static_nns()

        rb: ReplayBuffer = globals.REPLAY_BUFFER_LOADER['all']  # type: ignore
        act_key: str = globals.CONFIG.action_key  # type: ignore

        for obs_key in self.OBS_KEYS_TO_FIT:
            if obs_key in rb:
                self.fit_nn(rb[obs_key], nns['obs'][obs_key], obs_key)

        if act_key in rb:
            self.fit_nn(rb[act_key], nns['action'], act_key)

        return nns


class AIETAlignmentSim3BatchLoader(BatchLoader):
    """
    BatchLoader for the lh_forearm-only sim (no abb gofa), single-stage
    diffusion transformer, no aux losses.

    Zarr keys used (see aiet_alignment_sim_3.yaml):
      palm_pose_xyz_rpy           [T, 6] -- world pose of lh_palm, xyz + roll/pitch/yaw.
                                    NOT a model input (see AIETAlignmentSim3Sampler) --
                                    only used here/there to compute the action's
                                    relative-pose component; absent from obs_keys_to_use.
      gripper_value               [T]    -- single [0, 1] "how closed" scalar, stays
                                    absolute (both obs input and action component)
      wrist_camera_patch_features [T, 14, 14, 384] -- DINOv3 patch tokens, fed to
                                    the diffusion transformer's own spatial softmax,
                                    bypasses obs_encoder entirely (identity-normalized,
                                    same "don't normalize DINO features" precedent as
                                    aiet_erlenmeyer_flask_3 / aiet_alignment_sim_2)

    combined.zarr has no single "action" field -- config.action_key ("action")
    is a placeholder. The action is [rel_palm_pose_xyz_rpy (6), gripper_value (1)],
    built by relative-pose subtraction + concatenation in AIETAlignmentSim3Sampler
    for the actual per-sample trajectory used during training; get_fitted_nns below
    fits the normalizer over the same per-sample logic, aggregated across every
    episode via AIETAlignmentSim3Sampler.get_all_action_components (NOT
    reimplemented by hand-slicing the raw replay buffer -- the rb only has flat
    per-step arrays, not per-sample trajectories).
    """

    OBS_KEYS_TO_FIT = ["gripper_value"]

    def get_static_nns(self):
        nns = {}
        obs = {}
        obs_keys_to_load = globals.CONFIG.obs_keys_to_load  # type: ignore

        for obs_key in obs_keys_to_load:
            nb = globals.CONFIG.shape_meta[obs_key].shape  # type: ignore
            obs[obs_key] = get_identity_normalizer_from_stat(
                {'min': np.zeros(nb, dtype=np.float32)}
            )

        act_key = globals.CONFIG.action_key  # type: ignore
        nb_act = globals.CONFIG.shape_meta[act_key].shape  # type: ignore
        nns['action'] = get_identity_normalizer_from_stat(
            {'min': np.zeros(nb_act, dtype=np.float32)}
        )

        nns['obs'] = obs

        # the shared AIETErlenmeyerFlaskSampler always emits task_id/subtask_id
        # (with fallback defaults when a dataset doesn't have them, as here)
        nns['task_id'] = get_identity_normalizer_from_stat(
            {'min': np.array([0], dtype=np.float32)}
        )
        nns['subtask_id'] = get_identity_normalizer_from_stat(
            {'min': np.array([0], dtype=np.float32)}
        )

        return nns

    def get_all_action_components(self):
        """
        Delegates to AIETAlignmentSim3Sampler.get_all_action_components on
        every episode in the 'all' dataloader's sampler -- i.e. runs the
        exact same per-sample action-window logic real training samples go
        through (fill-back/fill-forward indexing, episode boundaries, the
        restrict_to_valid_future mask, etc., all courtesy of the real
        Indices/EpisodeSampler machinery), just aggregated across every
        valid sample in the dataset instead of one at a time. Deliberately
        NOT reimplemented by hand-slicing the raw replay buffer arrays --
        the replay buffer only has flat per-step data, not per-sample
        trajectories, so that reimplementation could silently drift from
        what get_action_trajectory actually produces for real samples.
        """
        dls: TrainAndVal = globals.DATALOADERS['all']  # type: ignore
        sampler = dls.sampler

        rel_palms = []
        grips = []
        for ep in sampler.get_ep_list():
            rel_palm, grip = ep.get_all_action_components()
            rel_palms.append(rel_palm)
            grips.append(grip)

        return np.concatenate(rel_palms, axis=0), np.concatenate(grips, axis=0)

    def get_fitted_nns(self):
        nns = self.get_static_nns()

        rb: ReplayBuffer = globals.REPLAY_BUFFER_LOADER['all']  # type: ignore

        for obs_key in self.OBS_KEYS_TO_FIT:
            if obs_key in rb:
                # rb[obs_key] (gripper_value) is 1-D [T] -- fit_nn's reshape(-1, dim)
                # takes dim from the LAST axis, which for a 1-D array is T itself,
                # not the 1-element feature dim. Add the trailing axis explicitly.
                self.fit_nn(rb[obs_key][:][:, None], nns['obs'][obs_key], obs_key)

        if "palm_pose_xyz_rpy" in rb and "gripper_value" in rb:
            # rel_palm/grip are computed over every valid sample's full
            # action_rel_indices window, so they aren't the same length as
            # each other row-for-row -- but SingleFieldLinearNormalizer.fit
            # computes min/max/scale/offset independently per (last-dim)
            # column, so fitting the two components separately and
            # concatenating their params is equivalent to fitting one array
            # with matching row counts.
            rel_palm, grip = self.get_all_action_components()
            palm_nn = get_identity_normalizer_from_stat({'min': np.zeros(6, dtype=np.float32)})
            self.fit_nn(rel_palm, palm_nn, "rel_palm_pose_xyz_rpy")

            grip_nn = get_identity_normalizer_from_stat({'min': np.zeros(1, dtype=np.float32)})
            self.fit_nn(grip[:, None], grip_nn, "gripper_value (action)")  # same 1-D trailing-axis fix as above

            for param_name in ["scale", "offset"]:
                nns['action'].params_dict[param_name] = torch.cat([
                    palm_nn.params_dict[param_name], grip_nn.params_dict[param_name],
                ])
            for stat_name in ["min", "max", "mean", "std"]:
                nns['action'].params_dict['input_stats'][stat_name] = torch.cat([
                    palm_nn.params_dict['input_stats'][stat_name],
                    grip_nn.params_dict['input_stats'][stat_name],
                ])

        return nns


class AIETErlenmeyerFlask4BatchLoader(BatchLoader):
    """
    Shared BatchLoader for aiet_erlenmeyer_flask_4's 3-way cotrain (real
    erlenmeyer, real full-avatar-cotrain, sim alignment) -- used for ALL
    THREE rb_ids, just pointed at a different rb_id/sampler pair per source
    in the yaml. Unlike aiet_erlenmeyer_flask_3 (which needed two different
    BatchLoader classes because its sim leg's action was joint-space while
    the real legs' wasn't), all three of `_4`'s sources now share the exact
    same schema and the exact same relative-palm-pose+gripper action
    (AIETErlenmeyerFlask4Real1Sampler/Real2Sampler/SimSampler are all thin
    AIETAlignmentSim3Sampler subclasses) -- so one BatchLoader class
    generalizing AIETAlignmentSim3BatchLoader's fitting logic from the
    hardcoded rb_id 'all' to self.rb_id suffices for every source.

    gripper_value and palm_pose_xyz_rpy, and the relative-palm-pose+gripper
    action, are fit by pooling every rb_id in normalizer_fit_rb_ids together
    (defaults to just [self.rb_id], an independent per-source fit -- pass the
    same explicit list on multiple streams' BatchLoaders in the yaml to make
    them deliberately share one normalizer instead, e.g. aiet_erlenmeyer_flask_8's
    normalizer_fit_rb_ids: [erlenmeyer, erlenmeyer_align, sim_alignment] on
    all three cotrain streams' batch_loaders AND default_batch_loader, so
    every one of them ends up on the exact same cache_key regardless of which
    instance's setup() happens to run first -- see BatchLoader.init_normalizers's
    docstring for the accidental-sharing bug this replaces). Delegates to
    get_all_action_components (NOT hand-sliced from the raw replay buffer --
    see that method's docstring for why). Every other obs key (the two
    patch-feature keys, biotac_lh) is identity-normalized, same "don't
    normalize DINO features" precedent as every other aiet_* BatchLoader.
    """

    OBS_KEYS_TO_FIT = [
        "palm_pose_xyz_rpy",
        "gripper_value"]

    def get_static_nns(self):
        nns = {}
        obs = {}
        obs_keys_to_load = globals.CONFIG.obs_keys_to_load  # type: ignore

        for obs_key in obs_keys_to_load:
            nb = globals.CONFIG.shape_meta[obs_key].shape  # type: ignore
            obs[obs_key] = get_identity_normalizer_from_stat(
                {'min': np.zeros(nb, dtype=np.float32)}
            )

        act_key = globals.CONFIG.action_key  # type: ignore
        nb_act = globals.CONFIG.shape_meta[act_key].shape  # type: ignore
        nns['action'] = get_identity_normalizer_from_stat(
            {'min': np.zeros(nb_act, dtype=np.float32)}
        )

        nns['obs'] = obs

        for key in ['task_id', 'subtask_id', 'data_source', 'ep_id']:
            nns[key] = get_identity_normalizer_from_stat(
                {'min': np.array([0], dtype=np.float32)}
            )

        return nns

    def get_all_action_components(self):
        """
        Same delegation as AIETAlignmentSim3BatchLoader.get_all_action_components,
        generalized from the hardcoded rb_id 'all' to self.normalizer_fit_rb_ids
        -- pools EVERY rb_id in that list (defaults to just [self.rb_id], an
        independent per-source fit) into one action normalizer. Pass the same
        normalizer_fit_rb_ids on multiple cotrain streams' BatchLoaders (see
        the yaml) to fit one shared normalizer across all of them instead.
        """
        rel_palms = []
        grips = []
        for rb_id in self.normalizer_fit_rb_ids:
            dls: TrainAndVal = globals.DATALOADERS[rb_id]  # type: ignore
            for ep in dls.sampler.get_ep_list():
                rel_palm, grip = ep.get_all_action_components()
                rel_palms.append(rel_palm)
                grips.append(grip)

        return np.concatenate(rel_palms, axis=0), np.concatenate(grips, axis=0)

    def get_fitted_nns(self):
        nns = self.get_static_nns()

        for obs_key in self.OBS_KEYS_TO_FIT:
            # pool this obs_key's raw values across every rb_id in
            # normalizer_fit_rb_ids that actually carries it -- e.g. sim's
            # zarr has palm_pose_xyz_rpy/gripper_value same as the real legs,
            # but a source missing a key entirely (rather than just having a
            # different range) is silently skipped for that key, same as the
            # original single-rb_id behavior's `if obs_key in rb` guard.
            pooled = []
            for rb_id in self.normalizer_fit_rb_ids:
                rb: ReplayBuffer = globals.REPLAY_BUFFER_LOADER[rb_id]  # type: ignore
                if obs_key in rb:
                    # rb[obs_key] (gripper_value) is 1-D [T] -- fit_nn's reshape(-1, dim)
                    # takes dim from the LAST axis, which for a 1-D array is T itself,
                    # not the 1-element feature dim. Add the trailing axis explicitly.
                    pooled.append(np.asarray(rb[obs_key][:])[:, None])
            if pooled:
                self.fit_nn(np.concatenate(pooled, axis=0), nns['obs'][obs_key], obs_key)

        # action is fit from get_all_action_components (already pools across
        # normalizer_fit_rb_ids), gated on the FIRST rb_id having both action
        # components -- same "does this cotrain even have palm/gripper" guard
        # as the original single-rb_id version, just checked once rather than
        # per source (all normalizer_fit_rb_ids members are expected to share
        # the same schema when they're being fit together in the first place).
        rb0: ReplayBuffer = globals.REPLAY_BUFFER_LOADER[self.normalizer_fit_rb_ids[0]]  # type: ignore
        if "palm_pose_xyz_rpy" in rb0 and "gripper_value" in rb0:
            rel_palm, grip = self.get_all_action_components()
            palm_nn = get_identity_normalizer_from_stat({'min': np.zeros(6, dtype=np.float32)})
            # 'gaussian', not 'limits' -- the action is what diffusion actually
            # noises/denoises; a target distribution closer to zero-mean/unit-
            # variance is a more consistent match for that Gaussian forward
            # process than one min/max-stretched to fill [-1, 1] by its
            # rarest outlier (obs keys stay 'limits', fit_nn's default, since
            # they're not what's being noised). See fit_nn's docstring for
            # the clip_sample interaction this trades in.
            self.fit_nn(rel_palm, palm_nn, "rel_palm_pose_xyz_rpy", mode='gaussian')

            grip_nn = get_identity_normalizer_from_stat({'min': np.zeros(1, dtype=np.float32)})
            self.fit_nn(grip[:, None], grip_nn, "gripper_value (action)", mode='gaussian')

            for param_name in ["scale", "offset"]:
                nns['action'].params_dict[param_name] = torch.cat([
                    palm_nn.params_dict[param_name], grip_nn.params_dict[param_name],
                ])
            for stat_name in ["min", "max", "mean", "std"]:
                nns['action'].params_dict['input_stats'][stat_name] = torch.cat([
                    palm_nn.params_dict['input_stats'][stat_name],
                    grip_nn.params_dict['input_stats'][stat_name],
                ])

        return nns


class TroubleshootClenchBatchLoader(BatchLoader):
    """
    BatchLoader for task 24 (erlenmeyer flask insertion).

    Zarr keys produced by gen_dataset.py:
      joint_commands        [T, 30]  — same layout, used as action
      wrist_cam_features    [T, 768] — DINOv2 ViT-B/14 CLS token
      biotac_lh             [T, D]   — already normalised to [0, 1]

    Joint limits in JOINT_LIMITS only cover 21 joints (no rf/lf), so both
    joint_states and joint_commands are fitted from data in get_fitted_nns.
    biotac_lh keeps its identity normalizer (already [0, 1]).
    
    For now, don't normalize the dinov2 features
    """

    def get_static_nns(self):
        """
        default identity normalizers
        
        nns['action'] is hard-coded in diffusion_model and must be used
        """
        nns = {}
        obs = {}
        obs_keys_to_load = globals.CONFIG.obs_keys_to_load  # type: ignore

        for obs_key in obs_keys_to_load:
            nb = globals.CONFIG.shape_meta[obs_key].shape  # type: ignore
            obs[obs_key] = get_identity_normalizer_from_stat(
                {'min': np.zeros(nb, dtype=np.float32)}
            )

        act_key = globals.CONFIG.action_key  # type: ignore
        nb_act = globals.CONFIG.shape_meta[act_key].shape  # type: ignore
        nns['action'] = get_identity_normalizer_from_stat(
            {'min': np.zeros(nb_act, dtype=np.float32)}
        )

        nns['obs'] = obs
        
        return nns

    def get_fitted_nns(self):
        nns = self.get_static_nns()

        rb: ReplayBuffer = globals.REPLAY_BUFFER_LOADER['all']  # type: ignore
        obs_keys_to_load: list = globals.CONFIG.obs_keys_to_load  # type: ignore
        act_key: str = globals.CONFIG.action_key  # type: ignore
        
        obs_keys_to_normalize = [
        ]

        for obs_key in obs_keys_to_normalize:
            assert obs_key in obs_keys_to_load, f"Observation key '{obs_key}' is not in the list of keys to load."
            self.fit_nn(rb[obs_key], nns['obs'][obs_key], obs_key)

        if act_key in rb:
            self.fit_nn(rb[act_key], nns['action'], act_key)

        return nns
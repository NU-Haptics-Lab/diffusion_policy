from typing import Optional
import numpy as np
import numba
import scipy.stats
import hydra
from diffusion_policy.common.replay_buffer import ReplayBuffer
import diffusion_policy.globals as globals
from diffusion_policy import utils
from collections import defaultdict
from tqdm import tqdm

import copy
from omegaconf import OmegaConf, open_dict
from operator import itemgetter

import diffusion_policy.globals as globals



def get_lower_bound_idx(sorted_array, value):
    # I believe value must be incremented by 1, otherwise we get an off-by-one error
    lb = np.searchsorted(sorted_array, value + 1)
    
    # must subtract by 1 because the sorted_array includes 0 as the first element
    lb2 = lb - 1
    
    # lb should be in [0, len(sorted_array)-1]
    return lb2
    

def get_history_indices(
    episode_ends, 
    index: int,
    history_indices
    ):
    """
    get the indices corresponding to the history of current index, as a function of a skip_amount.
    So, for example, if you want to provide the state history of the last two seconds, but spaced at 0.5s intervals, assuming the dataset frequency is 10hz and the current index is 100, then you'd call
    get_history_indices(ends, 100, 5, 3).
    
    Accounts for episode ends
    """
    indices = {}
    
    # find lower bound episode index
    lb_idx = get_lower_bound_idx(episode_ends, index)
    lb = episode_ends[lb_idx]
    
    for i in history_indices:
        j = index - i
        
        # valid index
        if j > lb:
            indices[i] = j
            
        # invalid index, is before the episode start
        else:
            indices[i] = None
            
    return indices

def get_next_index(
    episode_ends, 
    index: int,
    ):
    """
    Get the next index. Ensures it's not past the end of the episode, although this should be taken care of in `create_indices`
    """
    lb_idx = get_lower_bound_idx(episode_ends, index)
    ub_idx = lb_idx + 1
    
    if ub_idx > len(episode_ends):
        print("sampler_ql.get_next_index: Something went terribly wrong.")
        raise
        
    return index + 1

def get_not_done(
    episode_ends,
    index: int
    ):
    lb_idx = get_lower_bound_idx(episode_ends, index)
    ub_idx = lb_idx + 1
    
    ub_step_idx = episode_ends[ub_idx]
    
    # if this index is the 2nd to last index in this episode, then it's done (since it needs to get the next_state from the final index).
    if index == ub_step_idx - 1:
        not_done = False
    else:
        not_done = True
    
    return not_done

class Indices:
    """
    Class to generate training indices for each episode. Each episode has 1 instance of this class.
    
    
    """
    def __init__(self,
        rb_id: str,
        rb_offset: int,
        episode_end: int, 
        pad_before : int=0, 
        pad_after : int=0,
        debug : bool=True,
        ):
        self.rb_id = rb_id
        self.rb_offset = rb_offset
        self.episode_end = episode_end
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.debug = debug

        self.replay_buffer = globals.REPLAY_BUFFER_LOADER[self.rb_id]
        self.indices = []
        self.mask = None
        
    def create_indices(self, mask):
        """
        generate training indices based on padding before / after an episode and the mask.
        Positive and negative pad values are allowed
        """
        # make mapping from mask_indice to non-mask-indice
        self.mask = mask
        self.mask_indices = np.where(self.mask)[0]
        
        # set up start index
        start_idx = self.rb_offset

        # set up end index
        end_idx = self.episode_end

        # episode length is relative
        episode_length = end_idx - start_idx
        
        # ep length
        # self.ep_length = episode_length # SHOULDN"T be used
        self.mask_length = len(self.mask_indices)
        
        # # max future action
        # max_future_action = np.array(globals.CONFIG.action_rel_indices).max()

        # these are the training indices. Add padding before, add padding after
        self.train_indices = range(-self.pad_before, self.mask_length + self.pad_after)
        pass
        
    def get_mask_indices(self):
        """
        should ONLY be accessed by get_ep_idx_from_train_idx
        """
        return self.mask_indices
    
    def get_len_of_training_indices(self):
        return len(self.train_indices)
    
    def get_ep_idx_from_train_idx(self, train_idx):
        # must pass through the mask to get non-mask index
        nonmask_idx = self.get_mask_indices()[train_idx]
        return nonmask_idx

    def __len__(self):
        """
        This len includes padding. For a length that doesn't, use self.ep_length
        """
        return self.get_len_of_training_indices()
    
    # def key_zero_fill(self, key):
    #     input_arr = self.replay_buffer[key]
        
    #     data = np.zeros(shape=(self.sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
        
    #     return data

    def get_rb_indices(self, train_indices):
        """
        train_indices - mask-relative indices. Uses fill-back & fill-forward for indices out of bounds
        """
        # ensure it's numpy
        ti2 = np.array(train_indices).copy()

        # fill-back any indices less than zero
        mask = ti2 < 0
        ti2[mask] = 0

        # fill-forward any indices greater than mask length, minus one
        mask = ti2 > self.get_len_of_training_indices() - 1
        ti2[mask] = self.get_len_of_training_indices() - 1
        
        # get ep indices
        ep_indices = self.get_ep_idx_from_train_idx(ti2)

        # add on rb ep offset to make the indices rb-relative
        rb_indices = ep_indices + self.rb_offset

        return rb_indices
    
    def get_all_train_indices(self):
        return self.train_indices
    
    def get_all_rb_indices(self):
        train_indices = self.get_all_train_indices()
        all_rb_indices = self.get_rb_indices(train_indices)
        
        return all_rb_indices
    
    def get_sequence_by_train_indices_and_key(self, train_indices, key):
        """
        The caller requests data from `indices`, which may or may not exist in the episode this instance is associated with.
        If using the fill-back / fill-forward (default) option, then we first modify indices to be all valid indexes.
        We then use the indices to "fancy index" the r.b.

        indices - episode-relative indices. Meaning the only valid values are [0, len(episode)-1]
        """
        rb_indices = self.get_rb_indices(train_indices)

        # get this key's data from the r.b.
        assert(self.replay_buffer is not None)
        input_arr = self.replay_buffer[key]
        
        # # setup the item getter, more efficient than a for loop -- itemgetter doesn't maintain dimensions when the rb_indices is 1-long, so just use a for loop for simplicity
        # ig = itemgetter(*list(rb_indices))
        ls = []
        for i in rb_indices:

            ls.append(input_arr[i])

        # index the sample
        # sequence = np.array(ig(input_arr))
        sequence = np.array(ls)

        # we're done
        return sequence
    
    def get_all_key(self, key):
        indices = self.get_all_train_indices()
        
        seq = self.get_sequence_by_train_indices_and_key(indices, key)
        
        return seq


        
    
    # def get_sequence_by_key(self, idx, key):
    #     """
    #     given an index and a key, obtain the proper sequence of that index's data from the dataset.
    #     """
        
    #     # extract the buffer and sample start/end indices
    #     buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
    #         = self.indices[idx]
        
    #     # add on the episode's replay buffer offset
    #     buffer_start_idx += self.rb_offset
    #     buffer_end_idx   += self.rb_offset
    #     sample_start_idx += self.rb_offset
    #     sample_end_idx   += self.rb_offset
    
    #     # get this key's data from the r.b.
    #     input_arr = self.replay_buffer[key]

    #     # performance optimization, avoid small allocation if possible
    #     if key not in self.key_first_k:
    #         sample = input_arr[buffer_start_idx:buffer_end_idx]
    #     else:
    #         # performance optimization, only load used obs steps
    #         # number of data points
    #         n_data = buffer_end_idx - buffer_start_idx

    #         # key first k nb data points
    #         k_data = min(self.key_first_k[key], n_data)

    #         # fill value with Nan to catch bugs
    #         # the non-loaded region should never be used
    #         sample = np.full((n_data,) + input_arr.shape[1:], fill_value=np.nan, dtype=input_arr.dtype)

    #         # will throw if we try to access the non-loaded region
    #         try:
    #             # save the original data from the r.b. (SSD) into the sample in RAM
    #             sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
    #         except Exception as e:
    #             import pdb; pdb.set_trace()

    #     # save as data
    #     data = sample

    #     # padding before the sample start or after the sample end
    #     if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
    #         # reset data to be full of zeros
    #         data = self.key_zero_fill(key)

    #         # copy the first sample into the first sample_start_idx elements
    #         if sample_start_idx > 0:
    #             data[:sample_start_idx] = sample[0]

    #         # copy the last sample into the last `sequence_length - sample_end_idx` elements
    #         if sample_end_idx < self.sequence_length:
    #             data[sample_end_idx:] = sample[-1]

    #         # copy the sample into the correct elements, based off the sample start/end indices
    #         data[sample_start_idx:sample_end_idx] = sample
        
    #     return data
    
    # def get_history_by_key(self, indices, key):
    #     """
    #     Get the history of a key w.r.t input indices. Will return the specified history in a single array of structure [H, T, ...] --- where H == history, T == trajectory (a.k.a. sequence)
    #     """
    #     sequence = []

    #     for idx in indices:
    #         sequence.append(self.get_sequence_by_key(idx, key))

    #     # convert to np, basically adds a history dimension
    #     out = np.array(sequence)
    #     return out


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


    



class EpisodeSampler:
    """
    Get a sample from an episode
    """
    def __init__(self,
            # indices: Indices,
            # tr_offset,
            rb_id,
            rb_offset,
            rb_ep_end,
            mask,
            ):
        # self.tr_offset = tr_offset
        self.rb_id = rb_id
        self.rb_offset = rb_offset
        self.rb_ep_end = rb_ep_end
        self.mask = mask
        
        # my members
        self.qvals: np.ndarray = None # type:ignore
        
        # copy config for indices
        indices_cfg = copy.deepcopy(globals.CONFIG.common_indices) #type:ignore
        
        # update indices cfg
        with open_dict(indices_cfg):
            indices_cfg.rb_id = self.rb_id
            indices_cfg.rb_offset = int(self.rb_offset)
            indices_cfg.episode_end = int(self.rb_ep_end)

        # make using the config for this dataset. Could move this to the constructor?
        self.indices: Indices = hydra.utils.instantiate(indices_cfg)

        # make the training indices
        self.indices.create_indices(self.mask)

        # for effiency, only load the provided obs-keys
        self.obs_keys = globals.CONFIG.obs_keys_to_load #type:ignore

        self.action_key = "action"
        self.reward_key = "reward"

    def get_id(self, i):
        assert(i >= 0)
        assert(i < len(self))

        return i + self.rb_offset

    def __len__(self):
        return len(self.indices)
    
    def get(self, idx, key):
        raise
        return self.indices.get_sequence_by_indices_and_key(idx, key)
    
    def get_not_done(self, ep_idx):
        # last valid sample in the ep => second to last ep_idx, and don't forget python is zero-indexed.

        # sanity check: with 2 data-points [d1, d2], len(self) = 2, => ep_idx of 0 is done ... so we add 2
        done = ep_idx + 2 == len(self)
        not_done = not done
        
        # convert to np array, must add a dimension
        not_done = np.array([not_done])
        return not_done
    
    def get_reward(self, ep_idx):
        reward = self.get_key_sample("reward", ep_idx) # adds a dimension
        
        # TESTING -- reduce the existence penalty so I don't have to regen the dataset
        if False:
            reward[ reward < 0.0] = 0.0
        
        # convert to np array
        reward = np.array(reward)
        return reward

        return ep_indices

    # def get_history(self, key, idx):
    #     """
    #     ep_indices - relative indices within the episode
    #     """
    #     # get this key's data
    #     rb_key = self.rb[self.key]

    #     # access using numpy "fancy indexing". If zarr doesn't have that feature, just iterate through and concat
    #     data = rb_key[rb_indices]

    #     # done
    #     return data
    
    def get_obs_sample(self, ep_idx):
        # output dict
        sample = {}

        # make indices which are episode-relative
        indices = np.array(globals.CONFIG.obs_rel_indices) + ep_idx #type:ignore

        # iterate over obs keys
        for key in self.obs_keys:
            sample[key] = self.indices.get_sequence_by_train_indices_and_key(indices, key)
            
            # the haptics values can be fraught ......
            
        return sample
    
    def get_key_sample(self, key, ep_idx):
        """
        Get sequence by key
        """
        indices = [ep_idx]
        data = self.indices.get_sequence_by_train_indices_and_key(indices, key)
        return data
    
    def get_action_sample(self, ep_idx):
        """
        For an action, we want a sequence from ep_idx - n_obs_steps to ep_idx + horizon.
        """
        # make indices which are episode-relative
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx # type:ignore

        # get the sample
        sample = self.indices.get_sequence_by_train_indices_and_key(indices, "action")
        
        return sample
    
    def get_rb_index(self, ep_idx) -> np.ndarray:
        indices = [ep_idx]

        index = self.indices.get_rb_indices(indices)

        return np.array(index)
    
    def get_qval(self, ep_idx) -> np.ndarray:
        qval = self.qvals[ep_idx]

        assert(not np.isnan(qval))
        return np.array([qval])

    
    def get_sample(self, ep_idx):
        """
        return dict with keys (obs, action, reward, not_done, obs_next)
        """
        assert(ep_idx >= 0)
        # assert(ep_idx <= len(self)-2) # must be -2 since we get the next obs & action
        assert(ep_idx <= len(self)-1) # allow ep_idx to be == len(self)-1. In that case, obs_next will be a repeat of obs
        
        sample = {}
        sample["obs"] = self.get_obs_sample(ep_idx)
        
        # TODO: only load next obs (and action) if we're doing QL, otherwise it's a slowdown
        sample["obs_next"] = self.get_obs_sample(ep_idx + 1)

        sample["action"] = self.get_action_sample(ep_idx)
        sample["action_next"] = self.get_action_sample(ep_idx + 1)

        sample["reward"] = self.get_reward(ep_idx)

        sample["not_done"] = self.get_not_done(ep_idx)

        ## Meta data
        sample["rb_index"] = self.get_rb_index(ep_idx)
        
        # explicit q-val
        # sample["qval"] = self.get_qval(ep_idx)
        
        # ep len
        sample["ep_len"] = np.array([len(self)])

        return sample
    
    def get_all_rb_indices(self):
        return self.indices.get_all_rb_indices()
    
    def get_qvals(self):
        # get rewards
        indices = self.indices.get_all_train_indices()
        rewards = self.indices.get_sequence_by_train_indices_and_key(indices, "reward")
        
        qvals = []
        qval = 0.0
        
        discount = 0.975 # same as config
        
        for indice in reversed(indices):
            qval = rewards[indice] + discount * qval
            
            qvals.append(qval)
            
        qvals2 = np.array(qvals)
        
        # reverse
        qvals3 = np.flip(qvals2)
        
        self.qvals = qvals3
        
        assert(not np.any(np.isnan(self.qvals)))
        return qvals3
    
    def get_all_key(self, key):
        return self.indices.get_all_key(key)
        

class DatasetSampler:
    """
    This class takes a handle to a replay buffer and partitions it into episodes and data-points

    

    Here's the issue. In order to shuffle datasets, the data must be laid out in a single iterable. But the organization of our data is in episodes -> steps. So it makes sense to structure our sampler's around episode classes which contain step classes.

    But, to further compound the design problem, we use zarr to store our datasets on disk, which are linear in nature. We are only able to distinguish data in zarr into different episodes because of the `episode_ends` key in the zarr meta data.

    More complicated: the sampler provides the ability to pad the beginning or end of episodes, which means the training nb of datapoints may be different from the real nb of datapoints. I'll _rb_ to refer to replay buffer datapoints, and _tr_ to refer to training datapoints
    """
    def __init__(self,
            rb_id: str,
            ):
        # the dataset's aka replay-buffer
        self.rb_id = rb_id
        self.replay_buffer: ReplayBuffer = globals.REPLAY_BUFFER_LOADER[self.rb_id] #type:ignore
        
        self.initd = False
        self.ep_mask = None
        self.ep_samplers: list[EpisodeSampler] = []
        self.my_indices = []
        self.qvals = None
        self.inlier_mask = None
            
    def print_dataset_stats(self):
        # method vars
        per_ep = defaultdict(list)
        
        ## iterate over eps
        for ep in tqdm(self.ep_samplers, desc="print_dataset_stats"):
            s = ep.get_all_key('state')[:, 0:21]
            a = ep.get_all_key('action')
            r = ep.get_all_key('reward')
            
            j = utils.compute_jerk(s, a)
            per_ep['jerk'].append(j)
            per_ep['ep_len'].append(len(ep))
            
            total_r = r.sum()
            
            # successful?
            if total_r > 0.0:
                success = True
            else:
                success = False
                
            # success ep length
            if success:
                per_ep['success_ep_len'].append(len(ep))
            
        ##
        # compute averages
        for key, val in per_ep.items():
            mean, std, min, max = utils.compute_stats(val)
            
            print("{}--{} stats:".format(self.rb_id, key))
            print("\tmean: {}".format(mean))
            print("\tstd: {}".format(std))
            print("\tmin: {}".format(min))
            print("\tmax: {}".format(max))
            
        pass
            
            
            

    def Init(self,
              ep_mask
              ):
        self.ep_mask = ep_mask

        # episode classes
        self.ep_samplers: list[EpisodeSampler] = []

        # training episode ends. Copy from the ep sampler classes so we can use the efficient binary-search np.searchsorted method when converting from training index to episode
        self.tr_ep_offsets = []

        # make the inlier mask
        self.make_inliers()
        
        # compute / recompute episodes
        self.make_episodes()
        
        self.initd = True
        
        if False:
            self.print_dataset_stats()

    def make_episodes(self):
        """ Using the replay buffer's episode_ends, make episode sampler classes  """


        # first ep offset
        self.tr_ep_offsets = []
        rb_offset = 0
        tr_ep_offset = 0
        
        my_indices = np.array([])
        assert(self.inlier_mask is not None)

        # one episode sampler per episode
        for idx, episode_end in enumerate(self.replay_buffer.episode_ends): #type:ignore
            # if skip a.k.a. episode mask
            if self.ep_mask is None or self.ep_mask[idx]:
                # make the ep sampler
                ep_sampler = EpisodeSampler(
                    self.rb_id,
                    rb_offset,
                    episode_end,
                    self.inlier_mask[rb_offset:episode_end]
                )

                self.ep_samplers.append(ep_sampler)
                self.tr_ep_offsets.append(tr_ep_offset)

                # add the length of the training episode
                tr_ep_offset += len(ep_sampler)
                
                # save the indices
                my_indices = np.concatenate([my_indices, ep_sampler.get_all_rb_indices()])

            # set rb offset to the old episode_end
            rb_offset = episode_end
            
            
        # convert to np
        self.my_indices = np.array(my_indices, dtype=int)

    @property
    def episodes(self):
        return self.ep_samplers

    def get_episode_and_index(self, idx) -> tuple[EpisodeSampler, int]:
        # get the episode index
        ep_idx = get_lower_bound_idx(self.tr_ep_offsets, idx)

        # get the episode's training index offset
        tr_ep_offset = self.tr_ep_offsets[ep_idx]

        # get this ep's relative datapoint index
        dp_ep_idx = idx - tr_ep_offset

        # get the episode
        ep = self.ep_samplers[ep_idx]
        
        # never negative, never greater than len(ep)-1
        assert(dp_ep_idx >= 0)
        assert(dp_ep_idx <= len(ep)-1)
        
        return ep, dp_ep_idx

    def get_sample(self, ds_tr_idx: int) -> dict[str, np.ndarray]:
        """
        tr_idx - training dataset index
        
        return dict with keys (obs, action, reward, not_done, obs_next)
        """
        # convert absolute training idx into the episode and episode idx 
        ep, ep_tr_idx = self.get_episode_and_index(ds_tr_idx)

        # get the sample from the episode
        sample = ep.get_sample(ep_tr_idx)

        # we're done
        return sample

    def __len__(self):
        count = 0
        for ep in self.ep_samplers:
            count += len(ep)

        return count
    
    def get_key(self, key):
        assert(self.replay_buffer is not None)
        all_samples = self.replay_buffer[key]
        
        s = all_samples[self.my_indices]
        return s
    
    def make_qvals(self):
        qvals = np.array([])
        
        for ep_sampler in self.ep_samplers:
            qvals = np.concatenate([qvals, ep_sampler.get_qvals()])
            
        self.qvals = np.array(qvals)
        
    
    def get_qvals(self):
        # do this lazily since it's slow
        if self.qvals is None:
            self.make_qvals()
            
        return self.qvals
    
    def get_ep_lengths(self):
        l = [len(ep) for ep in self.ep_samplers]
        return l
    
    def get_ep_lengths_arr(self):
        l1 = self.get_ep_lengths()
        
        # ranges
        r = [np.array(range(l)) for l in l1]
        
        # expand 
        l2 = [(l.max()+1) * np.ones_like(l) for l in r]
        
        l3 = np.concatenate(l2)
        
        return l3
    
    def compute_stats(self, nb_std_devs = 3.5):
        # compute ds
        assert(self.replay_buffer is not None)
        s = self.replay_buffer['state']
        a = self.replay_buffer['action']
        
        nb_datapts = s.shape[0]
        nb_actions = a.shape[1]
        
        s2 = s[:, 0:nb_actions]
        
        ds = np.abs(s2-a)
        
        inlier_mask = np.ones((nb_datapts), dtype=np.bool_)
        
        for idx in range(nb_actions):
            ds2 = ds[:, idx]
        
            zscore = np.abs(scipy.stats.zscore(ds2))
            inliers = zscore < nb_std_devs
            
            # only keep inliers
            inlier_mask &= inliers
            
        # inlier-mask is now only the inliers for EVERY output action
        self.inlier_mask = inlier_mask
        
        print("{}: inlier_mask.sum(): {}".format(self.rb_id, self.inlier_mask.sum()))
        
    def calc_ds(self, inlier_mask):
        rb = self.replay_buffer
        rbs = np.array(rb['state'])
        rba = np.array(rb['action'])
        s = rbs[inlier_mask][:, 0:21]
        a = rba[inlier_mask]
        
        ds = np.abs(s - a) #type:ignore
        ds2 = ds.sum(axis=1)
        
        return ds2
        
    def make_inliers(self):
        """
        from data analysis, I've noticed that the largest jumps in joint state happen at the end of an episode ... so skip those ... I think there's a bug in my dataset generation script that's causing this.
        
        KEEP IN MIND: we train off trajectories ... not individual samples ... meaning that you can't simply cherrypick good/bad actions. You can only remove samples at the beginning / end of an episode.
        
        ANOTHER THING: ... Must turn this off if using q-learning, because then it won't see the reward (which is usually given on the final sample of an episode)
        """
        assert(self.replay_buffer is not None)
        
        lenrb = len(self.replay_buffer) #type:ignore
        
        inlier_mask = np.ones((lenrb), dtype=np.bool_)
        
        ends = np.array(self.replay_buffer.episode_ends)
        import matplotlib.pyplot as plt

        # init ds
        ds_before = self.calc_ds(inlier_mask)
        
        inlier_mask[ends - 1] = False
        
        ds_after = self.calc_ds(inlier_mask)
            
        # inlier-mask is now only the inliers for EVERY output action
        self.inlier_mask = inlier_mask
        
        print("{}: inlier_mask.sum(): {}".format(self.rb_id, self.inlier_mask.sum()))
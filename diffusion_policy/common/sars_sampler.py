from typing import Optional
import numpy as np
import numba
import hydra
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.globals import CONFIG
from diffusion_policy.globals import REPLAY_BUFFERS



def get_lower_bound_idx(sorted_array, value):
    lb = np.searchsorted(sorted_array, value) - 1
    return lb
    

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
    def __init__(self,
        rb_id: str,
        key_first_k,
        episode_length: int, 
        rb_offset: int,
        sequence_length : int, 
        episode_mask : np.ndarray,
        pad_before : int=0, 
        pad_after : int=0,
        debug : bool=True
        ):
        self.rb_id = rb_id
        self.key_first_k = key_first_k
        self.episode_length = episode_length
        self.rb_offset = rb_offset
        self.sequence_length = sequence_length
        self.episode_mask = episode_mask
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.debug = debug

        self.replay_buffer = REPLAY_BUFFERS[self.rb_id]
        self.indices = []
        
        
    def create_indices(self
        ) -> np.ndarray:
        """
        Iterate through the dataset episodes and corresponding r.b. indices to determine which indices correspond to which data-points.
        This is necessary because we're converting a dataset of waypoints into a dataset of trajectories, and we must be careful to only load the appropriate data.

        Output is list of lists of valid indices for each diffusion datapoint (a.k.a. a trajectory). So each element of the output `indices` is a list with [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx].

        buffer_start_idx - what data from the r.b. to start loading
        buffer_end_idx - what data from the r.b. to end loading
        sample_start_idx - how many additional datapoints to add before the start of the real data sequence
        sample_end_idx - how many additional datapoints to add after the end of the real data sequence
        """
        
        pad_before = min(max(self.pad_before, 0), self.sequence_length-1)
        pad_after = min(max(self.pad_after, 0), self.sequence_length-1)

        indices = list()
        valid_indices = list()

        # for R.L., we don't want to use the last datapoint in the episode because then we wouldn't have a valid next_state for the 2nd to last datapoint in the episode
        use_last_datapoint_in_episode = False
        
        if not self.episode_mask[i]:
            # skip episode
            return

        # set up start index
        start_idx = 0

        # set up end index
        end_idx = self.episode_length

        # episode length is relative
        episode_length = end_idx - start_idx
        
        # optional datapoint padding before the start of the episode, relative value
        min_start = -pad_before

        # since each sample is a trajectory with `sequence_length` waypoints, the last valid start index will be `sequence_length` indices before the end of the episode, plus an optional pad-after length.
        max_start = episode_length - self.sequence_length + pad_after

        # range end
        range_end = max_start+1
        
        # range stops one idx before end, so use max_start + 1.
        for idx in range(min_start, range_end):
            # buffer start corresponds to first real datapoint.
            buffer_start_idx = max(idx, 0) + start_idx

            # buffer end corresponds to the last real datapoint.
            buffer_end_idx = min(idx+ self.sequence_length, episode_length) + start_idx

            # start-offset is a relative value.
            start_offset = buffer_start_idx - (idx+start_idx)
            
            # end offset is a relative value.
            end_offset = (idx + self.sequence_length + start_idx) - buffer_end_idx

            # sample-start-idx is relative
            sample_start_idx = 0 + start_offset

            # sample-end-idx is relative
            sample_end_idx = self.sequence_length - end_offset

            # debug
            if self.debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)

            # add to the indices list
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
            
            def add_to_valid_indices(i, va):
                va.append(len(i) - 1)
            
            # add to the valid indices list
            if use_last_datapoint_in_episode:
                add_to_valid_indices(indices, valid_indices)

            # if we must skip the last datapoint in an episode
            else:
                # recall, range() ends one before the range-end value
                if idx == range_end - 1:
                    # skip it
                    pass
                # not at the end of the episode, so add to valid indices
                else:
                    add_to_valid_indices(indices, valid_indices)

                
        # convert to numpy
        indices = np.array(indices)

        # we're done
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def key_zero_fill(self, key):
        input_arr = self.replay_buffer[key]
        
        data = np.zeros(shape=(self.sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
        
        return data
    
    def get_sequence_by_key(self, idx, key):
        """
        given an index and a key, obtain the proper sequence of that index's data from the dataset.
        """
        
        # extract the buffer and sample start/end indices
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        
        # add on the episode replay buffer offset
        buffer_start_idx += self.rb_offset
        buffer_end_idx   += self.rb_offset
        sample_start_idx += self.rb_offset
        sample_end_idx   += self.rb_offset
    
        # get this key's data from the r.b.
        input_arr = self.replay_buffer[key]

        # performance optimization, avoid small allocation if possible
        if key not in self.key_first_k:
            sample = input_arr[buffer_start_idx:buffer_end_idx]
        else:
            # performance optimization, only load used obs steps
            # number of data points
            n_data = buffer_end_idx - buffer_start_idx

            # key first k nb data points
            k_data = min(self.key_first_k[key], n_data)

            # fill value with Nan to catch bugs
            # the non-loaded region should never be used
            sample = np.full((n_data,) + input_arr.shape[1:], fill_value=np.nan, dtype=input_arr.dtype)

            # will throw if we try to access the non-loaded region
            try:
                # save the original data from the r.b. (SSD) into the sample in RAM
                sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
            except Exception as e:
                import pdb; pdb.set_trace()

        # save as data
        data = sample

        # padding before the sample start or after the sample end
        if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
            # reset data to be full of zeros
            data = self.key_zero_fill(key)

            # copy the first sample into the first sample_start_idx elements
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]

            # copy the last sample into the last `sequence_length - sample_end_idx` elements
            if sample_end_idx < self.sequence_length:
                data[sample_end_idx:] = sample[-1]

            # copy the sample into the correct elements, based off the sample start/end indices
            data[sample_start_idx:sample_end_idx] = sample
        
        return data
    
    def get_history_by_key(self, indices, key):
        """
        Get the history of a key w.r.t input indices. Will return the specified history in a single array of structure [H, T, ...] --- where H == history, T == trajectory (a.k.a. sequence)
        """
        sequence = []

        for idx in indices:
            sequence.append(self.get_sequence_by_key(idx, key))

        # convert to np, basically adds a history dimension
        out = np.array(sequence)
        return out


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
    Get samples from an episode
    """
    def __init__(self,
            # indices: Indices,
            # tr_offset,
            rb_offset
            ):
        # self.tr_offset = tr_offset
        self.rb_offset = rb_offset

        # make using the config for this dataset
        self.indices: Indices = hydra.utils.instantiate(CONFIG.indices,
            rb_offset = self.rb_offset
                                                        )


        self.indices.create_indices()

        # hard-coded state keys, obtained from the rosbag-to-zarr dataset conversion script
        self.obs_keys = [
            "img",
            "img2",
            "state",
        ]

        self.action_key = "action"
        self.reward_key = "reward"

    def __len__(self):
        return len(self.indices)
    
    def get_not_done(self, ep_idx):
        # last valid sample in the ep => second to last ep_idx, and don't forget python is zero-indexed.

        # sanity check: with 2 data-points [d1, d2], len(self) = 2, => ep_idx of 0 is done ... so we add 2
        done = ep_idx + 2 == len(self)
        not_done = not done
        # not_done = np.array(not_done)
        return not_done
    
    def get_history_indices(self, ep_idx):
        """
        ep_idx is the current time tc
        """
        # something like this
        hist_arr = np.array([0, 5, 10])
        ep_indices = ep_idx - hist_arr

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
        # get the state episode indices
        ep_indices = self.get_history_indices(ep_idx)

        sample = {}

        # iterate over obs keys
        for key in self.obs_keys:
            sample[key] = self.indices.get_history_by_key(ep_indices, key)

        return sample
    
    def get_key_sample(self, key, ep_idx):
        data = self.indices.get_sequence_by_key(ep_idx, key)
        return data

    
    def get_sample(self, ep_idx):
        """
        return dict with keys (obs, action, reward, not_done, obs_next)
        """
        sample = {}
        sample["obs"] = self.get_obs_sample(ep_idx)
        sample["obs_next"] = self.get_obs_sample(ep_idx + 1)

        sample["action"] = self.get_key_sample(self.action_key, ep_idx)

        sample["reward"] = self.get_key_sample(self.reward_key, ep_idx)

        sample["not_done"] = self.get_not_done(ep_idx)

        return sample

class DatasetSampler:
    """
    Here's the issue. In order to shuffle datasets, the data must be laid out in a single iterable. But the organization of our data is in episodes -> steps. So it makes sense to structure our sampler's around episode classes which contain step classes.

    But, to further compound the design problem, we use zarr to store our datasets on disk, which are linear in nature. We are only able to distinguish data in zarr into different episodes because of the `episode_ends` key in the zarr meta data.

    More complicated: the sampler provides the ability to pad the beginning or end of episodes, which means the training nb of datapoints may be different from the real nb of datapoints. I'll _rb_ to refer to replay buffer datapoints, and _tr_ to refer to training datapoints
    """
    def __init__(self,
        rb_id: str, 
                 ):
        # the dataset's aka replay-buffer
        self.rb_id = rb_id
        self.replay_buffer = REPLAY_BUFFER_LOADER[self.rb_id]

        # episode classes
        self.ep_samplers = []

        # training episode ends. Copy from the ep sampler classes so we can use the efficient binary-search np.searchsorted method when converting from training index to episode
        self.tr_ep_offsets = []

        self.make_episodes()

    def make_episodes(self):
        """ Using the replay buffer's episode_ends, make episode sampler classes  """

        tr_ep_offset = 0

        # first ep offset
        self.tr_ep_offsets = [tr_ep_offset]

        # one episode sampler per episode
        for episode_end in self.replay_buffer.episode_ends:
            # make the ep sampler
            ep_sampler = EpisodeSampler(
                episode_end
            )

            # add the length of the training episode
            tr_ep_offset += len(ep_sampler)

            self.ep_samplers.append(ep_sampler)
            self.tr_ep_offsets.append(tr_ep_offset)

    def get_episode_and_index(self, idx):
        # get the episode index
        ep_idx = get_lower_bound_idx(self.tr_ep_offsets, idx)

        # get the episode's training index offset
        tr_ep_offset = self.tr_ep_offsets[ep_idx]

        # get this ep's relative datapoint index
        dp_ep_idx = idx - tr_ep_offset

        # get the episode
        ep = self.ep_samplers[ep_idx]
        
        return ep, dp_ep_idx

    def get_sample(self, tr_idx):
        """
        tr_idx - training dataset index
        """
        # convert absolute training idx into the episode and episode idx 
        ep, ep_idx = self.get_episode_and_index(tr_idx)

        # get the sample from the episode
        sample = ep.get_sample(ep_idx)

        # we're done
        return sample

    def __len__(self):
        count = 0
        for ep in self.ep_samplers:
            count += len(ep)

        return count
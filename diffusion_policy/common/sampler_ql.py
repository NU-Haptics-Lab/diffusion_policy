from typing import Optional
import numpy as np
import numba
from diffusion_policy.common.replay_buffer import ReplayBuffer

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

# @numba.jit(nopython=True) this is crashing the program
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True
    ) -> np.ndarray:
    
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    
    # iterate through each episode separately
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


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

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        history_indices=[]
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends=episode_ends, 
                sequence_length=sequence_length, 
                episode_mask=episode_mask,
                pad_before=pad_before, 
                pad_after=pad_after
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
        self.history_indices = history_indices
    
    def __len__(self):
        return len(self.indices)
    
    def key_zero_fill(self, key):
        input_arr = self.replay_buffer[key]
        
        data = np.zeros(shape=(self.sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
        
        return data
    
    def get_sequence_by_key(self, idx, key):
        """
        given an index and a key, obtain the proper sequence of that key's data from the dataset.
        """
        
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
    
        input_arr = self.replay_buffer[key]
        # performance optimization, avoid small allocation if possible
        if key not in self.key_first_k:
            sample = input_arr[buffer_start_idx:buffer_end_idx]
        else:
            # performance optimization, only load used obs steps
            n_data = buffer_end_idx - buffer_start_idx
            k_data = min(self.key_first_k[key], n_data)
            # fill value with Nan to catch bugs
            # the non-loaded region should never be used
            sample = np.full((n_data,) + input_arr.shape[1:], 
                fill_value=np.nan, dtype=input_arr.dtype)
            try:
                sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
            except Exception as e:
                import pdb; pdb.set_trace()
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
            # fill with zeros
            data = self.key_zero_fill(key)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < self.sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        
        return data
        
    def sample_sequence(self, idx):
        """
        Given an index, we want to return a sequence of data corresponding to that index. For B.C. we only need a sequence of (s, a), but for Q.L., we need a sequence of (s, a) a.k.a. the current state & current action, the single (r, not_done) values a.k.a. the current reward and current done, and a sequence of (s') a.k.a. the next state.
        
        Instead of saving s' with each s, which would about double the dataset size, let's be clever and simply save which indices correspond to s' when obtaining the indices for index idx.
        
        Furthermore, we may want to input a non-contiguous state history into our networks, so we want to have those indices available too.
        """
        
        result = dict()
        
        # hard-coded state keys, obtained from the rosbag-to-zarr dataset conversion script
        state_keys = [
            "img",
            "img2",
            "state",
        ]
        
        # get data for all keys for this index
        for key in self.keys:
            result[key] = self.get_sequence_by_key(idx, key)
            
        # get idx of next state
        idx_next = get_next_index(self.replay_buffer.episode_ends, idx)
            
        # get data for the next state
        for state in state_keys:
            result[state + "_next"] = self.get_sequence_by_key(idx_next, state)
            
        # get indices from the history
        history_indices = get_history_indices(self.replay_buffer.episode_ends, idx, self.history_indices)
        
        # get data from the history. Keys should be {3, 2, 1} for example, values should be {300, 305, 310} for example
        for key, value in enumerate(history_indices):
            # make the suffix
            suffix = "_" + str(key)
            
            # iterate over state keys in the dataset
            for state in state_keys:
                # invalid value means we have to pad using all zeros
                if value is None:
                    result[state + suffix] = self.key_zero_fill(key)
                    
                # valid data index
                else:
                    result[state + suffix] = self.get_sequence_by_key(idx_next, state)
            
        
        
        # we're done
        return result

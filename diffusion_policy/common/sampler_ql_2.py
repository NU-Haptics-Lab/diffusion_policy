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

def create_indices_once(
        epis
)

# @numba.jit(nopython=True) this is crashing the program
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, 
    pad_after: int=0,
    debug:bool=True
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
    
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    valid_indices = list()

    # for R.L., we don't want to use the last datapoint in the episode because then we wouldn't have a valid next_state for the 2nd to last datapoint in the episode
    use_last_datapoint_in_episode = False
    
    # iterate through each episode separately
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue

        # set up start index, which is absolute w.r.t. r.b.
        if i > 0:
            start_idx = episode_ends[i-1]
        else:
            start_idx = 0

        # set up end index, which is absolute w.r.t. r.b.
        end_idx = episode_ends[i]

        # episode length is relative
        episode_length = end_idx - start_idx
        
        # optional datapoint padding before the start of the episode, relative value
        min_start = -pad_before

        # since each sample is a trajectory with `sequence_length` waypoints, the last valid start index will be `sequence_length` indices before the end of the episode, plus an optional pad-after length. relative value
        max_start = episode_length - sequence_length + pad_after

        # range end
        range_end = max_start+1
        
        # range stops one idx before end, so use max_start + 1. idx is relative
        for idx in range(min_start, range_end):
            # buffer start corresponds to first real datapoint. Absolute value
            buffer_start_idx = max(idx, 0) + start_idx

            # buffer end corresponds to the last real datapoint. Absolute value
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx

            # start-offset is a relative value.
            start_offset = buffer_start_idx - (idx+start_idx)
            
            # end offset is a relative value.
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx

            # sample-start-idx is relative
            sample_start_idx = 0 + start_offset

            # sample-end-idx is relative
            sample_end_idx = sequence_length - end_offset

            # debug
            if debug:
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

class KeyStepSampler:
    """
    Sampler for a specific datapoint for a specific key. Allows 
    """
    def __init__(self, key, rb, rb_ep_idx):
        self.key = key
        self.rb = rb
        self.rb_ep_idx = rb_ep_idx

    def get_sample(self, ep_indices):
        """
        ep_indices - relative indices within the episode
        """
        # get this key's data
        rb_key = self.rb[self.key]

        # not yet supported
        if np.any(ep_indices < 0):
            raise

        # convert to replay buffer indices
        rb_indices = self.rb_ep_idx + ep_indices

        # access using numpy "fancy indexing". If zarr doesn't have that feature, just iterate through and concat
        data = rb_key[rb_indices]

        # done
        return data


# class StepSampler:
#     def __init__(self):
#         buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx

class EpisodeSampler:
    def __init__(self):
        self.indices = <>

        # hard-coded state keys, obtained from the rosbag-to-zarr dataset conversion script
        self.state_keys = [
            "img",
            "img2",
            "state",
        ]

    def __len__(self):
        return len(self.indices)
    
    def get_history_indices(self, ep_idx):
        """
        ep_idx is the current time tc
        """
        # something like this
        hist_arr = np.array([0, 5, 10])
        ep_indices = ep_idx - hist_arr

        return ep_indices
    
    def get_sample(self, ep_idx):
        """
        return dict with  (s, a, r, not_done, s')
        """
        # get the state episode indices
        ep_indices = self.get_history_indices(ep_idx)

        state_sample = {}

        # iterate over state keys
        for keystepsampler in self.key_step_samplers:
            skey = keystepsampler.key
            state_sample[skey] = keystepsampler.get_sample(ep_indices)



class SequenceSampler:
    """
    Here's the issue. In order to shuffle datasets, the data must be laid out in a single iterable. But the organization of our data is in episodes -> steps. So it makes sense to structure our sampler's around episode classes which contain step classes.

    But, to further compound the design problem, we use zarr to store our datasets on disk, which are linear in nature. We are only able to distinguish data in zarr into different episodes because of the `episode_ends` key in the zarr meta data.

    More complicated: the sampler provides the ability to pad the beginning or end of episodes, which means the training nb of datapoints may be different from the real nb of datapoints. I'll _rb_ to refer to replay buffer datapoints, and _tr_ to refer to training datapoints
    """
    def __init__(self,
        replay_buffer: ReplayBuffer, 
                 ):
        # the dataset's aka replay-buffer
        self.replay_buffer = replay_buffer

        # the training data-
        self.episode_ends = 

    def make_episodes(self):
        """ Using the replay buffer's episode_ends, make episode sampler classes  """

        for episode_end in self.replay_buffer.episode_ends:
            # make the ep sampler
            ep_sampler = EpisodeSampler()

    def get_episode_and_index(self, idx):
        # get the episode index
        ep_idx = get_lower_bound_idx(self.episode_ends, idx)

        # get the episode's replay buffer start index
        ep_rb_idx = self.episode_ends[ep_idx]

        # get this ep's relative datapoint index
        ep_dp_idx = idx - ep_rb_idx
        


    def get_sample(self, tr_idx):
        """
        tr_idx - training dataset index
        """
        # convert absolute idx into 
        ep, ep_idx = self.get_episode_and_index(tr_idx)

        # get the sample from the episode
        sample = ep.get_sample(ep_idx)

        # we're done
        return sample


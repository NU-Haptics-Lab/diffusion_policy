import numpy as np
import diffusion_policy.globals as globals
from diffusion_policy.common.sarsa_sampler import EpisodeSampler


class HondaSampler(EpisodeSampler):
    """
    Get a sample from an episode for the erlenmeyer flask BC policy.
    """

    def get_action_trajectory(self, ep_idx, key):
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx  # type: ignore
        return self.indices.get_sequence_by_train_indices_and_key(indices, key)

    def get_obs_sample(self, ep_idx):
        obs_sample = {}
        for obs_key in globals.CONFIG.obs_keys_to_load:  # type: ignore
            obs_sample[obs_key] = self.get_key_sample(obs_key, ep_idx)
        return obs_sample

    def get_sample(self, ep_idx):
        """
        note: nns['action'] is hard-coded in diffusion_model and must be used
        """
        
        assert ep_idx >= 0
        assert ep_idx <= len(self) - 1

        sample = {}

        sample["obs"] = self.get_obs_sample(ep_idx)

        act_key = globals.CONFIG.action_key  # type: ignore
        sample["action"] = self.get_action_trajectory(ep_idx, act_key)
        
        
        sample["task_id"] = self.get_key_sample("task_id", ep_idx)

        return sample


class Honda1Sampler(HondaSampler):
    """
    HondaSampler + subtask_id + a progress token input, for honda_1's
    full_task_rotate_lid.zarr specifically -- not all honda zarrs have a
    subtask_id key, so this stays separate from HondaSampler.

    progress is this sample's fraction of the way through its episode
    (0.0 at the first valid index, 1.0 at the last). It's only meaningful for
    full-task episodes (subtask_id == FULL_TASK_SUBTASK_ID); subtask-only
    episodes get progress_valid = 0.0, so the model's progress token (see
    DexNexTransformerForDiffusion's use_progress_token) gets masked out for
    them -- the numeric progress value is still computed either way, it's
    just not meaningful/used when invalid.
    """
    FULL_TASK_SUBTASK_ID = -1

    def get_sample(self, ep_idx):
        sample = super().get_sample(ep_idx)

        subtask_id = self.get_key_sample("subtask_id", ep_idx)
        sample["subtask_id"] = subtask_id

        progress = ep_idx / max(len(self) - 1, 1)
        is_full_task = bool(np.all(subtask_id == self.FULL_TASK_SUBTASK_ID))
        sample["progress"] = np.array([progress], dtype=np.float32)
        sample["progress_valid"] = np.array([1.0 if is_full_task else 0.0], dtype=np.float32)

        return sample
